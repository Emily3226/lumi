"""
rag/contest_ingestor.py

Contest-aware PDF ingestion for Waterloo Math contests.

Label-detection strategy (v3):
  Instead of an x-threshold + gap-clustering approach we now use
  "x-column pinning":
    1. Collect every span that looks like "N." with N in [1, expected]
       and x in the left 25% of the page, skipping instruction pages.
    2. Find the single dominant x-column via median of all candidates.
    3. Accept only spans within ±COL_TOL pts of that median.
  This is robust because every contest PDF typesets all problem numbers
  in exactly one column; the median is never corrupted by a stray match.

CIMC/CSMC split:
  Part A labels (1-6) are found before the "Part B" header.
  Part B labels (1-3) are found after it and renumbered +6 → stored as Q7-9.
  x-column detection uses only Part A pages so Part B can't add noise.
"""

from __future__ import annotations

import hashlib
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fitz


# ── Topic tagging ─────────────────────────────────────────────────────────────

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "algebra": ["algebra", "equation", "polynomial", "quadratic", "linear",
                "expression", "factor", "expand", "simplify", "solve for", "roots"],
    "number_theory": ["integer", "divisor", "prime", "modulo", "remainder", "gcd",
                      "lcm", "digit", "divisible", "multiple", "congruent"],
    "geometry": ["triangle", "circle", "rectangle", "polygon", "angle", "area",
                 "perimeter", "radius", "chord", "tangent", "parallel",
                 "perpendicular", "coordinate", "quadrilateral", "equilateral",
                 "isosceles", "hypotenuse", "diagonal"],
    "combinatorics": ["combination", "permutation", "probability", "count",
                      "arrangement", "choose", "ways", "paths", "pattern"],
    # Waterloo contest labels should avoid calculus; function-style problems are
    # better grouped under algebra unless there is a stronger topical signal.
    "sequences": ["arithmetic sequence", "geometric sequence", "sequence", "series",
                  "nth term", "n-th term", "recurrence", "fibonacci", "progression",
                  "common difference", "common ratio"],
    "inequalities": ["inequality", "maximum", "minimum", "optimize", "bound",
                     "absolute value"],
    "trigonometry": ["sine", "cosine", "tangent", r"\bsin\b", r"\bcos\b",
                     r"\btan\b", "radian", "degree", "trigonometric", "identity"],
    "logic": ["if and only if", "prove", "proof", "show that", "suppose",
              "contradiction", "induction"],
}


def tag_topics(text: str) -> list[str]:
    lower = text.lower()
    found: list[str] = []

    for topic, kws in TOPIC_KEYWORDS.items():
        hits = 0
        for k in kws:
            if re.search(k, lower):
                hits += 1

        if hits == 0:
            continue

        # Use stricter thresholds for historically noisy labels.
        if topic == "geometry" and hits < 2:
            continue
        if topic == "sequences":
            # Avoid false positives from generic words like "series" in prose.
            strong_seq = (
                "sequence" in lower
                or "recurrence" in lower
                or "fibonacci" in lower
                or "common ratio" in lower
                or "common difference" in lower
                or "n-th term" in lower
                or "nth term" in lower
            )
            if not strong_seq:
                continue

        found.append(topic)

    if not found:
        return ["algebra"]
    return found


# ── Contest metadata ──────────────────────────────────────────────────────────

FOLDER_CONTESTS: dict[str, list[str]] = {
    "CSIMC": ["CIMC", "CSMC"],
    "Euclid": ["Euclid"],
    "FGH": ["Fryer", "Galois", "Hypatia"],
    "Gauss": ["Gauss7", "Gauss8", "Gauss"],
    "PCF": ["Pascal", "Cayley", "Fermat"],
}

CONTEST_GRADES: dict[str, list[int]] = {
    "CIMC": [11, 12], "CSMC": [11, 12], "Euclid": [12],
    "Fryer": [9], "Galois": [10], "Hypatia": [11],
    "Gauss7": [7], "Gauss8": [8],
    "Pascal": [9], "Cayley": [10], "Fermat": [11],
}

CONTEST_QUESTION_COUNT: dict[str, int] = {
    "Euclid": 10,
    "Fryer": 4, "Galois": 4, "Hypatia": 4,
    "CIMC": 9, "CSMC": 9,
    "Gauss7": 25, "Gauss8": 25,
    "Pascal": 25, "Cayley": 25, "Fermat": 25,
}

# Tolerance (pts) for matching a span's x to the pinned column.
# Left-margin fraction: any label span left of this fraction of page width is a candidate
_LABEL_LEFT_FRAC = 0.22


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ContestFile:
    path: Path
    contest: str
    year: int
    is_solution: bool
    folder: str


@dataclass
class ProblemLocation:
    prob_num: int
    page_index: int   # 0-based
    y: float          # top-of-label y in PDF points
    x: float          # left edge of label
    font_size: float  # approximate label font size (used for crop trimming)


@dataclass
class ProblemChunk:
    chunk_id: str
    contest: str
    year: int
    problem_number: int
    problem_text: str
    solution_text: str
    topics: list[str]
    grades: list[int]
    source_file: str
    pdf_path: str
    solution_pdf_path: str
    page_number: int
    solution_page_number: int
    has_diagram: bool

    def to_document(self) -> str:
        return "\n".join([
            f"Contest: {self.contest} {self.year}",
            f"Problem {self.problem_number}",
            f"Topics: {', '.join(self.topics) or 'general'}",
            "",
            self.problem_text,
        ])

    def metadata(self) -> dict:
        return {
            "contest": self.contest,
            "year": str(self.year),
            "problem_number": str(self.problem_number),
            "topics": ",".join(self.topics),
            "grades": ",".join(str(g) for g in self.grades),
            "source_file": self.source_file,
            "pdf_path": self.pdf_path,
            "solution_pdf_path": self.solution_pdf_path,
            "page_number": str(self.page_number),
            "solution_page_number": str(self.solution_page_number),
            "has_solution": str(bool(self.solution_text)),
            "has_diagram": str(self.has_diagram),
            # Store solution text directly so the agent can retrieve it
            # without re-reading the PDF. Truncated to 8000 chars to stay
            # within ChromaDB metadata value limits.
            "solution_text": (self.solution_text or "")[:8000],
        }


# ── Core PDF utilities ────────────────────────────────────────────────────────

def _content_pages(doc: fitz.Document) -> list[int]:
    """
    0-based page indices to search for problems.
    Always skips page 0 (cover).
    Only skips the last page if it has no problem-number spans in left 25%.
    """
    n = len(doc)
    if n <= 1:
        return []

    last = doc[n - 1]
    pw = last.rect.width
    last_has_problems = any(
        re.match(r"^\d{1,2}\.$", span["text"].strip())
        and span["origin"][0] < pw * 0.25
        for block in last.get_text("dict")["blocks"]
        if block.get("type") == 0
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    )

    end = n if last_has_problems else n - 1
    return list(range(1, end))


def _is_instruction_page(doc: fitz.Document, page_index: int) -> bool:
    """True if this page is a cover/instructions page rather than a problem page."""
    if page_index == 0:
        return True
    page = doc[page_index]
    top_clip = fitz.Rect(0, 0, page.rect.width, 200)
    top_text = page.get_text("text", clip=top_clip).strip()
    markers = [
        "NOTE:", "Please read the instructions", "A Note about Bubbling",
        "Write all answers in the answer booklet", "Number of questions:",
    ]
    return any(m in top_text for m in markers)


def _find_part_b_start(doc: fitz.Document) -> tuple[int, float] | None:
    """Return (page_index, y) of the Part B header in a CIMC/CSMC PDF."""
    terms = ("Part B", "PART B", "Part B:", "Section B", "PARTIE B", "Partie B")
    for pi in _content_pages(doc):
        page = doc[pi]
        hits = []
        for term in terms:
            hits.extend(page.search_for(term))
        if hits:
            return pi, sorted(hits, key=lambda r: r.y0)[0].y0
    return None


# ── Label candidate collection ───────────────────────────────────────────────

def _collect_label_candidates(
    doc: fitz.Document,
    expected: int,
    page_indices: list[int],
    after: tuple[int, float] | None = None,
    before: tuple[int, float] | None = None,
) -> list[tuple[int, int, float, float, float]]:
    """
    Collect ALL spans that look like problem-number labels (matching "N." or "NN.")
    with number in [1, expected] found in the left _LABEL_LEFT_FRAC of the page.

    Returns list of (prob_num, page_index, y, x, font_size) tuples sorted by
    (page_index, y), with instruction-page items filtered out.
    """
    candidates = []

    for pi in page_indices:
        page = doc[pi]
        is_instr = _is_instruction_page(doc, pi)
        ph = page.rect.height
        pw = page.rect.width

        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    t = span["text"].strip()
                    x = span["origin"][0]
                    y = span["origin"][1]
                    fs = span.get("size", 10.0)

                    if not re.match(r"^\d{1,2}\.$", t):
                        continue
                    n = int(t.rstrip("."))
                    if n < 1 or n > expected:
                        continue
                    if x >= pw * _LABEL_LEFT_FRAC:
                        continue
                    # On instruction pages only accept labels in the lower 60%
                    if is_instr and y < ph * 0.4:
                        continue
                    # Boundary filters
                    if after is not None:
                        ap, ay = after
                        if pi < ap or (pi == ap and y <= ay):
                            continue
                    if before is not None:
                        bp, by = before
                        if pi > bp or (pi == bp and y >= by):
                            continue

                    candidates.append((n, pi, y, x, fs))

    # ── x-column pinning ─────────────────────────────────────────────────────
    # All real problem labels share a single left-margin column.  Any stray
    # match (e.g. a numbered list item indented further right) will have a
    # noticeably different x.  Compute the median x and reject outliers.
    if len(candidates) >= 3:
        import statistics as _stats
        median_x = _stats.median(c[3] for c in candidates)
        _COL_TOL = 18  # pts — allows for minor font/indent variation
        candidates = [c for c in candidates if abs(c[3] - median_x) <= _COL_TOL]

    return sorted(candidates, key=lambda c: (c[1], c[2]))


# ── Label scanning ────────────────────────────────────────────────────────────

def _scan_labels(
    doc: fitz.Document,
    expected: int,
    col_x: float,          # kept for API compat but ignored — we use left-margin filter
    page_indices: list[int],
    after: tuple[int, float] | None = None,
    before: tuple[int, float] | None = None,
) -> list[ProblemLocation]:
    """
    Find the first occurrence of each problem number label in [1, expected].
    Uses _collect_label_candidates (left-margin filter) then picks first
    occurrence of each number in document order.
    col_x is accepted for API compatibility but not used.
    """
    candidates = _collect_label_candidates(doc, expected, page_indices, after, before)

    found: dict[int, ProblemLocation] = {}
    for n, pi, y, x, fs in candidates:
        if n not in found:
            found[n] = ProblemLocation(prob_num=n, page_index=pi, y=y, x=x, font_size=fs)

    return sorted(found.values(), key=lambda loc: (loc.page_index, loc.y))


def _find_problem_labels(
    doc: fitz.Document,
    expected: int,
    col_x: float | None = None,  # kept for API compat, ignored
) -> list[ProblemLocation]:
    """
    Find all problem labels for a standard (non-CIMC) contest PDF.
    Uses left-margin candidate collection — no column pinning needed.
    """
    pages = _content_pages(doc)
    return _scan_labels(doc, expected, 0.0, pages)


def _find_csimc_labels(doc: fitz.Document) -> list[ProblemLocation]:
    """
    Find all 9 problem labels for a CIMC/CSMC PDF using Part A / Part B split.

    Part A: problems 1-6 (before the Part B header)
    Part B: problems 1-3 (after the Part B header) → renumbered as 7-9
    """
    part_b = _find_part_b_start(doc)
    pages = _content_pages(doc)

    # Part A labels: numbers 1-6, before Part B header
    labels_a = _scan_labels(
        doc, 6, 0.0, pages,
        before=part_b,
    )

    # Part B labels: numbers 1-3, after Part B header → renumber +6
    if part_b:
        labels_b_raw = _scan_labels(
            doc, 3, 0.0, pages,
            after=part_b,
        )
        labels_b = [
            ProblemLocation(
                prob_num=loc.prob_num + 6,
                page_index=loc.page_index,
                y=loc.y,
                x=loc.x,
                font_size=loc.font_size,
            )
            for loc in labels_b_raw
        ]
    else:
        labels_b = []

    return sorted(labels_a + labels_b, key=lambda loc: (loc.page_index, loc.y))


# Public alias used by the image router
def get_csimc_labels(doc: fitz.Document) -> dict[int, ProblemLocation]:
    return {loc.prob_num: loc for loc in _find_csimc_labels(doc)}


def _trim_solution_from_problem(text: str) -> str:
    """
    If a problem text accidentally contains solution content, trim it.
    Solutions usually start with "Solution:" or "Answer:" on a new line.
    """
    lower = text.lower()
    # Look for solution markers at the start of a line
    for marker in ["\nsolution:", "\nanswer:", "\nofficial solution:", "\nsol:"]:
        idx = lower.find(marker)
        if idx > 0:
            return text[:idx].strip()
    return text


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_text_between(
    doc: fitz.Document,
    start: ProblemLocation,
    end: ProblemLocation | None,
) -> str:
    parts = []
    content = set(_content_pages(doc))

    if end is None or end.page_index > start.page_index:
        page = doc[start.page_index]
        clip = fitz.Rect(0, start.y, page.rect.width, page.rect.height)
        parts.append(page.get_text("text", clip=clip))

        if end is not None:
            for pi in range(start.page_index + 1, end.page_index):
                if pi in content:
                    parts.append(doc[pi].get_text("text"))
            end_page = doc[end.page_index]
            clip2 = fitz.Rect(0, 0, end_page.rect.width, end.y)
            parts.append(end_page.get_text("text", clip=clip2))
        else:
            for pi in range(start.page_index + 1, len(doc) - 1):
                if pi in content:
                    parts.append(doc[pi].get_text("text"))
    else:
        page = doc[start.page_index]
        clip = fitz.Rect(0, start.y, page.rect.width, end.y)
        parts.append(page.get_text("text", clip=clip))

    return "\n".join(parts).strip()


def _page_has_diagram(doc: fitz.Document, page_index: int) -> bool:
    page = doc[page_index]
    return bool(page.get_drawings() or page.get_images())


def _extract_problems(
    doc: fitz.Document,
    contest: str,
) -> list[tuple[int, str]]:
    if contest in ("CIMC", "CSMC"):
        labels = _find_csimc_labels(doc)
    else:
        expected = CONTEST_QUESTION_COUNT.get(contest, 25)
        labels = _find_problem_labels(doc, expected)

    problems = []
    for i, loc in enumerate(labels):
        next_loc = labels[i + 1] if i + 1 < len(labels) else None
        raw_text = _extract_text_between(doc, loc, next_loc)
        # Trim any accidental solution content from the problem text
        clean_text = _trim_solution_from_problem(raw_text)
        problems.append((loc.prob_num, clean_text))
    return problems


# ── File discovery & pairing ──────────────────────────────────────────────────

def _parse_filename(filename: str) -> tuple[str, int, bool] | None:
    stem = Path(filename).stem

    # Solution files end with "Solution" — this is the authoritative check
    is_solution = filename.endswith("Solution.pdf") or stem.endswith("Solution")

    # Must start with a 4-digit year
    m = re.match(r"^(\d{4})", stem)
    if not m:
        return None
    year = int(m.group(1))

    # Strip year, then strip trailing/leading contest+solution type words
    remainder = stem[4:]
    remainder = re.sub(
        r"(?i)\s*[\-_]?\s*(solution|solutions|sol|answers?|exam)\s*$",
        "", remainder,
    ).strip(" -_")

    all_names = sorted(
        [n for names in FOLDER_CONTESTS.values() for n in names],
        key=len, reverse=True,
    )
    contest = None
    for name in all_names:
        if re.search(re.escape(name), remainder, re.I):
            contest = name
            break

    if contest == "Gauss":
        gm = re.search(r"Gauss\s*([78])", remainder, re.I)
        if gm:
            contest = f"Gauss{gm.group(1)}"

    return (contest, year, is_solution) if contest else None


def discover_contest_files(pdf_root: Path) -> list[ContestFile]:
    files: list[ContestFile] = []
    for folder_name in FOLDER_CONTESTS:
        folder_path = pdf_root / folder_name
        if not folder_path.exists():
            continue
        for pdf_file in sorted(folder_path.glob("*.pdf")):
            parsed = _parse_filename(pdf_file.name)
            if parsed:
                contest, year, is_solution = parsed
                files.append(ContestFile(
                    path=pdf_file, contest=contest, year=year,
                    is_solution=is_solution, folder=folder_name,
                ))
    return files


def pair_contest_files(
    files: list[ContestFile],
) -> Iterator[tuple[str, ContestFile | None, ContestFile | None]]:
    index: dict[tuple[str, int], dict[str, ContestFile]] = {}
    gauss_shared: dict[int, ContestFile] = {}

    for f in files:
        key = (f.contest, f.year)
        index.setdefault(key, {})
        if f.is_solution:
            index[key]["solution"] = f
            if f.contest == "Gauss":
                gauss_shared[f.year] = f
        else:
            index[key]["contest"] = f

    for variant in ("Gauss7", "Gauss8"):
        for (contest, year), group in index.items():
            if contest == variant and "solution" not in group:
                shared = gauss_shared.get(year)
                if shared:
                    group["solution"] = shared

    for key, group in sorted(index.items()):
        contest_name, _ = key
        if contest_name == "Gauss" and "contest" not in group:
            continue
        yield contest_name, group.get("contest"), group.get("solution")


# ── Main ingestion ────────────────────────────────────────────────────────────

def ingest_pair(
    contest_file: ContestFile | None,
    solution_file: ContestFile | None,
    contest_name: str | None = None,
) -> list[ProblemChunk]:
    ref = contest_file or solution_file
    assert ref is not None
    canonical = contest_name or ref.contest

    contest_doc = fitz.open(str(contest_file.path)) if contest_file else None
    solution_doc = fitz.open(str(solution_file.path)) if solution_file else None

    # Extract problems ONLY from the actual problem PDF, never from solution PDF
    # This ensures problems show correctly (not solutions)
    contest_problems = _extract_problems(contest_doc, canonical) if contest_doc else []
    solution_problems = _extract_problems(solution_doc, canonical) if solution_doc else []

    # Debug: print which files are being used
    if contest_file:
        print(f"    → Problem PDF: {contest_file.path.name}")
    else:
        print(f"    ⚠ No problem PDF found")
    if solution_file:
        print(f"    → Solution PDF: {solution_file.path.name}")

    # pdf_path must ONLY point to the problem PDF, never to solution PDF
    # If we don't have a problem PDF, problems won't render but we won't show solutions either
    #
    # Rather than storing an absolute local path (which only works on the
    # machine ingestion ran on), upload the PDF into MongoDB GridFS and
    # store a stable logical key ("<folder>/<filename>.pdf"). Serving code
    # (api/contest_image_router.py) resolves this key back to a local file
    # via rag/mongo_pdf_store.get_local_path(), downloading once and then
    # caching on disk.
    from rag.mongo_pdf_store import upload_pdf

    if contest_file:
        pdf_path_str = f"{contest_file.folder}/{contest_file.path.name}"
        upload_pdf(contest_file.path, pdf_path_str)
    else:
        pdf_path_str = ""

    if solution_file:
        sol_pdf_path_str = f"{solution_file.folder}/{solution_file.path.name}"
        upload_pdf(solution_file.path, sol_pdf_path_str)
    else:
        sol_pdf_path_str = ""

    expected = CONTEST_QUESTION_COUNT.get(canonical, 25)

    def _label_map(doc: fitz.Document) -> dict[int, ProblemLocation]:
        if canonical in ("CIMC", "CSMC"):
            return get_csimc_labels(doc)
        return {loc.prob_num: loc for loc in _find_problem_labels(doc, expected)}

    # Get labels from the correct PDFs (contest from problem PDF, solution from solution PDF)
    contest_labels = _label_map(contest_doc) if contest_doc else {}
    solution_labels = _label_map(solution_doc) if solution_doc else {}

    solution_map = dict(solution_problems)
    grades = CONTEST_GRADES.get(canonical, [])

    chunks: list[ProblemChunk] = []
    for prob_num, prob_text in contest_problems:
        sol_text = solution_map.get(prob_num, "")
        topics = tag_topics(prob_text)

        c_loc = contest_labels.get(prob_num)
        s_loc = solution_labels.get(prob_num)
        page_num = c_loc.page_index if c_loc else 1
        sol_page_num = s_loc.page_index if s_loc else 1
        has_diagram = _page_has_diagram(contest_doc, page_num) if contest_doc else False

        chunk_id = hashlib.md5(f"{canonical}_{ref.year}_{prob_num}".encode()).hexdigest()[:12]

        chunks.append(ProblemChunk(
            chunk_id=chunk_id,
            contest=canonical,
            year=ref.year,
            problem_number=prob_num,
            problem_text=prob_text,
            solution_text=sol_text,
            topics=topics,
            grades=grades,
            source_file=contest_file.path.name if contest_file else "",
            pdf_path=pdf_path_str,
            solution_pdf_path=sol_pdf_path_str,
            page_number=page_num,
            solution_page_number=sol_page_num,
            has_diagram=has_diagram,
        ))

    if contest_doc:
        contest_doc.close()
    if solution_doc:
        solution_doc.close()

    return chunks


def ingest_all(pdf_root: Path) -> list[ProblemChunk]:
    files = discover_contest_files(pdf_root)
    print(f"  Found {len(files)} PDF files")
    all_chunks: list[ProblemChunk] = []
    no_solution: list[str] = []

    for contest_name, contest_file, solution_file in pair_contest_files(files):
        ref = contest_file or solution_file
        label = f"{contest_name} {ref.year}"
        try:
            chunks = ingest_pair(contest_file, solution_file, contest_name=contest_name)
            expected = CONTEST_QUESTION_COUNT.get(contest_name, "?")
            count_flag = "" if len(chunks) == expected else f"  ⚠ expected {expected}"
            has_sol = solution_file is not None
            sol_flag = "" if has_sol else "  ⚠ NO SOLUTION PDF"
            with_sol = sum(1 for c in chunks if c.solution_text)
            sol_count = f"  ({with_sol}/{len(chunks)} with solution text)"
            print(f"  ✓ {label:<25} {len(chunks):>3} problems{count_flag}{sol_flag}{sol_count}")
            if not has_sol:
                no_solution.append(label)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  ✗ {label}: {e}")

    if no_solution:
        print(f"\n  ⚠ {len(no_solution)} contest(s) had no solution PDF paired:")
        for s in no_solution:
            print(f"      - {s}")
    return all_chunks