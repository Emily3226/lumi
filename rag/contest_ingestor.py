"""
rag/contest_ingestor.py

Contest-aware PDF ingestion for Waterloo Math contests.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fitz


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
    "calculus": ["limit", "derivative", "integral", "function", "continuous",
                 "rate of change", "slope", "tangent line"],
    "sequences": ["arithmetic", "geometric", "sequence", "series", "sum", "term",
                  "recurrence", "fibonacci", "progression"],
    "inequalities": ["inequality", "maximum", "minimum", "optimize", "bound",
                     "absolute value"],
    "trigonometry": ["sine", "cosine", "tangent", r"\bsin\b", r"\bcos\b",
                     r"\btan\b", "radian", "degree", "trigonometric", "identity"],
    "logic": ["if and only if", "prove", "proof", "show that", "suppose",
              "contradiction", "induction"],
}


def tag_topics(text: str) -> list[str]:
    lower = text.lower()
    return [
        topic
        for topic, keywords in TOPIC_KEYWORDS.items()
        if any(re.search(keyword, lower) for keyword in keywords)
    ]


FOLDER_CONTESTS: dict[str, list[str]] = {
    "CSIMC": ["CIMC", "CSMC"],
    "Euclid": ["Euclid"],
    "FGH": ["Fryer", "Galois", "Hypatia"],
    "Gauss": ["Gauss7", "Gauss8", "Gauss"],
    "PCF": ["Pascal", "Cayley", "Fermat"],
}

CONTEST_GRADES: dict[str, list[int]] = {
    "CIMC": [11, 12],
    "CSMC": [11, 12],
    "Euclid": [12],
    "Fryer": [9],
    "Galois": [10],
    "Hypatia": [11],
    "Gauss7": [7],
    "Gauss8": [8],
    "Pascal": [9],
    "Cayley": [10],
    "Fermat": [11],
}

CONTEST_QUESTION_COUNT: dict[str, int] = {
    "Euclid": 10,
    "Fryer": 4,
    "Galois": 4,
    "Hypatia": 4,
    "CIMC": 9,
    "CSMC": 9,
    "Gauss7": 25,
    "Gauss8": 25,
    "Pascal": 25,
    "Cayley": 25,
    "Fermat": 25,
}


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
    page_index: int
    y: float
    x: float


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
        }


def _content_pages(doc: fitz.Document) -> list[int]:
    n = len(doc)
    if n <= 1:
        return []

    last = doc[n - 1]
    pw = last.rect.width
    last_has_problems = False

    for block in last.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                x = span["origin"][0]
                if re.match(r"^\d{1,2}\.$", text) and x < pw * 0.25:
                    last_has_problems = True
                    break
            if last_has_problems:
                break
        if last_has_problems:
            break

    end = n if last_has_problems else n - 1
    return list(range(1, end))


def _is_instruction_page(doc: fitz.Document, page_index: int) -> bool:
    if page_index == 0:
        return True

    page = doc[page_index]
    top_clip = fitz.Rect(0, 0, page.rect.width, 200)
    top_text = page.get_text("text", clip=top_clip).strip()
    markers = [
        "NOTE:",
        "Please read the instructions",
        "A Note about Bubbling",
        "Write all answers in the answer booklet",
        "Number of questions:",
    ]
    return any(marker in top_text for marker in markers)


def _detect_prob_x_limit(doc: fitz.Document, expected: int) -> float:
    x_vals: list[float] = []

    for pi in _content_pages(doc):
        if _is_instruction_page(doc, pi):
            continue

        page = doc[pi]
        pw = page.rect.width

        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    x = span["origin"][0]
                    if not re.match(r"^\d{1,2}\.$", text):
                        continue

                    n = int(text.rstrip("."))
                    if 1 <= n <= expected and x < pw * 0.25:
                        x_vals.append(x)

    if not x_vals:
        return 76.0

    x_vals.sort()
    cluster_end = len(x_vals)

    for i in range(1, len(x_vals)):
        if x_vals[i] - x_vals[i - 1] > 10.0:
            cluster_end = i
            break

    return max(x_vals[:cluster_end]) + 3.0


def _find_problem_labels(
    doc: fitz.Document,
    expected: int,
    x_limit: float | None = None,
) -> list[ProblemLocation]:
    if x_limit is None:
        x_limit = _detect_prob_x_limit(doc, expected)

    found: dict[int, ProblemLocation] = {}

    for pi in _content_pages(doc):
        page = doc[pi]
        is_instr = _is_instruction_page(doc, pi)
        ph = page.rect.height

        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    x = span["origin"][0]
                    y = span["origin"][1]

                    if not re.match(r"^\d{1,2}\.$", text):
                        continue

                    n = int(text.rstrip("."))
                    if n < 1 or n > expected:
                        continue
                    if x > x_limit:
                        continue
                    if is_instr and y < ph * 0.4:
                        continue

                    found.setdefault(n, ProblemLocation(n, pi, y, x))

    return sorted(found.values(), key=lambda loc: (loc.page_index, loc.y))


def _find_problem_label_occurrences(
    doc: fitz.Document,
    expected: int,
    x_limit: float,
) -> list[ProblemLocation]:
    labels: list[ProblemLocation] = []

    for pi in _content_pages(doc):
        page = doc[pi]
        is_instr = _is_instruction_page(doc, pi)
        ph = page.rect.height

        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    x = span["origin"][0]
                    y = span["origin"][1]

                    if not re.match(r"^\d{1,2}\.$", text):
                        continue

                    n = int(text.rstrip("."))
                    if not (1 <= n <= expected):
                        continue
                    if x > x_limit:
                        continue
                    if is_instr and y < ph * 0.4:
                        continue

                    labels.append(ProblemLocation(n, pi, y, x))

    return sorted(labels, key=lambda loc: (loc.page_index, loc.y))


def _extract_text_between(
    doc: fitz.Document,
    start: ProblemLocation,
    end: ProblemLocation | None,
) -> str:
    parts = []
    content_pages = set(_content_pages(doc))

    if end is None or end.page_index > start.page_index:
        page = doc[start.page_index]
        clip = fitz.Rect(0, start.y, page.rect.width, page.rect.height)
        parts.append(page.get_text("text", clip=clip))

        if end is not None:
            for pi in range(start.page_index + 1, end.page_index):
                if pi in content_pages:
                    parts.append(doc[pi].get_text("text"))

            end_page = doc[end.page_index]
            clip2 = fitz.Rect(0, 0, end_page.rect.width, end.y)
            parts.append(end_page.get_text("text", clip=clip2))
        else:
            for pi in range(start.page_index + 1, len(doc) - 1):
                if pi in content_pages:
                    parts.append(doc[pi].get_text("text"))
    else:
        page = doc[start.page_index]
        clip = fitz.Rect(0, start.y, page.rect.width, end.y)
        parts.append(page.get_text("text", clip=clip))

    return "\n".join(parts).strip()


def _page_has_diagram(doc: fitz.Document, page_index: int) -> bool:
    page = doc[page_index]
    return bool(page.get_drawings() or page.get_images())


def _find_part_b_start(doc: fitz.Document) -> tuple[int, float] | None:
    terms = ("Part B", "PART B", "Part B:", "Section B", "PARTIE B", "Partie B")

    for pi in _content_pages(doc):
        page = doc[pi]
        hits = []
        for term in terms:
            hits.extend(page.search_for(term))
        if hits:
            hit = sorted(hits, key=lambda r: r.y0)[0]
            return pi, hit.y0

    return None


def _is_after_boundary(loc: ProblemLocation, boundary: tuple[int, float] | None) -> bool:
    if boundary is None:
        return False

    page, y = boundary
    return loc.page_index > page or (loc.page_index == page and loc.y > y)


def _csimc_labels(doc: fitz.Document) -> list[ProblemLocation]:
    boundary = _find_part_b_start(doc)
    x_limit = _detect_prob_x_limit(doc, 9)
    all_labels = _find_problem_label_occurrences(doc, 9, x_limit)

    labels_a = [
        label
        for label in all_labels
        if label.prob_num <= 6 and not _is_after_boundary(label, boundary)
    ]

    labels_b = [
        ProblemLocation(label.prob_num + 6, label.page_index, label.y, label.x)
        for label in all_labels
        if label.prob_num <= 3 and _is_after_boundary(label, boundary)
    ]

    return sorted(labels_a + labels_b, key=lambda loc: (loc.page_index, loc.y))


def _extract_csimc_problems(doc: fitz.Document) -> list[tuple[int, str]]:
    labels = _csimc_labels(doc)
    problems = []

    for i, loc in enumerate(labels):
        next_loc = labels[i + 1] if i + 1 < len(labels) else None
        problems.append((loc.prob_num, _extract_text_between(doc, loc, next_loc)))

    return problems


def _extract_problems(
    doc: fitz.Document,
    contest: str,
) -> list[tuple[int, str]]:
    if contest in ("CIMC", "CSMC"):
        return _extract_csimc_problems(doc)

    expected = CONTEST_QUESTION_COUNT.get(contest, 25)
    labels = _find_problem_labels(doc, expected)

    problems = []
    for i, loc in enumerate(labels):
        next_loc = labels[i + 1] if i + 1 < len(labels) else None
        problems.append((loc.prob_num, _extract_text_between(doc, loc, next_loc)))

    return problems


def _parse_filename(filename: str) -> tuple[str, int, bool] | None:
    stem = Path(filename).stem
    is_solution = bool(re.search(r"solution", stem, re.I))

    match = re.match(r"^(\d{4})", stem)
    if not match:
        return None

    year = int(match.group(1))
    remainder = re.sub(r"(?i)(contest|solution)\s*$", "", stem[4:]).strip()

    all_names = sorted(
        [name for names in FOLDER_CONTESTS.values() for name in names],
        key=len,
        reverse=True,
    )

    contest = None
    for name in all_names:
        if re.match(re.escape(name), remainder, re.I):
            contest = name
            break

    if contest == "Gauss":
        gauss_match = re.search(r"Gauss\s*([78])", remainder, re.I)
        if gauss_match:
            contest = f"Gauss{gauss_match.group(1)}"

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
                    path=pdf_file,
                    contest=contest,
                    year=year,
                    is_solution=is_solution,
                    folder=folder_name,
                ))

    return files


def pair_contest_files(
    files: list[ContestFile],
) -> Iterator[tuple[str, ContestFile | None, ContestFile | None]]:
    index: dict[tuple[str, int], dict[str, ContestFile]] = {}
    gauss_shared: dict[int, ContestFile] = {}

    for file in files:
        key = (file.contest, file.year)
        index.setdefault(key, {})

        if file.is_solution:
            index[key]["solution"] = file
            if file.contest == "Gauss":
                gauss_shared[file.year] = file
        else:
            index[key]["contest"] = file

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


def ingest_pair(
    contest_file: ContestFile | None,
    solution_file: ContestFile | None,
    contest_name: str | None = None,
) -> list[ProblemChunk]:
    ref = contest_file or solution_file
    assert ref is not None

    canonical = contest_name or ref.contest

    pdf_path_str = str(contest_file.path.resolve()) if contest_file else ""
    sol_pdf_path_str = str(solution_file.path.resolve()) if solution_file else ""

    contest_doc = fitz.open(str(contest_file.path)) if contest_file else None
    solution_doc = fitz.open(str(solution_file.path)) if solution_file else None

    contest_problems = _extract_problems(contest_doc, canonical) if contest_doc else []
    solution_problems = _extract_problems(solution_doc, canonical) if solution_doc else []

    expected = CONTEST_QUESTION_COUNT.get(canonical, 25)
    contest_labels: dict[int, ProblemLocation] = {}
    solution_labels: dict[int, ProblemLocation] = {}

    def _get_csimc_labels(doc: fitz.Document) -> dict[int, ProblemLocation]:
        return {label.prob_num: label for label in _csimc_labels(doc)}

    if contest_doc:
        if canonical in ("CIMC", "CSMC"):
            contest_labels = _get_csimc_labels(contest_doc)
        else:
            contest_labels = {
                label.prob_num: label
                for label in _find_problem_labels(contest_doc, expected)
            }

    if solution_doc:
        if canonical in ("CIMC", "CSMC"):
            solution_labels = _get_csimc_labels(solution_doc)
        else:
            solution_labels = {
                label.prob_num: label
                for label in _find_problem_labels(solution_doc, expected)
            }

    solution_map = dict(solution_problems)
    grades = CONTEST_GRADES.get(canonical, [])

    chunks: list[ProblemChunk] = []

    for prob_num, prob_text in contest_problems:
        sol_text = solution_map.get(prob_num, "")
        topics = tag_topics(prob_text)

        contest_loc = contest_labels.get(prob_num)
        solution_loc = solution_labels.get(prob_num)

        page_num = contest_loc.page_index if contest_loc else 1
        sol_page_num = solution_loc.page_index if solution_loc else 1
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

    for contest_name, contest_file, solution_file in pair_contest_files(files):
        ref = contest_file or solution_file
        label = f"{contest_name} {ref.year}"

        try:
            chunks = ingest_pair(contest_file, solution_file, contest_name=contest_name)
            expected = CONTEST_QUESTION_COUNT.get(contest_name, "?")
            flag = "" if len(chunks) == expected else f"  ⚠ expected {expected}"
            print(f"  ✓ {label:<25} {len(chunks):>3} problems{flag}")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  ✗ {label}: {e}")

    return all_chunks