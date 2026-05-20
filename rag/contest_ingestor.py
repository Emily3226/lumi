"""
rag/contest_ingestor.py  (rewritten)

Key changes vs previous version:
  - Strips junk pages: cover page, instructions page, "thank you" back page
  - Tracks which PDF page each problem lives on (for image rendering)
  - Stores absolute pdf_path in metadata so the image endpoint can find the file
  - Cleans extracted text: collapses broken newlines, fixes common math symbols
  - Strips solution text from problem chunks (solutions stored separately)
  - Smarter footer/blurb detection to avoid absorbing contest closers into Q10
"""

from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF


# ── Topic keyword taxonomy ────────────────────────────────────────────────────

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "algebra": ["algebra", "equation", "polynomial", "quadratic", "linear", "variable",
                "expression", "factor", "expand", "simplify", "solve for", "roots"],
    "number_theory": ["integer", "divisor", "prime", "modulo", "remainder", "gcd", "lcm",
                      "digit", "divisible", "factor", "multiple", "congruent"],
    "geometry": ["triangle", "circle", "rectangle", "polygon", "angle", "area", "perimeter",
                 "radius", "chord", "tangent", "parallel", "perpendicular", "coordinate",
                 "quadrilateral", "equilateral", "isosceles", "hypotenuse", "diagonal"],
    "combinatorics": ["combination", "permutation", "probability", "count", "arrangement",
                      "choose", "ways", "paths", "sequence", "series", "pattern"],
    "calculus": ["limit", "derivative", "integral", "function", "continuous", "differentiable",
                 "rate of change", "slope", "tangent line", "max", "min"],
    "sequences": ["arithmetic", "geometric", "sequence", "series", "sum", "term", "recurrence",
                  "fibonacci", "progression"],
    "inequalities": ["inequality", "greater than", "less than", "maximum", "minimum",
                     "optimize", "bound", "range", "absolute value"],
    "trigonometry": ["sine", "cosine", "tangent", "sin", "cos", "tan", "angle", "radian",
                     "degree", "trigonometric", "identity"],
    "logic": ["if and only if", "prove", "proof", "show that", "suppose", "assume",
              "contradiction", "induction"],
}


def tag_topics(text: str) -> list[str]:
    lower = text.lower()
    return [topic for topic, kws in TOPIC_KEYWORDS.items() if any(kw in lower for kw in kws)]


# ── Contest metadata ──────────────────────────────────────────────────────────

FOLDER_CONTESTS: dict[str, list[str]] = {
    "CSIMC": ["CIMC", "CSMC"],
    "Euclid": ["Euclid"],
    "FGH": ["Fryer", "Galois", "Hypatia"],
    "Gauss": ["Gauss7", "Gauss8", "Gauss"],
    "PCF": ["Pascal", "Cayley", "Fermat"],
}

CONTEST_GRADES: dict[str, list[int]] = {
    "CIMC": [11, 12], "CSMC": [11, 12],
    "Euclid": [12],
    "Fryer": [9], "Galois": [10], "Hypatia": [11],
    "Gauss7": [7], "Gauss8": [8],
    "Pascal": [9], "Cayley": [10], "Fermat": [11],
}

# Phrases that mark junk pages — if a page contains any of these, skip it
_JUNK_PAGE_PHRASES = [
    "thank you for writing",
    "cemc's contests",
    "do not open this booklet",
    "number of questions:",
    "calculating devices are allowed",
    "write all answers in the answer booklet",
    "a note about bubbling",
    "visit our website cemc",
    "for students...",
    "for teachers...",
    "free copies of past contests",
    "short answer parts indicated",
    "full solution parts indicated",
    "diagrams are not drawn to scale. they are intended as aids only",
]


def _is_junk_page(text: str) -> bool:
    lower = text.lower()
    return sum(1 for phrase in _JUNK_PAGE_PHRASES if phrase in lower) >= 2


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ContestFile:
    path: Path
    contest: str
    year: int
    is_solution: bool
    folder: str


@dataclass
class PageBlock:
    """Text from one PDF page, with junk filtered out."""
    page_index: int   # 0-based page number in the PDF
    text: str


@dataclass
class ProblemChunk:
    chunk_id: str
    contest: str
    year: int
    problem_number: int | None
    part: str | None
    problem_text: str       # cleaned text for search/embedding (no solution)
    solution_text: str      # kept separately; not shown by default
    topics: list[str]
    grades: list[int]
    source_file: str        # filename only e.g. "2025EuclidContest.pdf"
    pdf_path: str           # absolute path to the contest PDF (for image rendering)
    page_number: int        # 0-based page index in contest PDF where problem appears
    has_diagram: bool       # True if page likely contains a geometric diagram

    def to_document(self) -> str:
        """Text used for ChromaDB embedding — problem text only, no solution."""
        return "\n".join([
            f"Contest: {self.contest} {self.year}",
            f"Problem {self.problem_number}" + (f" Part {self.part}" if self.part else ""),
            f"Topics: {', '.join(self.topics) or 'general'}",
            "",
            self.problem_text,
        ])

    def metadata(self) -> dict:
        return {
            "contest": self.contest,
            "year": str(self.year),
            "problem_number": str(self.problem_number or 0),
            "part": self.part or "",
            "topics": ",".join(self.topics),
            "grades": ",".join(str(g) for g in self.grades),
            "source_file": self.source_file,
            "pdf_path": self.pdf_path,
            "page_number": str(self.page_number),
            "has_solution": str(bool(self.solution_text)),
            "has_diagram": str(self.has_diagram),
        }


# ── PDF text extraction ───────────────────────────────────────────────────────

# Common math symbol fixes for fitz text output
_MATH_FIXES = [
    # Superscripts that fitz breaks onto separate lines
    (re.compile(r'(\w)\n(\d)\n'), r'\1^\2 '),
    # Degree symbol
    (re.compile(r'◦'), '°'),
    # Triangle/angle symbols that come through as garbage
    (re.compile(r'\uf044|\uf0c6'), '△'),
    (re.compile(r'\uf0d0'), '∠'),
    # Square root
    (re.compile(r'\uf0d6|\u221a'), '√'),
    # Collapse lines that are clearly mid-sentence breaks (short orphan lines)
    (re.compile(r'(?<!\n)\n(?=[a-z,;)]|\d+ )'), ' '),
    # Multiple blank lines → single blank line
    (re.compile(r'\n{3,}'), '\n\n'),
    # Lines that are just whitespace
    (re.compile(r'\n[ \t]+\n'), '\n\n'),
]

# Phrases that signal the end of the last real problem
_END_OF_PROBLEMS_PHRASES = [
    "thank you for writing",
    "for students",
    "for teachers",
    "visit our website",
    "cemc.uwaterloo.ca",
    "good luck in your",
    "free copies of past",
]

# Diagram detection: look for image objects or geometry keywords on the page
_DIAGRAM_KEYWORDS = re.compile(
    r'\b(diagram|figure|shown|shaded|circle|triangle|rectangle|polygon|quadrilateral|hexagon)\b',
    re.I
)


def _clean_text(raw: str) -> str:
    """Apply math symbol fixes and whitespace cleanup to raw fitz text."""
    text = raw
    for pattern, replacement in _MATH_FIXES:
        text = pattern.sub(replacement, text)
    return text.strip()


def _page_has_diagram(page: fitz.Page, text: str) -> bool:
    """Return True if the page likely contains a geometric diagram."""
    # Check for embedded images on the page
    if page.get_images():
        return True
    # Check for drawing paths (vector graphics = diagrams)
    if page.get_drawings():
        return True
    # Check text for diagram keywords
    if _DIAGRAM_KEYWORDS.search(text):
        return True
    return False


def extract_pages(pdf_path: Path) -> list[PageBlock]:
    """
    Extract text from each page, skipping junk pages (cover, instructions, footer).
    Returns list of PageBlock with 0-based page indices preserved.
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        print(f"  ⚠ Could not open {pdf_path.name}: {e}")
        return []

    blocks: list[PageBlock] = []
    for i, page in enumerate(doc):
        raw = page.get_text("text")
        if _is_junk_page(raw):
            continue
        cleaned = _clean_text(raw)
        if cleaned.strip():
            blocks.append(PageBlock(page_index=i, text=cleaned))

    doc.close()
    return blocks


def _page_has_diagram_from_path(pdf_path: Path, page_index: int) -> bool:
    """Open PDF just to check one page for diagrams."""
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_index]
        has_img = bool(page.get_images() or page.get_drawings())
        doc.close()
        return has_img
    except Exception:
        return False


# ── Problem splitting ─────────────────────────────────────────────────────────

# Matches "1." or "1)" at start of line (problem number)
_PROB_RE = re.compile(r'(?m)^(\d{1,2})[.)]\s+\S')

# Matches trailing blurbs that signal end of real contest content
_END_BLURB_RE = re.compile(
    r'(?i)(thank you for writing|for students\.\.\.|for teachers\.\.\.|'
    r'visit our website|good luck in your|free copies of past)',
    re.I
)


def _split_into_problems(full_text: str) -> list[tuple[int, str]]:
    """
    Split full contest text (after junk pages removed) into
    (problem_number, problem_text) pairs.
    Strips any trailing footer blurb from the last problem.
    """
    matches = list(_PROB_RE.finditer(full_text))
    if len(matches) < 2:
        return []

    problems = []
    for i, match in enumerate(matches):
        num = int(match.group(1))
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        chunk = full_text[start:end].strip()

        # Strip trailing blurb from the last problem
        blurb = _END_BLURB_RE.search(chunk)
        if blurb:
            chunk = chunk[:blurb.start()].strip()

        if chunk:
            problems.append((num, chunk))

    return problems


def _find_problem_page(problem_text: str, pages: list[PageBlock]) -> int:
    """
    Find which page a problem lives on by matching its first ~60 chars of text.
    Returns 0-based page index, or 0 as fallback.
    """
    # Use first non-number words of the problem as a search key
    search_key = re.sub(r'^\d+[.)]\s*', '', problem_text).strip()[:60]
    search_key = re.sub(r'\s+', ' ', search_key).lower()

    best_page = pages[0].page_index if pages else 0
    for block in pages:
        if search_key[:30] in block.text.lower():
            best_page = block.page_index
            break

    return best_page


# ── File discovery ────────────────────────────────────────────────────────────

def _parse_filename(filename: str, folder: str) -> tuple[str, int, bool] | None:
    stem = Path(filename).stem
    is_solution = bool(re.search(r"solution", stem, re.I))
    year_match = re.match(r"^(\d{4})", stem)
    if not year_match:
        return None
    year = int(year_match.group(1))
    remainder = stem[4:]
    remainder = re.sub(r"(?i)(contest|solution)\s*$", "", remainder).strip()

    all_names = sorted(
        [name for names in FOLDER_CONTESTS.values() for name in names],
        key=len, reverse=True
    )
    contest = None
    for name in all_names:
        if re.match(re.escape(name), remainder, re.I):
            contest = name
            break

    if contest == "Gauss":
        grade_match = re.search(r"Gauss\s*([78])", remainder, re.I)
        if grade_match:
            contest = f"Gauss{grade_match.group(1)}"

    if not contest:
        return None
    return contest, year, is_solution


def discover_contest_files(pdf_root: Path) -> list[ContestFile]:
    files: list[ContestFile] = []
    for folder_name in FOLDER_CONTESTS:
        folder_path = pdf_root / folder_name
        if not folder_path.exists():
            continue
        for pdf_file in sorted(folder_path.glob("*.pdf")):
            parsed = _parse_filename(pdf_file.name, folder_name)
            if parsed:
                contest, year, is_solution = parsed
                files.append(ContestFile(
                    path=pdf_file, contest=contest, year=year,
                    is_solution=is_solution, folder=folder_name,
                ))
    return files


# ── Pairing (Gauss shared solution fix) ──────────────────────────────────────

def pair_contest_files(files: list[ContestFile]) -> Iterator[tuple[str, ContestFile | None, ContestFile | None]]:
    index: dict[tuple[str, int], dict[str, ContestFile]] = {}
    gauss_shared: dict[int, ContestFile] = {}

    for f in files:
        key = (f.contest, f.year)
        if key not in index:
            index[key] = {}
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
        contest_name, year = key
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
    canonical_contest = contest_name or ref.contest

    # Extract pages (junk filtered)
    contest_pages = extract_pages(contest_file.path) if contest_file else []
    solution_pages = extract_pages(solution_file.path) if solution_file else []

    contest_full = "\n".join(b.text for b in contest_pages)
    solution_full = "\n".join(b.text for b in solution_pages)

    contest_problems = _split_into_problems(contest_full)
    solution_problems = _split_into_problems(solution_full)

    # Build solution lookup by problem number
    solution_map: dict[int, str] = {num: text for num, text in solution_problems}

    grades = CONTEST_GRADES.get(canonical_contest, [])
    pdf_path_str = str(contest_file.path.resolve()) if contest_file else ""

    chunks: list[ProblemChunk] = []
    for prob_num, prob_text in contest_problems:
        sol_text = solution_map.get(prob_num, "")
        topics = tag_topics(prob_text)

        page_num = _find_problem_page(prob_text, contest_pages)
        has_diagram = _page_has_diagram_from_path(
            contest_file.path, page_num
        ) if contest_file else False

        raw_id = f"{canonical_contest}_{ref.year}_{prob_num}"
        chunk_id = hashlib.md5(raw_id.encode()).hexdigest()[:12]

        chunks.append(ProblemChunk(
            chunk_id=chunk_id,
            contest=canonical_contest,
            year=ref.year,
            problem_number=prob_num,
            part=None,
            problem_text=prob_text,
            solution_text=sol_text,
            topics=topics,
            grades=grades,
            source_file=contest_file.path.name if contest_file else "",
            pdf_path=pdf_path_str,
            page_number=page_num,
            has_diagram=has_diagram,
        ))

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
            print(f"  ✓ {label:<25} {len(chunks):>3} problems")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  ✗ {label}: {e}")
    return all_chunks