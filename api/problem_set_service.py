from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import fitz

from api.contest_image_router import _get_doc, _get_labels, _render_crop
from rag.contest_retriever import get_by_contest_year, list_available_contests


_CONTEST_RE = re.compile(
    r"\b(euclid|fryer|galois|hypatia|gauss\s*[78]?|pascal|cayley|fermat|cimc|csmc)\b",
    re.I,
)
_COUNT_RE = re.compile(r"\b(\d{1,2})\s+(?:problems?|questions?)\b", re.I)
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

_CONTEST_MAP = {
    "euclid": "Euclid",
    "fryer": "Fryer",
    "galois": "Galois",
    "hypatia": "Hypatia",
    "gauss7": "Gauss7",
    "gauss8": "Gauss8",
    "gauss": "Gauss",
    "pascal": "Pascal",
    "cayley": "Cayley",
    "fermat": "Fermat",
    "cimc": "CIMC",
    "csmc": "CSMC",
}


@dataclass
class ProblemSetResult:
    ok: bool
    reply: str
    problems: list[dict]
    pdf_url: str | None = None
    label: str | None = None
    solutions_url: str | None = None


def is_problem_set_request(text: str) -> bool:
    t = " ".join(text.lower().strip().split())
    has_generate = any(word in t for word in ("generate", "create", "make", "build"))
    has_target = any(word in t for word in ("problem set", "worksheet", "set of problems"))
    return has_generate and has_target


def _wants_solutions(text: str) -> bool:
    t = " ".join(text.lower().strip().split())
    return any(word in t for word in ("solution", "solutions", "answer key", "answers", "with answers"))


def _extract_count(text: str) -> int:
    m = _COUNT_RE.search(text)
    if m:
        return max(1, min(25, int(m.group(1))))
    m_any = re.search(r"\b(\d{1,2})\b", text)
    if m_any:
        return max(1, min(25, int(m_any.group(1))))
    return 5


def _extract_contests(text: str) -> list[str]:
    found: list[str] = []
    for m in _CONTEST_RE.finditer(text):
        key = m.group(1).lower().replace(" ", "")
        contest = _CONTEST_MAP.get(key)
        if contest and contest not in found:
            found.append(contest)
    return found


def _latest_year_map() -> dict[str, int]:
    available = list_available_contests()
    out: dict[str, int] = {}
    for item in available:
        years = []
        for y in item.get("years", []):
            try:
                years.append(int(y))
            except Exception:
                continue
        if years:
            out[item["contest"]] = max(years)
    return out


def _all_years_map() -> dict[str, list[int]]:
    available = list_available_contests()
    out: dict[str, list[int]] = {}
    for item in available:
        years: list[int] = []
        for y in item.get("years", []):
            try:
                years.append(int(y))
            except Exception:
                continue
        if years:
            out[item["contest"]] = sorted(set(years), reverse=True)
    return out


def _parse_contest_year_hints(text: str) -> dict[str, int]:
    # Supports patterns like "Euclid 2023, Fermat 2022"
    hints: dict[str, int] = {}
    for contest_match in _CONTEST_RE.finditer(text):
        start = contest_match.start()
        end = min(len(text), contest_match.end() + 12)
        window = text[start:end]
        year_match = _YEAR_RE.search(window)
        key = contest_match.group(1).lower().replace(" ", "")
        contest = _CONTEST_MAP.get(key)
        if contest and year_match:
            hints[contest] = int(year_match.group(1))
    return hints


def _order_rows_for_year(rows: list[dict], year: int) -> list[dict]:
    ordered = sorted(rows, key=lambda r: int(r.get("problem_number") or 999))
    if not ordered:
        return ordered
    offset = year % len(ordered)
    return ordered[offset:] + ordered[:offset]


def _pick_problems(contests: list[str], years_by_contest: dict[str, list[int]], count: int) -> list[dict]:
    pools: list[list[dict]] = []
    seen_keys: set[tuple[str, int, int]] = set()

    for contest in contests:
        years = years_by_contest.get(contest, [])
        for year in years:
            rows = get_by_contest_year(contest, year, n=40)
            valid = []
            for r in rows:
                prob_num = int(r.get("problem_number") or 0)
                if not prob_num or not r.get("pdf_path"):
                    continue
                key = (contest, year, prob_num)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                valid.append(r)

            if valid:
                pools.append(_order_rows_for_year(valid, year))

    selected: list[dict] = []
    while len(selected) < count and pools:
        next_round: list[list[dict]] = []
        for pool in pools:
            if len(selected) >= count:
                break
            if not pool:
                continue
            selected.append(pool.pop(0))
            if pool:
                next_round.append(pool)
        pools = next_round

    return selected


def _next_label(labels: dict[int, object], start_loc: object) -> object | None:
    sorted_locs = sorted(labels.values(), key=lambda loc: (loc.page_index, loc.y))
    for loc in sorted_locs:
        if loc.page_index > start_loc.page_index:
            return loc
        if loc.page_index == start_loc.page_index and loc.y > start_loc.y:
            return loc
    return None


def _add_problem_page(pdf: fitz.Document, problem: dict, scale: float = 2.0) -> None:
    contest = problem.get("contest", "Contest")
    year = int(problem.get("year", 0) or 0)
    number = int(problem.get("problem_number", 0) or 0)
    pdf_path = problem.get("pdf_path", "")

    doc = _get_doc(pdf_path)
    labels = _get_labels(pdf_path, contest)
    start_loc = labels.get(number)

    if start_loc is None:
        page_idx = max(0, int(problem.get("page_number", 1) or 1) - 1)
        page_idx = min(page_idx, len(doc) - 1)
        pix = doc[page_idx].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        png = pix.tobytes("png")
    else:
        next_loc = _next_label(labels, start_loc)
        png = _render_crop(doc, start_loc, next_loc, scale, is_last=next_loc is None)

    pix = fitz.Pixmap(png)
    a4_w, a4_h = 595.0, 842.0
    margin = 30.0
    top_reserved = 56.0
    max_w = a4_w - (margin * 2)
    max_h = a4_h - margin - top_reserved

    ratio = min(max_w / pix.width, max_h / pix.height)
    draw_w = pix.width * ratio
    draw_h = pix.height * ratio

    page = pdf.new_page(width=a4_w, height=a4_h)
    title = f"{contest} {year} - Problem {number}"
    page.insert_text((margin, 30), title, fontsize=13)
    rect = fitz.Rect(margin, top_reserved, margin + draw_w, top_reserved + draw_h)
    page.insert_image(rect, stream=png)


def _add_solution_page(pdf: fitz.Document, problem: dict, scale: float = 2.0) -> None:
    """Insert the official solution page for `problem` into `pdf` if available."""
    contest = problem.get("contest", "Contest")
    year = int(problem.get("year", 0) or 0)
    number = int(problem.get("problem_number", 0) or 0)
    sol_pdf_path = problem.get("solution_pdf_path", "") or problem.get("pdf_path", "")

    if not sol_pdf_path:
        raise ValueError("No solution PDF path for problem")

    doc = _get_doc(sol_pdf_path)
    page_idx = max(0, int(problem.get("solution_page_number", problem.get("page_number", 1)) or 1) - 1)
    page_idx = min(page_idx, len(doc) - 1)
    pix = doc[page_idx].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    png = pix.tobytes("png")

    pix = fitz.Pixmap(png)
    a4_w, a4_h = 595.0, 842.0
    margin = 30.0
    top_reserved = 56.0
    max_w = a4_w - (margin * 2)
    max_h = a4_h - margin - top_reserved

    ratio = min(max_w / pix.width, max_h / pix.height)
    draw_w = pix.width * ratio
    draw_h = pix.height * ratio

    page = pdf.new_page(width=a4_w, height=a4_h)
    title = f"{contest} {year} - Solution {number}"
    page.insert_text((margin, 30), title, fontsize=13)
    rect = fitz.Rect(margin, top_reserved, margin + draw_w, top_reserved + draw_h)
    page.insert_image(rect, stream=png)


def _output_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    out = root / "frontend" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_problem_set_from_text(text: str) -> ProblemSetResult:
    count = _extract_count(text)
    contests = _extract_contests(text)
    all_years = _all_years_map()

    if not contests:
        # If user doesn't specify contests, default to up to 3 indexed contests.
        contests = list(all_years.keys())[:3]

    year_hints = _parse_contest_year_hints(text)
    years_by_contest: dict[str, list[int]] = {}
    for contest in contests:
        if contest in year_hints:
            years_by_contest[contest] = [year_hints[contest]]
        else:
            years_by_contest[contest] = all_years.get(contest, [])

    selected = _pick_problems(contests, years_by_contest, count)
    if not selected:
        return ProblemSetResult(
            ok=False,
            reply=(
                "I couldn't build a problem set from that request. "
                "Please include indexed contest names like Euclid, Fermat, Cayley, or Gauss."
            ),
            problems=[],
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_contests = "-".join(c.lower() for c in contests[:3])
    filename = f"problem_set_{safe_contests}_{stamp}.pdf"
    out_path = _output_dir() / filename

    pdf = fitz.open()
    written = 0
    rendered: list[dict] = []
    try:
        for p in selected:
            try:
                _add_problem_page(pdf, p)
                written += 1
                rendered.append(p)
            except Exception:
                continue
        if written == 0:
            return ProblemSetResult(
                ok=False,
                reply="I found matching problems, but couldn't render them into a PDF.",
                problems=[],
            )
        pdf.save(str(out_path), deflate=True)
    finally:
        pdf.close()

    used_years = sorted({int(p.get("year") or 0) for p in rendered if p.get("year")})
    year_span = ""
    if used_years:
        year_span = f"; years {used_years[0]}-{used_years[-1]}"

    label = f"{written}-problem set ({', '.join(contests)}{year_span})"
    solutions_url = None
    # Only pay the (roughly doubled) render cost for a solutions PDF when the
    # user actually asked for solutions/answers - this was previously always
    # built, which is what made problem-set requests slow/timeout-prone.
    if _wants_solutions(text):
        sol_pdf = None
        try:
            sol_pdf = fitz.open()
            sol_written = 0
            for p in rendered:
                try:
                    if p.get("solution_pdf_path"):
                        _add_solution_page(sol_pdf, p)
                        sol_written += 1
                except Exception:
                    continue
            if sol_written > 0:
                sol_filename = f"solutions_set_{safe_contests}_{stamp}.pdf"
                sol_out = _output_dir() / sol_filename
                sol_pdf.save(str(sol_out), deflate=True)
                solutions_url = f"/frontend/generated/{sol_filename}"
        except Exception:
            # Ignore solution-generation failures — we still have the problems PDF
            solutions_url = None
        finally:
            if sol_pdf is not None:
                try:
                    sol_pdf.close()
                except Exception:
                    pass

    return ProblemSetResult(
        ok=True,
        reply=(
            f"I generated a mixed-year {label}. "
            f"Use the download button below to open the PDF."
        ),
        problems=rendered,
        pdf_url=f"/frontend/generated/{filename}",
        label=label,
        solutions_url=solutions_url,
    )