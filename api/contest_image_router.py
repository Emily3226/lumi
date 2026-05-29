from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path

import fitz
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

_label_cache: dict[tuple[str, str], list["ProblemLabel"]] = {}
_render_cache: dict[tuple[str, int, str, bool, float], dict] = {}

_ALLOWED_EXT = {".pdf"}
_PAD_TOP = 18
_PAD_BOTTOM = 22

CONTEST_QUESTION_COUNT: dict[str, int] = {
    "Euclid": 10,
    "Fryer": 4, "Galois": 4, "Hypatia": 4,
    "CIMC": 9, "CSMC": 9,
    "Gauss7": 25, "Gauss8": 25,
    "Pascal": 25, "Cayley": 25, "Fermat": 25,
}


@dataclass(frozen=True)
class ProblemLabel:
    prob_num: int
    page_index: int
    y: float
    x: float


def _safe(path: str) -> bool:
    p = Path(path)
    return p.is_absolute() and p.suffix.lower() in _ALLOWED_EXT and p.exists()


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


def _is_after_boundary(label: ProblemLabel, boundary: tuple[int, float] | None) -> bool:
    if boundary is None:
        return False
    page_index, y = boundary
    return label.page_index > page_index or (label.page_index == page_index and label.y > y)


def _find_label_occurrences(
    doc: fitz.Document,
    expected: int,
    x_limit: float,
) -> list[ProblemLabel]:
    labels: list[ProblemLabel] = []

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

                    labels.append(ProblemLabel(n, pi, y, x))

    return sorted(labels, key=lambda label: (label.page_index, label.y))


def _problem_labels_for_doc(
    doc: fitz.Document,
    contest: str,
    expected: int,
    x_limit: float,
) -> list[ProblemLabel]:
    occurrences = _find_label_occurrences(doc, expected, x_limit)

    if contest in ("CIMC", "CSMC"):
        boundary = _find_part_b_start(doc)

        labels_a = [
            label
            for label in occurrences
            if label.prob_num <= 6 and not _is_after_boundary(label, boundary)
        ]

        labels_b = [
            ProblemLabel(
                prob_num=label.prob_num + 6,
                page_index=label.page_index,
                y=label.y,
                x=label.x,
            )
            for label in occurrences
            if label.prob_num <= 3 and _is_after_boundary(label, boundary)
        ]

        return sorted(labels_a + labels_b, key=lambda label: (label.page_index, label.y))

    found: dict[int, ProblemLabel] = {}
    for label in occurrences:
        found.setdefault(label.prob_num, label)

    return sorted(found.values(), key=lambda label: (label.page_index, label.y))


def _find_label_locs(
    doc: fitz.Document,
    prob_num: int,
    contest: str,
    expected: int,
    x_limit: float,
    target: str,
) -> tuple[tuple[int, float] | None, tuple[int, float] | None]:
    cache_key = (target, contest)

    if cache_key not in _label_cache:
        _label_cache[cache_key] = _problem_labels_for_doc(doc, contest, expected, x_limit)

    labels = _label_cache[cache_key]

    for index, label in enumerate(labels):
        if label.prob_num != prob_num:
            continue

        start = (label.page_index, label.y)
        next_label = labels[index + 1] if index + 1 < len(labels) else None
        end = (next_label.page_index, next_label.y) if next_label else None
        return start, end

    return None, None


def _render_crop(
    doc: fitz.Document,
    start: tuple[int, float],
    end: tuple[int, float] | None,
    scale: float,
) -> bytes:
    start_page, start_y = start
    mat = fitz.Matrix(scale, scale)

    if end is not None and end[0] == start_page:
        page = doc[start_page]
        top = max(0, start_y - _PAD_TOP)
        bottom = min(end[1], page.rect.height)
        clip = fitz.Rect(0, top, page.rect.width, bottom)
        return page.get_pixmap(matrix=mat, clip=clip, alpha=False).tobytes("png")

    images: list[fitz.Pixmap] = []

    page0 = doc[start_page]
    clip0 = fitz.Rect(0, max(0, start_y - _PAD_TOP), page0.rect.width, page0.rect.height)
    images.append(page0.get_pixmap(matrix=mat, clip=clip0, alpha=False))

    end_page = end[0] if end else len(doc) - 1

    for pi in range(start_page + 1, end_page):
        if pi in _content_pages(doc):
            images.append(doc[pi].get_pixmap(matrix=mat, alpha=False))

    if end is not None and end[0] != start_page:
        page = doc[end[0]]
        clip = fitz.Rect(0, 0, page.rect.width, min(end[1] + _PAD_BOTTOM, page.rect.height))
        images.append(page.get_pixmap(matrix=mat, clip=clip, alpha=False))

    if len(images) == 1:
        return images[0].tobytes("png")

    width = max(pix.width for pix in images)
    height = sum(pix.height for pix in images)
    combined = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height))
    combined.set_rect(combined.irect, (255, 255, 255))

    y_off = 0
    for pix in images:
        combined.copy(pix, fitz.IRect(0, y_off, pix.width, y_off + pix.height))
        y_off += pix.height

    return combined.tobytes("png")


@router.get("/page-image")
def get_page_image(
    pdf_path: str = Query(...),
    prob_num: int = Query(..., ge=1),
    contest: str = Query(""),
    show_solution: bool = Query(False),
    solution_pdf_path: str = Query(""),
    scale: float = Query(2.0, ge=0.5, le=4.0),
):
    if show_solution and solution_pdf_path:
        if not _safe(solution_pdf_path):
            raise HTTPException(400, "Invalid solution PDF path.")
        target = solution_pdf_path
    else:
        if not _safe(pdf_path):
            raise HTTPException(400, "Invalid contest PDF path.")
        target = pdf_path

    render_key = (target, prob_num, contest, show_solution, scale)
    if render_key in _render_cache:
        return _render_cache[render_key]

    expected = CONTEST_QUESTION_COUNT.get(contest, 25)

    try:
        doc = fitz.open(target)
    except Exception as e:
        raise HTTPException(500, f"Cannot open PDF: {e}")

    x_limit = _detect_prob_x_limit(doc, expected)
    start, end = _find_label_locs(doc, prob_num, contest, expected, x_limit, target)

    if start is None:
        pages = _content_pages(doc)
        fallback = pages[min(prob_num - 1, len(pages) - 1)] if pages else 0
        pix = doc[fallback].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        response = {
            "image_base64": base64.b64encode(pix.tobytes("png")).decode(),
            "cropped": False,
            "page": fallback,
            "x_limit_used": x_limit,
        }
        doc.close()
        return response

    try:
        png = _render_crop(doc, start, end, scale)
    except Exception as e:
        doc.close()
        raise HTTPException(500, f"Render failed: {e}")

    doc.close()

    response = {
        "image_base64": base64.b64encode(png).decode(),
        "cropped": True,
        "page": start[0],
        "x_limit_used": x_limit,
    }
    _render_cache[render_key] = response
    return response