"""
api/contest_image_router.py

Renders precisely cropped PNG snippets of contest problems and solutions.

Crop boundaries (v3 — "exclusive next label"):
  The rendered region runs from (start_y - PAD_TOP) to (end_y - label_height - LABEL_GAP).
  This means the crop ends just *above* the next problem's number label, so neither
  the current problem's number bleeds into the previous slice nor the next problem's
  number bleeds into this one.

  For the last problem in a PDF (no next label) the crop extends to the last line of
  content, determined by scanning upward for the first non-white row.

Performance:
  fitz.Document objects are cached by path (LRU, max 8 in memory).
  Label maps are cached separately so they survive doc-cache eviction.
  Rendered PNGs are cached by (path, prob_num, contest, show_solution, scale).
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import fitz
from fastapi import APIRouter, HTTPException, Query

from rag.contest_ingestor import (
    _content_pages,
    _find_problem_labels,
    _find_csimc_labels,
    get_csimc_labels,
    ProblemLocation,
    CONTEST_QUESTION_COUNT,
)

router = APIRouter()

# ── Constants ─────────────────────────────────────────────────────────────────

_ALLOWED_EXT = {".pdf"}
_PAD_TOP = 20        # pts above the problem label's y to include (captures the number itself)
_LABEL_GAP = 6       # pts of breathing room kept below the last line before the next label
_DOC_CACHE_SIZE = 8

# ── Document cache (LRU) ──────────────────────────────────────────────────────

_doc_cache: dict[str, fitz.Document] = {}
_doc_order: list[str] = []


def _get_doc(path: str) -> fitz.Document:
    if path in _doc_cache:
        _doc_order.remove(path)
        _doc_order.append(path)
        return _doc_cache[path]
    while len(_doc_cache) >= _DOC_CACHE_SIZE:
        oldest = _doc_order.pop(0)
        try:
            _doc_cache.pop(oldest).close()
        except Exception:
            pass
    doc = fitz.open(path)
    _doc_cache[path] = doc
    _doc_order.append(path)
    return doc


# ── Label cache ───────────────────────────────────────────────────────────────

_label_cache: dict[tuple[str, str], dict[int, ProblemLocation]] = {}


def _get_labels(pdf_path: str, contest: str) -> dict[int, ProblemLocation]:
    key = (pdf_path, contest)
    if key in _label_cache:
        return _label_cache[key]
    doc = _get_doc(pdf_path)
    if contest in ("CIMC", "CSMC"):
        labels = get_csimc_labels(doc)
    else:
        expected = CONTEST_QUESTION_COUNT.get(contest, 25)
        labels = {loc.prob_num: loc for loc in _find_problem_labels(doc, expected)}
    _label_cache[key] = labels
    return labels


# ── Render cache ──────────────────────────────────────────────────────────────

_render_cache: dict[tuple, dict] = {}

# ── Safety ────────────────────────────────────────────────────────────────────

def _safe(path: str) -> bool:
    p = Path(path)
    return p.is_absolute() and p.suffix.lower() in _ALLOWED_EXT and p.exists()


# ── Cropping helpers ──────────────────────────────────────────────────────────

def _last_content_y(pix: fitz.Pixmap, threshold: int = 248) -> int:
    """
    Return the pixel-row index of the last row that contains non-white content.
    Used to trim the bottom of the final problem's strip (no next label to bound it).
    Keeps at least 15% of height to avoid over-trimming diagrams with light colours.
    """
    w, h, n = pix.width, pix.height, pix.n
    min_row = max(1, h * 15 // 100)
    samples = pix.samples
    for row in range(h - 1, min_row, -1):
        start = row * w * n
        row_bytes = samples[start: start + w * n]
        if any(b < threshold for b in row_bytes):
            return row
    return h - 1


def _trim_to_content(pix: fitz.Pixmap) -> fitz.Pixmap:
    """Crop a pixmap to its last content row (used for the final problem only)."""
    try:
        last_row = _last_content_y(pix)
        if last_row >= pix.height - 1:
            return pix
        trimmed = fitz.Pixmap(pix, fitz.IRect(0, 0, pix.width, max(1, last_row + 4)))
        return trimmed
    except Exception:
        # If trimming fails, return original pixmap
        return pix


def _render_crop(
    doc: fitz.Document,
    start_loc: ProblemLocation,
    end_loc: ProblemLocation | None,
    scale: float,
    is_last: bool,
    is_solution: bool = False,
) -> bytes:
    """
    Render the problem region as a PNG.

    start_loc : ProblemLocation of this problem's label
    end_loc   : ProblemLocation of the *next* problem's label (None if last)
    is_last   : True when there is no next label — use content-trim instead
    is_solution : True if rendering a solution (extend boundary further to capture multi-part solutions)

    For solutions, we extend the bottom boundary further down to ensure all
    parts of multi-part problems (like 2A, 2B, 2C) are included.
    """
    mat = fitz.Matrix(scale, scale)

    top_y = max(0, start_loc.y - _PAD_TOP)

    if end_loc is not None:
        # Stop just above the next label.
        # end_loc.font_size is the label's font size in pts; subtract it plus a gap.
        label_height = end_loc.font_size  # approximate — label baseline ≈ font_size above next y
        bottom_y = end_loc.y - label_height - _LABEL_GAP
        
        # For solutions, extend further down to capture multi-part problem solutions
        if is_solution:
            # Extend an extra 150 pixels worth of content to capture 2A, 2B, 2C etc.
            # At scale=2.0, this is ~75pts in document space
            bottom_y = min(bottom_y + 150 / scale, doc[end_loc.page_index].rect.height if end_loc.page_index < len(doc) else 9999)
    else:
        bottom_y = None  # will use full page then trim

    sp = start_loc.page_index
    ep = end_loc.page_index if end_loc is not None else None

    # ── Same page ─────────────────────────────────────────────────────────────
    if ep is None or ep == sp:
        page = doc[sp]
        bot = min(bottom_y, page.rect.height) if bottom_y is not None else page.rect.height
        clip = fitz.Rect(0, top_y, page.rect.width, bot)
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        if is_last:
            pix = _trim_to_content(pix)
        return pix.tobytes("png")

    # ── Multi-page ────────────────────────────────────────────────────────────
    strips: list[fitz.Pixmap] = []

    # First page: from top_y to page bottom
    page0 = doc[sp]
    clip0 = fitz.Rect(0, top_y, page0.rect.width, page0.rect.height)
    strips.append(page0.get_pixmap(matrix=mat, clip=clip0, alpha=False))

    # Middle pages: full page
    for pi in range(sp + 1, ep):
        if 0 < pi < len(doc):
            strips.append(doc[pi].get_pixmap(matrix=mat, alpha=False))

    # End page: from top to bottom_y
    if 0 < ep < len(doc):
        page_e = doc[ep]
        bot_e = min(bottom_y, page_e.rect.height) if bottom_y is not None else page_e.rect.height
        clip_e = fitz.Rect(0, 0, page_e.rect.width, bot_e)
        pix_e = page_e.get_pixmap(matrix=mat, clip=clip_e, alpha=False)
        if is_last:
            pix_e = _trim_to_content(pix_e)
        strips.append(pix_e)

    if not strips:
        pix = doc[sp].get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    if len(strips) == 1:
        return strips[0].tobytes("png")

    # Stitch vertically
    w = strips[0].width
    total_h = sum(p.height for p in strips)
    combined = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, w, total_h))
    combined.set_rect(combined.irect, (255, 255, 255))
    y_off = 0
    for pix in strips:
        combined.copy(pix, fitz.IRect(0, y_off, w, y_off + pix.height))
        y_off += pix.height
    return combined.tobytes("png")


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.get("/page-image")
def get_page_image(
    pdf_path: str = Query(...),
    prob_num: int = Query(..., ge=1),
    contest: str = Query(""),
    show_solution: bool = Query(False),
    solution_pdf_path: str = Query(""),
    scale: float = Query(2.0, ge=0.5, le=4.0),
):
    """Return a base64 PNG of the cropped problem or solution region."""
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

    try:
        doc = _get_doc(target)
    except Exception as e:
        raise HTTPException(500, f"Cannot open PDF: {e}")

    labels = _get_labels(target, contest)
    expected = CONTEST_QUESTION_COUNT.get(contest, 25)

    start_loc = labels.get(prob_num)

    if start_loc is None:
        pages = _content_pages(doc)
        fallback = pages[min(prob_num - 1, len(pages) - 1)] if pages else 0
        pix = doc[fallback].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        response = {
            "image_base64": base64.b64encode(pix.tobytes("png")).decode(),
            "cropped": False,
            "page": fallback,
        }
        _render_cache[render_key] = response
        return response

    # Find the next problem's label (used as the exclusive upper boundary)
    # Walk forward through sorted labels to find the immediate successor
    sorted_locs = sorted(labels.values(), key=lambda loc: (loc.page_index, loc.y))
    next_loc: ProblemLocation | None = None
    for loc in sorted_locs:
        if loc.page_index > start_loc.page_index or (
            loc.page_index == start_loc.page_index and loc.y > start_loc.y
        ):
            next_loc = loc
            break

    is_last = next_loc is None

    try:
        png = _render_crop(doc, start_loc, next_loc, scale, is_last, is_solution=show_solution)
    except Exception as e:
        raise HTTPException(500, f"Render failed: {e}")

    response = {
        "image_base64": base64.b64encode(png).decode(),
        "cropped": True,
        "page": start_loc.page_index,
    }
    _render_cache[render_key] = response
    return response