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
_CONTINUATION_HEADER_RE = re.compile(
    r"^(?:Page \d+|\d{4}\s+.*Contest(?: Solutions?)?|\d{4}\s+.*Solutions?)$",
    re.IGNORECASE,
)

# ── Document cache (LRU) ──────────────────────────────────────────────────────

_doc_cache: dict[str, fitz.Document] = {}
_doc_order: list[str] = []


def _get_doc(path: str) -> fitz.Document:
    """`path` is a logical GridFS key (e.g. "Euclid/2020Euclid.pdf"). We
    resolve it to a local file - downloading from GridFS into the disk
    cache on first use - then open + cache the fitz.Document as before.
    """
    if path in _doc_cache:
        _doc_order.remove(path)
        _doc_order.append(path)
        return _doc_cache[path]

    from rag.mongo_pdf_store import get_local_path
    local_path = get_local_path(path)
    if not local_path:
        raise HTTPException(status_code=404, detail=f"PDF not found in storage: {path}")

    while len(_doc_cache) >= _DOC_CACHE_SIZE:
        oldest = _doc_order.pop(0)
        try:
            _doc_cache.pop(oldest).close()
        except Exception:
            pass
    doc = fitz.open(local_path)
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


# ── Render cache (LRU, bounded) ─────────────────────────────────────────────
# NOTE: previously this was an unbounded dict that kept every rendered PNG
# (base64-encoded, at up to 4x scale) in memory for the life of the process.
# That's what was causing out-of-memory crashes, especially when generating
# problem sets which render many images in one go. Cap it like the doc cache.

_RENDER_CACHE_SIZE = 150
_render_cache: dict[tuple, dict] = {}
_render_cache_order: list[tuple] = []


def _render_cache_get(key: tuple) -> dict | None:
    if key in _render_cache:
        try:
            _render_cache_order.remove(key)
        except ValueError:
            pass
        _render_cache_order.append(key)
        return _render_cache[key]
    return None


def _render_cache_set(key: tuple, value: dict) -> None:
    while len(_render_cache) >= _RENDER_CACHE_SIZE:
        oldest = _render_cache_order.pop(0)
        _render_cache.pop(oldest, None)
    _render_cache[key] = value
    _render_cache_order.append(key)

# ── Safety ────────────────────────────────────────────────────────────────────

def _safe(path: str) -> bool:
    """`path` here is a logical GridFS key like "Euclid/2020Euclid.pdf"
    (see rag/mongo_pdf_store.py), not a local filesystem path. Guard against
    path traversal and make sure it's a PDF; existence in GridFS is checked
    later by _get_doc / mongo_pdf_store.get_local_path.
    """
    if not path or path.startswith("/") or ".." in path:
        return False
    p = Path(path)
    return p.suffix.lower() in _ALLOWED_EXT


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


def _trim_to_content(
    pix: fitz.Pixmap,
    top_pad: int = 8,
    bottom_pad: int = 8,
) -> fitz.Pixmap:
    """Crop a pixmap to its content band, keeping a small pad above and below."""
    try:
        first_row = None
        last_row = None
        w, h, n = pix.width, pix.height, pix.n
        samples = pix.samples
        for row in range(h):
            row_bytes = samples[row * w * n: (row + 1) * w * n]
            if any(b < 248 for b in row_bytes):
                if first_row is None:
                    first_row = row
                last_row = row
        if first_row is None or last_row is None:
            return pix
        top = max(0, first_row - top_pad)
        bottom = min(h - 1, last_row + bottom_pad)
        if top == 0 and bottom == h - 1:
            return pix
        new_h = max(1, bottom - top + 1)
        # fitz.Pixmap(src, irect) is broken in PyMuPDF >=1.25; use copy() instead.
        # Clipped pixmaps keep page-relative coordinates, so rebase the source to 0,0
        # before copying the kept rows into the trimmed destination.
        pix.set_origin(0, -top)
        trimmed = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, pix.width, new_h), False)
        trimmed.set_rect(trimmed.irect, (255, 255, 255))
        trimmed.copy(pix, fitz.IRect(0, 0, pix.width, new_h))
        return trimmed
    except Exception:
        return pix


def _continuation_top_y(page: fitz.Page) -> float:
    """Return the top y coordinate for a continuation page, skipping repeated headers."""
    lines: list[tuple[float, float, str]] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line.get("spans", [])).strip()
            if not text:
                continue
            y0 = min(span["bbox"][1] for span in line["spans"])
            y1 = max(span["bbox"][3] for span in line["spans"])
            lines.append((y0, y1, text))

    if not lines:
        return 0.0

    lines.sort(key=lambda item: (item[0], item[1]))
    for y0, y1, text in lines:
        if y0 > 120:
            break
        if _CONTINUATION_HEADER_RE.match(text):
            continue
        return max(0.0, y0 - 2.0)

    return max(0.0, lines[0][0] - 2.0)


def _render_page_strip(
    page: fitz.Page,
    mat: fitz.Matrix,
    clip: fitz.Rect,
) -> fitz.Pixmap:
    """Render a page strip and trim trailing blank space from the bottom."""
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    return _trim_to_content(pix)


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
    else:
        bottom_y = None  # will use full page then trim

    sp = start_loc.page_index
    if end_loc is not None:
        ep = end_loc.page_index
    else:
        # For the last problem, there is no next label boundary.
        # Solutions often continue onto pages that do not contain a new
        # problem label, so allow rendering through the document end.
        if is_solution:
            ep = len(doc) - 1
        else:
            content_pages = _content_pages(doc)
            ep = content_pages[-1] if content_pages else sp
        ep = max(sp, ep)

    # ── Same page ─────────────────────────────────────────────────────────────
    if ep == sp:
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
    strips.append(_render_page_strip(page0, mat, clip0))

    # Middle pages: full page
    for pi in range(sp + 1, ep):
        if 0 < pi < len(doc):
            page_mid = doc[pi]
            cont_top = _continuation_top_y(page_mid)
            clip_mid = fitz.Rect(0, cont_top, page_mid.rect.width, page_mid.rect.height)
            strips.append(_render_page_strip(page_mid, mat, clip_mid))

    # End page: from top to bottom_y
    if 0 < ep < len(doc):
        page_e = doc[ep]
        bot_e = min(bottom_y, page_e.rect.height) if bottom_y is not None else page_e.rect.height
        cont_top = _continuation_top_y(page_e)
        clip_e = fitz.Rect(0, cont_top, page_e.rect.width, bot_e)
        strips.append(_render_page_strip(page_e, mat, clip_e))

    if not strips:
        pix = doc[sp].get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    if len(strips) == 1:
        return strips[0].tobytes("png")

    # Stitch vertically.
    # Pixmap.copy(src, irect) reads src at those exact coordinates, so each
    # strip's origin must be shifted to its target y position before copying.
    w = strips[0].width
    total_h = sum(p.height for p in strips)
    combined = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, w, total_h))
    combined.set_rect(combined.irect, (255, 255, 255))
    y_off = 0
    for strip in strips:
        strip.set_origin(0, y_off)
        combined.copy(strip, fitz.IRect(0, y_off, w, y_off + strip.height))
        y_off += strip.height
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
    cached = _render_cache_get(render_key)
    if cached is not None:
        return cached

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
        _render_cache_set(render_key, response)
        return response
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
    _render_cache_set(render_key, response)
    return response


def prewarm_cache() -> None:
    """
    Pre-render every known problem's problem-page and solution-page images at
    startup, so the first real user to view any problem doesn't pay the
    multi-page-render cost themselves. Runs once, synchronously, at app boot.
    """
    import time
    from rag.contest_retriever import list_available_contests, get_by_contest_year

    start = time.monotonic()
    warmed = 0
    failed = 0

    try:
        available = list_available_contests()
    except Exception as e:
        print(f"[PREWARM] Could not list contests: {e}")
        return

    for item in available:
        contest = item.get("contest")
        years = item.get("years", [])
        for year_str in years:
            try:
                year = int(year_str)
            except Exception:
                continue
            try:
                rows = get_by_contest_year(contest, year, n=30)
            except Exception as e:
                print(f"[PREWARM] Failed to fetch {contest} {year}: {e}")
                continue

            for row in rows:
                pdf_path = row.get("pdf_path")
                solution_pdf_path = row.get("solution_pdf_path")
                prob_num = row.get("problem_number")
                if not pdf_path or not prob_num:
                    continue

                # Warm the problem-page render
                try:
                    _prewarm_one(pdf_path, prob_num, contest, False, solution_pdf_path)
                    warmed += 1
                except Exception as e:
                    failed += 1
                    print(f"[PREWARM] Failed problem render {contest} {year} Q{prob_num}: {e}")

                # Warm the solution-page render (this is the slow, multi-page one)
                if solution_pdf_path:
                    try:
                        _prewarm_one(pdf_path, prob_num, contest, True, solution_pdf_path)
                        warmed += 1
                    except Exception as e:
                        failed += 1
                        print(f"[PREWARM] Failed solution render {contest} {year} Q{prob_num}: {e}")

    elapsed = time.monotonic() - start
    print(f"[PREWARM] Done: {warmed} images cached, {failed} failed, took {elapsed:.1f}s")


def _prewarm_one(pdf_path: str, prob_num: int, contest: str, show_solution: bool, solution_pdf_path: str) -> None:
    """Render and cache a single problem/solution image, reusing the same logic as the endpoint."""
    target = solution_pdf_path if (show_solution and solution_pdf_path) else pdf_path
    if not _safe(target):
        return

    scale = 2.0
    render_key = (target, prob_num, contest, show_solution, scale)
    if _render_cache_get(render_key) is not None:
        return

    doc = _get_doc(target)
    labels = _get_labels(target, contest)
    start_loc = labels.get(prob_num)
    if start_loc is None:
        return

    sorted_locs = sorted(labels.values(), key=lambda loc: (loc.page_index, loc.y))
    next_loc = None
    for loc in sorted_locs:
        if loc.page_index > start_loc.page_index or (
            loc.page_index == start_loc.page_index and loc.y > start_loc.y
        ):
            next_loc = loc
            break
    is_last = next_loc is None

    png = _render_crop(doc, start_loc, next_loc, scale, is_last, is_solution=show_solution)
    _render_cache_set(render_key, {
        "image_base64": base64.b64encode(png).decode(),
        "cropped": True,
        "page": start_loc.page_index,
    })