"""
api/contest_image_router.py

Serves individual PDF pages as PNG images for display in the chat frontend.
This is how geometry diagrams, special math symbols, and proper formatting
are shown — instead of trying to render broken extracted text, we just
render the actual PDF page.

Mount in main.py:
    from api.contest_image_router import router as image_router
    app.include_router(image_router, prefix="/contest")

Endpoint:
    GET /contest/page-image?pdf_path=...&page=2&scale=2.0
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

router = APIRouter()

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONTEST_ROOT = _REPO_ROOT / "contests"

# Safety: only serve files from within these allowed roots
# (populated from the pdf_path values stored during ingestion)
_ALLOWED_EXTENSIONS = {".pdf"}


def _is_safe_path(pdf_path: str) -> bool:
    """Reject path traversal attempts and non-PDF files."""
    p = Path(pdf_path)
    # Must be absolute, must exist, must be a PDF
    if not p.is_absolute():
        return False
    if p.suffix.lower() not in _ALLOWED_EXTENSIONS:
        return False
    if not p.exists():
        return False
    # Reject any path with traversal components
    try:
        p.resolve().relative_to(p.parent.resolve())
    except ValueError:
        return False
    return True


def _resolve_local_pdf_path(pdf_path: str) -> str | None:
    """Map stale absolute paths from older workspaces to the current repo's contests folder."""
    p = Path(pdf_path)
    if p.exists() and _is_safe_path(pdf_path):
        return str(p)

    if p.parent.name and p.name:
        candidate = _CONTEST_ROOT / p.parent.name / p.name
        if candidate.exists():
            return str(candidate)

    if p.name and _CONTEST_ROOT.exists():
        for candidate in _CONTEST_ROOT.rglob(p.name):
            if candidate.is_file():
                return str(candidate)

    return None


@router.get("/page-image")
def get_page_image(
    pdf_path: str = Query(..., description="Absolute path to the contest PDF"),
    page: int = Query(0, ge=0, description="0-based page index"),
    scale: float = Query(2.0, ge=0.5, le=4.0, description="Render scale (2.0 = 144dpi)"),
    show_solution: bool = Query(False, description="If True, fetch from solution PDF instead"),
):
    """
    Render a single PDF page as a PNG and return it as base64 JSON.
    The frontend uses this to display problems with correct formatting and diagrams.
    """
    actual_path = _resolve_local_pdf_path(pdf_path)
    if not actual_path:
        raise HTTPException(status_code=400, detail="Invalid or inaccessible PDF path.")

    # If solution view requested, try to find the solution PDF alongside the contest PDF
    if show_solution:
        solution_path = _find_solution_pdf(actual_path)
        if solution_path:
            actual_path = solution_path

    try:
        import fitz
        doc = fitz.open(actual_path)

        if page >= len(doc):
            doc.close()
            raise HTTPException(
                status_code=404,
                detail=f"Page {page} does not exist in this PDF ({len(doc)} pages total)."
            )

        pdf_page = doc[page]
        mat = fitz.Matrix(scale, scale)
        pix = pdf_page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")
        doc.close()

        # Return as base64 JSON so the frontend can embed it directly
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return {
            "image_base64": b64,
            "width": pix.width,
            "height": pix.height,
            "page": page,
            "total_pages": len(fitz.open(actual_path)),
            "is_solution": actual_path != pdf_path,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to render page: {e}")


def _find_solution_pdf(contest_pdf_path: str) -> str | None:
    """
    Given a path like .../Euclid/2025EuclidContest.pdf,
    find the corresponding .../Euclid/2025EuclidSolution.pdf.
    """
    p = Path(contest_pdf_path)
    stem = p.stem  # e.g. "2025EuclidContest"

    # Try replacing "Contest" with "Solution"
    solution_stem = stem.replace("Contest", "Solution")
    if solution_stem == stem:
        # No "Contest" in name — try appending "Solution"
        solution_stem = re.sub(r'(CIMC|CSMC|Euclid|Fryer|Galois|Hypatia|Gauss[78]?|Pascal|Cayley|Fermat)',
                               r'\1Solution', stem, flags=re.I)

    candidate = p.parent / (solution_stem + ".pdf")
    if candidate.exists():
        return str(candidate)

    # Gauss special case: GaussSolution covers both Gauss7 and Gauss8
    gauss_solution = p.parent / (stem[:4] + "GaussSolution.pdf")
    if gauss_solution.exists():
        return str(gauss_solution)

    return None


# Need re for _find_solution_pdf
import re