"""
Drop in project root and run:
  python debug_solution.py "C:\path\to\2023PascalSolution.pdf" Pascal
Checks one solution PDF end-to-end to see where text extraction breaks.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import fitz
from rag.contest_ingestor import (
    _content_pages, _collect_label_candidates,
    _find_problem_labels, _extract_problems, CONTEST_QUESTION_COUNT
)

if len(sys.argv) < 2:
    print("Usage: python debug_solution.py <path_to_solution.pdf> [ContestName]")
    sys.exit(1)

pdf = Path(sys.argv[1])
contest = sys.argv[2] if len(sys.argv) > 2 else "Pascal"
expected = CONTEST_QUESTION_COUNT.get(contest, 25)

doc = fitz.open(str(pdf))
pages = _content_pages(doc)
print(f"Content pages: {pages}")
print(f"Expected: {expected} problems\n")

candidates = _collect_label_candidates(doc, expected, pages)
print(f"Raw candidates found: {len(candidates)}")
for c in candidates[:20]:
    print(f"  n={c[0]}  page={c[1]}  y={c[2]:.1f}  x={c[3]:.1f}")

print()
labels = _find_problem_labels(doc, expected)
print(f"Labels after dedup: {len(labels)}")
for l in labels:
    print(f"  prob={l.prob_num}  page={l.page_index}  y={l.y:.1f}")

print()
problems = _extract_problems(doc, contest)
print(f"Extracted problems: {len(problems)}")
for num, text in problems[:3]:
    print(f"  Q{num}: {repr(text[:80])}")

doc.close()