"""
Drop in project root and run:
  python diagnose_solution_pdf.py "C:\path\to\2023PascalSolution.pdf"

Shows the first 40 spans in the left 25% of each content page so you can
see exactly what label format the solution PDFs use.
"""
import sys
import re
from pathlib import Path
import fitz

if len(sys.argv) < 2:
    print("Usage: python diagnose_solution_pdf.py <path_to_solution.pdf>")
    sys.exit(1)

path = Path(sys.argv[1])
doc = fitz.open(str(path))
n = len(doc)
print(f"\nPDF: {path.name}  ({n} pages)\n")

for pi in range(1, min(n - 1, 6)):   # pages 1-5 (skip cover)
    page = doc[pi]
    pw = page.rect.width
    print(f"--- Page {pi} (width={pw:.0f}pt) ---")
    count = 0
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t = span["text"].strip()
                x = span["origin"][0]
                y = span["origin"][1]
                size = span.get("size", 0)
                if x < pw * 0.25 and t:   # left 25% only
                    marker = " <<<<" if re.match(r"^\d{1,2}\.?$", t) else ""
                    print(f"  x={x:6.1f}  y={y:6.1f}  size={size:4.1f}  {repr(t)}{marker}")
                    count += 1
                    if count >= 40:
                        break
            if count >= 40:
                break
        if count >= 40:
            break
    print()

doc.close()