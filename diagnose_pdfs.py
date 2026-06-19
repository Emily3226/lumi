"""
Run from the project root to see exactly how your PDF files are being parsed.
Usage: python diagnose_pdfs.py --pdf-root /path/to/your/contests
"""
import argparse
import re
import sys
from pathlib import Path

FOLDER_CONTESTS = {
    "CSIMC": ["CIMC", "CSMC"],
    "Euclid": ["Euclid"],
    "FGH": ["Fryer", "Galois", "Hypatia"],
    "Gauss": ["Gauss7", "Gauss8", "Gauss"],
    "PCF": ["Pascal", "Cayley", "Fermat"],
}

def _parse_filename(filename: str):
    stem = Path(filename).stem
    is_solution = bool(re.search(r"solution", stem, re.I))
    m = re.match(r"^(\d{4})", stem)
    if not m:
        return None
    year = int(m.group(1))
    remainder = re.sub(r"(?i)(contest|solution)\s*$", "", stem[4:]).strip()

    all_names = sorted(
        [n for names in FOLDER_CONTESTS.values() for n in names],
        key=len, reverse=True,
    )
    contest = None
    for name in all_names:
        if re.match(re.escape(name), remainder, re.I):
            contest = name
            break

    if contest == "Gauss":
        gm = re.search(r"Gauss\s*([78])", remainder, re.I)
        if gm:
            contest = f"Gauss{gm.group(1)}"

    return (contest, year, is_solution) if contest else None


parser = argparse.ArgumentParser()
parser.add_argument("--pdf-root", required=True)
args = parser.parse_args()
pdf_root = Path(args.pdf_root)

print(f"\nScanning: {pdf_root}\n")
print(f"{'FILE':<55} {'CONTEST':<10} {'YEAR':<6} {'SOLUTION?'}")
print("-" * 85)

unparsed = []
pairs: dict = {}

for folder in FOLDER_CONTESTS:
    folder_path = pdf_root / folder
    if not folder_path.exists():
        print(f"  [MISSING FOLDER] {folder}")
        continue
    for pdf in sorted(folder_path.glob("*.pdf")):
        result = _parse_filename(pdf.name)
        if result:
            contest, year, is_sol = result
            key = f"{contest}_{year}"
            pairs.setdefault(key, {"contest": [], "solution": []})
            pairs[key]["solution" if is_sol else "contest"].append(pdf.name)
            flag = "YES" if is_sol else "no"
            print(f"  {pdf.name:<53} {contest:<10} {year:<6} {flag}")
        else:
            unparsed.append(str(pdf))
            print(f"  {pdf.name:<53} *** COULD NOT PARSE ***")

print("\n--- PAIRING SUMMARY ---")
missing_solution = []
missing_contest = []
for key, group in sorted(pairs.items()):
    has_c = bool(group["contest"])
    has_s = bool(group["solution"])
    status = "OK" if has_c and has_s else ("NO SOLUTION" if has_c else "NO CONTEST PDF")
    if not has_s:
        missing_solution.append(key)
    if not has_c:
        missing_contest.append(key)
    marker = "" if (has_c and has_s) else "  <<<"
    print(f"  {key:<30} contest={has_c}  solution={has_s}  {status}{marker}")

print(f"\nTotal unparseable files: {len(unparsed)}")
print(f"Contests missing solution PDF: {len(missing_solution)}")
for k in missing_solution:
    print(f"  - {k}")