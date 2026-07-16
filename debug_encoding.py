"""
python debug_encoding.py "C:\path\to\2023PascalSolution.pdf"
Tests different text extraction modes to find which gives readable text.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import fitz

pdf = Path(sys.argv[1])
doc = fitz.open(str(pdf))
page = doc[1]  # page 2 (0-indexed), first content page

print("=== mode: text ===")
print(repr(page.get_text("text")[:300]))
print()

print("=== mode: text with flags ===")
print(repr(page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)[:300]))
print()

print("=== mode: html (strip tags) ===")
import re
html = page.get_text("html")
clean = re.sub(r'<[^>]+>', ' ', html)
clean = re.sub(r'\s+', ' ', clean).strip()
print(repr(clean[:300]))
print()

print("=== mode: dict char-level ===")
blocks = page.get_text("rawdict")["blocks"]
chars = []
for b in blocks:
    if b.get("type") != 0: continue
    for line in b.get("lines", []):
        for span in line.get("spans", []):
            for ch in span.get("chars", []):
                c = ch.get("c", "")
                if c.strip():
                    chars.append(c)
        chars.append(" ")
print(repr("".join(chars)[:300]))
print()

print("=== mode: xml ===")
xml = page.get_text("xml")
clean_xml = re.sub(r'<[^>]+>', '', xml)
clean_xml = re.sub(r'\s+', ' ', clean_xml).strip()
print(repr(clean_xml[:300]))

doc.close()