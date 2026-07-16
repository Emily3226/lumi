"""
scripts/ingest_contests.py

One-time (or periodic) script to parse Waterloo contest PDFs → ChromaDB.

Usage:
    python -m scripts.ingest_contests --pdf-root /path/to/your/contests/folder

The script is idempotent — re-running it upserts (updates existing chunks and
adds new ones), so you can safely run it again after adding new PDFs.

Options:
    --pdf-root   PATH   Root folder containing CSIMC/, Euclid/, FGH/, Gauss/, PCF/
    --clear              Wipe the existing ChromaDB collection before ingesting
    --dry-run            Parse PDFs but don't write to ChromaDB (useful for testing)
    --contest    NAME    Only ingest a specific contest (e.g. Euclid, Fryer)
    --year       YEAR    Only ingest a specific year
"""

from __future__ import annotations

import argparse
import sys
import os

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from rag.contest_ingestor import ingest_all, discover_contest_files, pair_contest_files, ingest_pair
from rag.contest_retriever import add_chunks, collection_count, COLLECTION_NAME
from api.db import get_db


def clear_collection() -> None:
    """Delete every document in the contest_chunks MongoDB collection."""
    try:
        result = get_db()[COLLECTION_NAME].delete_many({})
        print(f"  Cleared {result.deleted_count} existing document(s).")
    except Exception as e:
        print(f"  ⚠ Could not clear collection: {e}")


def main():
    parser = argparse.ArgumentParser(description="Ingest Waterloo contest PDFs into ChromaDB")
    parser.add_argument("--pdf-root", required=True, help="Root folder with CSIMC/, Euclid/, etc.")
    parser.add_argument("--clear", action="store_true", help="Wipe ChromaDB collection first")
    parser.add_argument("--dry-run", action="store_true", help="Parse but don't write to ChromaDB")
    parser.add_argument("--contest", help="Only ingest this contest (e.g. Euclid)")
    parser.add_argument("--year", type=int, help="Only ingest this year")
    args = parser.parse_args()

    pdf_root = Path(args.pdf_root)
    if not pdf_root.exists():
        print(f"ERROR: PDF root not found: {pdf_root}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Waterloo Contest Ingestion")
    print(f"  PDF root : {pdf_root}")
    print(f"  MongoDB  : {COLLECTION_NAME} collection (+ contest_pdfs GridFS bucket)")
    print(f"{'='*55}\n")

    if args.clear and not args.dry_run:
        print("Clearing existing MongoDB collection...")
        clear_collection()

    print("Discovering PDF files...")
    all_files = discover_contest_files(pdf_root)

    # Apply filters
    if args.contest:
        all_files = [f for f in all_files if f.contest.lower() == args.contest.lower()]
    if args.year:
        all_files = [f for f in all_files if f.year == args.year]

    if not all_files:
        print("No matching PDF files found. Check --pdf-root and folder structure.")
        sys.exit(0)

    print(f"Found {len(all_files)} PDF files. Parsing...\n")

    all_chunks = []
    errors = []

    for contest_name, contest_file, solution_file in pair_contest_files(all_files):
        ref = contest_file or solution_file
        label = f"{contest_name} {ref.year}"
        try:
            chunks = ingest_pair(contest_file, solution_file, contest_name=contest_name)
            print(f"  ✓ {label:<25} {len(chunks):>3} problems", end="")
            if not any(c.solution_text for c in chunks):
                print("  (no solutions found)", end="")
            print()
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  ✗ {label:<25} ERROR: {e}")
            errors.append((label, str(e)))

    print(f"\n{'─'*55}")
    print(f"Parsed {len(all_chunks)} problem chunks from {len(all_files)} files")
    if errors:
        print(f"  {len(errors)} errors:")
        for label, err in errors:
            print(f"    - {label}: {err}")

    if args.dry_run:
        print("\nDry run — skipping MongoDB write.")
        print("\nSample chunk (first one):")
        if all_chunks:
            c = all_chunks[0]
            print(f"  Contest : {c.contest} {c.year}")
            print(f"  Problem : {c.problem_number}")
            print(f"  Topics  : {c.topics}")
            print(f"  Text    : {c.problem_text[:200]}...")
        return

    if not all_chunks:
        print("\nNo chunks to index.")
        return

    print(f"\nUploading PDFs to GridFS and writing chunks + embeddings to MongoDB...")

    # Convert to dicts for the retriever
    chunk_dicts = [
        {
            "chunk_id": c.chunk_id,
            "document": c.to_document(),
            "metadata": c.metadata(),
        }
        for c in all_chunks
    ]

    add_chunks(chunk_dicts)

    final_count = collection_count()
    print(f"  ✓ MongoDB now contains {final_count} indexed problem chunks")
    print(f"\nDone! The contest knowledge base is ready.")
    print(f"Start the API server and try: 'show me a Euclid combinatorics problem'\n")


if __name__ == "__main__":
    main()
