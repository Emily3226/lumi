"""
scripts/migrate_to_mongo.py

One-time migration:
  1. Neon/Postgres (bookings, mentors, mentees, mentor_timeslots,
     historical_pairings) -> MongoDB Atlas
  2. Local ChromaDB contest collection (data/contest_chroma) -> MongoDB
     Atlas `contest_chunks` collection (embeddings copied as-is, no
     re-embedding needed)
  3. Local contest PDFs referenced by those chunks -> MongoDB GridFS

Run this ONCE, after setting both DATABASE_URL (old Postgres) and
MONGODB_URI (new Atlas cluster) in your environment / .env file.

Usage:
    python -m scripts.migrate_to_mongo --all
    python -m scripts.migrate_to_mongo --sql-only
    python -m scripts.migrate_to_mongo --chroma-only
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.db import get_db, next_id, ensure_indexes  # new Mongo db.py


def _old_pg_connection():
    """Connect straight to Postgres using psycopg2, independent of the
    (now Mongo-backed) api.db module. Only needed for this one-time
    migration - psycopg2-binary isn't in requirements.txt anymore, so:
        pip install psycopg2-binary --break-system-packages
    before running this script.
    """
    import psycopg2
    import psycopg2.extras

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Set DATABASE_URL to your OLD Neon connection string to migrate from it.")
    conn = psycopg2.connect(database_url)
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    return conn


def migrate_sql_data() -> None:
    print("== Migrating relational data (Postgres -> MongoDB) ==")
    pg = _old_pg_connection()
    db = get_db()
    ensure_indexes()

    with pg.cursor() as cur:
        cur.execute("SELECT * FROM mentors")
        mentors = cur.fetchall()
        for m in mentors:
            db["mentors"].update_one({"name": m["name"]}, {"$set": dict(m)}, upsert=True)
        print(f"  mentors: {len(mentors)}")

        cur.execute("SELECT * FROM mentees")
        mentees = cur.fetchall()
        for m in mentees:
            db["mentees"].update_one({"name": m["name"]}, {"$set": dict(m)}, upsert=True)
        print(f"  mentees: {len(mentees)}")

        cur.execute("SELECT * FROM mentor_timeslots")
        slots = cur.fetchall()
        max_slot_id = 0
        for s in slots:
            d = dict(s)
            max_slot_id = max(max_slot_id, d["id"])
            db["mentor_timeslots"].update_one({"id": d["id"]}, {"$set": d}, upsert=True)
        print(f"  mentor_timeslots: {len(slots)}")

        cur.execute("SELECT * FROM bookings")
        bookings = cur.fetchall()
        max_booking_id = 0
        for b in bookings:
            d = dict(b)
            d["created_at"] = str(d["created_at"])
            max_booking_id = max(max_booking_id, d["id"])
            db["bookings"].update_one({"id": d["id"]}, {"$set": d}, upsert=True)
        print(f"  bookings: {len(bookings)}")

        # Keep the id counters ahead of the highest migrated id so new
        # inserts via api/db.py's next_id() don't collide.
        db["counters"].update_one(
            {"_id": "mentor_timeslots"}, {"$max": {"seq": max_slot_id}}, upsert=True
        )
        db["counters"].update_one(
            {"_id": "bookings"}, {"$max": {"seq": max_booking_id}}, upsert=True
        )

        try:
            cur.execute("SELECT * FROM historical_pairings")
            historical = cur.fetchall()
            for h in historical:
                d = dict(h)
                db["historical_pairings"].update_one(
                    {"_id": d.get("id", None) or db["historical_pairings"].count_documents({}) + 1},
                    {"$set": d},
                    upsert=True,
                )
            print(f"  historical_pairings: {len(historical)}")
        except Exception as e:
            print(f"  (skipping historical_pairings: {e})")

    pg.close()
    print("Done with relational data.\n")


def migrate_chroma_and_pdfs() -> None:
    print("== Migrating contest chunks + PDFs (ChromaDB + local disk -> MongoDB) ==")
    import chromadb
    from rag.mongo_pdf_store import upload_pdf
    from rag.contest_retriever import COLLECTION_NAME

    chroma_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "contest_chroma")
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        old_collection = client.get_collection("waterloo_contests")
    except Exception as e:
        print(f"  No local ChromaDB collection found ({e}) - nothing to migrate.")
        return

    all_data = old_collection.get(include=["documents", "metadatas", "embeddings"])
    ids = all_data["ids"]
    docs = all_data["documents"]
    metas = all_data["metadatas"]
    embeddings = all_data["embeddings"]
    print(f"  Found {len(ids)} chunks in local ChromaDB.")

    db = get_db()
    uploaded_pdfs: set[str] = set()

    for chunk_id, doc_text, meta, embedding in zip(ids, docs, metas, embeddings):
        mongo_doc = {"_id": chunk_id, "document": doc_text, "embedding": [float(x) for x in embedding]}
        meta = dict(meta)

        # Old metadata may still have absolute local paths from before the
        # GridFS switch. Re-upload those files under the new logical key
        # ("<contest>/<filename>") and rewrite the metadata to match.
        for path_field in ("pdf_path", "solution_pdf_path"):
            old_path = meta.get(path_field, "")
            if old_path and os.path.isabs(old_path) and os.path.exists(old_path):
                folder = meta.get("contest", "misc")
                key = f"{folder}/{os.path.basename(old_path)}"
                if key not in uploaded_pdfs:
                    upload_pdf(old_path, key)
                    uploaded_pdfs.add(key)
                meta[path_field] = key

        mongo_doc.update(meta)
        db[COLLECTION_NAME].replace_one({"_id": chunk_id}, mongo_doc, upsert=True)

    print(f"  Migrated {len(ids)} chunks and {len(uploaded_pdfs)} PDF(s).")
    print(
        "\n  IMPORTANT: create the Atlas Vector Search index now (Atlas UI ->\n"
        "  your cluster -> Search -> Create Search Index -> JSON editor), on\n"
        f"  database '{db.name}', collection '{COLLECTION_NAME}':\n\n"
        "  {\n"
        '    "fields": [\n'
        "      {\n"
        '        "type": "vector",\n'
        '        "path": "embedding",\n'
        '        "numDimensions": 384,\n'
        '        "similarity": "cosine"\n'
        "      }\n"
        "    ]\n"
        "  }\n\n"
        "  Name the index 'contest_vector_index' (or set CONTEST_VECTOR_INDEX\n"
        "  to whatever you name it). Until it's built, queries automatically\n"
        "  fall back to a slower in-process scan.\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Migrate lumi from Postgres/Chroma/local-disk to MongoDB Atlas")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--sql-only", action="store_true")
    parser.add_argument("--chroma-only", action="store_true")
    args = parser.parse_args()

    if not any([args.all, args.sql_only, args.chroma_only]):
        parser.print_help()
        return

    if args.all or args.sql_only:
        migrate_sql_data()
    if args.all or args.chroma_only:
        migrate_chroma_and_pdfs()


if __name__ == "__main__":
    main()
