"""
models/clean_training_data.py

Diagnoses and (optionally) removes suspicious rows from the
`historical_pairings` table on Neon Postgres (see api/db.py) before
retraining the mentor matcher.

Background: the cycle-level stats showed avg_grade_gap == 0.0 for the
"Fall 2025" and "Summer 2025" cycles, while every other cycle averages
5.75-8.52. That strongly suggests those rows have mentor_grade and
mentee_grade both recorded as 0 (or otherwise identical placeholder
values), rather than real grades. Training on those rows teaches the
model that "grade gap doesn't matter", which produces nonsensical
matches (e.g. pairing a grade-12 mentor with a grade-9 mentee as if it
were ideal).

USAGE:
    # 1. Just show a report - makes NO changes
    python -m models.clean_training_data

    # 2. Show the report AND delete the flagged rows
    python -m models.clean_training_data --apply

    # 3. Then retrain on the cleaned data
    python -m models.train
"""

from __future__ import annotations

import argparse

from api.db import Connection, get_db


def _table_exists(conn: Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _columns(conn: Connection, table_name: str) -> list[str]:
    rows = conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ? "
        "ORDER BY ordinal_position",
        (table_name,),
    ).fetchall()
    return [r[0] for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the flagged rows (default is dry-run / report only).",
    )
    args = parser.parse_args()

    conn = get_db()

    if not _table_exists(conn, "historical_pairings"):
        print("⚠ No historical_pairings table found.")
        conn.close()
        return

    cols = _columns(conn, "historical_pairings")
    print(f"historical_pairings columns: {cols}\n")

    total = conn.execute("SELECT COUNT(*) FROM historical_pairings").fetchone()[0]
    print(f"Total rows: {total}")

    has_cycle = "cycle" in cols

    # --- Per-cycle breakdown (if a cycle column exists) -------------------
    if has_cycle:
        print("\nPer-cycle averages:")
        rows = conn.execute(
            """
            SELECT cycle,
                   COUNT(*) AS n,
                   AVG(ABS(COALESCE(mentor_grade, 0) - COALESCE(mentee_grade, 0))) AS avg_gap,
                   SUM(CASE WHEN COALESCE(mentor_grade, 0) = 0 AND COALESCE(mentee_grade, 0) = 0 THEN 1 ELSE 0 END) AS both_zero
            FROM historical_pairings
            GROUP BY cycle
            ORDER BY cycle
            """
        ).fetchall()
        for row in rows:
            print(
                f"  {row['cycle']!s:<14} n={row['n']:<5} "
                f"avg_grade_gap={row['avg_gap']:.2f}  both_grades_zero={row['both_zero']}"
            )

    # --- Identify suspicious rows ------------------------------------------
    # Flag rows where BOTH mentor_grade and mentee_grade are 0/NULL - these
    # cannot represent a real pairing (every real student/mentor has a grade
    # 9-13) and are the most likely source of the bogus avg_grade_gap == 0.0
    # cycles.
    suspicious_query = """
        SELECT *
        FROM historical_pairings
        WHERE COALESCE(mentor_grade, 0) = 0
          AND COALESCE(mentee_grade, 0) = 0
    """
    suspicious_rows = conn.execute(suspicious_query).fetchall()

    print(f"\nRows with mentor_grade == 0 AND mentee_grade == 0: {len(suspicious_rows)}")
    if has_cycle and suspicious_rows:
        from collections import Counter

        cycle_counts = Counter(row["cycle"] for row in suspicious_rows)
        for cycle, count in sorted(cycle_counts.items()):
            print(f"  {cycle!s:<14} {count} suspicious rows")

    if not suspicious_rows:
        print("\nNo rows matched the (grade==0, grade==0) heuristic.")
        print("If matching still looks wrong, share a few sample rows from")
        print("historical_pairings (especially for Fall 2025 / Summer 2025)")
        print("so we can figure out the right filter.")
        conn.close()
        return

    print("\nSample flagged rows:")
    for row in suspicious_rows[:5]:
        print(f"  {dict(row)}")

    if not args.apply:
        print(
            f"\nDry run only - {len(suspicious_rows)} rows would be deleted. "
            "Re-run with --apply to actually remove them."
        )
        conn.close()
        return

    # --- Backup before destructive change ----------------------------------
    # Neon supports instant branching/point-in-time restore from the
    # dashboard, so take a branch snapshot there before applying deletes if
    # you want an easy rollback path (Project -> Branches -> New Branch).
    print("\nProceeding with delete on Neon. Consider taking a Neon branch snapshot first if you haven't.")

    ids = [row["id"] for row in suspicious_rows]
    placeholders = ",".join("?" for _ in ids)
    conn.execute(f"DELETE FROM historical_pairings WHERE id IN ({placeholders})", ids)
    conn.commit()

    remaining = conn.execute("SELECT COUNT(*) FROM historical_pairings").fetchone()[0]
    print(f"Deleted {len(suspicious_rows)} rows. Remaining rows: {remaining}")

    if remaining < 8:
        print(
            "\n⚠ Warning: fewer than 8 rows remain. models/train.py requires "
            "at least 8 samples and will fall back to the heuristic scorer "
            "until you collect more clean training data."
        )
    else:
        print("\nNext step: retrain the model with:")
        print("  python -m models.train")

    conn.close()


if __name__ == "__main__":
    main()