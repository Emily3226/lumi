"""
scripts/import_and_train.py

Clean mentor, mentee, and pairing CSVs from the data directory, import the
normalized records into SQLite, generate a fairness summary, and retrain the
mentor matcher model.

Usage:
    c:/Users/ezhan/lumi/.venv/Scripts/python.exe scripts/import_and_train.py

Optional flags:
    --dry-run   Parse and report only, without writing to the database
    --no-train  Skip model training after import
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CLEAN_DIR = DATA_DIR / "cleaned"
APP_DB_PATH = DATA_DIR / "lumi.db"
TRAINING_DB_PATH = DATA_DIR / "training.db"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEASON_ORDER = {"fall": 0, "winter": 1, "summer": 2}
PAIRINGS_TABLE = "historical_pairings"


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ").strip()
    return re.sub(r"\s+", " ", text)


def normalize_name(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    return text.title()


def normalize_email(value: object) -> str:
    return normalize_text(value).lower()


def parse_grade(value: object) -> int | None:
    text = normalize_text(value).lower()
    if not text:
        return None

    grade_aliases = {
        "university": 13,
        "univ": 13,
        "college": 13,
        "post-secondary": 13,
        "post secondary": 13,
        "grade 9": 9,
        "grade 10": 10,
        "grade 11": 11,
        "grade 12": 12,
    }
    if text in grade_aliases:
        return grade_aliases[text]

    match = re.search(r"\b(13|12|11|10|9)\b", text)
    if match:
        return int(match.group(1))
    return None


def split_multi_value(value: object) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    parts = re.split(r"[,;/|]+", text)
    cleaned = []
    for part in parts:
        item = normalize_text(part)
        if item and item.lower() not in {"n/a", "na", "none", "-"}:
            cleaned.append(item)
    return cleaned


def normalize_subjects(value: object) -> str:
    items = split_multi_value(value)
    if not items:
        return ""
    seen = set()
    normalized = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            normalized.append(item.title())
    return ", ".join(normalized)


def primary_subject(subjects: str) -> str:
    parts = split_multi_value(subjects)
    return parts[0].title() if parts else ""


def detect_cycle(path: Path) -> tuple[str, tuple[int, int]]:
    match = re.search(r"(winter|summer|fall)\s*(\d{4})", path.stem, re.IGNORECASE)
    if not match:
        return "Legacy", (0, 0)
    season = match.group(1).lower()
    year = int(match.group(2))
    return f"{season.title()} {year}", (year, SEASON_ORDER.get(season, 99))


def detect_kind(path: Path) -> str | None:
    lower = path.name.lower()
    if lower.endswith("pairings.csv"):
        return "pairings"
    if lower.endswith("mentors.csv"):
        return "mentors"
    if lower.endswith("mentees.csv"):
        return "mentees"
    return None


def connect_app_db() -> sqlite3.Connection:
    conn = sqlite3.connect(APP_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def connect_training_db() -> sqlite3.Connection:
    conn = sqlite3.connect(TRAINING_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS historical_pairings (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle               TEXT,
            source_file         TEXT,
            mentor_name         TEXT,
            mentor_email        TEXT,
            mentor_grade        INTEGER,
            mentor_subjects     TEXT,
            mentor_notes        TEXT,
            mentee_name         TEXT,
            mentee_email        TEXT,
            mentee_grade        INTEGER,
            mentee_subjects     TEXT,
            subjects_satisfied  TEXT,
            subject_count       INTEGER,
            grade_gap           INTEGER,
            match_score         REAL,
            created_at          TEXT DEFAULT (datetime('now'))
        )
        """
    )


def subject_overlap_score(mentor_subjects: str, mentee_subjects: str) -> float:
    mentor_set = {item.lower() for item in split_multi_value(mentor_subjects)}
    mentee_set = {item.lower() for item in split_multi_value(mentee_subjects)}
    if not mentor_set or not mentee_set:
        return 0.0
    return 1.0 if mentor_set.intersection(mentee_set) else 0.0


def compute_match_score(mentor_grade: int | None, mentee_grade: int | None, subjects_satisfied: str) -> float:
    mentor_grade_value = float(mentor_grade or 0)
    mentee_grade_value = float(mentee_grade or 0)
    grade_gap = abs(mentor_grade_value - mentee_grade_value)
    senior_bonus = 1.0 if mentor_grade_value > mentee_grade_value else 0.0
    subject_count = len(split_multi_value(subjects_satisfied))
    subject_alignment = 1.0 if subject_count > 0 else 0.0
    coverage = min(subject_count / 3.0, 1.0)
    grade_similarity = max(0.0, 1.0 - (grade_gap / 4.0))
    score = (0.35 * coverage) + (0.30 * grade_similarity) + (0.20 * senior_bonus) + (0.15 * subject_alignment)
    return round(max(0.0, min(1.0, score)), 3)


def mentor_row_to_record(row: list[str], cycle: str, source_file: str) -> dict[str, object]:
    email = normalize_email(row[0]) if len(row) > 0 else ""
    name = normalize_name(row[1] if len(row) > 1 and row[1] else row[0] if row else "")
    grade = parse_grade(row[3] if len(row) > 3 else "")
    school = normalize_text(row[4] if len(row) > 4 else "")
    subjects = normalize_subjects(row[6] if len(row) > 6 else row[5] if len(row) > 5 else "")

    notes_parts = [
        normalize_text(row[5] if len(row) > 5 else ""),
        normalize_text(row[7] if len(row) > 7 else ""),
        normalize_text(row[8] if len(row) > 8 else ""),
        school,
    ]
    notes = " | ".join(part for part in notes_parts if part)

    return {
        "cycle": cycle,
        "source_file": source_file,
        "name": name,
        "email": email,
        "grade": grade,
        "subject": primary_subject(subjects),
        "subjects": subjects,
        "qualifications": notes,
    }


def mentee_row_to_record(row: list[str], cycle: str, source_file: str) -> dict[str, object]:
    email = normalize_email(row[0]) if len(row) > 0 else ""
    name = normalize_name(row[1] if len(row) > 1 and row[1] else row[0] if row else "")
    grade = parse_grade(row[3] if len(row) > 3 else "")
    subjects = normalize_subjects(row[5] if len(row) > 5 else row[4] if len(row) > 4 else "")
    school = normalize_text(row[4] if len(row) > 4 else "")

    notes_parts = [
        normalize_text(row[6] if len(row) > 6 else ""),
        normalize_text(row[7] if len(row) > 7 else ""),
        school,
        cycle,
    ]
    notes = " | ".join(part for part in notes_parts if part)

    return {
        "cycle": cycle,
        "source_file": source_file,
        "name": name,
        "email": email,
        "grade": grade,
        "subject": primary_subject(subjects),
        "subjects": subjects,
        "notes": notes,
    }


def write_clean_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CLEAN_DIR / f"{path.stem}_cleaned.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_lookup_key(value: str) -> str:
    return normalize_text(value).lower()


def collect_profile_lookups(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    for record in records:
        for key in (record.get("name"), record.get("email")):
            if key:
                lookup[build_lookup_key(str(key))] = record
    return lookup


def import_people(
    conn: sqlite3.Connection,
    files: list[Path],
    kind: str,
    dry_run: bool,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]], dict[str, int]]:
    all_records: list[dict[str, object]] = []
    counts = {"files": 0, "rows": 0, "skipped": 0}

    for path in files:
        cycle, _ = detect_cycle(path)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            source_rows = [row for row in reader if any(normalize_text(cell) for cell in row)]

        if not source_rows:
            continue

        counts["files"] += 1
        cleaned_rows: list[dict[str, object]] = []
        for row in source_rows:
            if kind == "mentors":
                record = mentor_row_to_record(row, cycle, path.name)
            else:
                record = mentee_row_to_record(row, cycle, path.name)

            if not record["name"]:
                counts["skipped"] += 1
                continue

            cleaned_rows.append(record)
            all_records.append(record)
            counts["rows"] += 1

        if cleaned_rows:
            fieldnames = ["cycle", "source_file", "name", "email", "grade", "subject", "subjects"]
            if kind == "mentors":
                fieldnames.append("qualifications")
            else:
                fieldnames.append("notes")
            write_clean_csv(path, fieldnames, cleaned_rows)

            if not dry_run:
                for record in cleaned_rows:
                    if kind == "mentors":
                        conn.execute(
                            "INSERT OR REPLACE INTO mentors (name, grade, qualifications, subject, available) VALUES (?, ?, ?, ?, 1)",
                            (
                                record["name"],
                                record["grade"] or 0,
                                record["qualifications"],
                                record["subject"],
                            ),
                        )
                    else:
                        conn.execute(
                            "INSERT OR REPLACE INTO mentees (name, grade, subject) VALUES (?, ?, ?)",
                            (
                                record["name"],
                                record["grade"] or 0,
                                record["subject"],
                            ),
                        )

    return all_records, collect_profile_lookups(all_records), counts


def import_pairings(
    conn: sqlite3.Connection,
    files: list[Path],
    mentor_lookup: dict[str, dict[str, object]],
    mentee_lookup: dict[str, dict[str, object]],
    dry_run: bool,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    counts = {"files": 0, "rows": 0, "skipped": 0}
    all_records: list[dict[str, object]] = []

    for path in files:
        cycle, _ = detect_cycle(path)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            source_rows = [row for row in reader if any(normalize_text(cell) for cell in row.values())]

        if not source_rows:
            continue

        counts["files"] += 1
        cleaned_rows: list[dict[str, object]] = []

        for row in source_rows:
            mentor_name = normalize_name(row.get("Mentor") or row.get("mentor") or "")
            mentor_email = normalize_email(row.get("Mentor email") or row.get("mentor email") or "")
            mentee_name = normalize_name(row.get("Mentee") or row.get("mentee") or "")
            mentee_email = normalize_email(row.get("Mentee email") or row.get("mentee email") or "")
            mentor_grade = parse_grade(row.get("Mentor grade") or row.get("mentor grade") or "")
            mentee_grade = parse_grade(row.get("Mentee grade") or row.get("mentee grade") or "")
            subjects_satisfied = normalize_subjects(row.get("Subjects satisfied") or row.get("subjects satisfied") or "")

            mentor_profile = mentor_lookup.get(build_lookup_key(mentor_email)) or mentor_lookup.get(build_lookup_key(mentor_name)) or {}
            mentee_profile = mentee_lookup.get(build_lookup_key(mentee_email)) or mentee_lookup.get(build_lookup_key(mentee_name)) or {}

            mentor_subjects = normalize_subjects(mentor_profile.get("subjects") or mentor_profile.get("subject") or "")
            mentee_subjects = normalize_subjects(mentee_profile.get("subjects") or mentee_profile.get("subject") or "")

            if not mentor_name or not mentee_name:
                counts["skipped"] += 1
                continue

            grade_gap = abs((mentor_grade or 0) - (mentee_grade or 0))
            match_score = compute_match_score(mentor_grade, mentee_grade, subjects_satisfied)
            subject_count = len(split_multi_value(subjects_satisfied))

            record = {
                "cycle": cycle,
                "source_file": path.name,
                "mentor_name": mentor_name,
                "mentor_email": mentor_email,
                "mentor_grade": mentor_grade,
                "mentor_subjects": mentor_subjects,
                "mentor_notes": normalize_text(mentor_profile.get("qualifications") or mentor_profile.get("notes") or ""),
                "mentee_name": mentee_name,
                "mentee_email": mentee_email,
                "mentee_grade": mentee_grade,
                "mentee_subjects": mentee_subjects,
                "subjects_satisfied": subjects_satisfied,
                "subject_count": subject_count,
                "grade_gap": grade_gap,
                "match_score": match_score,
            }
            cleaned_rows.append(record)
            all_records.append(record)
            counts["rows"] += 1

        if cleaned_rows:
            write_clean_csv(
                path,
                [
                    "cycle",
                    "source_file",
                    "mentor_name",
                    "mentor_email",
                    "mentor_grade",
                    "mentor_subjects",
                    "mentor_notes",
                    "mentee_name",
                    "mentee_email",
                    "mentee_grade",
                    "mentee_subjects",
                    "subjects_satisfied",
                    "subject_count",
                    "grade_gap",
                    "match_score",
                ],
                cleaned_rows,
            )

            if not dry_run:
                conn.execute("DELETE FROM historical_pairings WHERE source_file = ?", (path.name,))
                for record in cleaned_rows:
                    conn.execute(
                        f"""
                        INSERT INTO {PAIRINGS_TABLE}
                            (cycle, source_file, mentor_name, mentor_email, mentor_grade,
                             mentor_subjects, mentor_notes, mentee_name, mentee_email,
                             mentee_grade, mentee_subjects, subjects_satisfied,
                             subject_count, grade_gap, match_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record["cycle"],
                            record["source_file"],
                            record["mentor_name"],
                            record["mentor_email"],
                            record["mentor_grade"] or 0,
                            record["mentor_subjects"],
                            record["mentor_notes"],
                            record["mentee_name"],
                            record["mentee_email"],
                            record["mentee_grade"] or 0,
                            record["mentee_subjects"],
                            record["subjects_satisfied"],
                            record["subject_count"],
                            record["grade_gap"],
                            record["match_score"],
                        ),
                    )

    return all_records, counts


def build_fairness_report(pairings: list[dict[str, object]]) -> dict[str, object]:
    if not pairings:
        return {
            "pairings": 0,
            "cycles": {},
            "top_sacrifice_pairs": [],
            "top_mentor_load": [],
            "top_mentee_load": [],
        }

    cycle_stats: dict[str, dict[str, list[float] | int]] = defaultdict(lambda: {"count": 0, "grade_gaps": [], "scores": [], "subjects": []})
    mentor_load = Counter()
    mentee_load = Counter()

    for record in pairings:
        cycle = str(record["cycle"])
        cycle_stats[cycle]["count"] += 1
        cycle_stats[cycle]["grade_gaps"].append(float(record["grade_gap"]))
        cycle_stats[cycle]["scores"].append(float(record["match_score"]))
        cycle_stats[cycle]["subjects"].append(float(record["subject_count"]))
        mentor_load[str(record["mentor_name"])] += 1
        mentee_load[str(record["mentee_name"])] += 1

    cycles_report = {
        cycle: {
            "pairings": stats["count"],
            "avg_grade_gap": round(mean(stats["grade_gaps"]), 2) if stats["grade_gaps"] else 0.0,
            "avg_match_score": round(mean(stats["scores"]), 3) if stats["scores"] else 0.0,
            "avg_subjects_satisfied": round(mean(stats["subjects"]), 2) if stats["subjects"] else 0.0,
        }
        for cycle, stats in sorted(cycle_stats.items())
    }

    sacrifice_pairs = sorted(
        pairings,
        key=lambda record: (float(record["grade_gap"]), -float(record["match_score"])),
        reverse=True,
    )[:10]

    return {
        "pairings": len(pairings),
        "cycles": cycles_report,
        "top_sacrifice_pairs": [
            {
                "cycle": record["cycle"],
                "mentor": record["mentor_name"],
                "mentee": record["mentee_name"],
                "grade_gap": record["grade_gap"],
                "match_score": record["match_score"],
                "subjects_satisfied": record["subjects_satisfied"],
            }
            for record in sacrifice_pairs
        ],
        "top_mentor_load": mentor_load.most_common(10),
        "top_mentee_load": mentee_load.most_common(10),
    }


def discover_files() -> tuple[list[Path], list[Path], list[Path]]:
    mentor_files: list[Path] = []
    mentee_files: list[Path] = []
    pairing_files: list[Path] = []

    for path in DATA_DIR.glob("*.csv"):
        if "cleaned" in {part.lower() for part in path.parts}:
            continue
        kind = detect_kind(path)
        if kind == "mentors":
            mentor_files.append(path)
        elif kind == "mentees":
            mentee_files.append(path)
        elif kind == "pairings":
            pairing_files.append(path)

    sort_key = lambda item: detect_cycle(item)[1]
    mentor_files.sort(key=sort_key)
    mentee_files.sort(key=sort_key)
    pairing_files.sort(key=sort_key)
    return mentor_files, mentee_files, pairing_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean CSVs, import them into SQLite, and train the mentor matcher.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report only without writing to the database")
    parser.add_argument("--no-train", action="store_true", help="Skip retraining after import")
    args = parser.parse_args()

    if not DATA_DIR.exists():
        raise SystemExit(f"Data directory not found: {DATA_DIR}")

    mentor_files, mentee_files, pairing_files = discover_files()
    if not mentor_files and not mentee_files and not pairing_files:
        raise SystemExit(f"No CSV files found in {DATA_DIR}")

    conn = connect_db()
    ensure_tables(conn)

    print(f"Found {len(mentor_files)} mentor files, {len(mentee_files)} mentee files, {len(pairing_files)} pairing files.")
    print("Cleaning mentor and mentee data...")

    app_conn = connect_app_db()
    training_conn = connect_training_db()
    ensure_tables(training_conn)

    mentor_records, mentor_lookup, mentor_counts = import_people(app_conn, mentor_files, "mentors", args.dry_run)
    mentee_records, mentee_lookup, mentee_counts = import_people(app_conn, mentee_files, "mentees", args.dry_run)

    print("Cleaning pairings and building fairness report...")
    pairing_records, pairing_counts = import_pairings(training_conn, pairing_files, mentor_lookup, mentee_lookup, args.dry_run)

    if not args.dry_run:
        app_conn.commit()
        training_conn.commit()

    app_conn.close()
    training_conn.close()

    report = build_fairness_report(pairing_records)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    (CLEAN_DIR / "import_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nImport summary:")
    print(f"  Mentors imported: {mentor_counts['rows']}")
    print(f"  Mentees imported: {mentee_counts['rows']}")
    print(f"  Pairings imported: {pairing_counts['rows']}")
    print(f"  Pairings skipped: {pairing_counts['skipped']}")

    if report["cycles"]:
        print("\nCycle fairness summary:")
        for cycle, stats in report["cycles"].items():
            print(
                f"  {cycle}: {stats['pairings']} pairings, avg grade gap {stats['avg_grade_gap']}, "
                f"avg score {stats['avg_match_score']}, avg subjects satisfied {stats['avg_subjects_satisfied']}"
            )

    if report["top_sacrifice_pairs"]:
        print("\nLargest sacrifice pairs:")
        for record in report["top_sacrifice_pairs"][:5]:
            print(
                f"  {record['cycle']} | {record['mentor']} ↔ {record['mentee']} | "
                f"gap {record['grade_gap']} | score {record['match_score']} | {record['subjects_satisfied']}"
            )

    if not args.no_train and not args.dry_run:
        print("\nRetraining model from cleaned historical pairings...")
        from models.train import train_model

        result = train_model()
        if result is None:
            print("Model training skipped because there was not enough data yet.")
        else:
            print("Model retraining complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())