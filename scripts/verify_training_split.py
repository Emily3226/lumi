import sqlite3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
app_db = root / 'data' / 'lumi.db'
training_db = root / 'data' / 'training.db'

with sqlite3.connect(app_db) as conn:
    app_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='historical_pairings'"
    ).fetchone()

with sqlite3.connect(training_db) as conn:
    training_count = conn.execute("SELECT COUNT(*) FROM historical_pairings").fetchone()[0]

print(f'app_historical_pairings={app_exists}')
print(f'training_pairings={training_count}')