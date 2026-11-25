"""Download the support ticket dataset and load it into a SQLite table.

Run from the project root:  python src/ingest.py
Safe to re-run: the table is rebuilt from scratch each time.
"""

import sqlite3
from pathlib import Path

from datasets import load_dataset

DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "tickets.db"


def main():
    # 1. Download (cached after the first run) and convert to a pandas table
    df = load_dataset(DATASET, split="train").to_pandas()

    # 2. Clearer column names for our use case
    df = df.rename(columns={"instruction": "message", "response": "agent_response"})
    df.insert(0, "id", range(1, len(df) + 1))
    df = df[["id", "message", "category", "intent", "agent_response", "flags"]]

    # 3. Write to SQLite (creates the file if it doesn't exist)
    DB_PATH.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("tickets", conn, if_exists="replace", index=False)
        count = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]

    print(f"Loaded {count} tickets into {DB_PATH}")


if __name__ == "__main__":
    main()
