"""One-time migration: upload local CSV/JSON data to MongoDB Atlas.

Usage:
    MONGODB_URI="mongodb+srv://..." python scripts/migrate_to_mongodb.py

Collections created:
    season_matches   – rows from all league season_df.csv files
    league_tables    – rows from all league league_table.csv files
    corner_overrides – rows from corner_overrides.csv
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv(override=True)

import json
import os
import sys
from pathlib import Path

import certifi
import pandas as pd
from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
LEAGUES_DIR = DATA_DIR / "leagues"
CORNER_OVERRIDES_PATH = DATA_DIR / "corner_overrides.csv"


def _clean_for_mongo(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of dicts, replacing NaN with None."""
    records = df.to_dict("records")
    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
    return records


def migrate():
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI environment variable is not set. Cannot sync to MongoDB.")

    client = MongoClient(uri, tlsCAFile=certifi.where())
    db = client.get_default_database("backline")

    # ── Season matches ──
    print("Uploading season matches...")
    frames = []
    for league_dir in sorted(LEAGUES_DIR.iterdir()):
        csv_path = league_dir / "season_df.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["league_slug"] = league_dir.name
        frames.append(df)
        print(f"  {league_dir.name}: {len(df)} rows")

    if not frames:
        print("  No league data found, skipping.")
    else:
        all_matches = pd.concat(frames, ignore_index=True)
        records = _clean_for_mongo(all_matches)
        db.season_matches.drop()
        db.season_matches.insert_many(records)
        print(f"  Inserted {len(records)} total season match rows.")

        # Create indexes for common queries
        db.season_matches.create_index("match_id")
        db.season_matches.create_index("league_slug")
        db.season_matches.create_index("home_team_id")
        db.season_matches.create_index("away_team_id")
        print("  Created indexes on match_id, league_slug, home_team_id, away_team_id.")

    # ── League tables ──
    print("\nUploading league tables...")
    table_count = 0
    db.league_tables.drop()
    for league_dir in sorted(LEAGUES_DIR.iterdir()):
        table_path = league_dir / "league_table.csv"
        if not table_path.exists():
            continue
        df = pd.read_csv(table_path)
        df["league_slug"] = league_dir.name
        records = _clean_for_mongo(df)
        if records:
            db.league_tables.insert_many(records)
            table_count += len(records)
            print(f"  {league_dir.name}: {len(records)} rows")

    db.league_tables.create_index("league_slug")
    print(f"  Inserted {table_count} total league table rows.")

    # ── Corner overrides ──
    print("\nUploading corner overrides...")
    if CORNER_OVERRIDES_PATH.exists():
        df = pd.read_csv(CORNER_OVERRIDES_PATH)
        records = _clean_for_mongo(df)
        db.corner_overrides.drop()
        if records:
            db.corner_overrides.insert_many(records)
            db.corner_overrides.create_index("match_id")
        print(f"  Inserted {len(records)} corner override rows.")
    else:
        print("  No corner_overrides.csv found, skipping.")

    print("\nMigration complete!")
    client.close()


if __name__ == "__main__":
    migrate()
