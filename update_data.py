"""
update_data.py  –  Backline v2  –  Daily Incremental Data Updater
=================================================================

Run this script daily to fetch only *new* match results that aren't
already in your season_df.csv files.  Much faster than a full re-scrape.

Usage
-----
    python update_data.py                  # update all leagues
    python update_data.py --league "England Premier League"  # single league
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ScraperFC as sfc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Import helpers from the main scraper ────────────────────────────────────
from data.raw.scrape_data import (
    DEFAULT_ALL_LEAGUES,
    DEFAULT_LEAGUE_OUTPUT_BASE_DIR,
    _apply_derived_features,
    _build_league_table_from_season_df,
    _coerce_match_id,
    _default_output_paths_for_league,
    _extract_match_datetime,
    _flatten_match_stats_row,
    _is_played_match,
    _league_slug,
    _serialize_shot_payload,
    ensure_match_datetime_and_sort,
)


# ── Throttle ────────────────────────────────────────────────────────────────
SCRAPE_DELAY_SECONDS = 1.5  # polite delay between API calls


def _load_existing_season_df(csv_path: Path) -> pd.DataFrame | None:
    """Load and return the existing season CSV, or None if missing."""
    if not csv_path.exists():
        print(f"  [!!] No existing CSV at {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    # Normalise the index
    if "game_id" in df.columns:
        df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")
        df = df.dropna(subset=["game_id"])
        df["game_id"] = df["game_id"].astype(int)
        df = df.set_index("game_id")
    return df


def _compute_momentum_for_matches(ss, match_ids: list[int]) -> pd.DataFrame | None:
    """Scrape and compute PCA momentum for a list of match IDs."""
    if not match_ids:
        return None

    momentum_rows = []
    for i, mid in enumerate(match_ids, 1):
        print(f"    [{i}/{len(match_ids)}] Momentum for match {mid} ...")
        try:
            mom_df = ss.scrape_match_momentum(mid)
        except Exception as e:
            print(f"      [!!] Failed: {e}")
            continue

        if mom_df.empty or "minute" not in mom_df.columns or "value" not in mom_df.columns:
            continue

        try:
            pivoted = mom_df.pivot_table(
                index=pd.Series([mid], name="game_id"),
                columns="minute",
                values="value",
                aggfunc="first",
            )
            momentum_rows.append(pivoted)
        except Exception:
            continue
        time.sleep(SCRAPE_DELAY_SECONDS * 0.5)

    if not momentum_rows:
        return None

    momentum_combined = pd.concat(momentum_rows)
    all_minutes = [float(i) for i in range(1, 91)]
    momentum_combined = momentum_combined.reindex(columns=all_minutes, fill_value=0).fillna(0)

    # PCA + sigmoid
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(momentum_combined)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X_scaled).flatten()
    pc1_sigmoid = 20 / (1 + np.exp(-pc1)) - 10

    home_momentum = np.where(pc1_sigmoid >= 0, pc1_sigmoid, 10 - np.abs(pc1_sigmoid))
    away_momentum = np.where(pc1_sigmoid < 0, np.abs(pc1_sigmoid), 10 - np.abs(pc1_sigmoid))

    return pd.DataFrame(
        {
            "game_id": momentum_combined.index,
            "home_momentum": np.round(home_momentum, 2),
            "away_momentum": np.round(away_momentum, 2),
        }
    ).set_index("game_id")


def update_league(
    season: str,
    league: str,
    output_base_dir: Path = DEFAULT_LEAGUE_OUTPUT_BASE_DIR,
) -> int:
    """
    Incrementally update a single league's data.

    Returns the number of new matches added.
    """
    paths = _default_output_paths_for_league(league, output_base_dir)
    csv_path = paths["season_csv_path"]
    table_path = paths["league_table_csv_path"]
    league_dir = paths["league_dir"]
    league_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'-' * 60}")
    print(f"  Updating: {league}")
    print(f"  CSV:      {csv_path}")
    print(f"{'-' * 60}")

    # ── 1. Load existing data ──────────────────────────────────────────────
    existing_df = _load_existing_season_df(csv_path)
    existing_ids: set[int] = set()
    if existing_df is not None:
        existing_ids = set(int(x) for x in existing_df.index)
        print(f"  [OK] Loaded {len(existing_df)} existing matches")
    else:
        print("  [*] Starting fresh – no existing data found")

    # ── 2. Fetch latest match list from Sofascore ──────────────────────────
    ss = sfc.Sofascore()
    print(f"  [*] Fetching match list for {season} {league} ...")
    matches = ss.get_match_dicts(season, league)
    print(f"  [OK] Found {len(matches)} total matches in season")

    # ── 3. Identify NEW played matches ─────────────────────────────────────
    new_matches = []
    for m in matches:
        mid = _coerce_match_id(m.get("id"))
        if mid is None:
            continue
        if mid in existing_ids:
            continue
        if _is_played_match(m):
            new_matches.append(m)

    if not new_matches:
        print("  [OK] No new matches to scrape – data is up to date!")
        return 0

    print(f"  [*] {len(new_matches)} new match(es) to scrape\n")

    # ── 4. Scrape stats + shots for each new match ─────────────────────────
    new_rows = []
    failed = 0
    for i, match_info in enumerate(new_matches, 1):
        mid = match_info["id"]
        home = match_info.get("homeTeam", {}).get("name", "?")
        away = match_info.get("awayTeam", {}).get("name", "?")
        print(f"    [{i}/{len(new_matches)}] {home} vs {away} (id: {mid}) ...")

        try:
            match_stats = ss.scrape_team_match_stats(mid)
        except Exception as e:
            print(f"      [!!] Stats failed: {e}")
            failed += 1
            continue

        match_shots = None
        try:
            match_shots = ss.scrape_match_shots(mid)
        except Exception as e:
            print(f"      [!!] Shots failed: {e}")

        row = _flatten_match_stats_row(match_info, match_stats, match_shots=match_shots)
        new_rows.append(row)
        time.sleep(SCRAPE_DELAY_SECONDS)

    if not new_rows:
        print(f"  [!!] All {len(new_matches)} new matches failed to scrape")
        return 0

    print(f"\n  [OK] Scraped {len(new_rows)} new matches ({failed} failed)")

    # ── 5. Build DataFrame for new matches ─────────────────────────────────
    new_df = pd.DataFrame(new_rows)
    new_df = new_df.set_index("game_id")

    # ── 6. Momentum for new matches ────────────────────────────────────────
    print("  [*] Computing momentum for new matches ...")
    new_match_ids = [int(x) for x in new_df.index]
    mom_df = _compute_momentum_for_matches(ss, new_match_ids)
    if mom_df is not None:
        new_df = new_df.join(mom_df, how="left")
        print(f"  [OK] Momentum computed for {len(mom_df)} matches")
    else:
        new_df["home_momentum"] = np.nan
        new_df["away_momentum"] = np.nan

    # Add league_name column
    new_df["league_name"] = league

    # ── 7. Merge with existing data ────────────────────────────────────────
    if existing_df is not None:
        combined_df = pd.concat([existing_df, new_df], axis=0)
    else:
        combined_df = new_df

    # Remove duplicates (just in case)
    combined_df = combined_df[~combined_df.index.duplicated(keep="first")]

    # ── 8. Recompute derived features on full dataset ──────────────────────
    combined_df = combined_df.reset_index()
    combined_df = _apply_derived_features(combined_df)
    combined_df = ensure_match_datetime_and_sort(
        combined_df, season=season, league=league, matches=matches, ss=ss
    )

    # Re-set game_id as index for saving
    if "game_id" in combined_df.columns:
        combined_df = combined_df.set_index("game_id")

    # ── 9. Save updated CSV ────────────────────────────────────────────────
    combined_df.to_csv(csv_path)
    print(f"  [OK] Saved {len(combined_df)} matches → {csv_path}")

    # ── 10. Rebuild league table ───────────────────────────────────────────
    try:
        combined_df_reset = combined_df.reset_index()
        league_table = _build_league_table_from_season_df(combined_df_reset)
        league_table.to_csv(table_path, index=False)
        print(f"  [OK] League table saved → {table_path}")
    except Exception as e:
        print(f"  [!!] League table rebuild failed: {e}")

    return len(new_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Incrementally update Backline match data"
    )
    parser.add_argument(
        "--league",
        type=str,
        default=None,
        help='Specific league to update, e.g. "England Premier League". '
             "If omitted, all configured leagues are updated.",
    )
    parser.add_argument(
        "--season",
        type=str,
        default="25/26",
        help="Season identifier (default: 25/26)",
    )
    args = parser.parse_args()

    leagues = [args.league] if args.league else DEFAULT_ALL_LEAGUES

    print("=" * 60)
    print("  Backline v2  –  Daily Data Updater")
    print("=" * 60)
    print(f"  Season:  {args.season}")
    print(f"  Leagues: {len(leagues)}")
    print(f"  Time:    {pd.Timestamp.now()}")

    total_new = 0
    for league in leagues:
        try:
            count = update_league(args.season, league)
            total_new += count
        except Exception as e:
            print(f"\n  [X] ERROR updating {league}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  Done!  {total_new} new match(es) added across {len(leagues)} league(s).")
    print("=" * 60)


if __name__ == "__main__":
    main()
