import os
import pandas as pd
from data.raw.scrape_data import get_season_data

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")


def construct_league_table(season_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a league standings table from match results.

    Points system:
        - Win  (more normaltime goals)  → 3 pts
        - Draw (equal normaltime goals) → 1 pt each
        - Loss                          → 0 pts

    Parameters
    ----------
    season_df : pd.DataFrame
        DataFrame where each row is a match with columns:
        home_team, away_team, home_normaltime, away_normaltime.

    Returns
    -------
    pd.DataFrame
        League table sorted by points (desc), then goal difference (desc),
        then goals scored (desc).  Columns:
            team, played, won, drawn, lost, goals_for, goals_against,
            goal_difference, points
    """
    records: dict[str, dict] = {}

    def _ensure(team: str) -> None:
        if team not in records:
            records[team] = {
                "team": team,
                "played": 0,
                "won": 0,
                "drawn": 0,
                "lost": 0,
                "goals_for": 0,
                "goals_against": 0,
            }

    for _, row in season_df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_goals = int(row["home_normaltime"])
        away_goals = int(row["away_normaltime"])

        _ensure(home)
        _ensure(away)

        records[home]["played"] += 1
        records[away]["played"] += 1
        records[home]["goals_for"] += home_goals
        records[home]["goals_against"] += away_goals
        records[away]["goals_for"] += away_goals
        records[away]["goals_against"] += home_goals

        if home_goals > away_goals:          # home win
            records[home]["won"] += 1
            records[home]["points"] = records[home].get("points", 0) + 3
            records[away]["lost"] += 1
        elif away_goals > home_goals:        # away win
            records[away]["won"] += 1
            records[away]["points"] = records[away].get("points", 0) + 3
            records[home]["lost"] += 1
        else:                                 # draw
            records[home]["drawn"] += 1
            records[away]["drawn"] += 1
            records[home]["points"] = records[home].get("points", 0) + 1
            records[away]["points"] = records[away].get("points", 0) + 1

    table = pd.DataFrame(records.values())
    table["points"] = table.get("points", 0).fillna(0).astype(int)
    table["goal_difference"] = table["goals_for"] - table["goals_against"]
    table = table.sort_values(
        by=["points", "goal_difference", "goals_for"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    table.index += 1                        # 1-based ranking
    table.index.name = "rank"
    return table


if __name__ == "__main__":
    print("=" * 60)
    print("  Backline v2  –  Season Scraper & League Table Builder")
    print("=" * 60)

    # ── 1. Scrape season EPL 25/26 data ────────────────────────────────────
    print("\n[*] Starting match data scraping...")
    season_df = get_season_data()
    print(f"[✓] Completed scraping {len(season_df)} matches\n")

    # ── 2. Build league table from match results ─────────────────
    league_table = construct_league_table(season_df)

    print("\n")
    print("=" * 60)
    print("  LEAGUE TABLE")
    print("=" * 60)
    print(league_table.to_string())

    # ── 3. Save both DataFrames to data/raw/ ─────────────────────
    season_path = os.path.join(RAW_DIR, "season_df.csv")
    table_path = os.path.join(RAW_DIR, "league_table.csv")

    season_df.to_csv(season_path)
    league_table.to_csv(table_path)

    print(f"\n[✓] Season data saved  → {season_path}")
    print(f"[✓] League table saved → {table_path}")
