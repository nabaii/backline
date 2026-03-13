import argparse
from data.raw.scrape_data import (
    get_season_data, 
    _build_league_table_from_season_df,
    _default_output_paths_for_league
)

def rescrape_league(league, season="25/26"):
    print(f"\n{'='*60}")
    print(f"Starting FULL re-scrape for {league} ({season})...")
    print(f"{'='*60}")
    
    # 1. Scrape full season
    season_df = get_season_data(season=season, league=league)
    
    # 2. Build table
    league_table = _build_league_table_from_season_df(season_df)
    
    # 3. Save
    paths = _default_output_paths_for_league(league)
    paths["league_dir"].mkdir(parents=True, exist_ok=True)
    
    season_df.to_csv(paths["season_csv_path"])
    league_table.to_csv(paths["league_table_csv_path"], index=False)
    
    print(f"\n[âœ“] FULL rescrape complete for {league}!")
    print(f"  - Saved season_df to {paths['season_csv_path']}")
    print(f"  - Saved league_table to {paths['league_table_csv_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform a full re-scrape of a league")
    parser.add_argument("--league", type=str, required=True, help="League to rescrape (e.g. 'Germany Bundesliga')")
    parser.add_argument("--season", type=str, default="25/26")
    args = parser.parse_args()
    
    rescrape_league(args.league, args.season)
