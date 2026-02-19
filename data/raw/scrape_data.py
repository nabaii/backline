import pandas as pd
import ScraperFC as sfc
import numpy as np
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


DATE_COLUMN_CANDIDATES = (
    "match_datetime",
    "datetime",
    "date",
    "kickoff",
    "kickoff_time",
    "start_time",
    "startTimestamp",
    "start_timestamp",
)

DEFAULT_MULTI_LEAGUES = [
    "France Ligue 1",
    "Germany Bundesliga",
    "Italy Serie A",
    "Spain La Liga",
]
DEFAULT_ALL_LEAGUES = ["England Premier League"] + DEFAULT_MULTI_LEAGUES
DEFAULT_LEAGUE_OUTPUT_BASE_DIR = Path("data/raw/leagues")


def _league_slug(league: str) -> str:
    text = str(league or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "unknown_league"


def _default_output_paths_for_league(league: str, base_dir: str | Path = DEFAULT_LEAGUE_OUTPUT_BASE_DIR):
    root = Path(base_dir)
    league_dir = root / _league_slug(league)
    return {
        "league_dir": league_dir,
        "season_csv_path": league_dir / "season_df.csv",
        "league_table_csv_path": league_dir / "league_table.csv",
        "matches_json_path": league_dir / "matches_data.json",
    }


def _league_name_lookup():
    lookup = {}
    for league in DEFAULT_ALL_LEAGUES:
        lookup[_league_slug(league)] = league
    return lookup


def _display_team_name(name):
    if name is None:
        return ""
    text = str(name).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text.replace("_", " ").title()


def _coerce_match_id(value):
    """Return int match id when possible, else None."""
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_datetime_like(value):
    """Parse timestamp/string-like values into UTC pandas Timestamp."""
    if value is None:
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            return value.tz_localize("UTC")
        return value.tz_convert("UTC")

    if isinstance(value, dict):
        return pd.NaT

    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return pd.NaT
        number = float(value)
        unit = "ms" if abs(number) >= 1e12 else "s"
        return pd.to_datetime(number, unit=unit, utc=True, errors="coerce")

    text = str(value).strip()
    if not text:
        return pd.NaT

    # Numeric strings can still represent epoch values.
    try:
        number = float(text)
        if np.isfinite(number):
            unit = "ms" if abs(number) >= 1e12 else "s"
            parsed = pd.to_datetime(number, unit=unit, utc=True, errors="coerce")
            if not pd.isna(parsed):
                return parsed
    except ValueError:
        pass

    return pd.to_datetime(text, utc=True, errors="coerce")


def _extract_match_datetime(match_dict):
    """Extract and parse the best available datetime from one Sofascore match dict."""
    if not isinstance(match_dict, dict):
        return pd.NaT

    timestamp_keys = (
        "startTimestamp",
        "start_timestamp",
        "matchTimestamp",
        "kickoffTimestamp",
        "timestamp",
    )
    datetime_keys = (
        "startDate",
        "date",
        "datetime",
        "start_time",
        "startTime",
        "kickoff",
        "time",
    )

    for key in timestamp_keys + datetime_keys:
        if key in match_dict:
            parsed = _parse_datetime_like(match_dict.get(key))
            if not pd.isna(parsed):
                return parsed

    return pd.NaT


def _resolve_match_id_series(df: pd.DataFrame):
    """Find a match-id series from index/columns, aligned to dataframe index."""
    if df.index.name in {"game_id", "match_id", "id"}:
        return pd.Series(df.index, index=df.index).map(_coerce_match_id)

    for col in ("game_id", "match_id", "id"):
        if col in df.columns:
            return df[col].map(_coerce_match_id)

    return None


def ensure_match_datetime_and_sort(
    season_df: pd.DataFrame,
    season: str = "25/26",
    league: str = "England Premier League",
    matches=None,
    ss=None,
):
    """
    Ensure `match_datetime` exists and sort rows chronologically.

    If no usable date column exists, this fetches season matches from Sofascore
    and maps by match id (`game_id`/`match_id`) to enrich the dataframe.
    """
    df = season_df.copy()
    match_ids = _resolve_match_id_series(df)
    if match_ids is None:
        raise ValueError("Could not find match ID field (expected game_id/match_id/id).")

    # 1) Try existing date-like columns first.
    for col in DATE_COLUMN_CANDIDATES:
        if col in df.columns:
            parsed = df[col].map(_parse_datetime_like)
            if parsed.notna().any():
                df["match_datetime"] = parsed
                break

    # 2) Fallback: enrich from Sofascore match dicts if still missing.
    if "match_datetime" not in df.columns or not df["match_datetime"].notna().any():
        if matches is None:
            if ss is None:
                ss = sfc.Sofascore()
            print(
                f"[*] No usable date column found. Fetching match metadata via "
                f"Sofascore.get_match_dicts('{season}', '{league}')..."
            )
            matches = ss.get_match_dicts(season, league)

        dt_by_id = {}
        for match in matches:
            mid = _coerce_match_id(match.get("id"))
            dt = _extract_match_datetime(match)
            if mid is not None and not pd.isna(dt):
                dt_by_id[mid] = dt

        df["match_datetime"] = match_ids.map(dt_by_id)
        matched_count = int(df["match_datetime"].notna().sum())
        print(f"[*] Datetime enrichment matched {matched_count}/{len(df)} rows by match id.")

    df["match_datetime"] = pd.to_datetime(df["match_datetime"], utc=True, errors="coerce")
    if df["match_datetime"].notna().any():
        df = df.sort_values("match_datetime", kind="mergesort", na_position="last")
    else:
        # Last-resort deterministic order.
        df = df.sort_index()

    return df


def get_season_data(season='25/26', league='England Premier League'):
    """
    Scrape every match in the given season/league and return a single
    DataFrame where each row is one match and columns are the flattened
    match statistics, score/team metadata, derived features, and
    home/away momentum computed via PCA + sigmoid scaling.

    Parameters
    ----------
    season : str
        Season identifier accepted by ScraperFC, e.g. '24/25'.
    league : str
        League name accepted by ScraperFC, e.g. 'England Premier League'.

    Returns
    -------
    pd.DataFrame
        Season dataframe indexed by ``game_id``, sorted by index.
    """
    ss = sfc.Sofascore()

    # â”€â”€ 1. Fetch the list of match dicts for the season â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    matches = ss.get_match_dicts(season, league)

    # Build a lookup so we can quickly find metadata by match id
    match_lookup = {}
    for item in matches:
        match_id = item['id']
        match_lookup[match_id] = item

    # â”€â”€ 2. Scrape stats for every match & flatten in-memory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    all_rows = []
    total = len(match_lookup)
    failed_count = 0

    print(f"\n[*] Scraping match statistics for {total} matches...")
    for i, (match_id, match_info) in enumerate(match_lookup.items(), start=1):
        home_team = match_info['homeTeam']['name']
        away_team = match_info['awayTeam']['name']

        print(f"  [{i}/{total}] Scraping {home_team} vs {away_team} "
              f"(id: {match_id}) ...")

        try:
            match_stats = ss.scrape_team_match_stats(match_id)
        except Exception as e:
            print(f"    [!] Failed to scrape match {match_id}: {e}")
            failed_count += 1
            continue

        match_shots = None
        try:
            match_shots = ss.scrape_match_shots(match_id)
        except Exception as e:
            print(f"    [!] Failed to scrape shots for match {match_id}: {e}")

        row_dict = _flatten_match_stats_row(match_info, match_stats, match_shots=match_shots)

        all_rows.append(row_dict)

    # â”€â”€ 4. Assemble a single DataFrame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    successful_count = len(all_rows)
    print(f"[âœ“] Match statistics scraped: {successful_count}/{total} successful"
          f" ({failed_count} failed)\n")
    
    season_df = pd.DataFrame(all_rows)
    season_df = season_df.set_index('game_id')
    season_df = ensure_match_datetime_and_sort(
        season_df,
        season=season,
        league=league,
        matches=matches,
        ss=ss,
    )

    # â”€â”€ 5. Create derived features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    season_df['xg_diff'] = season_df['expected_goals_home'] - season_df['expected_goals_away']
    season_df['xg_diff_home'] = season_df['expected_goals_home'] - season_df['expected_goals_away']
    season_df['xg_diff_away'] = season_df['expected_goals_away'] - season_df['expected_goals_home']
    season_df['total_goals'] = season_df['home_normaltime'] + season_df['away_normaltime']
    season_df['total_xg'] = season_df['expected_goals_home'] + season_df['expected_goals_away']

    # Field tilt
    ftp_sum = season_df['final_third_phase_home'] + season_df['final_third_phase_away']
    season_df['field_tilt_home'] = season_df['final_third_phase_home'] / ftp_sum
    season_df['field_tilt_away'] = season_df['final_third_phase_away'] / ftp_sum

    # Win/draw flags
    season_df['home_win'] = (season_df['game_winner'] == 1).astype(int)
    season_df['away_win'] = (season_df['game_winner'] == 2).astype(int)
    season_df['draw']     = (season_df['game_winner'] == 3).astype(int)

    # â”€â”€ 6. Scrape momentum data & compute PCA-based momentum â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"[*] Scraping momentum data for {len(season_df)} matches...")
    momentum_rows = []
    momentum_failed = 0

    for i, m in enumerate(matches, start=1):
        mid = m['id']
        if mid not in season_df.index:
            continue  # skip matches we failed to scrape stats for

        completed = len(momentum_rows) + 1
        print(f"  [{completed}/{len(season_df)}] Momentum for match {mid} ...")
        try:
            mom_df = ss.scrape_match_momentum(mid)
        except Exception as e:
            print(f"    [!] Failed: {e}")
            momentum_failed += 1
            continue

        if mom_df.empty or 'minute' not in mom_df.columns or 'value' not in mom_df.columns:
            print(f"    [!] Empty or missing columns -- skipping")
            momentum_failed += 1
            continue

        # Pivot: minute values become columns, single row per match
        try:
            pivoted = mom_df.pivot_table(
                index=pd.Series([mid], name='game_id'),
                columns='minute',
                values='value',
                aggfunc='first'
            )
            momentum_rows.append(pivoted)
        except Exception as e:
            print(f"    [!] Pivot failed: {e}")
            continue

    if momentum_rows:
        momentum_combined = pd.concat(momentum_rows)
        # Standardize to minutes 1â€“90 (fill missing minutes with 0)
        all_minutes = [float(i) for i in range(1, 91)]
        momentum_combined = momentum_combined.reindex(columns=all_minutes, fill_value=0)
        momentum_combined = momentum_combined.fillna(0)

        # â”€â”€ PCA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(momentum_combined)

        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X_scaled).flatten()

        # â”€â”€ Sigmoid scaling to [-10, +10] â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Scale PC1 scores to [0, 1] range first using StandardScaler
        pc1_scaler = StandardScaler()
        pc1_scores = pc1_scaler.fit_transform(pc1.reshape(-1, 1)).flatten()

        # Sigmoid: maps to [-10, +10]
        pc1_sigmoid = 20 / (1 + np.exp(-pc1)) - 10

        # â”€â”€ Home / Away momentum via np.where â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Positive sigmoid â†’ home had more momentum
        # Negative sigmoid â†’ away had more momentum
        home_momentum = np.where(
            pc1_sigmoid >= 0,
            pc1_sigmoid,
            10 - np.abs(pc1_sigmoid)
        )
        away_momentum = np.where(
            pc1_sigmoid < 0,
            np.abs(pc1_sigmoid),
            10 - np.abs(pc1_sigmoid)
        )

        # Build a small DataFrame to merge back
        mom_result = pd.DataFrame({
            'game_id':        momentum_combined.index,
            'home_momentum':  np.round(home_momentum, 2),
            'away_momentum':  np.round(away_momentum, 2),
        }).set_index('game_id')

        # Merge momentum into season dataframe
        season_df = season_df.join(mom_result, how='left')

        print(f"[âœ“] Momentum computed for {len(mom_result)} matches "
              f"({momentum_failed} failed, explained variance: {pca.explained_variance_ratio_[0]:.2%})\n")
    else:
        print(f"[!] No momentum data was scraped -- columns will be NaN\n")
        season_df['home_momentum'] = np.nan
        season_df['away_momentum'] = np.nan

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n[OK] Done -- scraped {len(season_df)} matches.  "
          f"Shape: {season_df.shape}")

    return season_df


def enrich_existing_season_csv(
    csv_path=None,
    season="25/26",
    league="England Premier League",
    output_base_dir=DEFAULT_LEAGUE_OUTPUT_BASE_DIR,
):
    """Enrich existing season CSV with `match_datetime` and sort chronologically."""
    if csv_path is None:
        csv_path = _default_output_paths_for_league(league, output_base_dir)["season_csv_path"]
    df = pd.read_csv(csv_path)
    df = ensure_match_datetime_and_sort(df, season=season, league=league)
    df.to_csv(csv_path, index=False)
    print(f"[OK] Updated {csv_path} with match_datetime and chronological ordering.")
    return df


def _safe_nested_get(payload, path, default=None):
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _normalize_stat_name(name):
    return (
        str(name)
        .lower()
        .replace(" ", "_")
        .replace("%", "")
        .replace("(", "")
        .replace(")", "")
    )


def _is_played_match(match_info):
    home_normaltime = _safe_nested_get(match_info, ("homeScore", "normaltime"))
    away_normaltime = _safe_nested_get(match_info, ("awayScore", "normaltime"))

    if home_normaltime is not None and away_normaltime is not None:
        return True

    status = match_info.get("status")
    if isinstance(status, dict):
        status_tokens = [
            str(status.get("type", "")),
            str(status.get("description", "")),
            str(status.get("code", "")),
        ]
        status_text = " ".join(status_tokens).lower()
        if any(token in status_text for token in ("finished", "full", "ended", "aet", "after penalties")):
            return True
    return False


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def _serialize_shot_payload(shot_df: pd.DataFrame):
    """
    Build nested home/away shot payload JSON strings:
    {"xg": [...], "count": N}
    """
    empty_payload = json.dumps({"xg": [], "count": 0}, ensure_ascii=False)
    if shot_df is None or shot_df.empty:
        return empty_payload, empty_payload
    if "isHome" not in shot_df.columns or "xg" not in shot_df.columns:
        return empty_payload, empty_payload

    frame = shot_df.copy()
    frame["isHomeBool"] = frame["isHome"].map(_to_bool)
    frame["xg"] = pd.to_numeric(frame["xg"], errors="coerce")
    frame = frame.dropna(subset=["xg"])

    home_xg = [float(v) for v in frame[frame["isHomeBool"]]["xg"].tolist()]
    away_xg = [float(v) for v in frame[~frame["isHomeBool"]]["xg"].tolist()]

    home_payload = json.dumps({"xg": home_xg, "count": len(home_xg)}, ensure_ascii=False)
    away_payload = json.dumps({"xg": away_xg, "count": len(away_xg)}, ensure_ascii=False)
    return home_payload, away_payload


def _refresh_shot_payloads_for_existing_rows(df: pd.DataFrame, ss, force: bool = False) -> pd.DataFrame:
    """
    Backfill or refresh nested home_shots/away_shots payloads for existing matches.
    When force=False, only rows with missing/blank payloads are refreshed.
    """
    out = df.copy()
    if "game_id" not in out.columns:
        return out

    if "home_shots" not in out.columns:
        out["home_shots"] = None
    if "away_shots" not in out.columns:
        out["away_shots"] = None

    def needs_refresh(value):
        if value is None:
            return True
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return True
        return False

    total_rows = len(out)
    refreshed = 0
    failed = 0
    for idx, row in out.iterrows():
        game_id = _coerce_match_id(row.get("game_id"))
        if game_id is None:
            continue

        if not force:
            if not (needs_refresh(row.get("home_shots")) or needs_refresh(row.get("away_shots"))):
                continue

        try:
            shot_df = ss.scrape_match_shots(game_id)
            home_payload, away_payload = _serialize_shot_payload(shot_df)
            out.at[idx, "home_shots"] = home_payload
            out.at[idx, "away_shots"] = away_payload
            refreshed += 1
        except Exception:
            failed += 1

    print(f"[*] Shot payload refresh complete: refreshed={refreshed}, failed={failed}, total_rows={total_rows}")
    return out


def _flatten_match_stats_row(match_info, match_stats, match_shots=None):
    row_dict = {}
    for _, row in match_stats.iterrows():
        feature = _normalize_stat_name(row.get("name", ""))
        if not feature:
            continue
        row_dict[f"{feature}_home"] = row.get("homeValue")
        row_dict[f"{feature}_away"] = row.get("awayValue")

    home_team_name = _safe_nested_get(match_info, ("homeTeam", "name"), default="")
    away_team_name = _safe_nested_get(match_info, ("awayTeam", "name"), default="")

    row_dict["home_h1_goals"] = _safe_nested_get(match_info, ("homeScore", "period1"), default=0)
    row_dict["home_h2_goals"] = _safe_nested_get(match_info, ("homeScore", "period2"), default=0)
    row_dict["home_normaltime"] = _safe_nested_get(match_info, ("homeScore", "normaltime"), default=0)
    row_dict["away_h1_goals"] = _safe_nested_get(match_info, ("awayScore", "period1"), default=0)
    row_dict["away_h2_goals"] = _safe_nested_get(match_info, ("awayScore", "period2"), default=0)
    row_dict["away_normaltime"] = _safe_nested_get(match_info, ("awayScore", "normaltime"), default=0)
    row_dict["game_id"] = match_info.get("id")
    row_dict["match_datetime"] = _extract_match_datetime(match_info)
    row_dict["game_winner"] = match_info.get("winnerCode")
    row_dict["home_team"] = str(home_team_name).lower().replace(" ", "_")
    row_dict["away_team"] = str(away_team_name).lower().replace(" ", "_")
    row_dict["nameCode_home"] = _safe_nested_get(match_info, ("homeTeam", "nameCode"))
    row_dict["nameCode_away"] = _safe_nested_get(match_info, ("awayTeam", "nameCode"))
    row_dict["league_id"] = _safe_nested_get(match_info, ("tournament", "uniqueTournament", "id"))
    row_dict["season_id"] = _safe_nested_get(match_info, ("season", "id"))
    row_dict["home_team_id"] = _safe_nested_get(match_info, ("homeTeam", "id"))
    row_dict["away_team_id"] = _safe_nested_get(match_info, ("awayTeam", "id"))
    home_shots_payload, away_shots_payload = _serialize_shot_payload(match_shots)
    row_dict["home_shots"] = home_shots_payload
    row_dict["away_shots"] = away_shots_payload

    return row_dict


def _apply_derived_features(df):
    out = df.copy()

    if "home_shots" not in out.columns:
        out["home_shots"] = json.dumps({"xg": [], "count": 0}, ensure_ascii=False)
    else:
        out["home_shots"] = out["home_shots"].fillna(json.dumps({"xg": [], "count": 0}, ensure_ascii=False))

    if "away_shots" not in out.columns:
        out["away_shots"] = json.dumps({"xg": [], "count": 0}, ensure_ascii=False)
    else:
        out["away_shots"] = out["away_shots"].fillna(json.dumps({"xg": [], "count": 0}, ensure_ascii=False))

    for col in ("expected_goals_home", "expected_goals_away", "home_normaltime", "away_normaltime"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if {"expected_goals_home", "expected_goals_away"}.issubset(out.columns):
        out["xg_diff"] = out["expected_goals_home"] - out["expected_goals_away"]
        out["xg_diff_home"] = out["expected_goals_home"] - out["expected_goals_away"]
        out["xg_diff_away"] = out["expected_goals_away"] - out["expected_goals_home"]
        out["total_xg"] = out["expected_goals_home"] + out["expected_goals_away"]

    if {"home_normaltime", "away_normaltime"}.issubset(out.columns):
        out["total_goals"] = out["home_normaltime"] + out["away_normaltime"]

    if {"final_third_phase_home", "final_third_phase_away"}.issubset(out.columns):
        home_phase = pd.to_numeric(out["final_third_phase_home"], errors="coerce")
        away_phase = pd.to_numeric(out["final_third_phase_away"], errors="coerce")
        phase_sum = home_phase + away_phase
        out["field_tilt_home"] = np.where(phase_sum > 0, home_phase / phase_sum, np.nan)
        out["field_tilt_away"] = np.where(phase_sum > 0, away_phase / phase_sum, np.nan)

    if "game_winner" in out.columns:
        winner = pd.to_numeric(out["game_winner"], errors="coerce")
        out["home_win"] = (winner == 1).astype(int)
        out["away_win"] = (winner == 2).astype(int)
        out["draw"] = (winner == 3).astype(int)

    return out


def _build_league_table_from_season_df(season_df):
    required = {"home_team", "away_team", "home_normaltime", "away_normaltime", "game_winner"}
    missing = sorted(required - set(season_df.columns))
    if missing:
        raise ValueError(f"Cannot build league table; season_df is missing columns: {missing}")

    df = season_df.copy()
    df["home_normaltime"] = pd.to_numeric(df["home_normaltime"], errors="coerce")
    df["away_normaltime"] = pd.to_numeric(df["away_normaltime"], errors="coerce")
    df["game_winner"] = pd.to_numeric(df["game_winner"], errors="coerce")
    df = df.dropna(subset=["home_team", "away_team", "home_normaltime", "away_normaltime", "game_winner"])

    home_rows = pd.DataFrame(
        {
            "team": df["home_team"],
            "won": (df["game_winner"] == 1).astype(int),
            "drawn": (df["game_winner"] == 3).astype(int),
            "lost": (df["game_winner"] == 2).astype(int),
            "goals_for": df["home_normaltime"].astype(int),
            "goals_against": df["away_normaltime"].astype(int),
        }
    )
    away_rows = pd.DataFrame(
        {
            "team": df["away_team"],
            "won": (df["game_winner"] == 2).astype(int),
            "drawn": (df["game_winner"] == 3).astype(int),
            "lost": (df["game_winner"] == 1).astype(int),
            "goals_for": df["away_normaltime"].astype(int),
            "goals_against": df["home_normaltime"].astype(int),
        }
    )

    table = pd.concat([home_rows, away_rows], ignore_index=True)
    table = (
        table.groupby("team", as_index=False)
        .agg(
            won=("won", "sum"),
            drawn=("drawn", "sum"),
            lost=("lost", "sum"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
        )
    )
    table["played"] = table["won"] + table["drawn"] + table["lost"]
    table["points"] = table["won"] * 3 + table["drawn"]
    table["goal_difference"] = table["goals_for"] - table["goals_against"]
    table = table.sort_values(
        by=["points", "goal_difference", "goals_for", "team"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    table["rank"] = table.index + 1
    table = table[
        ["rank", "team", "played", "won", "drawn", "lost", "goals_for", "goals_against", "points", "goal_difference"]
    ]
    return table


def _build_multi_league_table_from_season_df(season_df):
    if season_df.empty:
        return pd.DataFrame(
            columns=[
                "league",
                "rank",
                "team",
                "played",
                "won",
                "drawn",
                "lost",
                "goals_for",
                "goals_against",
                "points",
                "goal_difference",
            ]
        )

    group_col = "league_name" if "league_name" in season_df.columns else "league_id"
    tables = []
    for league_value, group in season_df.groupby(group_col):
        league_table = _build_league_table_from_season_df(group)
        league_table.insert(0, "league", str(league_value))
        tables.append(league_table)

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def _match_fixture_names(match_info):
    home_name = _safe_nested_get(match_info, ("homeTeam", "name"), default=None)
    away_name = _safe_nested_get(match_info, ("awayTeam", "name"), default=None)
    home_name = _display_team_name(home_name) or "Home"
    away_name = _display_team_name(away_name) or "Away"
    return home_name, away_name


def _load_matches_json(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, list) else None
    except Exception:
        return None


def _discover_season_df_files(output_base_dir: str | Path, include_root: bool = True):
    records = []
    base_dir = Path(output_base_dir)
    if base_dir.exists():
        for csv_path in sorted(base_dir.glob("*/season_df.csv")):
            records.append(
                {
                    "season_csv_path": csv_path,
                    "matches_json_path": csv_path.parent / "matches_data.json",
                    "league_slug": csv_path.parent.name,
                    "scope": "league_folder",
                }
            )

    if include_root:
        root_season = Path("data/raw/season_df.csv")
        if root_season.exists():
            records.append(
                {
                    "season_csv_path": root_season,
                    "matches_json_path": Path("data/raw/matches_data.json"),
                    "league_slug": None,
                    "scope": "root",
                }
            )

    return records


def _infer_league_name_for_dataset(df: pd.DataFrame, league_slug: str | None, root_league: str):
    if "league_name" in df.columns:
        values = [str(v).strip() for v in df["league_name"].dropna().unique().tolist() if str(v).strip()]
        if len(values) == 1:
            return values[0]

    if league_slug:
        lookup = _league_name_lookup()
        if league_slug in lookup:
            return lookup[league_slug]

    return root_league


def _resolve_match_id_column(df: pd.DataFrame):
    if "game_id" in df.columns:
        return "game_id"
    if "match_id" in df.columns:
        return "match_id"
    if "id" in df.columns:
        return "id"
    return None


def _build_fixture_map_from_df(df: pd.DataFrame):
    fixture_map = {}
    id_col = _resolve_match_id_column(df)
    if id_col is None:
        return fixture_map

    for _, row in df.iterrows():
        mid = _coerce_match_id(row.get(id_col))
        if mid is None:
            continue
        home = _display_team_name(row.get("home_team"))
        away = _display_team_name(row.get("away_team"))
        if home or away:
            fixture_map[mid] = (home or "Home", away or "Away")
    return fixture_map


def _build_fixture_map_from_matches(matches):
    fixture_map = {}
    for match_info in matches or []:
        mid = _coerce_match_id(match_info.get("id"))
        if mid is None:
            continue
        fixture_map[mid] = _match_fixture_names(match_info)
    return fixture_map


def _format_missing_rows(rows):
    if not rows:
        return ["- None"]
    return [
        f"- `{row['match_id']}`: {row['home_team']} vs {row['away_team']}{row.get('extra', '')}"
        for row in rows
    ]


def generate_missing_data_report(
    season="25/26",
    output_base_dir=DEFAULT_LEAGUE_OUTPUT_BASE_DIR,
    report_path="text.md",
    include_root=True,
    root_league="England Premier League",
    fetch_missing_match_json=True,
    persist_fetched_match_json=True,
):
    """
    Audit season dataframes and produce a markdown report of:
    - Missing played matches (present in matches JSON, absent in season_df)
    - Rows with missing momentum values
    """
    dataset_records = _discover_season_df_files(output_base_dir=output_base_dir, include_root=include_root)
    if not dataset_records:
        raise FileNotFoundError(f"No season_df.csv files found under {output_base_dir}")

    ss = sfc.Sofascore() if fetch_missing_match_json else None
    report_lines = [
        "# Missing Data Audit",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}",
        f"- Season: {season}",
        f"- Output base dir: {Path(output_base_dir)}",
        "",
        "## Summary",
        "",
        "| Dataset | League | Played Matches (JSON) | Rows In season_df | Missing Match Rows | Missing Momentum Rows |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    summaries = []
    details_by_dataset = []

    for record in dataset_records:
        season_path = Path(record["season_csv_path"])
        matches_json_path = Path(record["matches_json_path"])
        df = pd.read_csv(season_path)
        league_name = _infer_league_name_for_dataset(df, record.get("league_slug"), root_league)
        match_id_col = _resolve_match_id_column(df)
        if match_id_col is None:
            raise ValueError(f"Could not find match id column in {season_path}")

        match_ids = df[match_id_col].map(_coerce_match_id)
        df_match_ids = {mid for mid in match_ids.tolist() if mid is not None}
        matches_payload = _load_matches_json(matches_json_path)
        matches_source = "local_json"

        if matches_payload is None and fetch_missing_match_json and ss is not None and league_name:
            print(f"[*] Fetching missing matches JSON for report: {league_name} ({season})")
            matches_payload = ss.get_match_dicts(season, league_name)
            matches_source = "fetched_live"
            if persist_fetched_match_json:
                matches_json_path.parent.mkdir(parents=True, exist_ok=True)
                with matches_json_path.open("w", encoding="utf-8") as handle:
                    json.dump(matches_payload, handle, ensure_ascii=False, indent=2)
                print(f"[OK] Saved matches JSON for report: {matches_json_path}")
        elif matches_payload is None:
            matches_source = "missing"

        played_match_ids = set()
        missing_match_rows = []
        fixture_by_match_id = _build_fixture_map_from_df(df)

        if matches_payload is not None:
            fixture_by_match_id.update(_build_fixture_map_from_matches(matches_payload))
            played_matches = [m for m in matches_payload if _is_played_match(m)]
            played_match_ids = {
                mid for mid in (_coerce_match_id(m.get("id")) for m in played_matches) if mid is not None
            }
            missing_ids = sorted(played_match_ids - df_match_ids)
            for mid in missing_ids:
                home, away = fixture_by_match_id.get(mid, ("Home", "Away"))
                missing_match_rows.append({"match_id": mid, "home_team": home, "away_team": away, "extra": ""})

        missing_momentum_rows = []
        has_home_momentum = "home_momentum" in df.columns
        has_away_momentum = "away_momentum" in df.columns
        if has_home_momentum or has_away_momentum:
            home_momentum = pd.to_numeric(df["home_momentum"], errors="coerce") if has_home_momentum else pd.Series(np.nan, index=df.index)
            away_momentum = pd.to_numeric(df["away_momentum"], errors="coerce") if has_away_momentum else pd.Series(np.nan, index=df.index)
            missing_mask = home_momentum.isna() | away_momentum.isna()
            missing_df = df[missing_mask].copy()
            if not missing_df.empty:
                missing_df["_match_id"] = missing_df[match_id_col].map(_coerce_match_id)
                missing_df = missing_df.dropna(subset=["_match_id"]).copy()
                missing_df["_match_id"] = missing_df["_match_id"].astype(int)
                missing_df = missing_df.sort_values("_match_id", kind="mergesort")
                for _, row in missing_df.iterrows():
                    mid = int(row["_match_id"])
                    home, away = fixture_by_match_id.get(
                        mid,
                        (_display_team_name(row.get("home_team")) or "Home", _display_team_name(row.get("away_team")) or "Away"),
                    )
                    hm = row.get("home_momentum")
                    am = row.get("away_momentum")
                    missing_momentum_rows.append(
                        {
                            "match_id": mid,
                            "home_team": home,
                            "away_team": away,
                            "extra": f" (home_momentum={hm}, away_momentum={am})",
                        }
                    )

        summaries.append(
            {
                "dataset": str(season_path),
                "league": league_name or "Unknown",
                "played_count": len(played_match_ids),
                "row_count": int(len(df)),
                "missing_match_count": len(missing_match_rows),
                "missing_momentum_count": len(missing_momentum_rows),
                "matches_source": matches_source,
                "matches_json_path": str(matches_json_path),
            }
        )
        details_by_dataset.append(
            {
                "dataset": str(season_path),
                "league": league_name or "Unknown",
                "matches_source": matches_source,
                "matches_json_path": str(matches_json_path),
                "missing_match_rows": missing_match_rows,
                "missing_momentum_rows": missing_momentum_rows,
            }
        )

    for summary in summaries:
        report_lines.append(
            f"| `{summary['dataset']}` | {summary['league']} | {summary['played_count']} | "
            f"{summary['row_count']} | {summary['missing_match_count']} | {summary['missing_momentum_count']} |"
        )

    report_lines.append("")
    report_lines.append("## Details")
    report_lines.append("")

    for detail in details_by_dataset:
        report_lines.append(f"### `{detail['dataset']}`")
        report_lines.append("")
        report_lines.append(f"- League: {detail['league']}")
        report_lines.append(f"- Matches source: {detail['matches_source']}")
        report_lines.append(f"- Matches JSON path: `{detail['matches_json_path']}`")
        report_lines.append("")
        report_lines.append(f"#### Missing Match Rows ({len(detail['missing_match_rows'])})")
        report_lines.append("")
        report_lines.extend(_format_missing_rows(detail["missing_match_rows"]))
        report_lines.append("")
        report_lines.append(f"#### Missing Momentum Rows ({len(detail['missing_momentum_rows'])})")
        report_lines.append("")
        report_lines.extend(_format_missing_rows(detail["missing_momentum_rows"]))
        report_lines.append("")

    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[OK] Missing data report written to: {output_path}")
    return output_path


def scrape_multi_league_season_df(
    season="25/26",
    leagues=None,
    output_base_dir=DEFAULT_LEAGUE_OUTPUT_BASE_DIR,
    combined_season_csv_path=None,
    combined_league_table_csv_path=None,
):
    """
    Scrape multiple leagues for the same season and persist each league into:
    <output_base_dir>/<league_slug>/season_df.csv
    <output_base_dir>/<league_slug>/league_table.csv

    Optionally persist combined outputs when combined_* paths are provided.
    """
    target_leagues = list(leagues) if leagues else list(DEFAULT_MULTI_LEAGUES)
    if not target_leagues:
        raise ValueError("No leagues provided for multi-league scrape.")

    base_dir = Path(output_base_dir)
    all_frames = []
    for league in target_leagues:
        print(f"\n[***] Starting scrape for: {league} ({season})")
        league_df = get_season_data(season=season, league=league).copy()
        if league_df.index.name == "game_id":
            league_df = league_df.reset_index()
        if "game_id" not in league_df.columns and "match_id" in league_df.columns:
            league_df = league_df.rename(columns={"match_id": "game_id"})
        league_df["league_name"] = league
        league_df = _apply_derived_features(league_df)
        if "match_datetime" in league_df.columns:
            league_df["match_datetime"] = pd.to_datetime(league_df["match_datetime"], utc=True, errors="coerce")
            league_df = league_df.sort_values(
                by=["match_datetime", "game_id"],
                ascending=[True, True],
                na_position="last",
                kind="mergesort",
            )
        elif "game_id" in league_df.columns:
            league_df = league_df.sort_values(by=["game_id"], kind="mergesort")

        league_paths = _default_output_paths_for_league(league, base_dir)
        season_path = Path(league_paths["season_csv_path"])
        table_path = Path(league_paths["league_table_csv_path"])
        season_path.parent.mkdir(parents=True, exist_ok=True)
        league_df.to_csv(season_path, index=False)
        print(f"[OK] Saved {league} season_df: {season_path} (rows={len(league_df)})")

        league_table = _build_league_table_from_season_df(league_df)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        league_table.to_csv(table_path, index=False)
        print(f"[OK] Saved {league} league_table: {table_path} (teams={len(league_table)})")

        all_frames.append(league_df)

    combined = pd.concat(all_frames, ignore_index=True, sort=False)
    if "game_id" in combined.columns:
        combined["game_id"] = combined["game_id"].map(_coerce_match_id)
        combined = combined.dropna(subset=["game_id"])
        combined = combined.drop_duplicates(subset=["game_id"], keep="last")

    combined = _apply_derived_features(combined)

    if "match_datetime" in combined.columns:
        combined["match_datetime"] = pd.to_datetime(combined["match_datetime"], utc=True, errors="coerce")
        combined = combined.sort_values(
            by=["match_datetime", "game_id"],
            ascending=[True, True],
            na_position="last",
            kind="mergesort",
        )
    elif "game_id" in combined.columns:
        combined = combined.sort_values(by=["game_id"], kind="mergesort")

    multi_league_table = _build_multi_league_table_from_season_df(combined)

    if combined_season_csv_path:
        combined_season_path = Path(combined_season_csv_path)
        combined_season_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(combined_season_path, index=False)
        print(f"[OK] Combined multi-league season_df saved: {combined_season_path} (rows={len(combined)})")

    if combined_league_table_csv_path:
        combined_table_path = Path(combined_league_table_csv_path)
        combined_table_path.parent.mkdir(parents=True, exist_ok=True)
        multi_league_table.to_csv(combined_table_path, index=False)
        print(f"[OK] Combined multi-league table saved: {combined_table_path} (rows={len(multi_league_table)})")

    return combined, multi_league_table


def update_season_and_league_table(
    season_csv_path=None,
    league_table_csv_path=None,
    matches_json_path=None,
    season="25/26",
    league="England Premier League",
    refresh_shots_for_existing=False,
    output_base_dir=DEFAULT_LEAGUE_OUTPUT_BASE_DIR,
):
    """
    Incrementally update season_df.csv by appending only missing played matches
    from Sofascore matches JSON, then rebuild league_table.csv.
    """
    default_paths = _default_output_paths_for_league(league, output_base_dir)
    season_path = Path(season_csv_path) if season_csv_path else Path(default_paths["season_csv_path"])
    league_table_path = (
        Path(league_table_csv_path) if league_table_csv_path else Path(default_paths["league_table_csv_path"])
    )
    matches_path = Path(matches_json_path) if matches_json_path else Path(default_paths["matches_json_path"])

    if not season_path.exists():
        raise FileNotFoundError(f"Season CSV not found: {season_path}")

    season_df = pd.read_csv(season_path)
    if "game_id" not in season_df.columns and "match_id" in season_df.columns:
        season_df = season_df.rename(columns={"match_id": "game_id"})

    existing_ids_series = season_df["game_id"].map(_coerce_match_id) if "game_id" in season_df.columns else pd.Series(dtype="float64")
    existing_ids = {mid for mid in existing_ids_series.tolist() if mid is not None}

    ss = sfc.Sofascore()
    print(f"[*] Fetching matches JSON via Sofascore.get_match_dicts('{season}', '{league}')...")
    matches = ss.get_match_dicts(season, league)
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    with matches_path.open("w", encoding="utf-8") as handle:
        json.dump(matches, handle, ensure_ascii=False, indent=2)
    print(f"[OK] Saved matches JSON to: {matches_path}")

    missing_matches = []
    for match_info in matches:
        match_id = _coerce_match_id(match_info.get("id"))
        if match_id is None or match_id in existing_ids:
            continue
        if not _is_played_match(match_info):
            continue
        missing_matches.append(match_info)

    def _missing_sort_key(match_info):
        dt = _extract_match_datetime(match_info)
        if pd.isna(dt):
            return (1, pd.Timestamp.max.tz_localize("UTC"))
        return (0, dt)

    missing_matches.sort(key=_missing_sort_key)
    print(f"[*] Existing matches: {len(existing_ids)} | Missing played matches: {len(missing_matches)}")

    new_rows = []
    for idx, match_info in enumerate(missing_matches, start=1):
        match_id = _coerce_match_id(match_info.get("id"))
        home_team = _safe_nested_get(match_info, ("homeTeam", "name"), default="home")
        away_team = _safe_nested_get(match_info, ("awayTeam", "name"), default="away")
        print(f"  [{idx}/{len(missing_matches)}] Scraping stats for {home_team} vs {away_team} (id: {match_id})")
        try:
            stats_df = ss.scrape_team_match_stats(match_id)
            match_shots = None
            try:
                match_shots = ss.scrape_match_shots(match_id)
            except Exception as exc:
                print(f"    [!] Failed to scrape shots for match {match_id}: {exc}")
            row = _flatten_match_stats_row(match_info, stats_df, match_shots=match_shots)
            new_rows.append(row)
        except Exception as exc:
            print(f"    [!] Failed to scrape match {match_id}: {exc}")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        updated = pd.concat([season_df, new_df], ignore_index=True, sort=False)
        updated["game_id"] = updated["game_id"].map(_coerce_match_id)
        updated = updated.dropna(subset=["game_id"]).drop_duplicates(subset=["game_id"], keep="last")
        updated = _apply_derived_features(updated)
        if "home_momentum" in updated.columns:
            updated["home_momentum"] = pd.to_numeric(updated["home_momentum"], errors="coerce").fillna(5.0)
        if "away_momentum" in updated.columns:
            updated["away_momentum"] = pd.to_numeric(updated["away_momentum"], errors="coerce").fillna(5.0)
        updated = ensure_match_datetime_and_sort(updated, season=season, league=league, matches=matches, ss=ss)

        original_columns = list(season_df.columns)
        extra_columns = [c for c in updated.columns if c not in original_columns]
        updated = updated[original_columns + extra_columns]
    else:
        updated = _apply_derived_features(season_df.copy())
        updated = ensure_match_datetime_and_sort(updated, season=season, league=league, matches=matches, ss=ss)

    if refresh_shots_for_existing:
        updated = _refresh_shot_payloads_for_existing_rows(updated, ss=ss, force=True)
    updated = _apply_derived_features(updated)

    updated.to_csv(season_path, index=False)
    print(f"[OK] season_df updated: {season_path} (rows={len(updated)})")

    league_table = _build_league_table_from_season_df(updated)
    league_table_path.parent.mkdir(parents=True, exist_ok=True)
    league_table.to_csv(league_table_path, index=False)
    print(f"[OK] league_table updated: {league_table_path} (teams={len(league_table)})")

    return updated, league_table


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape/update season and league table CSVs.")
    parser.add_argument("--season", default="25/26", help="Season identifier, e.g. 25/26")
    parser.add_argument("--league", default="England Premier League", help="League name for ScraperFC")
    parser.add_argument(
        "--output-base-dir",
        default=str(DEFAULT_LEAGUE_OUTPUT_BASE_DIR),
        help="Base directory for per-league folders (default: data/raw/leagues)",
    )
    parser.add_argument(
        "--season-csv",
        default=None,
        help=(
            "Optional explicit season_df path. "
            "If omitted, uses <output-base-dir>/<league_slug>/season_df.csv."
        ),
    )
    parser.add_argument(
        "--league-table-csv",
        default=None,
        help=(
            "Optional explicit league table path. "
            "If omitted, uses <output-base-dir>/<league_slug>/league_table.csv."
        ),
    )
    parser.add_argument(
        "--matches-json",
        default=None,
        help=(
            "Optional explicit matches JSON path. "
            "If omitted, uses <output-base-dir>/<league_slug>/matches_data.json."
        ),
    )
    parser.add_argument(
        "--combined-season-csv",
        default=None,
        help="Optional path to save a combined multi-league season_df in multi-league-scrape mode.",
    )
    parser.add_argument(
        "--combined-league-table-csv",
        default=None,
        help="Optional path to save a combined multi-league table in multi-league-scrape mode.",
    )
    parser.add_argument(
        "--mode",
        choices=["enrich-datetime", "update", "multi-league-scrape"],
        default="update",
        help=(
            "enrich-datetime: only ensure match_datetime/sort; "
            "update: append missing matches + rebuild league table; "
            "multi-league-scrape: scrape each league into its own folder under output-base-dir"
        ),
    )
    parser.add_argument(
        "--refresh-shots",
        action="store_true",
        help="When used with --mode update, backfill/refresh home_shots and away_shots payloads for existing rows.",
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=DEFAULT_MULTI_LEAGUES,
        help=(
            "League strings for multi-league-scrape mode. "
            "Defaults to: France Ligue 1, Germany Bundesliga, Italy Serie A, Spain La Liga"
        ),
    )
    args = parser.parse_args()

    if args.mode == "enrich-datetime":
        enrich_existing_season_csv(
            csv_path=args.season_csv,
            season=args.season,
            league=args.league,
            output_base_dir=args.output_base_dir,
        )
    elif args.mode == "multi-league-scrape":
        scrape_multi_league_season_df(
            season=args.season,
            leagues=args.leagues,
            output_base_dir=args.output_base_dir,
            combined_season_csv_path=args.combined_season_csv,
            combined_league_table_csv_path=args.combined_league_table_csv,
        )
    else:
        update_season_and_league_table(
            season_csv_path=args.season_csv,
            league_table_csv_path=args.league_table_csv,
            matches_json_path=args.matches_json,
            season=args.season,
            league=args.league,
            refresh_shots_for_existing=args.refresh_shots,
            output_base_dir=args.output_base_dir,
        )

