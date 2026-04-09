"""
update_data.py  â€“  Backline v2  â€“  Daily Incremental Data Updater
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
import ast

import numpy as np
import pandas as pd
import ScraperFC as sfc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# â”€â”€ Import helpers from the main scraper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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


# â”€â”€ Throttle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SCRAPE_DELAY_SECONDS = 1.5  # polite delay between API calls
ENRICH_DELAY_SECONDS = 0.2

SET_PIECE_SITUATIONS = {"corner", "set-piece", "setpiece", "free-kick", "freekick"}
COUNTER_ATTACK_SITUATIONS = {"fast-break", "fastbreak"}
ASSISTED_SITUATIONS = {"assisted"}

# Sofascore playerCoordinates are normalized to ~0..100.
# We treat low-x central channels as a danger zone.
DANGER_ZONE_X_MAX = 20.0
DANGER_ZONE_Y_MIN = 33.0
DANGER_ZONE_Y_MAX = 67.0

SHOT_MOMENTUM_FEATURE_COLUMNS = [
    "hq_shot_volume_home",
    "hq_shot_volume_away",
    "shot_quality_variance_home",
    "shot_quality_variance_away",
    "counter_attack_shot_count_home",
    "counter_attack_shot_count_away",
    "counter_attack_shot_ratio_home",
    "counter_attack_shot_ratio_away",
    "set_piece_reliance_home",
    "set_piece_reliance_away",
    "assisted_flow_ratio_home",
    "assisted_flow_ratio_away",
    "box_dominance_ratio_home",
    "box_dominance_ratio_away",
    "periphery_shot_ratio_home",
    "periphery_shot_ratio_away",
    "sustained_pressure_index_home",
    "sustained_pressure_index_away",
    "momentum_correlation_home",
    "momentum_correlation_away",
    "momentum_auc_home",
    "momentum_auc_away",
    "momentum_symmetry_home",
    "momentum_symmetry_away",
    "basketball_index_home",
    "basketball_index_away",
]


def _safe_float(value, default=np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isnan(out):
        return float(default)
    return out


def _safe_int(value, default: int | None = None) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalize_situation(value) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _parse_dict_like(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _extract_xy_from_coordinates(value) -> tuple[float, float]:
    payload = _parse_dict_like(value)
    x = _safe_float(payload.get("x"), default=np.nan)
    y = _safe_float(payload.get("y"), default=np.nan)
    return x, y


def _extract_shot_minute(row: pd.Series) -> int | None:
    time_seconds = _safe_int(row.get("timeSeconds"), default=None)
    if time_seconds is not None:
        return max(1, int(round(time_seconds / 60.0)))
    minute = _safe_int(row.get("time"), default=None)
    if minute is not None:
        return max(1, minute)
    return None


def _extract_team_shot_records(shot_df: pd.DataFrame | None) -> dict[str, list[dict[str, float | str | int | None]]]:
    records = {"home": [], "away": []}
    if shot_df is None or shot_df.empty:
        return records
    if "isHome" not in shot_df.columns:
        return records

    frame = shot_df.copy()
    frame["isHomeBool"] = frame["isHome"].map(lambda v: str(v).strip().lower() in {"true", "1", "yes", "y"})
    frame["xg_num"] = pd.to_numeric(frame.get("xg"), errors="coerce")

    for _, row in frame.iterrows():
        side = "home" if bool(row.get("isHomeBool")) else "away"
        xg = _safe_float(row.get("xg_num"), default=np.nan)
        situation = _normalize_situation(row.get("situation"))
        minute = _extract_shot_minute(row)
        x, y = _extract_xy_from_coordinates(row.get("playerCoordinates"))
        records[side].append(
            {
                "xg": xg,
                "situation": situation,
                "minute": minute,
                "x": x,
                "y": y,
            }
        )
    return records


def _compute_team_shot_metrics(records: list[dict[str, float | str | int | None]]) -> dict[str, float]:
    total = len(records)
    xg_values = [float(r["xg"]) for r in records if pd.notna(r.get("xg"))]
    hq_volume = float(sum(1 for xg in xg_values if xg >= 0.15))
    shot_quality_variance = float(np.std(xg_values, ddof=0)) if xg_values else np.nan

    counter_count = float(
        sum(1 for r in records if _normalize_situation(r.get("situation")) in COUNTER_ATTACK_SITUATIONS)
    )
    set_piece_count = float(
        sum(1 for r in records if _normalize_situation(r.get("situation")) in SET_PIECE_SITUATIONS)
    )
    assisted_count = float(
        sum(1 for r in records if _normalize_situation(r.get("situation")) in ASSISTED_SITUATIONS)
    )

    counter_ratio = counter_count / total if total > 0 else np.nan
    set_piece_ratio = set_piece_count / total if total > 0 else np.nan
    assisted_ratio = assisted_count / total if total > 0 else np.nan

    coord_records = [r for r in records if pd.notna(r.get("x")) and pd.notna(r.get("y"))]
    coord_total = len(coord_records)
    danger_count = 0
    for r in coord_records:
        x = float(r["x"])
        y = float(r["y"])
        if x <= DANGER_ZONE_X_MAX and DANGER_ZONE_Y_MIN <= y <= DANGER_ZONE_Y_MAX:
            danger_count += 1
    danger_ratio = (danger_count / coord_total) if coord_total > 0 else np.nan
    periphery_ratio = (1.0 - danger_ratio) if pd.notna(danger_ratio) else np.nan

    return {
        "hq_shot_volume": hq_volume,
        "shot_quality_variance": shot_quality_variance,
        "counter_attack_shot_count": counter_count,
        "counter_attack_shot_ratio": counter_ratio,
        "set_piece_reliance": set_piece_ratio,
        "assisted_flow_ratio": assisted_ratio,
        "box_dominance_ratio": danger_ratio,
        "periphery_shot_ratio": periphery_ratio,
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    try:
        return float(np.corrcoef(a, b)[0, 1])
    except Exception:
        return np.nan


def _build_momentum_series(momentum_df: pd.DataFrame | None) -> pd.Series | None:
    if momentum_df is None or momentum_df.empty:
        return None
    if "minute" not in momentum_df.columns or "value" not in momentum_df.columns:
        return None

    frame = momentum_df.copy()
    frame["minute"] = pd.to_numeric(frame["minute"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["minute", "value"])
    if frame.empty:
        return None

    frame["minute"] = frame["minute"].astype(int)
    frame = frame[frame["minute"] >= 1]
    if frame.empty:
        return None

    grouped = frame.groupby("minute", as_index=True)["value"].mean().sort_index()
    max_minute = int(max(90, grouped.index.max()))
    minute_axis = pd.Index(range(1, max_minute + 1), dtype=int)
    series = grouped.reindex(minute_axis, fill_value=0.0)
    return series.astype(float)


def _window_pressure_for_shot(series: pd.Series, minute: int, is_home: bool) -> float:
    if minute <= 0 or series.empty:
        return np.nan
    oriented = series if is_home else -series
    window_start = max(int(oriented.index.min()), int(minute) - 5)
    window_end = min(int(oriented.index.max()), int(minute))
    if window_start > window_end:
        return np.nan
    window = oriented.loc[window_start:window_end]
    if window.empty:
        return np.nan
    return float(window.mean())


def _compute_match_shot_and_momentum_metrics(
    shot_df: pd.DataFrame | None,
    momentum_df: pd.DataFrame | None,
) -> dict[str, float]:
    records_by_side = _extract_team_shot_records(shot_df)
    home_shot = _compute_team_shot_metrics(records_by_side["home"])
    away_shot = _compute_team_shot_metrics(records_by_side["away"])

    out: dict[str, float] = {}
    for key, value in home_shot.items():
        out[f"{key}_home"] = value
    for key, value in away_shot.items():
        out[f"{key}_away"] = value

    momentum_series = _build_momentum_series(momentum_df)
    if momentum_series is None:
        out["sustained_pressure_index_home"] = np.nan
        out["sustained_pressure_index_away"] = np.nan
        out["momentum_correlation_home"] = np.nan
        out["momentum_correlation_away"] = np.nan
        out["momentum_auc_home"] = np.nan
        out["momentum_auc_away"] = np.nan
        out["momentum_symmetry_home"] = np.nan
        out["momentum_symmetry_away"] = np.nan
        out["basketball_index_home"] = np.nan
        out["basketball_index_away"] = np.nan
        return out

    values = momentum_series.to_numpy(dtype=float)
    minutes = momentum_series.index.to_numpy(dtype=float)

    home_curve = np.clip(values, 0.0, None)
    away_curve = np.clip(-values, 0.0, None)
    out["momentum_auc_home"] = float(np.trapezoid(home_curve, minutes))
    out["momentum_auc_away"] = float(np.trapezoid(away_curve, minutes))

    symmetry = _safe_corr(home_curve, away_curve)
    out["momentum_symmetry_home"] = symmetry
    out["momentum_symmetry_away"] = symmetry

    signs = np.sign(values)
    non_zero_signs = signs[signs != 0]
    switches = int(np.sum(non_zero_signs[1:] != non_zero_signs[:-1])) if len(non_zero_signs) > 1 else 0
    out["basketball_index_home"] = float(switches)
    out["basketball_index_away"] = float(switches)

    def _team_pressure_and_corr(team: str, is_home: bool) -> tuple[float, float]:
        pressures = []
        shot_xg = []
        for shot in records_by_side[team]:
            minute = shot.get("minute")
            xg = shot.get("xg")
            if minute is None or pd.isna(xg):
                continue
            pressure = _window_pressure_for_shot(momentum_series, int(minute), is_home=is_home)
            if pd.isna(pressure):
                continue
            pressures.append(float(pressure))
            shot_xg.append(float(xg))

        if not pressures:
            return np.nan, np.nan
        sustained = float(np.mean(pressures))
        corr = _safe_corr(np.asarray(shot_xg, dtype=float), np.asarray(pressures, dtype=float))
        return sustained, corr

    sustained_home, corr_home = _team_pressure_and_corr("home", is_home=True)
    sustained_away, corr_away = _team_pressure_and_corr("away", is_home=False)
    out["sustained_pressure_index_home"] = sustained_home
    out["sustained_pressure_index_away"] = sustained_away
    out["momentum_correlation_home"] = corr_home
    out["momentum_correlation_away"] = corr_away
    return out


def _enrich_season_with_shot_and_momentum_metrics(
    season_df: pd.DataFrame,
    ss,
    force_refresh: bool = False,
    shot_cache: dict[int, pd.DataFrame] | None = None,
    momentum_cache: dict[int, pd.DataFrame] | None = None,
    max_matches: int | None = None,
) -> pd.DataFrame:
    out = season_df.copy()
    if "game_id" not in out.columns:
        return out

    for col in SHOT_MOMENTUM_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    if "home_shots" not in out.columns:
        out["home_shots"] = None
    if "away_shots" not in out.columns:
        out["away_shots"] = None

    total_rows = len(out)
    processed = 0
    skipped = 0
    failed = 0

    for idx, row in out.iterrows():
        match_id = _coerce_match_id(row.get("game_id"))
        if match_id is None:
            skipped += 1
            continue

        if max_matches is not None and processed >= max_matches:
            break

        has_all_metrics = all(pd.notna(row.get(col)) for col in SHOT_MOMENTUM_FEATURE_COLUMNS)
        if has_all_metrics and not force_refresh:
            skipped += 1
            continue

        shot_df = None
        if shot_cache is not None:
            shot_df = shot_cache.get(match_id)
        if shot_df is None:
            try:
                shot_df = ss.scrape_match_shots(match_id)
            except Exception:
                shot_df = None

        if shot_df is not None and not shot_df.empty:
            home_payload, away_payload = _serialize_shot_payload(shot_df)
            out.at[idx, "home_shots"] = home_payload
            out.at[idx, "away_shots"] = away_payload

        momentum_df = None
        if momentum_cache is not None:
            momentum_df = momentum_cache.get(match_id)
        if momentum_df is None:
            try:
                momentum_df = ss.scrape_match_momentum(match_id)
            except Exception:
                momentum_df = None

        try:
            metrics = _compute_match_shot_and_momentum_metrics(shot_df, momentum_df)
            for col in SHOT_MOMENTUM_FEATURE_COLUMNS:
                out.at[idx, col] = metrics.get(col, np.nan)
            processed += 1
        except Exception:
            failed += 1

        if processed % 25 == 0:
            print(
                f"  [*] Shot/momentum enrichment progress: processed={processed}, "
                f"skipped={skipped}, failed={failed}, total_rows={total_rows}"
            )
        time.sleep(ENRICH_DELAY_SECONDS)

    print(
        f"  [OK] Shot/momentum enrichment complete: processed={processed}, "
        f"skipped={skipped}, failed={failed}, total_rows={total_rows}"
    )
    return out


def _refresh_epl_fixture_cache_csv() -> None:
    """Refresh local EPL fixture CSV used by backend fixture snapshot fallback."""
    try:
        from backend.backend_api import _fetch_live_fixture_snapshot

        snapshot = _fetch_live_fixture_snapshot()
        print(
            "  [OK] Refreshed EPL fixture CSV "
            f"({len(snapshot.fixtures)} fixtures, source={snapshot.source})"
        )
    except Exception as e:
        print(f"  [!!] EPL fixture CSV refresh failed: {e}")


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


def _compute_momentum_for_matches(
    ss,
    match_ids: list[int],
    collect_raw: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame | None, dict[int, pd.DataFrame]] | None:
    """Scrape and compute PCA momentum for a list of match IDs."""
    if not match_ids:
        if collect_raw:
            return None, {}
        return None

    momentum_rows = []
    raw_by_match_id: dict[int, pd.DataFrame] = {}
    for i, mid in enumerate(match_ids, 1):
        print(f"    [{i}/{len(match_ids)}] Momentum for match {mid} ...")
        try:
            mom_df = ss.scrape_match_momentum(mid)
        except Exception as e:
            print(f"      [!!] Failed: {e}")
            continue

        if mom_df.empty or "minute" not in mom_df.columns or "value" not in mom_df.columns:
            continue

        if collect_raw:
            raw_by_match_id[mid] = mom_df.copy()

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
        if collect_raw:
            return None, raw_by_match_id
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

    result = pd.DataFrame(
        {
            "game_id": momentum_combined.index,
            "home_momentum": np.round(home_momentum, 2),
            "away_momentum": np.round(away_momentum, 2),
        }
    ).set_index("game_id")
    if collect_raw:
        return result, raw_by_match_id
    return result


def update_league(
    season: str,
    league: str,
    output_base_dir: Path = DEFAULT_LEAGUE_OUTPUT_BASE_DIR,
    enrich_shot_momentum: bool = False,
    force_enrich: bool = False,
    max_enrich_matches: int | None = None,
) -> int:
    """
    Incrementally update a single league's data.

    Returns the number of new matches added.
    """
    paths = _default_output_paths_for_league(league, output_base_dir)
    csv_path = paths["season_csv_path"]
    table_path = paths["league_table_csv_path"]
    matches_json_path = paths["matches_json_path"]
    league_dir = paths["league_dir"]
    league_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'-' * 60}")
    print(f"  Updating: {league}")
    print(f"  CSV:      {csv_path}")
    print(f"{'-' * 60}")

    # 1. Load existing data
    existing_df = _load_existing_season_df(csv_path)
    existing_ids: set[int] = set()
    if existing_df is not None:
        existing_ids = set(int(x) for x in existing_df.index)
        print(f"  [OK] Loaded {len(existing_df)} existing matches")
    else:
        print("  [*] Starting fresh - no existing data found")

    # 2. Fetch latest match list from Sofascore
    ss = sfc.Sofascore()
    print(f"  [*] Fetching match list for {season} {league} ...")
    matches = ss.get_match_dicts(season, league)
    print(f"  [OK] Found {len(matches)} total matches in season")
    try:
        matches_json_path.parent.mkdir(parents=True, exist_ok=True)
        with matches_json_path.open("w", encoding="utf-8") as handle:
            json.dump(matches, handle, ensure_ascii=False, indent=2)
        print(f"  [OK] Saved matches JSON -> {matches_json_path}")
    except Exception as e:
        print(f"  [!!] Failed to save matches JSON at {matches_json_path}: {e}")

    # 3. Identify NEW played matches
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
        print("  [OK] No new matches to scrape - data is up to date!")
        if not enrich_shot_momentum:
            return 0
    else:
        print(f"  [*] {len(new_matches)} new match(es) to scrape\n")

    # 4. Scrape stats + shots for each new match
    new_rows = []
    shot_cache: dict[int, pd.DataFrame] = {}
    momentum_cache: dict[int, pd.DataFrame] = {}
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
            if match_shots is not None and not match_shots.empty:
                shot_cache[int(mid)] = match_shots.copy()
        except Exception as e:
            print(f"      [!!] Shots failed: {e}")

        row = _flatten_match_stats_row(match_info, match_stats, match_shots=match_shots)
        new_rows.append(row)
        time.sleep(SCRAPE_DELAY_SECONDS)

    if new_matches and not new_rows and existing_df is None:
        print(f"  [!!] All {len(new_matches)} new matches failed to scrape")
        return 0

    if new_matches and new_rows:
        print(f"\n  [OK] Scraped {len(new_rows)} new matches ({failed} failed)")

    # 5. Build DataFrame for new matches
    new_df = None
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        new_df = new_df.set_index("game_id")

        # 6. Momentum for new matches
        print("  [*] Computing momentum for new matches ...")
        new_match_ids = [int(x) for x in new_df.index]
        momentum_result = _compute_momentum_for_matches(ss, new_match_ids, collect_raw=True)
        mom_df, raw_momentum_cache = momentum_result
        momentum_cache.update(raw_momentum_cache)
        if mom_df is not None:
            new_df = new_df.join(mom_df, how="left")
            print(f"  [OK] Momentum computed for {len(mom_df)} matches")
        else:
            new_df["home_momentum"] = np.nan
            new_df["away_momentum"] = np.nan

        # Add league_name column
        new_df["league_name"] = league

    # 7. Merge with existing data
    if existing_df is not None and new_df is not None:
        combined_df = pd.concat([existing_df, new_df], axis=0)
    elif existing_df is not None:
        combined_df = existing_df.copy()
    elif new_df is not None:
        combined_df = new_df.copy()
    else:
        print("  [!!] No data available to persist for this league.")
        return 0

    # Remove duplicates (just in case)
    combined_df = combined_df[~combined_df.index.duplicated(keep="first")]

    # 8. Recompute derived features on full dataset
    combined_df = combined_df.reset_index()
    if "league_name" not in combined_df.columns:
        combined_df["league_name"] = league
    else:
        combined_df["league_name"] = combined_df["league_name"].fillna(league)

    combined_df = _apply_derived_features(combined_df)
    combined_df = ensure_match_datetime_and_sort(
        combined_df, season=season, league=league, matches=matches, ss=ss
    )

    if enrich_shot_momentum:
        print("  [*] Enriching advanced shot/momentum features ...")
        combined_df = _enrich_season_with_shot_and_momentum_metrics(
            season_df=combined_df,
            ss=ss,
            force_refresh=force_enrich,
            shot_cache=shot_cache,
            momentum_cache=momentum_cache,
            max_matches=max_enrich_matches,
        )

    # Re-set game_id as index for saving
    if "game_id" in combined_df.columns:
        combined_df = combined_df.set_index("game_id")

    # 9. Save updated CSV
    combined_df.to_csv(csv_path)
    print(f"  [OK] Saved {len(combined_df)} matches -> {csv_path}")

    # 10. Rebuild league table
    try:
        combined_df_reset = combined_df.reset_index()
        league_table = _build_league_table_from_season_df(combined_df_reset)
        league_table.to_csv(table_path, index=False)
        print(f"  [OK] League table saved -> {table_path}")
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
    parser.add_argument(
        "--enrich-shot-momentum",
        action="store_true",
        help=(
            "Scrape per-match shots + momentum curves and enrich season_df with "
            "advanced home/away chance profile and temporal momentum features."
        ),
    )
    parser.add_argument(
        "--force-enrich",
        action="store_true",
        help="Recompute enrichment features even when target columns already exist.",
    )
    parser.add_argument(
        "--max-enrich-matches",
        type=int,
        default=None,
        help="Optional cap on enriched matches per league (for partial backfills/testing).",
    )
    parser.add_argument(
        "--skip-mongo-sync",
        action="store_true",
        help="Skip MongoDB Atlas sync and only refresh local CSV/cache files.",
    )
    args = parser.parse_args()

    leagues = [args.league] if args.league else DEFAULT_ALL_LEAGUES

    print("=" * 60)
    print("  Backline v2  â€“  Daily Data Updater")
    print("=" * 60)
    print(f"  Season:  {args.season}")
    print(f"  Leagues: {len(leagues)}")
    print(f"  Enrich:  {args.enrich_shot_momentum}")
    print(f"  Time:    {pd.Timestamp.now()}")

    total_new = 0
    for league in leagues:
        try:
            count = update_league(
                args.season,
                league,
                enrich_shot_momentum=args.enrich_shot_momentum,
                force_enrich=args.force_enrich,
                max_enrich_matches=args.max_enrich_matches,
            )
            total_new += count
        except Exception as e:
            print(f"\n  [X] ERROR updating {league}: {e}")
            import traceback
            traceback.print_exc()

    print("\n  [*] Refreshing EPL fixture cache CSV ...")
    _refresh_epl_fixture_cache_csv()

    print(f"\n{'=' * 60}")
    print(f"  Done!  {total_new} new match(es) added across {len(leagues)} league(s).")
    print("=" * 60)


    skip_mongo_sync = args.skip_mongo_sync or os.environ.get("BACKLINE_SKIP_MONGO_SYNC", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if skip_mongo_sync:
        print("\n  [*] Skipping MongoDB Atlas sync.")
    else:
        print("\n  [*] Syncing updated data to MongoDB Atlas ...")
        try:
            from scripts.migrate_to_mongodb import migrate

            migrate()
            print("  [OK] MongoDB sync complete.")
        except Exception as e:
            print(f"  [!!] MongoDB sync failed:\n{e}")

if __name__ == "__main__":
    main()
