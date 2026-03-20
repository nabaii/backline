"""
team_clustering.py - K-means clustering over shot/momentum team profiles.

Usage examples:
    python cluster_teams.py --scope by_league --k auto
    python cluster_teams.py --league-slug england_premier_league --k 5
    python cluster_teams.py --scope by_league --k auto --k-min 2 --k-max 6 --min-cluster-size 2
    python cluster_teams.py --scope global --k auto --k-min 3 --k-max 8 --min-matches 8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


LEAGUES_ROOT = Path("data/raw/leagues")
DEFAULT_OUTPUT_PATH = Path("data/raw/team_clusters.csv")
DEFAULT_PROFILE_OUTPUT_PATH = Path("data/raw/team_cluster_profiles.csv")

# Requested shot/momentum feature families.
FEATURE_BASE_COLUMNS = [
    "hq_shot_volume",
    "shot_quality_variance",
    "counter_attack_shot_ratio",
    "set_piece_reliance",
    "assisted_flow_ratio",
    "box_dominance_ratio",
    "sustained_pressure_index",
    "momentum_correlation",
    "momentum_auc",
    "momentum_symmetry",
    "basketball_index",
]


@dataclass
class LeagueDataset:
    league_slug: str
    league_name: str
    season_path: Path
    season_df: pd.DataFrame


def _discover_league_datasets(league_slugs: set[str] | None = None) -> list[LeagueDataset]:
    datasets: list[LeagueDataset] = []
    for season_path in sorted(LEAGUES_ROOT.glob("*/season_df.csv")):
        slug = season_path.parent.name
        if league_slugs and slug not in league_slugs:
            continue
        df = pd.read_csv(season_path)
        league_name = _resolve_league_name(df, slug)
        datasets.append(
            LeagueDataset(
                league_slug=slug,
                league_name=league_name,
                season_path=season_path,
                season_df=df,
            )
        )
    return datasets


def _resolve_league_name(df: pd.DataFrame, fallback_slug: str) -> str:
    if "league_name" not in df.columns:
        return fallback_slug
    values = (
        df["league_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    if not values:
        return fallback_slug
    return values[0]


def _ensure_side_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for base in FEATURE_BASE_COLUMNS:
        for side in ("home", "away"):
            col = f"{base}_{side}"
            if col not in out.columns:
                out[col] = np.nan
    return out


def _team_side_view(df: pd.DataFrame, side: str) -> pd.DataFrame:
    if side not in {"home", "away"}:
        raise ValueError("side must be 'home' or 'away'")

    team_col = "home_team" if side == "home" else "away_team"
    match_col = "game_id" if "game_id" in df.columns else ("match_id" if "match_id" in df.columns else None)
    if match_col is None:
        match_ids = pd.Series(np.arange(len(df)), index=df.index)
    else:
        match_ids = df[match_col]

    rows = pd.DataFrame(
        {
            "team": df[team_col] if team_col in df.columns else pd.Series("", index=df.index),
            "match_id": pd.to_numeric(match_ids, errors="coerce"),
            "side": side,
        }
    )

    for base in FEATURE_BASE_COLUMNS:
        source_col = f"{base}_{side}"
        rows[base] = pd.to_numeric(df[source_col], errors="coerce")

    rows["team"] = rows["team"].astype(str).str.strip().str.lower()
    rows = rows.replace({"team": {"": np.nan, "nan": np.nan, "none": np.nan}})
    rows = rows.dropna(subset=["team"])
    return rows


def build_team_feature_frame(season_df: pd.DataFrame, min_matches: int) -> pd.DataFrame:
    working = _ensure_side_feature_columns(season_df)
    home_rows = _team_side_view(working, "home")
    away_rows = _team_side_view(working, "away")
    stacked = pd.concat([home_rows, away_rows], ignore_index=True)

    grouped = (
        stacked.groupby("team", as_index=False)
        .agg(
            matches=("match_id", "count"),
            **{base: (base, "mean") for base in FEATURE_BASE_COLUMNS},
        )
    )
    grouped = grouped[grouped["matches"] >= int(min_matches)].reset_index(drop=True)
    return grouped


def _prepare_feature_matrix(
    df: pd.DataFrame,
    min_feature_std: float,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    usable_features = [col for col in FEATURE_BASE_COLUMNS if col in df.columns and df[col].notna().any()]
    if not usable_features:
        raise ValueError("No usable feature columns found for clustering.")

    matrix = df[usable_features].copy()
    for col in usable_features:
        median_val = pd.to_numeric(matrix[col], errors="coerce").median()
        if pd.isna(median_val):
            median_val = 0.0
        matrix[col] = pd.to_numeric(matrix[col], errors="coerce").fillna(median_val)

    # Drop near-constant features to reduce noise amplification after standardization.
    std_by_feature = matrix.std(axis=0, ddof=0)
    kept_features = [col for col in usable_features if float(std_by_feature.get(col, 0.0)) > float(min_feature_std)]
    dropped_features = [col for col in usable_features if col not in kept_features]
    if not kept_features:
        # Never allow an empty matrix; fallback to all usable features.
        kept_features = usable_features
        dropped_features = []

    return matrix[kept_features], kept_features, dropped_features


def _parse_requested_k(k_value: int | str) -> int | None:
    if isinstance(k_value, int):
        return max(1, int(k_value))

    text = str(k_value).strip().lower()
    if text == "auto":
        return None
    try:
        return max(1, int(text))
    except ValueError as exc:
        raise ValueError(f"Invalid --k value: {k_value!r}. Use an integer or 'auto'.") from exc


def _fit_kmeans_once(
    x_scaled: np.ndarray,
    n_clusters: int,
    random_state: int,
    n_init: int,
) -> tuple[KMeans, np.ndarray, float | None]:
    model = KMeans(n_clusters=int(n_clusters), random_state=int(random_state), n_init=int(n_init))
    labels = model.fit_predict(x_scaled)
    silhouette: float | None = None
    if n_clusters > 1 and len(x_scaled) > n_clusters:
        silhouette = float(silhouette_score(x_scaled, labels))
    return model, labels, silhouette


def _select_cluster_count(
    x_scaled: np.ndarray,
    requested_k: int | None,
    k_min: int,
    k_max: int,
    min_cluster_size: int,
    max_cluster_share: float,
    random_state: int,
    n_init: int,
) -> tuple[int, list[dict[str, Any]], str]:
    n_teams = int(len(x_scaled))
    if n_teams <= 1:
        return 1, [], "single_team"

    if requested_k is not None:
        k_fixed = max(1, min(int(requested_k), n_teams))
        return k_fixed, [], "fixed_k"

    k_low = max(2, int(k_min))
    k_high = min(int(k_max), n_teams - 1)
    if k_high < k_low:
        # If we cannot evaluate multiple k values, fallback to 2 (or 1 for tiny datasets).
        fallback_k = 2 if n_teams >= 2 else 1
        return fallback_k, [], "auto_degenerate_range"

    evaluations: list[dict[str, Any]] = []
    for k in range(k_low, k_high + 1):
        model, labels, silhouette = _fit_kmeans_once(
            x_scaled=x_scaled,
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
        )
        counts = np.bincount(labels, minlength=k)
        min_size_observed = int(counts.min()) if len(counts) else 0
        max_share_observed = float(counts.max() / n_teams) if len(counts) else 1.0
        sil_value = float(silhouette) if silhouette is not None and np.isfinite(silhouette) else float("-inf")
        valid = bool(
            min_size_observed >= int(min_cluster_size)
            and max_share_observed <= float(max_cluster_share)
        )
        evaluations.append(
            {
                "k": int(k),
                "silhouette": sil_value,
                "inertia": float(model.inertia_),
                "cluster_sizes": counts.tolist(),
                "min_cluster_size_observed": min_size_observed,
                "max_cluster_share_observed": max_share_observed,
                "valid": valid,
            }
        )

    valid_candidates = [item for item in evaluations if item["valid"]]
    pool = valid_candidates if valid_candidates else evaluations
    # Primary objective: maximize silhouette. Secondary objective: prefer smaller k for interpretability.
    best = max(pool, key=lambda item: (item["silhouette"], -item["k"]))
    reason = "auto_valid_constraints" if valid_candidates else "auto_fallback_no_valid_constraints"
    return int(best["k"]), evaluations, reason


def run_kmeans(
    team_df: pd.DataFrame,
    k: int | str,
    k_min: int,
    k_max: int,
    min_cluster_size: int,
    max_cluster_share: float,
    min_feature_std: float,
    random_state: int,
    n_init: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], int, float | None, float, dict[str, Any]]:
    if team_df.empty:
        raise ValueError("Team feature frame is empty.")

    matrix, feature_cols, dropped_low_variance = _prepare_feature_matrix(
        team_df,
        min_feature_std=min_feature_std,
    )

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(matrix)

    requested_k = _parse_requested_k(k)
    n_clusters, k_evaluations, k_selection_reason = _select_cluster_count(
        x_scaled=x_scaled,
        requested_k=requested_k,
        k_min=k_min,
        k_max=k_max,
        min_cluster_size=min_cluster_size,
        max_cluster_share=max_cluster_share,
        random_state=random_state,
        n_init=n_init,
    )

    model, labels, silhouette = _fit_kmeans_once(
        x_scaled=x_scaled,
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )

    assignments = team_df.copy()
    assignments["cluster_id"] = labels
    assignments["distance_to_centroid"] = model.transform(x_scaled).min(axis=1)

    counts = np.bincount(labels, minlength=n_clusters)
    cluster_size_lookup = {idx: int(size) for idx, size in enumerate(counts)}
    assignments["cluster_size"] = assignments["cluster_id"].map(cluster_size_lookup)

    cluster_profiles = (
        assignments.groupby("cluster_id", as_index=False)
        .agg(
            teams_in_cluster=("team", "count"),
            **{col: (col, "mean") for col in feature_cols},
        )
        .sort_values("cluster_id", kind="mergesort")
        .reset_index(drop=True)
    )

    diagnostics: dict[str, Any] = {
        "requested_k": k,
        "k_effective": int(n_clusters),
        "k_selection_reason": k_selection_reason,
        "k_evaluations": k_evaluations,
        "dropped_low_variance_features": dropped_low_variance,
        "min_cluster_size_observed": int(counts.min()) if len(counts) else 0,
        "max_cluster_share_observed": float(counts.max() / len(team_df)) if len(counts) else 1.0,
    }

    return (
        assignments,
        cluster_profiles,
        feature_cols,
        n_clusters,
        float(silhouette) if silhouette is not None else None,
        float(model.inertia_),
        diagnostics,
    )


def cluster_by_league(
    datasets: list[LeagueDataset],
    k: int | str,
    k_min: int,
    k_max: int,
    min_cluster_size: int,
    max_cluster_share: float,
    min_feature_std: float,
    min_matches: int,
    random_state: int,
    n_init: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignment_frames: list[pd.DataFrame] = []
    profile_frames: list[pd.DataFrame] = []

    for ds in datasets:
        team_df = build_team_feature_frame(ds.season_df, min_matches=min_matches)
        if team_df.empty:
            print(f"[!] Skipping {ds.league_slug}: no teams after min_matches filter.")
            continue

        assignments, profiles, used_features, k_eff, silhouette, inertia, diag = run_kmeans(
            team_df=team_df,
            k=k,
            k_min=k_min,
            k_max=k_max,
            min_cluster_size=min_cluster_size,
            max_cluster_share=max_cluster_share,
            min_feature_std=min_feature_std,
            random_state=random_state,
            n_init=n_init,
        )

        assignments.insert(0, "league_slug", ds.league_slug)
        assignments.insert(1, "league_name", ds.league_name)
        assignments.insert(2, "scope", "by_league")
        assignments["k_effective"] = int(k_eff)
        assignments["k_selection_reason"] = str(diag.get("k_selection_reason", ""))

        profiles.insert(0, "league_slug", ds.league_slug)
        profiles.insert(1, "league_name", ds.league_name)
        profiles.insert(2, "scope", "by_league")
        profiles["k_effective"] = int(k_eff)
        profiles["silhouette_score"] = silhouette
        profiles["inertia"] = inertia
        profiles["requested_k"] = str(diag.get("requested_k"))
        profiles["k_selection_reason"] = str(diag.get("k_selection_reason", ""))
        profiles["min_cluster_size_observed"] = int(diag.get("min_cluster_size_observed", 0))
        profiles["max_cluster_share_observed"] = float(diag.get("max_cluster_share_observed", 1.0))
        profiles["dropped_low_variance_features"] = ",".join(diag.get("dropped_low_variance_features", []))
        evaluations = diag.get("k_evaluations", [])
        profiles["k_evaluation_summary"] = ";".join(
            f"k={item['k']},sil={item['silhouette']:.4f},valid={int(item['valid'])},"
            f"min={item['min_cluster_size_observed']},max_share={item['max_cluster_share_observed']:.3f}"
            for item in evaluations
        )
        profiles["features_used"] = ",".join(used_features)

        assignment_frames.append(assignments)
        profile_frames.append(profiles)

        print(
            f"[OK] {ds.league_slug}: teams={len(team_df)}, k={k_eff}, "
            f"silhouette={silhouette if silhouette is not None else 'n/a'}, "
            f"reason={diag.get('k_selection_reason')}"
        )

    if not assignment_frames:
        return pd.DataFrame(), pd.DataFrame()

    return (
        pd.concat(assignment_frames, ignore_index=True),
        pd.concat(profile_frames, ignore_index=True),
    )


def cluster_global(
    datasets: list[LeagueDataset],
    k: int | str,
    k_min: int,
    k_max: int,
    min_cluster_size: int,
    max_cluster_share: float,
    min_feature_std: float,
    min_matches: int,
    random_state: int,
    n_init: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_frames: list[pd.DataFrame] = []
    for ds in datasets:
        team_df = build_team_feature_frame(ds.season_df, min_matches=min_matches)
        if team_df.empty:
            continue
        team_df.insert(0, "league_slug", ds.league_slug)
        team_df.insert(1, "league_name", ds.league_name)
        team_frames.append(team_df)

    if not team_frames:
        return pd.DataFrame(), pd.DataFrame()

    combined = pd.concat(team_frames, ignore_index=True)
    assignments, profiles, used_features, k_eff, silhouette, inertia, diag = run_kmeans(
        team_df=combined,
        k=k,
        k_min=k_min,
        k_max=k_max,
        min_cluster_size=min_cluster_size,
        max_cluster_share=max_cluster_share,
        min_feature_std=min_feature_std,
        random_state=random_state,
        n_init=n_init,
    )

    assignments.insert(2, "scope", "global")
    assignments["k_effective"] = int(k_eff)
    assignments["k_selection_reason"] = str(diag.get("k_selection_reason", ""))
    profiles.insert(0, "scope", "global")
    profiles["k_effective"] = int(k_eff)
    profiles["silhouette_score"] = silhouette
    profiles["inertia"] = inertia
    profiles["requested_k"] = str(diag.get("requested_k"))
    profiles["k_selection_reason"] = str(diag.get("k_selection_reason", ""))
    profiles["min_cluster_size_observed"] = int(diag.get("min_cluster_size_observed", 0))
    profiles["max_cluster_share_observed"] = float(diag.get("max_cluster_share_observed", 1.0))
    profiles["dropped_low_variance_features"] = ",".join(diag.get("dropped_low_variance_features", []))
    evaluations = diag.get("k_evaluations", [])
    profiles["k_evaluation_summary"] = ";".join(
        f"k={item['k']},sil={item['silhouette']:.4f},valid={int(item['valid'])},"
        f"min={item['min_cluster_size_observed']},max_share={item['max_cluster_share_observed']:.3f}"
        for item in evaluations
    )
    profiles["features_used"] = ",".join(used_features)
    profiles["league_slug"] = "all"
    profiles["league_name"] = "All Leagues"

    print(
        f"[OK] global: teams={len(combined)}, k={k_eff}, "
        f"silhouette={silhouette if silhouette is not None else 'n/a'}, "
        f"reason={diag.get('k_selection_reason')}"
    )
    return assignments, profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-means clustering over team shot/momentum profiles.")
    parser.add_argument(
        "--league-slug",
        type=str,
        default=None,
        help="Optional comma-separated league slug(s), e.g. england_premier_league,spain_la_liga",
    )
    parser.add_argument(
        "--scope",
        choices=["by_league", "global"],
        default="by_league",
        help="Cluster within each league separately, or across all leagues globally.",
    )
    parser.add_argument(
        "--k",
        type=str,
        default="auto",
        help="Cluster count (integer) or 'auto' to select k by silhouette + cluster-size constraints.",
    )
    parser.add_argument("--k-min", type=int, default=2, help="Minimum k to test in auto mode.")
    parser.add_argument("--k-max", type=int, default=6, help="Maximum k to test in auto mode.")
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum allowed cluster size in auto-k validation.",
    )
    parser.add_argument(
        "--max-cluster-share",
        type=float,
        default=0.75,
        help="Maximum allowed share of teams in a single cluster during auto-k validation.",
    )
    parser.add_argument(
        "--min-feature-std",
        type=float,
        default=1e-3,
        help="Drop near-constant features whose std <= this threshold before scaling.",
    )
    parser.add_argument("--min-matches", type=int, default=5, help="Minimum team match rows required.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for K-means.")
    parser.add_argument("--n-init", type=int, default=25, help="K-means n_init parameter.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Assignments CSV output path.")
    parser.add_argument(
        "--profile-output",
        type=str,
        default=str(DEFAULT_PROFILE_OUTPUT_PATH),
        help="Cluster profile CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    league_slugs = None
    if args.league_slug:
        league_slugs = {part.strip() for part in str(args.league_slug).split(",") if part.strip()}

    datasets = _discover_league_datasets(league_slugs=league_slugs)
    if not datasets:
        raise FileNotFoundError("No league season_df.csv files found for the selected scope.")

    if args.scope == "global":
        assignments, profiles = cluster_global(
            datasets=datasets,
            k=args.k,
            k_min=args.k_min,
            k_max=args.k_max,
            min_cluster_size=args.min_cluster_size,
            max_cluster_share=args.max_cluster_share,
            min_feature_std=args.min_feature_std,
            min_matches=args.min_matches,
            random_state=args.random_state,
            n_init=args.n_init,
        )
    else:
        assignments, profiles = cluster_by_league(
            datasets=datasets,
            k=args.k,
            k_min=args.k_min,
            k_max=args.k_max,
            min_cluster_size=args.min_cluster_size,
            max_cluster_share=args.max_cluster_share,
            min_feature_std=args.min_feature_std,
            min_matches=args.min_matches,
            random_state=args.random_state,
            n_init=args.n_init,
        )

    if assignments.empty:
        raise RuntimeError("No clustering output produced.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_path, index=False)

    profile_path = Path(args.profile_output)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(profile_path, index=False)

    print(f"[OK] Saved team clusters -> {output_path}")
    print(f"[OK] Saved cluster profiles -> {profile_path}")


if __name__ == "__main__":
    main()
