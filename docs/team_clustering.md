# Team Clustering Deep Dive

Last updated: 2026-03-04

## 1) Purpose

`backend/team_clustering.py` clusters football teams using team-level aggregates of the shot and momentum features derived in `season_df.csv`.

Primary output files:

1. `data/raw/team_clusters.csv` (team-level assignments)
2. `data/raw/team_cluster_profiles.csv` (cluster centroids/profiles)


## 2) Why non-EPL looked problematic

A fixed `k=4` across all leagues can produce unstable or unbalanced clusters because:

1. Different leagues have different natural group structure.
2. Team counts differ (`18` vs `20`), changing feasible partition quality.
3. Outlier teams can force singleton clusters at higher `k`.
4. Plotting only two axes (for example `hq_shot_volume` vs `momentum_auc`) is a projection of an 11-dimensional clustering space, so visual overlap in 2D does not imply clustering failure.

This is not classical supervised “overfitting”; it is mostly a cluster count/model-selection mismatch and projection interpretation issue.


## 3) Current Pipeline (Function-by-Function)

## 3.1 Dataset Discovery

`_discover_league_datasets(...)`:

1. Scans `data/raw/leagues/*/season_df.csv`.
2. Loads each file into memory.
3. Produces `LeagueDataset` records (slug, name, path, dataframe).

## 3.2 Team Feature Construction

`build_team_feature_frame(...)`:

1. Ensures all required side columns exist (`*_home`, `*_away`).
2. Builds home and away views with `_team_side_view(...)`.
3. Stacks both views so each row is one team-match-side observation.
4. Aggregates means per team for all clustering features.
5. Applies `min_matches` threshold.

Result: one row per team, one numeric value per feature family.

## 3.3 Feature Matrix Preparation

`_prepare_feature_matrix(...)`:

1. Keeps only features present with non-null data.
2. Median-imputes missing values.
3. Drops near-constant features (`std <= min_feature_std`) before scaling.

Rationale: near-constant features become noisy after z-score normalization and can distort K-means distance geometry.

## 3.4 Cluster Count (`k`) Selection

`_parse_requested_k(...)`:

1. Supports fixed integer `k` or `"auto"`.

`_select_cluster_count(...)`:

1. If fixed `k`, uses clipped fixed value.
2. If auto:
   - evaluates candidate `k` in `[k_min, k_max]` (bounded by sample size),
   - computes silhouette and cluster-size diagnostics for each candidate,
   - validates candidates using:
     - `min_cluster_size`
     - `max_cluster_share`
3. Selects best valid candidate by:
   - highest silhouette,
   - tie-breaker: lower `k` for interpretability.
4. Falls back to best silhouette if no candidate passes constraints.

This avoids forcing one-size-fits-all `k` and prevents pathological one-team clusters by default.

## 3.5 K-means Execution

`run_kmeans(...)`:

1. Builds cleaned matrix.
2. Standardizes with `StandardScaler`.
3. Selects `k` (fixed or auto).
4. Fits KMeans.
5. Computes:
   - `cluster_id`
   - `distance_to_centroid`
   - `cluster_size`
   - silhouette/inertia
6. Returns assignments, cluster profiles, and diagnostics.

## 3.6 Scope Execution

`cluster_by_league(...)`:

1. Runs independent clustering per league dataset.
2. Writes diagnostics into profile output:
   - requested/effective `k`
   - selection reason
   - candidate evaluation summary
   - min/max cluster size metrics
   - dropped low-variance features

`cluster_global(...)`:

1. Combines all leagues and clusters in one shared feature space.
2. Emits same diagnostics.


## 4) CLI Parameters (Current)

Core parameters:

1. `--scope` (`by_league` or `global`)
2. `--k` (integer or `auto`)
3. `--k-min`, `--k-max` (auto-k range)
4. `--min-cluster-size`
5. `--max-cluster-share`
6. `--min-feature-std`
7. `--min-matches`
8. `--league-slug`

Example:

`python cluster_teams.py --scope by_league --k auto --k-min 2 --k-max 6 --min-cluster-size 2 --max-cluster-share 0.75`


## 5) Output Diagnostics

### `team_clusters.csv`

Includes:

1. team identity (`league_slug`, `team`)
2. feature values
3. `cluster_id`
4. `distance_to_centroid`
5. `cluster_size`
6. `k_effective`
7. `k_selection_reason`

### `team_cluster_profiles.csv`

Includes:

1. per-cluster feature means
2. `k_effective`, `requested_k`
3. `silhouette_score`, `inertia`
4. `min_cluster_size_observed`, `max_cluster_share_observed`
5. `dropped_low_variance_features`
6. `k_evaluation_summary`
7. `features_used`


## 6) Practical Interpretation Notes

1. Use profile outputs to name cluster archetypes.
2. Avoid judging full clustering quality only from a single 2D scatter.
3. If you still see unstable assignments:
   - increase `min_matches`,
   - tighten `max_cluster_share`,
   - adjust `k_max`,
   - raise `min_feature_std` slightly.


## 7) Similar Teams Filter Integration

The workspace UI filter `Opponent Set -> Similar Teams (PCA Cluster)` uses the same feature family but runs a global PCA+KMeans similarity model inside `backend/backend_api.py`:

1. Build team-level global frame from all leagues.
2. Standardize all clustering features.
3. Project into PCA space (target explained variance `0.90`).
4. Auto-select `k` using silhouette with balance constraints.
5. For each upcoming opponent, keep matches against teams in that opponent's PCA cluster (nearest subset).
