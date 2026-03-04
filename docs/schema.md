# Backline v2 Data Schema

Last updated: 2026-03-04

## 1) Scope

This document defines the primary file and object schemas used by Backline v2:

1. League season datasets (`season_df.csv`)
2. League tables (`league_table.csv`)
3. League match metadata snapshots (`matches_data.json`)
4. EPL fixture cache (`EPL_fixture.csv`)
5. Core backend request/filter objects (`EvidenceRequest`, `FilterSpec`)
6. Team clustering outputs (`team_clusters.csv`, `team_cluster_profiles.csv`)


## 2) File Contracts

| Path pattern | Format | Primary key | Purpose |
| --- | --- | --- | --- |
| `data/raw/leagues/<league_slug>/season_df.csv` | CSV | `game_id` | Match-level analytics source used by backend filtering/workspaces |
| `data/raw/leagues/<league_slug>/league_table.csv` | CSV | `team` (per file) | Aggregated team table for a single league |
| `data/raw/leagues/<league_slug>/matches_data.json` | JSON array | `id` per item | Raw Sofascore season match metadata snapshot |
| `data/raw/EPL_fixture.csv` | CSV | `fixture_id` | EPL fixture cache/fallback used by fixture endpoints |


## 3) `season_df.csv` Schema

Current observed column count in EPL file: `156`.

### 3.1 Identity and Metadata

| Column | Type | Notes |
| --- | --- | --- |
| `game_id` | integer | Match identifier, canonical key for season rows |
| `match_datetime` | datetime (UTC string) | Canonical match time used for sorting |
| `league_id` | integer/string | Provider league id |
| `season_id` | integer/string | Provider season id |
| `league_name` | string | Human-readable league name |
| `home_team`, `away_team` | string | Normalized team names (snake-style) |
| `home_team_id`, `away_team_id` | integer | Team ids |
| `nameCode_home`, `nameCode_away` | string | Team short code |

### 3.2 Match Outcome and Derived Match Metrics

| Column | Type | Notes |
| --- | --- | --- |
| `home_normaltime`, `away_normaltime` | numeric | Full-time goals |
| `home_h1_goals`, `home_h2_goals` | numeric | Home first/second-half goals |
| `away_h1_goals`, `away_h2_goals` | numeric | Away first/second-half goals |
| `game_winner` | integer | 1=home, 2=away, 3=draw |
| `home_win`, `away_win`, `draw` | 0/1 integer | Derived flags |
| `xg_diff`, `xg_diff_home`, `xg_diff_away` | float | xG delta features |
| `total_goals`, `total_xg` | float | Match-level totals |
| `field_tilt_home`, `field_tilt_away` | float | Derived from final third phase share |
| `home_momentum`, `away_momentum` | float | PCA-based momentum scalars |

### 3.3 Base Home/Away Stat Pairs

Most raw performance stats are stored in paired columns:

1. `<metric>_home`
2. `<metric>_away`

Examples:

1. `ball_possession_home`, `ball_possession_away`
2. `expected_goals_home`, `expected_goals_away`
3. `total_shots_home`, `total_shots_away`
4. `corner_kicks_home`, `corner_kicks_away`
5. `yellow_cards_home`, `yellow_cards_away`
6. `red_cards_home`, `red_cards_away`

### 3.4 Shot Payload Columns

| Column | Type | Notes |
| --- | --- | --- |
| `home_shots` | JSON string | Stored payload with xG array and count |
| `away_shots` | JSON string | Stored payload with xG array and count |

Current payload shape:

```json
{
  "xg": [0.12, 0.05, 0.33],
  "count": 3
}
```

### 3.5 Advanced Shot and Temporal Momentum Enrichment Columns

The updater now supports per-match enrichment from `scrape_match_shots(match_id)` and `scrape_match_momentum(match_id)`.

For each metric below, both `_home` and `_away` columns exist.

| Metric base name | Type | Definition |
| --- | --- | --- |
| `hq_shot_volume` | float/int | Count of shots with `xg >= 0.15` |
| `shot_quality_variance` | float | Standard deviation of shot xG values |
| `counter_attack_shot_count` | float/int | Count of shots with `situation == fast-break` |
| `counter_attack_shot_ratio` | float | Counter-attack shot count / total team shots |
| `set_piece_reliance` | float | Share of shots from set-piece situations (`corner`, `set-piece`, `free-kick`) |
| `assisted_flow_ratio` | float | Share of shots with `situation == assisted` |
| `box_dominance_ratio` | float | Share of shots from defined danger-zone coordinates |
| `periphery_shot_ratio` | float | Complement of box dominance ratio |
| `sustained_pressure_index` | float | Mean momentum in 5-minute window preceding each shot |
| `momentum_correlation` | float | Correlation between shot xG and pre-shot pressure values |
| `momentum_auc` | float | Area under momentum curve for the side |
| `momentum_symmetry` | float | Correlation between home and away momentum peak profiles (duplicated per side) |
| `basketball_index` | float/int | Number of momentum sign switches (duplicated per side) |


## 4) `league_table.csv` Schema

Observed columns:

1. `rank`
2. `team`
3. `played`
4. `won`
5. `drawn`
6. `lost`
7. `goals_for`
8. `goals_against`
9. `points`
10. `goal_difference`
11. `average_corners`
12. `average_possession`
13. `expected_goals_for`
14. `expected_goals_against`
15. `momentum`
16. `expected_goal_difference`

Notes:

1. This table is rebuilt from `season_df.csv` after updates.
2. One file represents one league.


## 5) `matches_data.json` Schema

`matches_data.json` is an array of raw Sofascore match objects.

Top-level common keys include:

1. `id`
2. `startTimestamp`
3. `status`
4. `homeTeam`
5. `awayTeam`
6. `homeScore`
7. `awayScore`
8. `season`
9. `tournament`
10. `winnerCode`

Important nested keys:

1. `status`: `type`, `description`, `code`
2. `homeTeam`/`awayTeam`: `id`, `name`, `nameCode`, plus provider metadata
3. `homeScore`/`awayScore`: includes `normaltime`, period scores


## 6) `EPL_fixture.csv` Schema

Observed columns:

1. `gameweek`
2. `kickoff_time`
3. `home_team`
4. `away_team`
5. `home_team_score`
6. `away_team_score`
7. `finished`
8. `fixture_id`
9. `league_id`
10. `home_team_id`
11. `away_team_id`


## 7) Backend Object Schemas

### 7.1 `EvidenceRequest` (`backend/contracts/evidence.py`)

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `match_id` | int | yes | Anchor match id |
| `filters` | `List[FilterSpec]` | yes | Declarative filters |
| `perspective` | `'home' | 'away'` | yes | Team perspective |
| `bet_type` | string | yes | Market/workspace type |
| `required_features` | `List[str]` or null | no | Additional columns to materialize |
| `time_scope` | any or null | no | Optional time context |

### 7.2 `FilterSpec` (`backend/contracts/filter_spec.py`)

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `key` | string | yes | Logical filter id |
| `kind` | `'column' | 'context'` | yes | Filter execution mode |
| `operator` | `>`, `>=`, `<`, `<=`, `==`, `!=`, `between`, `in` | yes | Comparator |
| `value` | any | yes | Comparator value |
| `field` | string or null | conditional | Required for `kind='column'` |
| `perspective` | `'home' | 'away'` or null | no | Optional side marker |
| `display_name` | string or null | no | UI label |
| `required_columns` | tuple/list of strings | no | Additional columns needed for execution |


## 8) Operational Notes

1. `season_df.csv` is the source of truth for analytics filtering.
2. New enrichment fields are nullable until backfilled.
3. Full enrichment can be executed with:
   - `python update_data.py --enrich-shot-momentum`
4. For controlled backfills/testing:
   - `python update_data.py --enrich-shot-momentum --max-enrich-matches 50`


## 9) Team Clustering Schema

Team clustering is implemented in:

1. `backend/team_clustering.py`
2. `cluster_teams.py` (root wrapper)
3. Detailed behavior notes: `docs/team_clustering.md`

### 9.1 Input Contract

The clustering pipeline reads each league `season_df.csv` and expects team-level means from these base features:

1. `hq_shot_volume`
2. `shot_quality_variance`
3. `counter_attack_shot_ratio`
4. `set_piece_reliance`
5. `assisted_flow_ratio`
6. `box_dominance_ratio`
7. `sustained_pressure_index`
8. `momentum_correlation`
9. `momentum_auc`
10. `momentum_symmetry`
11. `basketball_index`

For each match row, side-specific columns (`*_home`, `*_away`) are folded into one team-level feature frame.

### 9.2 k Selection Logic

`team_clustering.py` supports two modes:

1. Fixed `k`: pass integer `--k 4`
2. Auto `k`: pass `--k auto` (default)

Auto `k` evaluates `k` in `[--k-min, --k-max]` and scores candidates using silhouette. It enforces:

1. `--min-cluster-size` (default `2`)
2. `--max-cluster-share` (default `0.75`)

If no candidate satisfies constraints, the script falls back to the highest silhouette candidate and marks reason as `auto_fallback_no_valid_constraints`.

### 9.3 Feature Stability Guard

Before scaling:

1. Missing feature values are median-imputed.
2. Near-constant features (std <= `--min-feature-std`, default `1e-3`) are dropped.

This avoids noise amplification after `StandardScaler`.

### 9.4 `data/raw/team_clusters.csv` Schema

Primary per-team assignment output. Key columns:

1. `league_slug`
2. `league_name`
3. `scope` (`by_league` or `global`)
4. `team`
5. `matches`
6. Team feature columns (the 11 base features)
7. `cluster_id`
8. `distance_to_centroid`
9. `cluster_size`
10. `k_effective`
11. `k_selection_reason`

### 9.5 `data/raw/team_cluster_profiles.csv` Schema

Per-cluster aggregate output. Key columns:

1. `league_slug`
2. `league_name`
3. `scope`
4. `cluster_id`
5. `teams_in_cluster`
6. Mean feature values for each base feature
7. `k_effective`
8. `silhouette_score`
9. `inertia`
10. `requested_k`
11. `k_selection_reason`
12. `min_cluster_size_observed`
13. `max_cluster_share_observed`
14. `dropped_low_variance_features`
15. `k_evaluation_summary`
16. `features_used`

### 9.6 CLI Reference (Current)

Common commands:

1. `python cluster_teams.py --scope by_league --k auto`
2. `python cluster_teams.py --scope by_league --k auto --k-min 2 --k-max 6 --min-cluster-size 2 --max-cluster-share 0.75`
3. `python cluster_teams.py --scope global --k auto --k-min 3 --k-max 8`
4. `python cluster_teams.py --league-slug england_premier_league --k 4`
