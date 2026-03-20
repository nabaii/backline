# Backline v2 - Full Application Breakdown

Last updated: 2026-03-01

## 1) What this app is

Backline v2 is a football analytics workspace that:

1. Scrapes and stores match-level data for multiple leagues.
2. Serves that data through a Flask API.
3. Computes filtered evidence sets for betting-style questions (1X2, Over/Under, Double Chance, BTTS, Corners, Home O/U, Away O/U).
4. Renders an interactive Kitchen UI in React with chart overlays, split views, and filter controls.

This document explains how each part works and why it is designed that way.


## 2) Runtime architecture

The runtime is a two-part web app:

1. Backend: Flask (`backend/backend_api.py`)
2. Frontend: Vite + React (`frontend/src/*`)

Data is file-based (CSV and JSON), not database-backed.

Primary flow:

1. Update job writes league CSVs and fixture caches under `data/raw`.
2. Backend loads those files into memory and exposes `/api/*`.
3. Frontend calls `/api/*`, builds per-match workspaces, and draws charts.


## 3) Repository map (source and responsibility)

### Root wrappers and deployment

1. `main.py`: wrapper that runs `backend.main` (legacy scrape/table script).
2. `backend_api.py`: wrapper that runs Flask app from `backend.backend_api`.
3. `update_data.py`: wrapper that runs `backend.update_data`.
4. `Procfile`: production command (`gunicorn backend.backend_api:app ...`).
5. `render.yaml`: Render build/start config for backend + frontend static build.
6. `requirements.txt`: Python runtime dependencies.

### Backend core

1. `backend/backend_api.py`: main API server, fixture source logic, workspace endpoints, static frontend serving.
2. `backend/update_data.py`: incremental multi-league updater (new matches only) + fixture cache refresh.
3. `backend/main.py`: older full scrape and league table script.

### Backend analytics domain

1. `backend/builders/match_analytics_builder.py`: converts raw rows into `MatchAnalytics` objects.
2. `backend/store/analytics_store.py`: in-memory evidence query and filter execution engine.
3. `backend/filters/filters.py`: filter builders that produce `FilterSpec`.
4. `backend/bet_type/one_x_two/one_x_two.py`: 1X2 workspace logic.
5. `backend/bet_type/double_chance/double_chance.py`: Double Chance workspace logic.
6. `backend/bet_type/over_under/over_under.py`: Over/Under workspace logic.
7. `backend/bet_type/corners/corners.py`: Corners workspace logic.
8. `backend/contracts/*.py`: abstract contracts and DTO types (`EvidenceRequest`, `FilterSpec`, workspace interfaces, etc).
9. `backend/chart/chart_spec.py`: chart contract types.
10. `backend/metrics/metric_spec.py`: metric contract types.

### Data pipeline source

1. `data/raw/scrape_data.py`: full-season scrape logic, flattening, derived features, league-table generation, report tools, alternative update path.

### Frontend core

1. `frontend/src/main.jsx`: React bootstrap.
2. `frontend/src/App.jsx`: top-level app shell.
3. `frontend/src/api/backendApi.js`: API client wrappers.
4. `frontend/src/components/KitchenPage.jsx`: page-level state and fixture loading flow.
5. `frontend/src/components/BetTypeWorkspace.jsx`: workspace tab logic, filter state, request dispatch.
6. `frontend/src/components/ChartArea.jsx`: chart construction, overlays, line controls, data shaping.
7. `frontend/src/components/FilterDropdown.jsx`: filter UI and range slider behavior.
8. `frontend/src/components/FixtureList.jsx`: fixture sidebar list.
9. `frontend/src/components/LeagueSelector.jsx`: league tabs.
10. `frontend/src/components/MetricsPanel.jsx`: metrics summary cards.
11. `frontend/src/components/WorkspaceLoadingPlaceholder.jsx`: loading shell.
12. `frontend/src/utils/premierLeagueLogos.js`: team and league logo resolution, date/time formatting helpers.
13. `frontend/src/styles.css`: all styling, theme variables, responsive layout rules.

### Static assets and generated build

1. `frontend/public/logos/*`: local logo assets.
2. `frontend/dist/*`: built static frontend output, served by Flask in production.
3. `data/raw/leagues/*`: per-league live data files.
4. `data/raw/EPL_fixture.csv`: EPL fixture cache used by backend snapshot fallback.


## 4) Data model and file contracts

## 4.1 Primary data files

1. `data/raw/leagues/<league_slug>/season_df.csv`
2. `data/raw/leagues/<league_slug>/league_table.csv`
3. `data/raw/leagues/<league_slug>/matches_data.json`
4. `data/raw/EPL_fixture.csv`
5. `data/raw/corner_overrides.csv` (optional manual corrections)

`season_df.csv` is the master analytics source. Current structure includes about 130 columns per league, including:

1. Base match stats (possession, xG, shots, cards, corners, etc).
2. Match metadata (`game_id`, teams, team IDs, league IDs, date/time).
3. Derived metrics (`xg_diff`, `total_goals`, `field_tilt_*`).
4. Momentum (`home_momentum`, `away_momentum`).
5. Shot payload columns (`home_shots`, `away_shots`) as JSON strings.

`league_table.csv` contains:

1. `rank`, `team`, `played`, `won`, `drawn`, `lost`.
2. `goals_for`, `goals_against`, `points`, `goal_difference`.
3. `average_corners`, `average_possession`, `expected_goals_for`, `expected_goals_against`, `momentum`, `expected_goal_difference`.

`matches_data.json` stores raw list output from `Sofascore.get_match_dicts(season, league)` for each league.

`EPL_fixture.csv` stores official FPL fixture rows:

1. `gameweek`, `kickoff_time`, `home_team`, `away_team`.
2. scores, finished flag.
3. `fixture_id`, `league_id`, and team IDs.

## 4.2 Why file-based storage

1. Zero database setup.
2. Easy deployment on low-cost hosting.
3. Easy local inspectability and debugging.
4. Tradeoff: no transactional safety or query indexing.


## 5) Scraping and update pipeline

Two main scripts exist:

1. Full-season or utility operations in `data/raw/scrape_data.py`.
2. Daily incremental updater in `backend/update_data.py` (invoked via root `update_data.py`).

## 5.1 Full scrape path (`get_season_data`)

For a given `season` and `league`:

1. Fetch match list with `Sofascore.get_match_dicts`.
2. For each match:
   - scrape team match stats.
   - scrape shot events.
   - flatten into one row (`_flatten_match_stats_row`).
3. Assemble DataFrame.
4. Ensure `match_datetime` and chronological sorting (`ensure_match_datetime_and_sort`).
5. Compute derived features (`xg_diff`, `total_goals`, field tilt, win/draw flags).
6. Scrape momentum timeline per match.
7. Compute PCA component and map to home/away momentum values via sigmoid scaling.

Why this shape:

1. Flat rows simplify feature filtering in pandas.
2. Momentum from minute-level data is compressed to two scalar features for fast querying.

## 5.2 Incremental update path (`backend/update_data.py`)

`update_league` does:

1. Load existing `season_df.csv`.
2. Fetch current season match list.
3. Persist `matches_data.json` for that league.
4. Identify new played matches not already in CSV.
5. Scrape stats, shots, and momentum only for new matches.
6. Merge with existing data.
7. Recompute derived features for full league dataset.
8. Re-sort chronologically.
9. Save updated `season_df.csv`.
10. Rebuild and save `league_table.csv`.

After all leagues:

1. Refresh `data/raw/EPL_fixture.csv` using live FPL (`_refresh_epl_fixture_cache_csv`).

Why incremental:

1. Much faster than full re-scrape.
2. Lower load on third-party sources.
3. Better fit for daily operations.


## 6) Backend deep dive

Main server: `backend/backend_api.py` (`create_app()`).

## 6.1 App setup

1. Flask app with simple CORS headers (`Access-Control-Allow-Origin`, `Headers`, `Methods`).
2. API routes under `/api/*`.
3. Static SPA serving fallback from `frontend/dist`.

## 6.2 In-memory caches

1. `_snapshot_cache`: fixture snapshot cache with TTL.
2. `_sofascore_fixture_cache`: non-EPL live fixtures cache.
3. `_understat_fixture_cache`: non-EPL fallback fixture cache.
4. `_sofascore_season_cache`: tournament/season ID mapping cache.
5. `lru_cache` wrappers for heavy data transforms (`_load_raw_df`, `_build_store`, rank maps, etc).

Cache TTLs:

1. FPL snapshot: 60s.
2. Sofascore non-EPL fixture cache: 600s.
3. Understat fixture cache: 900s.

Why:

1. Reduces repeated network calls.
2. Keeps API latency stable.
3. Avoids hard dependency on upstream service availability for every request.

## 6.3 Fixture snapshot source chain

For EPL:

1. Preferred source: live FPL (`_fetch_live_fixture_snapshot`) when force-refreshing.
2. Non-blocking request path usually uses cache first.
3. If no in-memory cache, backend loads `data/raw/EPL_fixture.csv`.
4. If CSV missing, fallback to historical snapshot derived from `season_df.csv`.

For non-EPL (`/api/fixtures`):

1. Try live Sofascore (`_get_live_sofascore_league_fixtures`).
2. If empty, try Understat (`_get_understat_league_fixtures`).
3. If still empty, use local `matches_data.json` fallback (`_get_local_matches_json_league_fixtures`).

Note:

1. `matches_data.json` from `get_match_dicts` tends to include played matches, so local fallback may produce little/no upcoming data for non-EPL. Primary non-EPL fixture experience is intended to be live provider driven.

Why this layered chain:

1. Live accuracy when providers are available.
2. Controlled degradation when providers fail.
3. Keeps Kitchen UI usable even when network upstream is unstable.

## 6.4 Raw analytics load and store build

`_load_raw_df`:

1. Reads all `data/raw/leagues/*/season_df.csv`.
2. Adds `league_slug` from folder name.
3. Renames `game_id` to `match_id` if needed.
4. Falls back to legacy single-file path if league folders are absent.

`_build_store`:

1. Iterates unique `match_id`.
2. Uses `MatchAnalyticsBuilder` to build one `MatchAnalytics` object per match.
3. Ingests all into `AnalyticsStore`.

Why:

1. Converts CSV rows into a normalized object model once.
2. Makes repeated evidence queries faster and consistent.

## 6.5 Analytics object model

`MatchAnalytics` stores:

1. match ID, league ID.
2. home/away team IDs.
3. feature dictionary.

`for_perspective("away")` swaps `home_*` and `away_*` features for perspective-aware analysis.

Why:

1. Avoids duplicating separate home/away materialization logic everywhere.
2. Keeps filter semantics consistent.

## 6.6 Filter system

Flow:

1. Frontend sends either:
   - explicit `evidenceFilters` array, or
   - UI `filters` object.
2. Backend `_parse_filters`:
   - if `evidenceFilters` exists, deserialize directly.
   - else build default filters from `filters` object using `_default_evidence_filters`.

Filter classes in `backend/filters/filters.py` define:

1. key
2. display name
3. required columns
4. `build(...) -> FilterSpec`

`AnalyticsStore` executes filters:

1. Column filters via resolved series and operator logic.
2. Context filters (`head_to_head`, `last_n_games`) via specialized behavior.
3. Special shot-xG filter (`team_shot_xg`) parses `home_shots` / `away_shots` JSON payloads and counts shots meeting threshold.

Why this design:

1. Declarative specs keep frontend/backend contracts stable.
2. Store remains generic while bet-type workspaces remain thin.

## 6.7 Opponent ranking filter stage

After evidence retrieval, backend may apply ranking filters from `filters` payload:

1. Loads league table via `_load_league_table(league_id)`.
2. Builds rank maps for selected metric (`xGD`, `xGF`, `xGA`, position, corners, momentum, possession).
3. Resolves opponent per row based on venue.
4. Filters rows to selected rank ranges.

Why post-evidence:

1. Depends on external league table context.
2. Easier to apply after core evidence set is materialized.

## 6.8 Team anchor match strategy

Each workspace endpoint resolves an anchor match for each requested team:

1. Try preferred perspective in same league.
2. Fallback to any perspective if needed.
3. Use that anchor `match_id` + perspective in `EvidenceRequest`.

Why:

1. Ensures evidence query has a concrete match context.
2. Works around provider team ID mismatches by using name-based candidate IDs.

## 6.9 Workspace endpoint behavior

Common endpoint pattern:

1. Validate `match_id`.
2. Resolve fixture for fallback team/league metadata.
3. Parse filters.
4. Resolve home/away anchor matches.
5. Query store (via workspace class or direct `EvidenceRequest`).
6. Sort history chronologically.
7. Apply opponent rank filters.
8. Compute summary metrics.
9. Build `recent_matches` payload for chart.
10. Return JSON with `workspace`, `sample_size(s)`, `metrics`, `chartSeries`, `recent_matches`, `notes`.

Endpoints:

1. `/api/workspace/1x2`
2. `/api/workspace/over_under`
3. `/api/workspace/double_chance`
4. `/api/workspace/btts`
5. `/api/workspace/home_ou`
6. `/api/workspace/away_ou`
7. `/api/workspace/corners`

Bet outcome logic:

1. 1X2:
   - Win -> `1.0`
   - Draw -> `0.5`
   - Loss -> `0.1`
2. Double Chance:
   - Hit if `goals_scored >= opponent_goals`.
3. BTTS:
   - Hit if both teams scored (`goals_scored > 0 and opponent_goals > 0`).
4. Over/Under:
   - Hit if `total_goals > line`.
5. Home O/U:
   - Hit if home-side `goals_scored > line` (home sample only).
6. Away O/U:
   - Hit if away-side `goals_scored > line` (away sample only).
7. Corners:
   - Hit if resolved `total_corners > line` (with optional manual override by `corner_overrides.csv`).

Why separate endpoints:

1. Keeps each market payload explicit.
2. Allows distinct metric summaries and chart shaping without frontend condition explosion.

## 6.10 Recent match payload builders

Backend builds chart-ready rows in:

1. `_build_recent_matches`
2. `_build_recent_matches_over_under`
3. `_build_recent_matches_team_goals_over_under`
4. `_build_recent_matches_double_chance`
5. `_build_recent_matches_btts`
6. `_build_recent_matches_corners`

Each row includes:

1. display labels (`chart_label`, `fixture_display`)
2. match datetime ISO string
3. venue/opponent metadata
4. result metric fields
5. overlay-ready numeric features (momentum, xG, goals, possession, field tilt, opponent rank-derived values)

Why backend-generated:

1. Frontend remains mostly rendering-focused.
2. Keeps chart semantics consistent across clients.


## 7) AnalyticsStore internals

Key methods:

1. `ingest`: store `MatchAnalytics` by `match_id`.
2. `query`:
   - resolve team ID from anchor match + perspective.
   - materialize all games for that team.
   - apply filters.
   - return `EvidenceSubsetImpl`.
3. `_materialize_team_games`: builds one row per relevant match with venue-aware columns.
4. `_get_filter_series`: resolves filter series for team/opponent context (momentum, xG, possession, field tilt).
5. `_apply_operator_filter`: handles comparison operators (`>`, `>=`, `<`, `<=`, `==`, `!=`, `between`, `in`).
6. `_apply_context_filter`: head-to-head and last-n games logic.

Why this abstraction:

1. Centralized filter execution.
2. Bet-type workspaces can focus on outcome transformations.
3. Simple pandas-based implementation with low overhead for current scale.


## 8) Frontend deep dive

## 8.1 App bootstrap

1. `frontend/src/main.jsx` mounts `App`.
2. `App` renders `KitchenPage`.

## 8.2 API client

`frontend/src/api/backendApi.js`:

1. `requestJson` wraps fetch + uniform JSON error handling.
2. Exposes endpoint-specific calls:
   - `getLeagues`, `getFixturesForLeague`, `getMatch`.
   - workspace calls for each market endpoint.
3. Supports `VITE_API_BASE` override; Vite dev proxy routes `/api` to local backend.

## 8.3 KitchenPage state machine

State:

1. leagues
2. selected league
3. fixtures
4. selected match
5. error/loading flags
6. fixture cache per league (`Map`)
7. remembered selected match per league (`Map`)

Behavior:

1. On mount:
   - fetch leagues.
   - select first league.
   - warm fixture cache for other leagues in background.
2. On league change:
   - use cached fixtures if present.
   - else fetch from `/api/fixtures`.
3. Select first available fixture (or previous selection if still present).
4. Pass selected fixture metadata into `BetTypeWorkspace`.

Why:

1. Fast league switching.
2. Reduces duplicate network calls.
3. Keeps user context per league.

## 8.4 BetTypeWorkspace behavior

Responsibilities:

1. Track active bet type tab.
2. Manage draft vs applied filter state.
3. Manage draft vs applied line values for O/U and Corners.
4. Build `evidenceFilters` payload from filter UI.
5. Request workspace JSON from correct backend endpoint.
6. Cache workspace responses by composite key (match/team/bet type/filters/lines).

Why separate draft/applied states:

1. Users can adjust sliders without immediate re-query.
2. API requests occur only when user applies changes.

## 8.5 FilterDropdown behavior

Features:

1. Category tabs: Suggested, Split, Stats, Opponent Rankings.
2. Dual-ended range sliders for numeric filters.
3. Venue selector.
4. Shot xG threshold and min-shot controls.
5. Split view toggle (`both`, `home`, `away`).
6. Overlay filter activation for chart line overlays.

Why:

1. Lets user tune evidence set and visual overlays independently.
2. Range-first controls align with `between` filter contract used by backend.

## 8.6 ChartArea behavior

`ChartArea.jsx` converts backend `recent_matches` into plotted bars:

1. Build bars per market (1X2, Double Chance, BTTS, O/U, Corners).
2. Normalize and clamp line values.
3. Compute hit rates, graph averages, season averages.
4. Draw chart using Recharts `ComposedChart`.
5. Support optional overlay line from active filter metric.
6. Provide draggable line paddle for O/U and Corners thresholds.
7. Show custom axis ticks with logos and compact date labels.

Why custom shaping:

1. Backend payloads differ by market.
2. Frontend needs unified chart primitives (`label`, `value`, `color`) for consistent rendering.

## 8.7 Metrics panel

`MetricsPanel` renders sample size and side-by-side home/away summary blocks with market-specific labels.

Why separate component:

1. Keeps chart component focused on visualization.
2. Easier responsive duplication (desktop and mobile placements).

## 8.8 Logo and date utilities

`premierLeagueLogos.js`:

1. Maps normalized team names and aliases to local logo paths.
2. Falls back to Sofascore image endpoints by team/tournament ID.
3. Formats fixture date/time labels.

Why:

1. Local assets improve visual consistency and reduce runtime dependency.
2. Aliases handle name variations from different data providers.

## 8.9 Styling system

`styles.css` defines:

1. Theme variables (`--bg`, `--surface`, `--primary`, etc).
2. Desktop layout (league header + fixture sidebar + workspace pane).
3. Mobile adaptations (bottom fixture menu and responsive chart sizing).

Why:

1. Single stylesheet keeps component JS cleaner.
2. CSS variables allow broad theme changes with low maintenance.


## 9) Why it functions this way (design rationale)

1. CSV-first architecture:
   - chosen for low ops complexity.
   - tradeoff is weaker concurrency and no advanced querying.
2. In-memory analytics store:
   - fast for current dataset size.
   - avoids repeated per-request CSV scans once cached.
3. Multi-source fixture strategy:
   - maximizes uptime despite provider instability.
4. Separate workspace endpoints:
   - explicit business logic per market.
5. Post-query opponent rank filtering:
   - requires league table context, so applied after evidence materialization.
6. Anchor match approach:
   - practical way to drive perspective-specific history without dedicated fixture DB joins.
7. Draft/applied UI filter model:
   - avoids network spam while dragging ranges.


## 10) Current operational snapshot (as of 2026-03-01)

Per-league season row counts:

1. EPL: 277
2. Ligue 1: 211
3. Bundesliga: 215
4. Serie A: 269
5. La Liga: 257

`matches_data.json` entries:

1. EPL: 277
2. Ligue 1: 208
3. Bundesliga: 215
4. Serie A: 269
5. La Liga: 254

`EPL_fixture.csv` rows: 379


## 11) Operational runbook

Local backend:

```bash
python backend_api.py
```

Local frontend:

```bash
cd frontend
npm install
npm run dev
```

Incremental data refresh (all leagues):

```bash
python update_data.py
```

Single league refresh:

```bash
python update_data.py --league "England Premier League"
```

Full scraping utility script examples:

```bash
python data/raw/scrape_data.py --mode update --league "England Premier League" --season 25/26
python data/raw/scrape_data.py --mode multi-league-scrape --season 25/26
```


## 12) Known limitations and risks

1. Non-EPL local fixture fallback uses `matches_data.json` from `get_match_dicts`, which can skew toward played matches, not full future fixture sets.
2. No relational DB means no transactional updates or historical versioning.
3. In-memory cache can become stale between update runs unless scripts are executed regularly.
4. Team identity reconciliation across providers relies on name normalization and alias maps, which can miss edge cases.
5. Some code paths are legacy (`backend/main.py`, root single-file CSV fallbacks) and should be treated as compatibility layers.


## 13) Extension guide

To add a new betting market cleanly:

1. Create a workspace class under `backend/bet_type/...` implementing `BetTypeWorkspace`.
2. Define allowed filters and outcome enrichment logic.
3. Add endpoint in `backend/backend_api.py`.
4. Extend frontend tabs in `BetTypeWorkspace.jsx`.
5. Add chart conversion logic in `ChartArea.jsx`.
6. Add metric rendering branch in `MetricsPanel.jsx`.

To add a new filter:

1. Add filter class in `backend/filters/filters.py`.
2. Extend `buildEvidenceFilters` in frontend.
3. Handle series resolution in `AnalyticsStore._get_filter_series` if needed.
4. Add UI control in `FilterDropdown.jsx`.


## 14) End-to-end request example (Over/Under)

1. User selects fixture in Kitchen page.
2. Frontend posts `/api/workspace/over_under` with:
   - match/team IDs
   - line
   - applied filter object and generated `evidenceFilters`.
3. Backend resolves fixture metadata and anchor match IDs.
4. Backend `OverUnderWorkspace.get_evidence(...)` queries `AnalyticsStore`.
5. Store materializes all relevant games for each team, applies filters.
6. Backend computes over/under metrics and recent match payload rows.
7. Frontend receives JSON and transforms to chart bars.
8. User drags line or applies filters, frontend re-queries and redraws.


## 15) Summary

Backline v2 is a pragmatic, file-backed analytics system:

1. Data ingestion is ScraperFC-driven and incremental.
2. Backend logic is split between fixture sourcing, evidence querying, and market-specific aggregation.
3. Frontend is stateful and interactive, with caching and market-aware charting.
4. The architecture favors low operational overhead and predictable behavior over heavy infrastructure.

