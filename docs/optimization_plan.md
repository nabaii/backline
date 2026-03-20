# Performance Optimization Plan

The project suffers from significant initial lag due to inefficient data processing in the backend and redundant concurrent network requests in the frontend. This plan outlines fixes to address these bottlenecks.

## Proposed Changes

### Frontend Optimization

#### [MODIFY] [KitchenPage.jsx](file:///c:/Users/enaic/OneDrive/Desktop/backline/backline_v2/frontend/src/components/KitchenPage.jsx)
- **Remove "Warm Cache" logic**: Stop fetching fixtures for all leagues in parallel upon startup. This currently triggers 5+ concurrent heavy backend requests, causing server-side resource exhaustion.
- **Lazy Load League Data**: Only fetch fixtures for a league when it is explicitly selected by the user.

### Backend Optimization

#### [MODIFY] [backend_api.py](file:///c:/Users/enaic/OneDrive/Desktop/backline/backline_v2/backend/backend_api.py)
- **Optimize [_build_store](file:///c:/Users/enaic/OneDrive/Desktop/backline/backline_v2/backend/backend_api.py#795-807)**:
    - Current implementation performs a full scan of the dataset ($O(N^2)$) for every match during initialization.
    - Rewrite to use `raw_df.iterrows()` or `raw_df.groupby` to achieve $O(N)$ performance.
- **Improve Fixture Fetching**:
    - Add a simple file-based cache for fixtures (e.g., `data/cache/fixtures_{league}.json`) to avoid redundant external API calls across server restarts on Render.
    - Implement a basic request deduplication logic for concurrent fixture fetches for the same league.

## Verification Plan

### Automated Tests
- I will write a simple benchmark script in `tests/benchmark_initialization.py` to measure the time taken for [_build_store](file:///c:/Users/enaic/OneDrive/Desktop/backline/backline_v2/backend/backend_api.py#795-807) before and after optimization.
- Command to run: `python tests/benchmark_initialization.py`

### Manual Verification
1.  **Local Startup Check**: Run the project locally and verify that the initial page load is significantly faster.
2.  **Network Audit**: Use browser developer tools to confirm that only the necessary league fixtures are fetched on startup.
3.  **Redundant Refresh Test**: Trigger multiple fixture refreshes and ensure they hit the cache or are handled efficiently.
