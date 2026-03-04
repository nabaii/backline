# Figma Handoff Workflow (Backline v2)

Last updated: 2026-03-04

## Goal

Create a predictable design-to-code pipeline so Figma edits can be implemented quickly with minimal ambiguity.

This workflow is aligned to the current frontend:

1. React components in `frontend/src/components`
2. Global styling in `frontend/src/styles.css`
3. Chart rendering in `frontend/src/components/ChartArea.jsx`

## 1) Breakpoint and Frame Setup

Build Figma frames to match current responsive behavior:

1. Desktop: `1440 x 1024` (applies to widths `>= 1101`)
2. Tablet: `1024 x 1366` (applies to widths `701..1100`)
3. Mobile: `390 x 844` (applies to widths `<= 700`)
4. Small mobile QA frame: `375 x 812` (captures `<= 380` typography/tight spacing behavior)

Key responsive rules already in code:

1. Main desktop split (`fixtures + workspace`) collapses at `1100px`.
2. Mobile behavior changes at `700px`:
   - fixture sidebar hidden
   - bottom fixture menu shown
   - workspace constrained to max width `430px`
3. Extra compact adjustments at `380px`.

## 2) Figma File Structure

Use this page structure:

1. `00_Foundations`
2. `10_Desktop`
3. `11_Tablet`
4. `12_Mobile`
5. `20_Components`
6. `30_States_Interactions`
7. `99_Handoff_Notes`

Use Auto Layout for all reusable blocks.

## 3) Component Naming Convention

Use this naming pattern in Figma components:

`BL/<Area>/<Component>/<Variant>`

Examples:

1. `BL/Shell/KitchenHeader/Default`
2. `BL/Nav/LeagueTab/Active`
3. `BL/Fixture/FixtureItem/Selected`
4. `BL/Workspace/BetTab/Active`
5. `BL/Filter/CategoryTab/Default`
6. `BL/Filter/Pill/Modified`
7. `BL/Chart/TeamBlock/OverUnder`
8. `BL/Metrics/Panel/Desktop`

State variants should always use the same property keys:

1. `state`: `default | hover | active | selected | disabled`
2. `size`: `desktop | tablet | mobile | compact`
3. `theme`: `default` (reserve for future themes)

## 4) Repo Component Map

Map each Figma component to current code targets:

| Figma component | React file | Primary CSS hooks |
| --- | --- | --- |
| `BL/Shell/KitchenHeader` | `frontend/src/components/KitchenPage.jsx` | `.kitchen-header`, `.brand-block` |
| `BL/Nav/LeagueSelector` | `frontend/src/components/LeagueSelector.jsx` | `.league-tabs`, `.league-tab` |
| `BL/Fixture/FixtureList` | `frontend/src/components/FixtureList.jsx` | `.fixture-list`, `.fixture-item` |
| `BL/Workspace/BetTabs` | `frontend/src/components/BetTypeWorkspace.jsx` | `.workspace-header`, `.bettabs` |
| `BL/Workspace/FilterDrawer` | `frontend/src/components/BetTypeWorkspace.jsx` | `.filters-drawer`, `.filters-drawer-header` |
| `BL/Filter/FilterDropdown` | `frontend/src/components/FilterDropdown.jsx` | `.filter-dropdown`, `.filter-category-tabs`, `.filter-pill-btn` |
| `BL/Chart/ChartArea` | `frontend/src/components/ChartArea.jsx` | `.chart-area`, `.team-chart-block`, `.pm-*` |
| `BL/Metrics/MetricsPanel` | `frontend/src/components/MetricsPanel.jsx` | `.metrics-panel`, `.team-metrics` |
| `BL/Fixture/BottomMenu` | `frontend/src/components/KitchenPage.jsx` | `.fixture-bottom-menu`, `.fixture-bottom-menu-item` |
| `BL/Workspace/LoadingShell` | `frontend/src/components/WorkspaceLoadingPlaceholder.jsx` | `.workspace-loading-*` |

## 5) Token Mapping (Design -> Code)

Current base tokens live in `:root` in `frontend/src/styles.css`.

### 5.1 Color Tokens

| Figma token | CSS variable | Value |
| --- | --- | --- |
| `BL/Color/Bg/Base` | `--bg` | `#0e0e11` |
| `BL/Color/Bg/Soft` | `--bg-soft` | `#131316` |
| `BL/Color/Surface/Base` | `--surface` | `#1a1a1f` |
| `BL/Color/Surface/Alt` | `--surface-alt` | `#1f1f25` |
| `BL/Color/Surface/Elev` | `--surface-elev` | `#242429` |
| `BL/Color/Border/Base` | `--border` | `#2a2a30` |
| `BL/Color/Border/Light` | `--border-light` | `#35353d` |
| `BL/Color/Text/Primary` | `--text` | `#edeef2` |
| `BL/Color/Text/Secondary` | `--text-secondary` | `#9b9ca6` |
| `BL/Color/Text/Muted` | `--muted` | `#6e7080` |
| `BL/Color/Brand/Primary` | `--primary` | `#f8c629` |
| `BL/Color/Brand/PrimaryDim` | `--primary-dim` | `rgba(248, 198, 41, 0.15)` |
| `BL/Color/Accent/Blue` | `--accent-blue` | `#4aa3df` |
| `BL/Color/State/Success` | `--success` | `#2ecc71` |
| `BL/Color/State/Danger` | `--danger` | `#e74c3c` |
| `BL/Color/State/Draw` | `--draw` | `#f39c12` |

### 5.2 Shape and Motion Tokens

| Figma token | CSS variable | Value |
| --- | --- | --- |
| `BL/Radius/Sm` | `--radius-sm` | `7px` |
| `BL/Radius/Md` | `--radius` | `10px` |
| `BL/Radius/Lg` | `--radius-lg` | `14px` |
| `BL/Motion/Fast` | `--transition` | `150ms ease` |

### 5.3 Typography Tokens

| Figma token | Source |
| --- | --- |
| `BL/Type/UI/*` | `Inter` (`styles.css` import) |
| `BL/Type/Chart/*` | `Aldrich` (`styles.css` import + chart labels) |

### 5.4 Chart Semantic Colors (hardcoded in `ChartArea.jsx`)

| Figma token | Value | Meaning |
| --- | --- | --- |
| `BL/Chart/Over` | `#2ecc71` | Over / positive hit |
| `BL/Chart/Under` | `#e74c3c` | Under / miss |
| `BL/Chart/Draw` | `#f39c12` | Draw (1X2) |
| `BL/Chart/Overlay` | `#2d6bff` | Overlay line |
| `BL/Chart/Line` | `#f8c629` | O/U threshold line |

## 6) Required Deliverables From Figma Back to Dev

When handing design back for implementation, include:

1. Figma file link
2. Page + frame name(s) changed
3. Component names changed (using `BL/...` naming)
4. Token changes (old -> new)
5. Interaction notes (hover, active, transitions, drag behavior)
6. Priority list (`P0`, `P1`, `P2`)

Use the template in `docs/figma_handoff_return_template.md`.

## 7) Implementation Rules For Fast Turnaround

1. Do not rename component roots (`.kitchen-*`, `.workspace-*`, `.filter-*`, `.pm-*`) unless explicitly requested.
2. Prefer token updates first, layout updates second, behavior updates third.
3. Keep breakpoint behavior compatible with existing `1100`, `700`, `380` thresholds unless the handoff explicitly changes them.
4. If a Figma update cannot map cleanly to current structure, add a note under `99_Handoff_Notes` with proposed code tradeoff.

## 8) Suggested Handoff Cycle

1. Design pass in Figma with the page/component/token structure above.
2. Fill return template.
3. Dev implementation pass.
4. Side-by-side QA at 1440, 1024, 390, and 375 widths.
5. Final polish pass for interaction parity.
