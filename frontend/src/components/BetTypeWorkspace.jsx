import React, { useEffect, useRef, useState } from 'react'
import api from '../api/backendApi'
import FilterDropdown from './FilterDropdown'
import ChartArea from './ChartArea'
import MetricsPanel from './MetricsPanel'
import WorkspaceLoadingPlaceholder from './WorkspaceLoadingPlaceholder'

const OVER_UNDER_LINE_STEP = 0.5
const OVER_UNDER_LINE_MIN = 0.5
const OVER_UNDER_LINE_MAX = 8.5
const DEFAULT_OVER_UNDER_LINE = 2.5
const CORNERS_LINE_STEP = 0.5
const CORNERS_LINE_MIN = 0.5
const CORNERS_LINE_MAX = 20.5
const DEFAULT_CORNERS_LINE = 8.5
const DEFAULT_CHART_TEAM_VIEW = 'both'
const BET_TYPE_ONE_X_TWO = '1X2'
const BET_TYPE_OVER_UNDER = 'over_under'
const BET_TYPE_DOUBLE_CHANCE = 'double_chance'
const BET_TYPE_BTTS = 'btts'
const BET_TYPE_HOME_OU = 'home_ou'
const BET_TYPE_AWAY_OU = 'away_ou'
const BET_TYPE_CORNERS = 'corners'

function defaultTeamViewForBetType(betType) {
  if (betType === BET_TYPE_HOME_OU) return DEFAULT_CHART_TEAM_VIEW
  if (betType === BET_TYPE_AWAY_OU) return DEFAULT_CHART_TEAM_VIEW
  return DEFAULT_CHART_TEAM_VIEW
}

function createDefaultFilters() {
  return {
    home_away: 'all',
    team_momentum_range: [0, 10],
    opponent_momentum_range: [0, 10],
    total_match_goals_range: [0, 10],
    team_goals_range: [0, 10],
    opposition_goals_range: [0, 10],
    team_xg_range: [0, 5],
    opposition_xg_range: [0, 5],
    team_possession_range: [0, 100],
    opposition_possession_range: [0, 100],
    field_tilt_range: [0, 1],
    opponent_rank_xgd_range: [1, 20],
    opponent_rank_xgf_range: [1, 20],
    opponent_rank_xga_range: [1, 20],
    opponent_rank_position_range: [1, 20],
    opponent_rank_corners_range: [1, 20],
    opponent_rank_momentum_range: [1, 20],
    opponent_rank_possession_range: [1, 20],
    shot_xg_threshold: 0.3,
    shot_xg_min_shots: 0,
  }
}

function normalizeOverUnderLine(rawValue) {
  const numeric = Number(rawValue)
  if (Number.isNaN(numeric)) return DEFAULT_OVER_UNDER_LINE
  const clamped = Math.min(OVER_UNDER_LINE_MAX, Math.max(OVER_UNDER_LINE_MIN, numeric))
  return Math.round(clamped / OVER_UNDER_LINE_STEP) * OVER_UNDER_LINE_STEP
}

function normalizeCornersLine(rawValue) {
  const numeric = Number(rawValue)
  if (Number.isNaN(numeric)) return DEFAULT_CORNERS_LINE
  const clamped = Math.min(CORNERS_LINE_MAX, Math.max(CORNERS_LINE_MIN, numeric))
  return Math.round(clamped / CORNERS_LINE_STEP) * CORNERS_LINE_STEP
}

function buildEvidenceFilters(filters) {
  const evidenceFilters = []

  if (filters.home_away && filters.home_away !== 'all') {
    evidenceFilters.push({
      key: 'venue',
      kind: 'column',
      field: 'venue',
      operator: '==',
      value: filters.home_away,
      display_name: 'Match Venue',
      required_columns: ['venue'],
    })
  }

  if (Array.isArray(filters.team_momentum_range) && filters.team_momentum_range.length === 2) {
    evidenceFilters.push({
      key: 'team_momentum',
      kind: 'column',
      field: 'home_momentum',
      operator: 'between',
      value: filters.team_momentum_range,
      display_name: 'Team Momentum',
      required_columns: ['home_momentum', 'away_momentum'],
    })
  }

  if (Array.isArray(filters.opponent_momentum_range) && filters.opponent_momentum_range.length === 2) {
    evidenceFilters.push({
      key: 'opponent_momentum',
      kind: 'column',
      field: 'away_momentum',
      operator: 'between',
      value: filters.opponent_momentum_range,
      display_name: 'Opponent Momentum',
      required_columns: ['home_momentum', 'away_momentum'],
    })
  }

  if (Array.isArray(filters.total_match_goals_range) && filters.total_match_goals_range.length === 2) {
    evidenceFilters.push({
      key: 'total_match_goals',
      kind: 'column',
      field: 'total_goals',
      operator: 'between',
      value: filters.total_match_goals_range,
      display_name: 'Total Match Goals',
      required_columns: ['total_goals'],
    })
  }

  if (Array.isArray(filters.team_goals_range) && filters.team_goals_range.length === 2) {
    evidenceFilters.push({
      key: 'goals_scored',
      kind: 'column',
      field: 'goals_scored',
      operator: 'between',
      value: filters.team_goals_range,
      display_name: 'Team Goals',
      required_columns: ['goals_scored', 'venue'],
    })
  }

  if (Array.isArray(filters.opposition_goals_range) && filters.opposition_goals_range.length === 2) {
    evidenceFilters.push({
      key: 'goals_conceded',
      kind: 'column',
      field: 'opponent_goals',
      operator: 'between',
      value: filters.opposition_goals_range,
      display_name: 'Opposition Goals',
      required_columns: ['opponent_goals', 'venue'],
    })
  }

  if (Array.isArray(filters.team_xg_range) && filters.team_xg_range.length === 2) {
    evidenceFilters.push({
      key: 'team_xg',
      kind: 'column',
      field: 'team_xg',
      operator: 'between',
      value: filters.team_xg_range,
      display_name: 'Team xG',
      required_columns: ['expected_goals_home', 'expected_goals_away', 'venue'],
    })
  }

  if (Array.isArray(filters.opposition_xg_range) && filters.opposition_xg_range.length === 2) {
    evidenceFilters.push({
      key: 'opponent_xg',
      kind: 'column',
      field: 'opponent_xg',
      operator: 'between',
      value: filters.opposition_xg_range,
      display_name: 'Opposition xG',
      required_columns: ['expected_goals_home', 'expected_goals_away', 'venue'],
    })
  }

  if (Array.isArray(filters.team_possession_range) && filters.team_possession_range.length === 2) {
    evidenceFilters.push({
      key: 'team_possession',
      kind: 'column',
      field: 'ball_possession_home',
      operator: 'between',
      value: filters.team_possession_range,
      display_name: 'Team Possession',
      required_columns: ['ball_possession_home', 'ball_possession_away', 'venue'],
    })
  }

  if (Array.isArray(filters.opposition_possession_range) && filters.opposition_possession_range.length === 2) {
    evidenceFilters.push({
      key: 'opponent_possession',
      kind: 'column',
      field: 'ball_possession_away',
      operator: 'between',
      value: filters.opposition_possession_range,
      display_name: 'Opponent Possession',
      required_columns: ['ball_possession_home', 'ball_possession_away', 'venue'],
    })
  }

  if (Array.isArray(filters.field_tilt_range) && filters.field_tilt_range.length === 2) {
    evidenceFilters.push({
      key: 'field_tilt',
      kind: 'column',
      field: 'field_tilt_home',
      operator: 'between',
      value: filters.field_tilt_range,
      display_name: 'Field Tilt',
      required_columns: ['field_tilt_home', 'field_tilt_away', 'venue'],
    })
  }

  const shotXgThreshold = Number(filters.shot_xg_threshold)
  const shotXgMinShots = Number(filters.shot_xg_min_shots)
  if (Number.isFinite(shotXgThreshold) && Number.isFinite(shotXgMinShots) && shotXgMinShots > 0) {
    evidenceFilters.push({
      key: 'team_shot_xg',
      kind: 'column',
      field: 'home_shots',
      operator: '>=',
      value: {
        min_xg: shotXgThreshold,
        min_shots: Math.max(0, Math.floor(shotXgMinShots)),
      },
      display_name: 'Team Shot xG Count',
      required_columns: ['home_shots', 'away_shots', 'venue'],
    })
  }

  return evidenceFilters
}

function buildWorkspaceCacheKey({
  matchId,
  homeTeamId,
  awayTeamId,
  homeTeamName,
  awayTeamName,
  leagueId,
  betType,
  filters,
  overUnderLine,
  cornersLine,
}) {
  return JSON.stringify({
    matchId,
    homeTeamId,
    awayTeamId,
    homeTeamName,
    awayTeamName,
    leagueId,
    betType,
    filters,
    overUnderLine,
    cornersLine,
  })
}

export default function BetTypeWorkspace({
  matchId,
  homeTeamId,
  awayTeamId,
  homeTeamName,
  awayTeamName,
  leagueId,
}) {
  const [betType, setBetType] = useState(BET_TYPE_ONE_X_TWO)
  const [workspace, setWorkspace] = useState(null)
  const [draftFilters, setDraftFilters] = useState(createDefaultFilters)
  const [appliedFilters, setAppliedFilters] = useState(createDefaultFilters)
  const [draftOverUnderLine, setDraftOverUnderLine] = useState(DEFAULT_OVER_UNDER_LINE)
  const [appliedOverUnderLine, setAppliedOverUnderLine] = useState(DEFAULT_OVER_UNDER_LINE)
  const [draftCornersLine, setDraftCornersLine] = useState(DEFAULT_CORNERS_LINE)
  const [appliedCornersLine, setAppliedCornersLine] = useState(DEFAULT_CORNERS_LINE)
  const [chartTeamView, setChartTeamView] = useState(DEFAULT_CHART_TEAM_VIEW)
  const [isFiltersOpen, setIsFiltersOpen] = useState(false)
  const [activeOverlayFilter, setActiveOverlayFilter] = useState(null)
  const [error, setError] = useState(null)
  const workspaceCacheRef = useRef(new Map())
  const hasPendingFilterChanges = JSON.stringify(draftFilters) !== JSON.stringify(appliedFilters)

  const applyFilters = () => {
    if (!hasPendingFilterChanges) return
    setAppliedFilters({ ...draftFilters })
  }

  const clearFilters = () => {
    const defaults = createDefaultFilters()
    setDraftFilters(defaults)
    setAppliedFilters(defaults)
  }

  const updateOverUnderLineDraft = (nextValue) => {
    setDraftOverUnderLine(normalizeOverUnderLine(nextValue))
  }

  const commitOverUnderLine = (nextValue) => {
    const normalized = normalizeOverUnderLine(nextValue)
    setDraftOverUnderLine(normalized)
    setAppliedOverUnderLine(current => (current === normalized ? current : normalized))
  }

  const updateCornersLineDraft = (nextValue) => {
    setDraftCornersLine(normalizeCornersLine(nextValue))
  }

  const commitCornersLine = (nextValue) => {
    const normalized = normalizeCornersLine(nextValue)
    setDraftCornersLine(normalized)
    setAppliedCornersLine(current => (current === normalized ? current : normalized))
  }

  useEffect(() => {
    if (!matchId || !homeTeamId || !awayTeamId) return
    setError(null)
    const evidenceFilters = buildEvidenceFilters(appliedFilters)
    const cacheKey = buildWorkspaceCacheKey({
      matchId,
      homeTeamId,
      awayTeamId,
      homeTeamName,
      awayTeamName,
      leagueId,
      betType,
      filters: appliedFilters,
      overUnderLine: appliedOverUnderLine,
      cornersLine: appliedCornersLine,
    })
    const cachedWorkspace = workspaceCacheRef.current.get(cacheKey)
    if (cachedWorkspace) {
      setWorkspace(cachedWorkspace)
      return
    }

    const requestPayload = {
      match_id: matchId,
      home_team_id: homeTeamId,
      away_team_id: awayTeamId,
      home_team_name: homeTeamName,
      away_team_name: awayTeamName,
      league_id: leagueId,
      limit: 0,
      filters: appliedFilters,
      evidenceFilters,
    }
    const requestPromise = (() => {
      if (betType === BET_TYPE_OVER_UNDER) {
        return api.getWorkspaceOverUnder({ ...requestPayload, line: appliedOverUnderLine })
      }
      if (betType === BET_TYPE_DOUBLE_CHANCE) {
        return api.getWorkspaceDoubleChance(requestPayload)
      }
      if (betType === BET_TYPE_BTTS) {
        return api.getWorkspaceBtts(requestPayload)
      }
      if (betType === BET_TYPE_CORNERS) {
        return api.getWorkspaceCorners({ ...requestPayload, line: appliedCornersLine })
      }
      if (betType === BET_TYPE_HOME_OU) {
        return api.getWorkspaceHomeOu({ ...requestPayload, line: appliedOverUnderLine })
      }
      if (betType === BET_TYPE_AWAY_OU) {
        return api.getWorkspaceAwayOu({ ...requestPayload, line: appliedOverUnderLine })
      }
      return api.getWorkspace1X2(requestPayload)
    })()

    requestPromise
      .then(r => {
        workspaceCacheRef.current.set(cacheKey, r)
        setWorkspace(r)
      })
      .catch(err => {
        setWorkspace(null)
        setError(err?.message || 'Failed to load workspace')
      })
  }, [
    matchId,
    homeTeamId,
    awayTeamId,
    homeTeamName,
    awayTeamName,
    leagueId,
    betType,
    appliedFilters,
    appliedOverUnderLine,
    appliedCornersLine,
  ])

  useEffect(() => {
    if (!matchId) return
    const defaults = createDefaultFilters()
    setDraftFilters(defaults)
    setAppliedFilters(defaults)
    setChartTeamView(defaultTeamViewForBetType(betType))
  }, [matchId])

  if (error) {
    return (
      <div>
        <div><strong>Workspace failed to load.</strong></div>
        <div>{error}</div>
      </div>
    )
  }

  if (!workspace) return <WorkspaceLoadingPlaceholder activeBetType={betType} />

  return (
    <div className="bettype-workspace">
      <div className="workspace-header">
        <div className="bettabs">
          <button className={betType === BET_TYPE_ONE_X_TWO ? 'active' : ''} onClick={() => setBetType(BET_TYPE_ONE_X_TWO)}>1X2</button>
          <button className={betType === BET_TYPE_DOUBLE_CHANCE ? 'active' : ''} onClick={() => setBetType(BET_TYPE_DOUBLE_CHANCE)}>Double Chance</button>
          <button className={betType === BET_TYPE_BTTS ? 'active' : ''} onClick={() => setBetType(BET_TYPE_BTTS)}>BTTS</button>
          <button className={betType === BET_TYPE_OVER_UNDER ? 'active' : ''} onClick={() => setBetType(BET_TYPE_OVER_UNDER)}>Over/Under</button>
          <button className={betType === BET_TYPE_HOME_OU ? 'active' : ''} onClick={() => setBetType(BET_TYPE_HOME_OU)}>Home O/U</button>
          <button className={betType === BET_TYPE_AWAY_OU ? 'active' : ''} onClick={() => setBetType(BET_TYPE_AWAY_OU)}>Away O/U</button>
          <button className={betType === BET_TYPE_CORNERS ? 'active' : ''} onClick={() => setBetType(BET_TYPE_CORNERS)}>Corners</button>
        </div>
      </div>

      <div className={`workspace-body ${isFiltersOpen ? 'filters-open' : 'filters-closed'}`}>
        <div className="workspace-main">
          <div className="workspace-chart-stack">
            <ChartArea
              recentMatches={workspace.recent_matches}
              betType={workspace?.workspace?.bet_type || betType}
              overUnderLine={draftOverUnderLine}
              onOverUnderLineDraftChange={updateOverUnderLineDraft}
              onOverUnderLineCommit={commitOverUnderLine}
              cornersLine={draftCornersLine}
              onCornersLineDraftChange={updateCornersLineDraft}
              onCornersLineCommit={commitCornersLine}
              teamView={chartTeamView}
              isFiltersOpen={isFiltersOpen}
              onToggleFilters={() => setIsFiltersOpen(current => !current)}
              activeOverlayFilter={activeOverlayFilter}
            />
            <div className="metrics-panel-desktop-wrap">
              <MetricsPanel
                betType={workspace?.workspace?.bet_type || betType}
                metrics={workspace.metrics}
                sampleSize={workspace.sample_size}
                sampleSizes={workspace.sample_sizes}
              />
            </div>
          </div>
        </div>
        <aside className={`filters-drawer ${isFiltersOpen ? 'open' : 'closed'}`}>
          {isFiltersOpen ? (
            <>
              <div className="filters-drawer-header">
                <strong>Filters</strong>
                <button type="button" onClick={() => setIsFiltersOpen(false)}>✕</button>
              </div>
              <FilterDropdown
                value={draftFilters}
                onChange={setDraftFilters}
                onApply={applyFilters}
                onClear={clearFilters}
                hasPendingChanges={hasPendingFilterChanges}
                splitView={chartTeamView}
                onSplitViewChange={setChartTeamView}
                activeOverlayFilter={activeOverlayFilter}
                onOverlayFilterChange={setActiveOverlayFilter}
              />
            </>
          ) : null}
        </aside>
        <div className="metrics-panel-mobile-wrap">
          <MetricsPanel
            betType={workspace?.workspace?.bet_type || betType}
            metrics={workspace.metrics}
            sampleSize={workspace.sample_size}
            sampleSizes={workspace.sample_sizes}
          />
        </div>
      </div>
    </div>
  )
}
