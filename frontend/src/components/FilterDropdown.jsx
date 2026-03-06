import React, { useEffect, useState, useCallback } from 'react'

const VIEW_BOTH = 'both'
const VIEW_HOME = 'home'
const VIEW_AWAY = 'away'
const CATEGORY_SUGGESTED = 'suggested'
const CATEGORY_SPLIT = 'split'
const CATEGORY_STATS = 'stats'
const CATEGORY_OPPONENT_RANKINGS = 'opponent_rankings'

const RANGE_CONFIGS = {
  team_momentum_range: { label: 'Momentum', min: 0, max: 10, step: 0.1, precision: 1 },
  opponent_momentum_range: { label: 'Opp Momentum', min: 0, max: 10, step: 0.1, precision: 1 },
  total_match_goals_range: { label: 'Total Goals', min: 0, max: 10, step: 1, precision: 0 },
  team_goals_range: { label: 'Goals', min: 0, max: 10, step: 1, precision: 0 },
  opposition_goals_range: { label: 'Opp Goals', min: 0, max: 10, step: 1, precision: 0 },
  team_xg_range: { label: 'xG', min: 0, max: 5, step: 0.1, precision: 1 },
  opposition_xg_range: { label: 'Opp xG', min: 0, max: 5, step: 0.1, precision: 1 },
  team_possession_range: { label: 'Possession', min: 0, max: 100, step: 1, precision: 0 },
  opposition_possession_range: { label: 'Opp Poss', min: 0, max: 100, step: 1, precision: 0 },
  field_tilt_range: { label: 'Field Tilt', min: 0, max: 1, step: 0.01, precision: 2 },
}

const OPPONENT_RANKING_RANGE_CONFIGS = {
  opponent_rank_xgd_range: { label: 'Opp Rank xGD', min: 1, max: 20, step: 1, precision: 0 },
  opponent_rank_xgf_range: { label: 'Opp Rank xGF', min: 1, max: 20, step: 1, precision: 0 },
  opponent_rank_xga_range: { label: 'Opp Rank xGA', min: 1, max: 20, step: 1, precision: 0 },
  opponent_rank_position_range: { label: 'Opp Rank Position', min: 1, max: 20, step: 1, precision: 0 },
  opponent_rank_corners_range: { label: 'Opp Rank Corners', min: 1, max: 20, step: 1, precision: 0 },
  opponent_rank_momentum_range: { label: 'Opp Rank Momentum', min: 1, max: 20, step: 1, precision: 0 },
  opponent_rank_possession_range: { label: 'Opp Rank Poss', min: 1, max: 20, step: 1, precision: 0 },
}

const FILTER_KEYS = [...Object.keys(RANGE_CONFIGS), ...Object.keys(OPPONENT_RANKING_RANGE_CONFIGS)]
const STATS_FILTER_KEYS = [
  'team_momentum_range',
  'opponent_momentum_range',
  'total_match_goals_range',
  'opposition_goals_range',
  'team_xg_range',
  'opposition_xg_range',
  'team_possession_range',
  'opposition_possession_range',
  'field_tilt_range',
]
const OPPONENT_RANKING_FILTER_KEYS = Object.keys(OPPONENT_RANKING_RANGE_CONFIGS)

function getRangeConfig(key) {
  return RANGE_CONFIGS[key] || OPPONENT_RANKING_RANGE_CONFIGS[key] || null
}

function defaultRangeFor(config) {
  return [config.min, config.max]
}

function normalizeRange(range, config) {
  const fallback = defaultRangeFor(config)
  if (!Array.isArray(range) || range.length !== 2) return fallback
  const min = Number(range[0])
  const max = Number(range[1])
  if (Number.isNaN(min) || Number.isNaN(max)) return fallback
  const clampedMin = Math.max(config.min, Math.min(config.max, min))
  const clampedMax = Math.max(config.min, Math.min(config.max, max))
  return [Math.min(clampedMin, clampedMax), Math.max(clampedMin, clampedMax)]
}

function toPercent(value, config) {
  return ((value - config.min) / (config.max - config.min)) * 100
}

function isRangeModified(range, config) {
  const def = defaultRangeFor(config)
  const norm = normalizeRange(range, config)
  return norm[0] !== def[0] || norm[1] !== def[1]
}

function clampGamesCount(rawValue) {
  const numeric = Number(rawValue)
  if (!Number.isFinite(numeric)) return 21
  return Math.max(1, Math.min(60, Math.floor(numeric)))
}

function isSimilarTeamsEnabled(filters = {}) {
  const mode = String(filters.similar_teams_mode || '').trim().toLowerCase()
  if (mode === 'pca_cluster') return true
  return filters.similar_teams === true
}

function DualRangeSlider({ range, config, onDraftChange, onCommit }) {
  const [minValue, maxValue] = range
  const left = toPercent(minValue, config)
  const right = toPercent(maxValue, config)

  const handleLeft = (rawValue) => {
    const nextLeft = Number(rawValue)
    if (nextLeft > maxValue) {
      onDraftChange([maxValue, maxValue])
      return
    }
    onDraftChange([Number(nextLeft.toFixed(config.precision)), maxValue])
  }

  const handleRight = (rawValue) => {
    const nextRight = Number(rawValue)
    if (nextRight < minValue) {
      onDraftChange([minValue, minValue])
      return
    }
    onDraftChange([minValue, Number(nextRight.toFixed(config.precision))])
  }

  const commit = () => onCommit(range)

  return (
    <div className="dual-range">
      <div className="dual-range-track" />
      <div
        className="dual-range-selected"
        style={{ left: `${left}%`, width: `${Math.max(right - left, 0)}%` }}
      />
      <input
        type="range"
        min={config.min}
        max={config.max}
        step={config.step}
        value={minValue}
        onChange={e => handleLeft(e.target.value)}
        onPointerUp={commit}
        onBlur={commit}
        className="dual-range-input"
      />
      <input
        type="range"
        min={config.min}
        max={config.max}
        step={config.step}
        value={maxValue}
        onChange={e => handleRight(e.target.value)}
        onPointerUp={commit}
        onBlur={commit}
        className="dual-range-input"
      />
    </div>
  )
}

export default function FilterDropdown({
  value,
  onChange,
  onApply,
  onClear,
  hasPendingChanges,
  splitView = VIEW_BOTH,
  onSplitViewChange,
  activeOverlayFilter,
  onOverlayFilterChange,
  homeTeamName,
  awayTeamName,
}) {
  const update = (k, v) => onChange({ ...value, [k]: v })

  // Track which filter pills are expanded (show slider)
  const [expandedFilters, setExpandedFilters] = useState(new Set())
  const [activeCategory, setActiveCategory] = useState(CATEGORY_SUGGESTED)

  // Draft states for each range filter
  const [drafts, setDrafts] = useState(() => {
    const d = {}
    for (const key of FILTER_KEYS) {
      const config = getRangeConfig(key)
      if (!config) continue
      d[key] = normalizeRange(value[key], config)
    }
    return d
  })

  // Sync drafts when value changes externally
  useEffect(() => {
    setDrafts(prev => {
      const next = { ...prev }
      for (const key of FILTER_KEYS) {
        const config = getRangeConfig(key)
        if (!config) continue
        const norm = normalizeRange(value[key], config)
        if (prev[key]?.[0] !== norm[0] || prev[key]?.[1] !== norm[1]) {
          next[key] = norm
        }
      }
      return next
    })
  }, [value])

  const toggleFilter = useCallback((key) => {
    setExpandedFilters(prev => {
      const next = new Set(prev)
      if (next.has(key)) {
        next.delete(key)
      } else {
        next.add(key)
      }
      return next
    })
    // Stats and opponent ranking filters are valid overlay candidates.
    const isOverlayCandidate = (
      Object.prototype.hasOwnProperty.call(RANGE_CONFIGS, key)
      || Object.prototype.hasOwnProperty.call(OPPONENT_RANKING_RANGE_CONFIGS, key)
    )
    if (typeof onOverlayFilterChange === 'function' && isOverlayCandidate) {
      if (activeOverlayFilter === key) {
        onOverlayFilterChange(null)
      } else {
        onOverlayFilterChange(key)
      }
    }
  }, [activeOverlayFilter, onOverlayFilterChange])

  const updateDraft = (key, nextRange) => {
    setDrafts(prev => ({ ...prev, [key]: nextRange }))
  }

  const commitRange = (key, committedRange) => {
    const config = getRangeConfig(key)
    if (!config) return
    const normalized = normalizeRange(committedRange, config)
    const current = normalizeRange(value[key], config)
    if (normalized[0] === current[0] && normalized[1] === current[1]) return
    update(key, normalized)
  }

  const shotXgThreshold = Number.isFinite(Number(value.shot_xg_threshold))
    ? Number(value.shot_xg_threshold)
    : 0.3
  const shotXgMinShots = Number.isFinite(Number(value.shot_xg_min_shots))
    ? Math.max(0, Math.floor(Number(value.shot_xg_min_shots)))
    : 0
  const gamesMode = String(value.games_mode || 'max').trim().toLowerCase()
  const gamesCount = clampGamesCount(value.games_count)
  const similarTeamsEnabled = isSimilarTeamsEnabled(value)

  return (
    <div className="filter-dropdown">
      <div className="filter-top-row">
        <div className="filter-top-label">Season</div>
        <div className="filter-chip-row">
          <button
            type="button"
            className="filter-chip-btn active"
            onClick={() => onChange({ ...value, season: '25/26' })}
          >
            25/26
          </button>
        </div>
      </div>

      <div className="filter-top-row">
        <div className="filter-top-label">Games</div>
        <div className="filter-chip-row">
          <button
            type="button"
            className={`filter-chip-btn ${gamesMode === '10' ? 'active' : ''}`}
            onClick={() => onChange({ ...value, games_mode: '10', games_count: 10 })}
          >
            10
          </button>
          <button
            type="button"
            className={`filter-chip-btn ${gamesMode === '20' ? 'active' : ''}`}
            onClick={() => onChange({ ...value, games_mode: '20', games_count: 20 })}
          >
            20
          </button>
          <button
            type="button"
            className={`filter-chip-btn ${gamesMode === 'max' ? 'active' : ''}`}
            onClick={() => onChange({ ...value, games_mode: 'max' })}
          >
            Max
          </button>
          <div className={`filter-games-custom ${gamesMode === 'custom' ? 'active' : ''}`}>
            <button
              type="button"
              className="filter-games-step"
              aria-label="Decrease games"
              onClick={() => onChange({
                ...value,
                games_mode: 'custom',
                games_count: clampGamesCount(gamesCount - 1),
              })}
            >
              –
            </button>
            <span className="filter-games-value">{gamesCount}</span>
            <button
              type="button"
              className="filter-games-step"
              aria-label="Increase games"
              onClick={() => onChange({
                ...value,
                games_mode: 'custom',
                games_count: clampGamesCount(gamesCount + 1),
              })}
            >
              +
            </button>
            <span className="filter-games-lock" aria-hidden="true">🔒</span>
          </div>
        </div>
      </div>

      {/* Chart View — global, sits between Games and category tabs */}
      {typeof onSplitViewChange === 'function' ? (
        <div className="filter-top-row" style={{ marginTop: 2 }}>
          <div className="filter-top-label" style={{ flex: '0 0 auto', whiteSpace: 'nowrap' }}>Chart View</div>
          <div className="split-toggle" style={{ display: 'flex', gap: 4 }}>
            <button
              type="button"
              className={`split-toggle-btn ${splitView === VIEW_BOTH ? 'active' : ''}`}
              onClick={() => onSplitViewChange(VIEW_BOTH)}
            >
              Both
            </button>
            <button
              type="button"
              className={`split-toggle-btn ${splitView === VIEW_HOME ? 'active' : ''}`}
              onClick={() => onSplitViewChange(VIEW_HOME)}
            >
              {homeTeamName || 'Home'}
            </button>
            <button
              type="button"
              className={`split-toggle-btn ${splitView === VIEW_AWAY ? 'active' : ''}`}
              onClick={() => onSplitViewChange(VIEW_AWAY)}
            >
              {awayTeamName || 'Away'}
            </button>
          </div>
        </div>
      ) : null}

      {/* Category tabs */}
      <div className="filter-category-tabs" style={{ marginTop: 6 }}>
        <button
          type="button"
          className={`filter-category-tab ${activeCategory === CATEGORY_SUGGESTED ? 'active' : ''}`}
          onClick={() => setActiveCategory(CATEGORY_SUGGESTED)}
        >
          Suggested
        </button>
        <button
          type="button"
          className={`filter-category-tab ${activeCategory === CATEGORY_SPLIT ? 'active' : ''}`}
          onClick={() => setActiveCategory(CATEGORY_SPLIT)}
        >
          Split
        </button>
        <button
          type="button"
          className={`filter-category-tab ${activeCategory === CATEGORY_STATS ? 'active' : ''}`}
          onClick={() => setActiveCategory(CATEGORY_STATS)}
        >
          Stats
        </button>
        <button
          type="button"
          className={`filter-category-tab ${activeCategory === CATEGORY_OPPONENT_RANKINGS ? 'active' : ''}`}
          onClick={() => setActiveCategory(CATEGORY_OPPONENT_RANKINGS)}
        >
          Opp Rank
        </button>
      </div>

      {/* ── Suggested tab: curated pills only ── */}
      {activeCategory === CATEGORY_SUGGESTED ? (
        <>
          <div className="filter-pills-grid">
            {/* H2H */}
            <button
              type="button"
              className={`filter-pill-btn ${value.h2h ? 'active' : ''}`}
              onClick={() => onChange({ ...value, h2h: !value.h2h })}
            >
              H2H
            </button>
            {/* Home venue */}
            <button
              type="button"
              className={`filter-pill-btn ${(value.home_away || 'all') === 'home' ? 'active' : ''}`}
              onClick={() => update('home_away', (value.home_away || 'all') === 'home' ? 'all' : 'home')}
            >
              Home
            </button>
            {/* Away venue */}
            <button
              type="button"
              className={`filter-pill-btn ${(value.home_away || 'all') === 'away' ? 'active' : ''}`}
              onClick={() => update('home_away', (value.home_away || 'all') === 'away' ? 'all' : 'away')}
            >
              Away
            </button>
            {/* vs Similar Teams */}
            <button
              type="button"
              className={`filter-pill-btn ${similarTeamsEnabled ? 'active' : ''}`}
              onClick={() => onChange({
                ...value,
                similar_teams_mode: similarTeamsEnabled ? 'off' : 'pca_cluster',
                similar_teams: !similarTeamsEnabled,
              })}
            >
              vs Similar Teams
            </button>
            {/* Opp xGD rank */}
            {(() => {
              const key = 'opponent_rank_xgd_range'
              const config = getRangeConfig(key)
              if (!config) return null
              const isExpanded = expandedFilters.has(key)
              const isModified = isRangeModified(value[key], config)
              const isOverlay = activeOverlayFilter === key
              let cls = 'filter-pill-btn'
              if (isExpanded || isOverlay) cls += ' active'
              if (isModified) cls += ' modified'
              return (
                <button key={key} type="button" className={cls} onClick={() => toggleFilter(key)}>
                  {config.label}
                </button>
              )
            })()}
          </div>
          {/* Expandable slider for Opp xGD rank */}
          {(() => {
            const key = 'opponent_rank_xgd_range'
            const config = getRangeConfig(key)
            if (!config) return null
            const isExpanded = expandedFilters.has(key)
            const draft = drafts[key] || defaultRangeFor(config)
            return (
              <div key={key} className={`filter-slider-section ${isExpanded ? 'expanded' : ''}`}>
                <div className="filter-slider-label">{config.label}</div>
                <DualRangeSlider
                  range={draft}
                  config={config}
                  onDraftChange={(next) => updateDraft(key, next)}
                  onCommit={(next) => commitRange(key, next)}
                />
                <div className="filter-slider-values">
                  <span>{draft[0].toFixed(config.precision)}</span>
                  <span>{draft[1].toFixed(config.precision)}</span>
                </div>
              </div>
            )
          })()}
        </>
      ) : null}

      {/* ── Split tab ── */}
      {activeCategory === CATEGORY_SPLIT ? (
        <>
          <div className="filter-control">
            <label>Venue</label>
            <div className="filter-pills-grid" style={{ marginTop: 2 }}>
              <button
                type="button"
                className={`filter-pill-btn ${(value.home_away || 'all') === 'home' ? 'active' : ''}`}
                onClick={() => update('home_away', (value.home_away || 'all') === 'home' ? 'all' : 'home')}
              >
                Home
              </button>
              <button
                type="button"
                className={`filter-pill-btn ${(value.home_away || 'all') === 'away' ? 'active' : ''}`}
                onClick={() => update('home_away', (value.home_away || 'all') === 'away' ? 'all' : 'away')}
              >
                Away
              </button>
            </div>
          </div>
          <div className="filter-pills-grid" style={{ marginTop: 4 }}>
            <button
              type="button"
              className={`filter-pill-btn ${similarTeamsEnabled ? 'active' : ''}`}
              onClick={() => onChange({
                ...value,
                similar_teams_mode: similarTeamsEnabled ? 'off' : 'pca_cluster',
                similar_teams: !similarTeamsEnabled,
              })}
            >
              vs Similar Teams
            </button>
            <button
              type="button"
              className={`filter-pill-btn ${value.h2h ? 'active' : ''}`}
              onClick={() => onChange({ ...value, h2h: !value.h2h })}
            >
              H2H
            </button>
          </div>
        </>
      ) : null}

      {/* ── Stats / Opp Rank tabs: full range pill grid ── */}
      {activeCategory === CATEGORY_STATS || activeCategory === CATEGORY_OPPONENT_RANKINGS ? (
        <>
          <div className="filter-pills-grid">
            {(activeCategory === CATEGORY_STATS
              ? STATS_FILTER_KEYS
              : OPPONENT_RANKING_FILTER_KEYS).map(key => {
                const config = getRangeConfig(key)
                if (!config) return null
                const isExpanded = expandedFilters.has(key)
                const isModified = isRangeModified(value[key], config)
                const isOverlay = activeOverlayFilter === key
                let className = 'filter-pill-btn'
                if (isExpanded || isOverlay) className += ' active'
                if (isModified) className += ' modified'
                return (
                  <button
                    key={key}
                    type="button"
                    className={className}
                    onClick={() => toggleFilter(key)}
                  >
                    {config.label}
                  </button>
                )
              })}
            {/* NPG toggle (Stats only) */}
            {activeCategory === CATEGORY_STATS ? (
              <button
                type="button"
                className={`filter-pill-btn ${value.npg_toggle ? 'active' : ''}`}
                onClick={() => onChange({ ...value, npg_toggle: !value.npg_toggle })}
              >
                NPG
              </button>
            ) : null}
          </div>

          {/* Expandable sliders */}
          {(activeCategory === CATEGORY_STATS
            ? STATS_FILTER_KEYS
            : OPPONENT_RANKING_FILTER_KEYS).map(key => {
              const config = getRangeConfig(key)
              if (!config) return null
              const isExpanded = expandedFilters.has(key)
              const draft = drafts[key] || defaultRangeFor(config)
              return (
                <div key={key} className={`filter-slider-section ${isExpanded ? 'expanded' : ''}`}>
                  <div className="filter-slider-label">{config.label}</div>
                  <DualRangeSlider
                    range={draft}
                    config={config}
                    onDraftChange={(next) => updateDraft(key, next)}
                    onCommit={(next) => commitRange(key, next)}
                  />
                  <div className="filter-slider-values">
                    <span>{draft[0].toFixed(config.precision)}</span>
                    <span>{draft[1].toFixed(config.precision)}</span>
                  </div>
                </div>
              )
            })}

          {/* Shot xG controls (Stats only) */}
          {activeCategory === CATEGORY_STATS ? (
            <>
              <div className="filter-control" style={{ marginTop: 4 }}>
                <label>Shot xG Threshold</label>
                <input
                  type="number"
                  min={0}
                  max={2}
                  step={0.01}
                  value={shotXgThreshold}
                  onChange={(event) => {
                    const next = Number(event.target.value)
                    update('shot_xg_threshold', Number.isFinite(next) ? next : 0)
                  }}
                />
              </div>

              <div className="filter-control">
                <label>Min Shots At/Above Threshold</label>
                <input
                  type="number"
                  min={0}
                  max={30}
                  step={1}
                  value={shotXgMinShots}
                  onChange={(event) => {
                    const next = Number(event.target.value)
                    const safe = Number.isFinite(next) ? next : 0
                    update('shot_xg_min_shots', Math.max(0, Math.floor(safe)))
                  }}
                />
              </div>
            </>
          ) : null}
        </>
      ) : null}

      {/* Actions */}
      <div className="filter-actions">
        <button type="button" className="clear-filters-btn" onClick={onClear}>
          Clear
        </button>
        <button
          type="button"
          className="apply-filters-btn"
          onClick={onApply}
          disabled={!hasPendingChanges}
        >
          Apply
        </button>
      </div>
    </div>
  )
}
