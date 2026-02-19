import React, { useEffect, useState, useCallback } from 'react'

const VIEW_BOTH = 'both'
const VIEW_HOME = 'home'
const VIEW_AWAY = 'away'

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

const FILTER_KEYS = Object.keys(RANGE_CONFIGS)

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
}) {
  const update = (k, v) => onChange({ ...value, [k]: v })

  // Track which filter pills are expanded (show slider)
  const [expandedFilters, setExpandedFilters] = useState(new Set())

  // Draft states for each range filter
  const [drafts, setDrafts] = useState(() => {
    const d = {}
    for (const key of FILTER_KEYS) {
      d[key] = normalizeRange(value[key], RANGE_CONFIGS[key])
    }
    return d
  })

  // Sync drafts when value changes externally
  useEffect(() => {
    setDrafts(prev => {
      const next = { ...prev }
      for (const key of FILTER_KEYS) {
        const norm = normalizeRange(value[key], RANGE_CONFIGS[key])
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
    // Also set this as the active overlay filter
    if (typeof onOverlayFilterChange === 'function') {
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
    const config = RANGE_CONFIGS[key]
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

  return (
    <div className="filter-dropdown">
      {/* Category tabs */}
      <div className="filter-category-tabs">
        <button type="button" className="filter-category-tab active">Suggested</button>
        <button type="button" className="filter-category-tab">Splits</button>
        <button type="button" className="filter-category-tab">Stats</button>
      </div>

      {/* Split view toggle */}
      {typeof onSplitViewChange === 'function' ? (
        <div className="filter-control">
          <label>Chart View</label>
          <div className="split-toggle">
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
              Home
            </button>
            <button
              type="button"
              className={`split-toggle-btn ${splitView === VIEW_AWAY ? 'active' : ''}`}
              onClick={() => onSplitViewChange(VIEW_AWAY)}
            >
              Away
            </button>
          </div>
        </div>
      ) : null}

      {/* Venue filter */}
      <div className="filter-control">
        <label>Venue</label>
        <select value={value.home_away || 'all'} onChange={e => update('home_away', e.target.value)}>
          <option value="all">All</option>
          <option value="home">Home</option>
          <option value="away">Away</option>
        </select>
      </div>

      {/* Filter pill buttons */}
      <div className="filter-pills-grid">
        {FILTER_KEYS.map(key => {
          const config = RANGE_CONFIGS[key]
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
      </div>

      {/* Expandable sliders for active pills */}
      {FILTER_KEYS.map(key => {
        const config = RANGE_CONFIGS[key]
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

      {/* Shot xG controls */}
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
