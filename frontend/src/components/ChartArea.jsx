import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react'
import {
  ComposedChart, Bar, XAxis, YAxis, Tooltip, Line,
  ResponsiveContainer, ReferenceLine, Cell
} from 'recharts'
import {
  getTeamLogo,
  formatShortMatchDate,
  formatPropsMadnessMatchDateParts,
} from '../utils/premierLeagueLogos'

const BET_TYPE_OVER_UNDER = 'over_under'
const BET_TYPE_HOME_OU = 'home_ou'
const BET_TYPE_AWAY_OU = 'away_ou'
const BET_TYPE_DOUBLE_CHANCE = 'double_chance'
const BET_TYPE_BTTS = 'btts'
const BET_TYPE_CORNERS = 'corners'
const VIEW_BOTH = 'both'
const VIEW_HOME = 'home'
const VIEW_AWAY = 'away'

const OVER_UNDER_LINE_MIN = 0.5
const OVER_UNDER_LINE_MAX = 8.5
const OVER_UNDER_LINE_STEP = 0.5
const CORNERS_LINE_MIN = 0.5
const CORNERS_LINE_MAX = 20.5
const CORNERS_LINE_STEP = 0.5
const Y_AXIS_STEP = 0.5
const Y_AXIS_BASE_TICK_COUNT = 4
const CHART_HEIGHT = 330
const CHART_HEIGHT_MOBILE = 260
const CHART_MARGIN_DESKTOP = { top: 12, right: 10, left: 4, bottom: 44 }
const CHART_MARGIN_MOBILE = { top: 12, right: 8, left: 2, bottom: 44 }
const X_AXIS_HEIGHT = 74
const X_AXIS_PADDING_DESKTOP = 14
const X_AXIS_PADDING_MOBILE = 8
const Y_AXIS_WIDTH_DESKTOP = 34
const Y_AXIS_WIDTH_MOBILE = 26
const RIGHT_Y_AXIS_WIDTH_DESKTOP = 34
const RIGHT_Y_AXIS_WIDTH_MOBILE = 26
const MOBILE_CHART_BREAKPOINT = 700

// Map filter keys to the match data fields we can overlay
const OVERLAY_FIELD_MAP = {
  team_momentum_range: { field: 'team_momentum_range', label: 'Momentum', color: '#2d6bff' },
  opponent_momentum_range: { field: 'opponent_momentum_range', label: 'Opp Momentum', color: '#2d6bff' },
  total_match_goals_range: { field: 'total_match_goals_range', label: 'Total Goals', color: '#2d6bff' },
  team_goals_range: { field: 'team_goals_range', label: 'Goals', color: '#2d6bff' },
  opposition_goals_range: { field: 'opposition_goals_range', label: 'Opp Goals', color: '#2d6bff' },
  team_xg_range: { field: 'team_xg', label: 'xG', color: '#2d6bff' },
  opposition_xg_range: { field: 'opponent_xg', label: 'Opp xG', color: '#2d6bff' },
  team_possession_range: { field: 'team_possession_range', label: 'Possession', color: '#2d6bff' },
  opposition_possession_range: { field: 'opposition_possession_range', label: 'Opp Poss', color: '#2d6bff' },
  field_tilt_range: { field: 'field_tilt_range', label: 'Field Tilt', color: '#2d6bff' },
  opponent_rank_xgd_range: { field: 'opponent_rank_xgd_range', label: 'Opp Rank xGD', color: '#2d6bff' },
  opponent_rank_xgf_range: { field: 'opponent_rank_xgf_range', label: 'Opp Rank xGF', color: '#2d6bff' },
  opponent_rank_xga_range: { field: 'opponent_rank_xga_range', label: 'Opp Rank xGA', color: '#2d6bff' },
  opponent_rank_position_range: { field: 'opponent_rank_position_range', label: 'Opp Rank Position', color: '#2d6bff' },
  opponent_rank_corners_range: { field: 'opponent_rank_corners_range', label: 'Opp Rank Corners', color: '#2d6bff' },
  opponent_rank_momentum_range: { field: 'opponent_rank_momentum_range', label: 'Opp Rank Momentum', color: '#2d6bff' },
  opponent_rank_possession_range: { field: 'opponent_rank_possession_range', label: 'Opp Rank Poss', color: '#2d6bff' },
}

function normalizeLineValue(rawValue) {
  const numeric = Number(rawValue)
  if (Number.isNaN(numeric)) return 2.5
  const clamped = Math.min(OVER_UNDER_LINE_MAX, Math.max(OVER_UNDER_LINE_MIN, numeric))
  return Math.round(clamped / OVER_UNDER_LINE_STEP) * OVER_UNDER_LINE_STEP
}

function normalizeCornersLineValue(rawValue) {
  const numeric = Number(rawValue)
  if (Number.isNaN(numeric)) return 8.5
  const clamped = Math.min(CORNERS_LINE_MAX, Math.max(CORNERS_LINE_MIN, numeric))
  return Math.round(clamped / CORNERS_LINE_STEP) * CORNERS_LINE_STEP
}

function mapOneXTwoValue(result) {
  if (result === 'W') return { value: 1.0, color: '#2ecc71' }
  if (result === 'D') return { value: 0.5, color: '#f39c12' }
  return { value: 0.1, color: '#e74c3c' }
}

function isOverUnderFamilyBetType(betType) {
  const normalized = String(betType).toLowerCase()
  return normalized === BET_TYPE_OVER_UNDER || normalized === BET_TYPE_HOME_OU || normalized === BET_TYPE_AWAY_OU
}

function isCornersBetType(betType) {
  return String(betType).toLowerCase() === BET_TYPE_CORNERS
}

function extractMatchDate(rawMatch = {}) {
  return (
    rawMatch.match_date
    || rawMatch.match_datetime
    || rawMatch.kickoff
    || rawMatch.kickoff_time
    || rawMatch.date
    || null
  )
}

function buildBaseMatchFields(match, fallbackOpponent) {
  const opponentName = match.opponent_name || fallbackOpponent
  const dateValue = extractMatchDate(match)
  const dateParts = formatPropsMadnessMatchDateParts(dateValue)
  const opponentId = Number(match.opponent_id)
  const opponentLogo = getTeamLogo(
    opponentName,
    Number.isFinite(opponentId) ? opponentId : null,
    match.league_id
  )

  return {
    fixture_display: match.fixture_display || `${match.team_name || ''} vs ${opponentName}`,
    team_name: match.team_name || '',
    opponent_name: opponentName,
    opponent_logo: opponentLogo,
    venue: match.venue || '',
    match_date: dateValue,
    date_month: dateParts.month,
    date_day: dateParts.day,
    date_label: formatShortMatchDate(dateValue),
    // Carry through raw match data for overlay access
    _raw: match,
  }
}

function buildOneXTwoBars(matches = []) {
  return matches.map((m, idx) => {
    const mapped = mapOneXTwoValue(m.result)
    const fallbackOpponent = `Opponent ${m.opponent_id ?? idx + 1}`
    const label = m.chart_label || (m.venue === 'away' ? `@ ${m.opponent_name || fallbackOpponent}` : `vs ${m.opponent_name || fallbackOpponent}`)
    return {
      label,
      value: mapped.value,
      color: mapped.color,
      result: m.result,
      match_id: m.match_id,
      ...buildBaseMatchFields(m, fallbackOpponent),
    }
  })
}

function buildDoubleChanceBars(matches = []) {
  return matches.map((m, idx) => {
    const fallbackOpponent = `Opponent ${m.opponent_id ?? idx + 1}`
    const label = m.chart_label || (m.venue === 'away' ? `@ ${m.opponent_name || fallbackOpponent}` : `vs ${m.opponent_name || fallbackOpponent}`)
    const hitFlag = m.double_chance_result || (m.result === 'W' || m.result === 'D' ? 'H' : 'M')
    return {
      label,
      value: hitFlag === 'H' ? 1.0 : 0.1,
      color: hitFlag === 'H' ? '#2ecc71' : '#e74c3c',
      result: m.result || '',
      double_chance_result: hitFlag,
      match_id: m.match_id,
      ...buildBaseMatchFields(m, fallbackOpponent),
    }
  })
}

function buildBttsBars(matches = []) {
  return matches.map((m, idx) => {
    const fallbackOpponent = `Opponent ${m.opponent_id ?? idx + 1}`
    const label = m.chart_label || (m.venue === 'away' ? `@ ${m.opponent_name || fallbackOpponent}` : `vs ${m.opponent_name || fallbackOpponent}`)
    const isBtts = m.btts_result
      ? m.btts_result === 'Y'
      : (Number(m.goals_scored ?? 0) > 0 && Number(m.opponent_goals ?? 0) > 0)
    return {
      label,
      value: isBtts ? 1.0 : 0.1,
      color: isBtts ? '#2ecc71' : '#e74c3c',
      btts_result: isBtts ? 'Y' : 'N',
      goals_scored: Number(m.goals_scored ?? 0),
      opponent_goals: Number(m.opponent_goals ?? 0),
      match_id: m.match_id,
      ...buildBaseMatchFields(m, fallbackOpponent),
    }
  })
}

function buildOverUnderBars(matches = [], line = 2.5, betType = BET_TYPE_OVER_UNDER) {
  const isTeamGoalsMode = String(betType).toLowerCase() === BET_TYPE_HOME_OU || String(betType).toLowerCase() === BET_TYPE_AWAY_OU
  return matches.map((m, idx) => {
    const rawGoals = Number(isTeamGoalsMode ? (m.team_goals ?? 0) : (m.total_goals ?? 0))
    const safeGoals = Number.isFinite(rawGoals) ? rawGoals : 0
    const plottedGoals = safeGoals === 0 ? 0.1 : safeGoals
    const fallbackOpponent = `Opponent ${m.opponent_id ?? idx + 1}`
    const label = m.chart_label || (m.venue === 'away' ? `@ ${m.opponent_name || fallbackOpponent}` : `vs ${m.opponent_name || fallbackOpponent}`)
    const result = safeGoals > line ? 'O' : 'U'
    return {
      label,
      value: plottedGoals,
      total_goals: isTeamGoalsMode ? undefined : safeGoals,
      team_goals: isTeamGoalsMode ? safeGoals : undefined,
      goals_metric_name: isTeamGoalsMode ? 'Team Goals' : 'Total Goals',
      color: result === 'O' ? '#2ecc71' : '#e74c3c',
      over_under_result: result,
      match_id: m.match_id,
      ...buildBaseMatchFields(m, fallbackOpponent),
    }
  })
}

function getTotalCornersValue(row = {}) {
  const directTotal = Number(row.total_corners ?? row._raw?.total_corners)
  if (Number.isFinite(directTotal)) return directTotal

  const homeCorners = Number(
    row.home_corners
      ?? row._raw?.home_corners
      ?? row.corners_home
      ?? row._raw?.corners_home
      ?? row.corner_kicks_home
      ?? row._raw?.corner_kicks_home
      ?? 0
  )
  const awayCorners = Number(
    row.away_corners
      ?? row._raw?.away_corners
      ?? row.corners_away
      ?? row._raw?.corners_away
      ?? row.corner_kicks_away
      ?? row._raw?.corner_kicks_away
      ?? 0
  )
  const safeHome = Number.isFinite(homeCorners) ? homeCorners : 0
  const safeAway = Number.isFinite(awayCorners) ? awayCorners : 0
  return safeHome + safeAway
}

function buildCornerBars(matches = [], line = 8.5) {
  return matches.map((m, idx) => {
    const safeCorners = getTotalCornersValue(m)
    const fallbackOpponent = `Opponent ${m.opponent_id ?? idx + 1}`
    const label = m.chart_label || (m.venue === 'away' ? `@ ${m.opponent_name || fallbackOpponent}` : `vs ${m.opponent_name || fallbackOpponent}`)
    const result = safeCorners > line ? 'O' : 'U'
    return {
      label,
      value: safeCorners,
      total_corners: safeCorners,
      color: result === 'O' ? '#2ecc71' : '#e74c3c',
      corners_over_under_result: result,
      match_id: m.match_id,
      ...buildBaseMatchFields(m, fallbackOpponent),
    }
  })
}

function readSeriesMax(data = []) {
  return data.reduce((maxValue, row) => {
    const candidate = Number(row?.value ?? 0)
    return Number.isFinite(candidate) ? Math.max(maxValue, candidate) : maxValue
  }, 0)
}

function roundAxisValue(value) {
  const rounded = Math.round(Number(value) * 10) / 10
  if (!Number.isFinite(rounded)) return 0
  return Object.is(rounded, -0) ? 0 : rounded
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

function containsApprox(values, candidate, epsilon = 1e-6) {
  return values.some(value => Math.abs(value - candidate) < epsilon)
}

function toStepUnits(value, step = Y_AXIS_STEP) {
  return Math.round(Number(value) / step)
}

function fromStepUnits(units, step = Y_AXIS_STEP) {
  return roundAxisValue(units * step)
}

function buildYAxisScale({
  min = 0,
  max = 1,
  referenceValue = null,
  step = Y_AXIS_STEP,
  baseTickCount = Y_AXIS_BASE_TICK_COUNT,
}) {
  const safeMin = Number.isFinite(min) ? min : 0
  const safeMax = Number.isFinite(max) ? Math.max(max, safeMin) : safeMin
  const safeBaseTickCount = Math.max(2, Math.floor(baseTickCount))

  const minUnits = Math.ceil(safeMin / step)
  let maxLabelUnits = Math.floor(safeMax / step)
  const minUnitsForBaseTicks = minUnits + (safeBaseTickCount - 1)
  if (maxLabelUnits < minUnitsForBaseTicks) {
    maxLabelUnits = minUnitsForBaseTicks
  }

  const rangeUnits = maxLabelUnits - minUnits
  const baseUnits = [minUnits]
  for (let idx = 1; idx < safeBaseTickCount - 1; idx++) {
    const ratio = idx / (safeBaseTickCount - 1)
    baseUnits.push(minUnits + Math.round(rangeUnits * ratio))
  }
  baseUnits.push(maxLabelUnits)

  const enforcedUnits = []
  for (let idx = 0; idx < baseUnits.length; idx++) {
    const previous = enforcedUnits[idx - 1]
    const remainingSlots = baseUnits.length - idx - 1
    const maxAllowed = maxLabelUnits - remainingSlots
    let next = baseUnits[idx]

    if (idx === 0) {
      next = minUnits
    } else if (idx === baseUnits.length - 1) {
      next = maxLabelUnits
    } else {
      next = Math.max(previous + 1, Math.min(next, maxAllowed))
    }

    enforcedUnits.push(next)
  }

  const uniqueTicks = enforcedUnits.map(units => fromStepUnits(units, step))

  if (
    Number.isFinite(referenceValue)
    && referenceValue >= fromStepUnits(minUnits, step)
    && referenceValue <= safeMax + 1e-9
  ) {
    const referenceTick = fromStepUnits(toStepUnits(referenceValue, step), step)
    if (!containsApprox(uniqueTicks, referenceTick)) {
      uniqueTicks.push(referenceTick)
    }
  }

  const axisMax = Math.max(safeMax, fromStepUnits(maxLabelUnits, step))
  return {
    ticks: uniqueTicks.sort((a, b) => a - b),
    axisMax: roundAxisValue(axisMax),
  }
}

function formatYAxisTick(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return ''
  if (Math.abs(numeric - Math.round(numeric)) < 1e-6) {
    return String(Math.round(numeric))
  }
  return numeric.toFixed(1).replace(/\.0$/, '')
}

function computeBarCategoryGap(totalBars) {
  if (totalBars > 24) return '2%'
  if (totalBars > 14) return '6%'
  return '10%'
}

function computeBarSize(plotWidth, totalBars) {
  const barsCount = Math.max(totalBars, 1)
  const gapUnits = barsCount > 24 ? 2 : barsCount > 14 ? 4 : 6
  const rawBarSize = Math.floor(plotWidth / (barsCount + gapUnits))
  return Math.max(2, Math.min(40, rawBarSize))
}

function buildOverlayYAxisScale(data = []) {
  const values = data
    .map(row => Number(row?.overlayValue))
    .filter(value => Number.isFinite(value))
  if (!values.length) return null

  const rawMin = Math.min(...values)
  const rawMax = Math.max(...values)

  let axisMin = rawMin
  let axisMax = rawMax
  if (Math.abs(rawMax - rawMin) < 1e-6) {
    const pad = Math.max(1, Math.abs(rawMax) * 0.2)
    axisMin = rawMin - pad
    axisMax = rawMax + pad
  } else {
    const pad = (rawMax - rawMin) * 0.15
    axisMin = rawMin - pad
    axisMax = rawMax + pad
  }

  if (values.every(value => value >= 0)) {
    axisMin = Math.max(0, axisMin)
  }

  const safeMin = roundAxisValue(axisMin)
  const safeMax = roundAxisValue(Math.max(axisMax, safeMin + 0.1))
  const tickCount = 4
  const ticks = []
  for (let idx = 0; idx < tickCount; idx++) {
    const ratio = tickCount === 1 ? 0 : idx / (tickCount - 1)
    ticks.push(roundAxisValue(safeMin + ((safeMax - safeMin) * ratio)))
  }

  return {
    min: safeMin,
    max: safeMax,
    ticks: Array.from(new Set(ticks)).sort((a, b) => a - b),
  }
}

function buildVisibleTickIndexSet(totalBars, containerWidth) {
  const safeBars = Math.max(1, Number(totalBars) || 1)
  if (safeBars <= 1) return new Set([0])

  const safeWidth = Math.max(1, Number(containerWidth) || 1)
  const maxLabelsByWidth = clamp(Math.floor(safeWidth / 72), 4, 8)
  const targetLabels = Math.min(safeBars, maxLabelsByWidth)

  if (safeBars <= targetLabels) {
    return new Set(Array.from({ length: safeBars }, (_, idx) => idx))
  }

  const indexSet = new Set([0, safeBars - 1])
  const denominator = Math.max(targetLabels - 1, 1)
  for (let i = 0; i < targetLabels; i++) {
    const ratio = i / denominator
    const index = Math.round(ratio * (safeBars - 1))
    indexSet.add(clamp(index, 0, safeBars - 1))
  }
  return indexSet
}

function MatchAxisTick({ x, y, payload, data = [], containerWidth = 1000 }) {
  const row = data[payload?.index]
  if (!row) return null

  const totalBars = Math.max(1, data.length)
  const effectiveWidth = Math.max(1, containerWidth)
  const perTickWidth = effectiveWidth / totalBars
  const compactMode = perTickWidth < 14
  const superCompactMode = perTickWidth < 11

  const index = payload?.index ?? 0
  const isPhoneSizedChart = effectiveWidth <= MOBILE_CHART_BREAKPOINT
  const shouldShowTickIcon = !(isPhoneSizedChart && totalBars > 15)
  let shouldRenderDate = true

  if (isPhoneSizedChart) {
    const visibleTickIndexSet = buildVisibleTickIndexSet(totalBars, effectiveWidth)
    shouldRenderDate = visibleTickIndexSet.has(index)
  } else {
    let showDateEvery = 1
    if (perTickWidth < 10) showDateEvery = 4
    else if (perTickWidth < 12) showDateEvery = 3
    else if (perTickWidth < 16) showDateEvery = 2

    const isLastTick = index === totalBars - 1
    shouldRenderDate = (index % showDateEvery) === 0 || isLastTick
  }

  const logoSize = shouldShowTickIcon
    ? clamp(perTickWidth * 0.72, superCompactMode ? 8 : 10, 22)
    : 0
  const logoOffset = logoSize / 2

  const fontPx = superCompactMode ? 6 : compactMode ? 7 : 9
  const fontSize = `${fontPx}px`

  const textY1 = shouldShowTickIcon
    ? logoSize + (superCompactMode ? 7 : 9)
    : (superCompactMode ? 8 : 10)
  const textY2 = textY1 + (superCompactMode ? 7 : 8)
  const dayOnly = superCompactMode

  return (
    <g transform={`translate(${x},${y})`} className="chart-axis-tick">
      {shouldShowTickIcon ? (
        row.opponent_logo ? (
          <image href={row.opponent_logo} x={-logoOffset} y={0} width={logoSize} height={logoSize} />
        ) : (
          <circle cx={0} cy={logoOffset} r={logoOffset - 1} fill="#1a1a1f" stroke="#2a2a30" strokeWidth={1} />
        )
      ) : null}
      {shouldRenderDate ? (
        <>
          {!dayOnly ? (
            <text x={0} y={textY1} textAnchor="middle" style={{ fontSize, fill: '#9b9ca6' }} className="chart-axis-date-month">
              {row.date_month || '--'}
            </text>
          ) : null}
          <text
            x={0}
            y={dayOnly ? textY1 : textY2}
            textAnchor="middle"
            style={{ fontSize, fill: '#9b9ca6', fontWeight: 500 }}
            className="chart-axis-date-day"
          >
            {row.date_day || '--'}
          </text>
        </>
      ) : null}
    </g>
  )
}

function computeHitRate(rows, betType) {
  const normalizedBetType = String(betType).toLowerCase()
  const total = rows.length
  if (total === 0) return { percent: null, hits: 0, total: 0 }

  let hits = 0
  if (isOverUnderFamilyBetType(normalizedBetType)) {
    hits = rows.filter(row => row.over_under_result === 'O').length
  } else if (isCornersBetType(normalizedBetType)) {
    hits = rows.filter(row => row.corners_over_under_result === 'O').length
  } else if (normalizedBetType === BET_TYPE_BTTS) {
    hits = rows.filter(row => row.btts_result === 'Y').length
  } else if (normalizedBetType === BET_TYPE_DOUBLE_CHANCE) {
    hits = rows.filter(row => row.double_chance_result === 'H').length
  } else {
    hits = rows.filter(row => row.result === 'W').length
  }

  const percent = (hits / total) * 100
  return { percent, hits, total }
}

function getHitRateToneClass(percent) {
  if (!Number.isFinite(percent)) return ''
  if (percent > 50) return 'hit-rate-positive'
  if (percent < 50) return 'hit-rate-negative'
  return ''
}

function computeGraphAvg(rows, betType) {
  const normalizedBetType = String(betType).toLowerCase()
  let sum = 0
  let count = 0
  for (const row of rows) {
    const value = isCornersBetType(normalizedBetType)
      ? getTotalCornersValue(row)
      : Number(row?.value)
    if (Number.isFinite(value)) {
      sum += value
      count++
    }
  }
  return count > 0 ? sum / count : null
}

function computeSeasonAvg(rows, betType) {
  const normalizedBetType = String(betType).toLowerCase()
  const isTeamGoalsMode = normalizedBetType === BET_TYPE_HOME_OU || normalizedBetType === BET_TYPE_AWAY_OU

  let sum = 0
  let count = 0
  for (const row of rows) {
    let value = null

    if (isOverUnderFamilyBetType(normalizedBetType)) {
      value = Number(
        isTeamGoalsMode
          ? (row.team_goals ?? row._raw?.goals_scored)
          : (row.total_goals ?? row._raw?.total_goals)
      )
    } else if (isCornersBetType(normalizedBetType)) {
      value = getTotalCornersValue(row)
    } else {
      value = Number(row.value)
    }

    if (Number.isFinite(value)) {
      sum += value
      count++
    }
  }
  return count > 0 ? sum / count : null
}

function getAverageMetricLabel(betType) {
  const normalizedBetType = String(betType).toLowerCase()
  const isTeamGoalsMode = normalizedBetType === BET_TYPE_HOME_OU || normalizedBetType === BET_TYPE_AWAY_OU

  if (isOverUnderFamilyBetType(normalizedBetType)) {
    return isTeamGoalsMode ? 'Team Goals Avg' : 'Total Goals Avg'
  }
  if (isCornersBetType(normalizedBetType)) {
    return 'Corners Avg'
  }
  if (normalizedBetType === BET_TYPE_DOUBLE_CHANCE) {
    return 'Double Chance Avg'
  }
  if (normalizedBetType === BET_TYPE_BTTS) {
    return 'BTTS Avg'
  }
  return 'Outcome Avg'
}

function formatLineDisplayValue(value, isCorners = false) {
  const safe = isCorners ? normalizeCornersLineValue(value) : normalizeLineValue(value)
  return safe.toFixed(1).replace(/\.0$/, '')
}

function getLineSummaryLabel(betType, line) {
  const normalizedBetType = String(betType).toLowerCase()
  const isCorners = isCornersBetType(normalizedBetType)
  const lineText = formatLineDisplayValue(line, isCorners)

  if (normalizedBetType === BET_TYPE_OVER_UNDER) {
    return `Over/Under ${lineText} Total Goals`
  }
  if (normalizedBetType === BET_TYPE_HOME_OU) {
    return `Home Over/Under ${lineText} Team Goals`
  }
  if (normalizedBetType === BET_TYPE_AWAY_OU) {
    return `Away Over/Under ${lineText} Team Goals`
  }
  if (isCorners) {
    return `Over/Under ${lineText} Total Corners`
  }
  return null
}

// Enrich chart data with overlay field values
function enrichWithOverlay(data, overlayConfig) {
  if (!overlayConfig) return data
  return data.map(row => {
    const raw = row._raw || {}
    const directValue = Number(raw[overlayConfig.field])
    let fallbackValue = null
    if (!Number.isFinite(directValue) && (overlayConfig.field === 'team_xg' || overlayConfig.field === 'opponent_xg')) {
      const homeXg = Number(raw.expected_goals_home)
      const awayXg = Number(raw.expected_goals_away)
      const venue = String(row?.venue || raw?.venue || '').toLowerCase()
      if (Number.isFinite(homeXg) && Number.isFinite(awayXg)) {
        if (overlayConfig.field === 'team_xg') {
          fallbackValue = venue === 'away' ? awayXg : homeXg
        } else {
          fallbackValue = venue === 'away' ? homeXg : awayXg
        }
      }
    }
    return {
      ...row,
      overlayValue: Number.isFinite(directValue)
        ? directValue
        : (Number.isFinite(fallbackValue) ? fallbackValue : null),
    }
  })
}

// Compute overlay graph avg and season avg
function computeOverlayAvg(data) {
  let sum = 0
  let count = 0
  for (const row of data) {
    if (row.overlayValue != null) {
      sum += row.overlayValue
      count++
    }
  }
  return count > 0 ? (sum / count) : null
}

function TeamBarChart({
  title,
  data,
  betType,
  line,
  onReferenceLineClick,
  onReferenceLineContextMenu,
  onReferenceLineDragChange,
  onReferenceLineDragCommit,
  layoutVersion,
  overlayConfig,
  isFiltersOpen,
  onToggleFilters,
}) {
  const containerRef = useRef(null)
  const chartCanvasRef = useRef(null)
  const dragListenersRef = useRef({ move: null, up: null })
  const [containerWidth, setContainerWidth] = useState(0)

  useEffect(() => {
    const node = containerRef.current
    if (!node) return undefined

    const updateWidth = () => {
      const nextWidth = node.clientWidth || 0
      setContainerWidth(nextWidth)
    }

    updateWidth()

    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver((entries) => {
        const nextWidth = entries?.[0]?.contentRect?.width ?? node.clientWidth ?? 0
        setContainerWidth(nextWidth)
      })
      observer.observe(node)
      return () => observer.disconnect()
    }

    window.addEventListener('resize', updateWidth)
    return () => window.removeEventListener('resize', updateWidth)
  }, [])

  useEffect(() => {
    const node = containerRef.current
    if (!node) return
    setContainerWidth(node.clientWidth || 0)
  }, [layoutVersion, data.length])

  const normalizedBetType = String(betType).toLowerCase()
  const isOverUnder = isOverUnderFamilyBetType(normalizedBetType)
  const isCorners = isCornersBetType(normalizedBetType)
  const isLineType = isOverUnder || isCorners
  const isDoubleChance = normalizedBetType === BET_TYPE_DOUBLE_CHANCE
  const isBtts = normalizedBetType === BET_TYPE_BTTS
  const rawSeriesMax = readSeriesMax(data)
  const referenceLineValue = isLineType ? line : (!isDoubleChance && !isBtts ? 0.5 : null)
  const rawDomainMax = isLineType
    ? Math.max(rawSeriesMax, line, isCorners ? CORNERS_LINE_MIN : OVER_UNDER_LINE_MIN)
    : 1.5
  const { ticks: yAxisTicks, axisMax: yAxisMax } = buildYAxisScale({
    min: 0,
    max: rawDomainMax,
    referenceValue: referenceLineValue,
  })
  const hitRate = computeHitRate(data, normalizedBetType)
  const hitRateToneClass = getHitRateToneClass(hitRate.percent)
  const graphAvg = computeGraphAvg(data, normalizedBetType)
  const seasonAvg = computeSeasonAvg(data, normalizedBetType)
  const averageMetricLabel = getAverageMetricLabel(normalizedBetType)
  const lineSummaryLabel = getLineSummaryLabel(normalizedBetType, line)
  const hasLineSummaryLabel = Boolean(lineSummaryLabel)
  const hasOverlay = Boolean(overlayConfig && data.some(d => d.overlayValue != null))
  const overlayAvg = hasOverlay ? computeOverlayAvg(data) : null
  const overlayScale = hasOverlay ? buildOverlayYAxisScale(data) : null
  const firstRow = data[0] || null
  const teamIdFromRaw = Number(
    firstRow?.venue === 'home'
      ? firstRow?._raw?.home_team_id
      : firstRow?._raw?.away_team_id
  )
  const teamLogo = firstRow
    ? getTeamLogo(firstRow.team_name, Number.isFinite(teamIdFromRaw) ? teamIdFromRaw : null, firstRow?._raw?.league_id)
    : null

  const isMobileChart = containerWidth > 0 && containerWidth <= MOBILE_CHART_BREAKPOINT
  const chartHeight = isMobileChart ? CHART_HEIGHT_MOBILE : CHART_HEIGHT
  const chartMargin = isMobileChart ? CHART_MARGIN_MOBILE : CHART_MARGIN_DESKTOP
  const xAxisPadding = isMobileChart ? X_AXIS_PADDING_MOBILE : X_AXIS_PADDING_DESKTOP
  const yAxisWidth = isMobileChart ? Y_AXIS_WIDTH_MOBILE : Y_AXIS_WIDTH_DESKTOP
  const rightYAxisWidth = hasOverlay ? (isMobileChart ? RIGHT_Y_AXIS_WIDTH_MOBILE : RIGHT_Y_AXIS_WIDTH_DESKTOP) : 0
  const yAxisTickStyle = {
    fill: '#9b9ca6',
    fontSize: isMobileChart ? 12 : 13,
    fontFamily: 'Aldrich, sans-serif',
    textAnchor: 'end',
    dx: isMobileChart ? -4 : -2,
    dy: -6,
  }
  const rightYAxisTickStyle = {
    fill: '#8a90a2',
    fontSize: isMobileChart ? 10 : 11,
    fontFamily: 'Aldrich, sans-serif',
    textAnchor: 'start',
    dx: isMobileChart ? 2 : 3,
    dy: -4,
  }
  const linePaddleLeft = isMobileChart ? 0 : 8

  const plotWidth = Math.max((containerWidth || 0) - (yAxisWidth + rightYAxisWidth + (xAxisPadding * 2)), 100)
  const barCategoryGap = computeBarCategoryGap(data.length)
  const barSize = computeBarSize(plotWidth, data.length)
  const plotTop = chartMargin.top
  const plotBottom = chartHeight - chartMargin.bottom - X_AXIS_HEIGHT
  const plotHeight = Math.max(1, plotBottom - plotTop)
  const lineRatio = yAxisMax > 0 ? clamp(line / yAxisMax, 0, 1) : 0
  const lineY = plotTop + ((1 - lineRatio) * plotHeight)

  const teardownDragListeners = useCallback(() => {
    const listeners = dragListenersRef.current
    if (listeners.move) {
      window.removeEventListener('pointermove', listeners.move)
    }
    if (listeners.up) {
      window.removeEventListener('pointerup', listeners.up)
    }
    dragListenersRef.current = { move: null, up: null }
  }, [])

  useEffect(() => () => teardownDragListeners(), [teardownDragListeners])

  const getLineValueFromClientY = useCallback((clientY) => {
    const canvas = chartCanvasRef.current
    if (!canvas) return line

    const rect = canvas.getBoundingClientRect()
    const localY = clientY - rect.top
    const clampedY = clamp(localY, plotTop, plotBottom)
    const ratio = (plotBottom - clampedY) / plotHeight
    const rawValue = ratio * yAxisMax
    return isCorners ? normalizeCornersLineValue(rawValue) : normalizeLineValue(rawValue)
  }, [line, plotBottom, plotHeight, plotTop, yAxisMax, isCorners])

  const handlePaddlePointerDown = useCallback((event) => {
    if (!isLineType || typeof onReferenceLineDragChange !== 'function') return

    event.preventDefault()
    event.stopPropagation()
    teardownDragListeners()

    let latestValue = getLineValueFromClientY(event.clientY)
    onReferenceLineDragChange(latestValue)

    const handlePointerMove = (moveEvent) => {
      latestValue = getLineValueFromClientY(moveEvent.clientY)
      onReferenceLineDragChange(latestValue)
    }

    const handlePointerUp = () => {
      teardownDragListeners()
      if (typeof onReferenceLineDragCommit === 'function') {
        onReferenceLineDragCommit(latestValue)
      }
    }

    dragListenersRef.current = { move: handlePointerMove, up: handlePointerUp }
    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', handlePointerUp, { once: true })
  }, [getLineValueFromClientY, isLineType, onReferenceLineDragChange, onReferenceLineDragCommit, teardownDragListeners])

  const yAxisTickFormatter = (value) => {
    const numeric = Number(value)
    if (isLineType && Number.isFinite(numeric) && Math.abs(numeric - line) < 1e-6) {
      // Paddle shows the line value in place of the axis tick label.
      return ''
    }
    return formatYAxisTick(value)
  }

  if (!data.length) {
    return <div className="chart-empty">No matches available.</div>
  }

  return (
    <div className="team-chart-block" ref={containerRef}>
      {/* ── PropsMadness-style header ── */}
      <div className="pm-chart-header">
        <div className="pm-identity">
          {teamLogo ? (
            <img src={teamLogo} alt={firstRow?.team_name || title} className="pm-logo" />
          ) : (
            <div className="pm-logo-fallback">{(firstRow?.team_name || title || '?').slice(0, 1)}</div>
          )}
          <div className="pm-name-block">
            <span className="pm-team-name">{firstRow?.team_name || title}</span>
            <span className={`pm-subtitle ${hasLineSummaryLabel ? 'pm-subtitle--line' : ''}`}>{lineSummaryLabel || title}</span>
          </div>
        </div>

        {hitRate.percent != null ? (
          <div className="pm-hit-rate">
            <span className="pm-hit-rate-label">HIT RATE</span>
            <span className={`pm-hit-rate-value ${hitRateToneClass}`}>{hitRate.percent.toFixed(1)}%</span>
            <span className="pm-hit-rate-count">({hitRate.hits}/{hitRate.total})</span>
          </div>
        ) : null}

        <div className="pm-metrics-row">
          {seasonAvg != null ? (
            <div className="pm-metric-cell">
              <span className="pm-metric-label">SEASON AVG</span>
              <span className="pm-metric-value">{seasonAvg.toFixed(2)}</span>
            </div>
          ) : null}
          {graphAvg != null ? (
            <div className="pm-metric-cell">
              <span className="pm-metric-label">GRAPH AVG</span>
              <span className="pm-metric-value">{graphAvg.toFixed(2)}</span>
            </div>
          ) : null}
          {hasOverlay && overlayAvg != null ? (
            <div className="pm-metric-cell">
              <span className="pm-metric-label">{overlayConfig.label}</span>
              <span className="pm-metric-value">{overlayAvg.toFixed(2)}</span>
            </div>
          ) : null}
          {typeof onToggleFilters === 'function' ? (
            <button
              className={`filters-toggle-btn ${isFiltersOpen ? 'active' : ''}`}
              onClick={onToggleFilters}
              style={{ marginLeft: 8, alignSelf: 'center' }}
            >
              Filters
            </button>
          ) : null}
        </div>
      </div>

      {/* ── Bar chart ── */}
      <div className="pm-chart-body">
        <div className="pm-chart-canvas" ref={chartCanvasRef}>
          <ResponsiveContainer width="100%" height={chartHeight}>
            <ComposedChart data={data} barCategoryGap={barCategoryGap} margin={chartMargin}>
              <XAxis
                dataKey="label"
                interval={0}
                tickLine={false}
                axisLine={false}
                height={X_AXIS_HEIGHT}
                tick={<MatchAxisTick data={data} containerWidth={containerWidth} />}
                padding={{ left: xAxisPadding, right: xAxisPadding }}
              />
              <YAxis
                yAxisId="left"
                domain={[0, yAxisMax]}
                ticks={yAxisTicks}
                interval={0}
                allowDecimals
                tickFormatter={yAxisTickFormatter}
                tickLine={false}
                axisLine={false}
                width={yAxisWidth}
                tickMargin={2}
                tick={yAxisTickStyle}
              />
              {hasOverlay && overlayScale ? (
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  domain={[overlayScale.min, overlayScale.max]}
                  ticks={overlayScale.ticks}
                  interval={0}
                  allowDecimals
                  tickFormatter={formatYAxisTick}
                  tickLine={false}
                  axisLine={false}
                  width={rightYAxisWidth}
                  tickMargin={2}
                  tick={rightYAxisTickStyle}
                />
              ) : null}
              <Tooltip
                cursor={{ fill: 'rgba(255, 255, 255, 0.04)' }}
                contentStyle={{ background: '#111114', borderColor: '#2a2a30', borderRadius: 10, color: '#edeef2', fontSize: '0.8rem' }}
                formatter={(val, name, item) => {
                  if (name === 'overlayValue' && overlayConfig) {
                    return [formatYAxisTick(val), overlayConfig.label]
                  }
                  const metricName = item?.payload?.goals_metric_name
                  if (isOverUnder) {
                    return [Number(val).toFixed(1), metricName || 'Goals']
                  }
                  if (isCorners) {
                    const totalCorners = getTotalCornersValue(item?.payload || {})
                    return [Number(totalCorners).toFixed(1), 'Total Corners']
                  }
                  if (isBtts) {
                    return [Number(val).toFixed(1), 'BTTS']
                  }
                  if (isDoubleChance) {
                    return [Number(val).toFixed(1), 'Double Chance']
                  }
                  return [Number(val).toFixed(1), 'Value']
                }}
                labelFormatter={(_, payload) => {
                  const row = payload?.[0]?.payload
                  if (!row) return ''
                  if (isOverUnder) {
                    const metricLabel = row.goals_metric_name || 'Goals'
                    const metricValue = row.total_goals ?? row.team_goals ?? 0
                    return `${row.fixture_display} | ${row.date_label} | ${metricLabel} ${Number(metricValue).toFixed(1)} | ${row.over_under_result}`
                  }
                  if (isCorners) {
                    return `${row.fixture_display} | ${row.date_label} | Total Corners ${getTotalCornersValue(row).toFixed(1)}`
                  }
                  if (isDoubleChance) {
                    return `${row.fixture_display} | ${row.date_label} | ${row.result || ''} | ${row.double_chance_result === 'H' ? 'Hit' : 'Miss'}`
                  }
                  if (isBtts) {
                    const bttsLabel = row.btts_result === 'Y' ? 'BTTS Yes' : 'BTTS No'
                    return `${row.fixture_display} | ${row.date_label} | ${bttsLabel} | ${Number(row.goals_scored ?? 0)}-${Number(row.opponent_goals ?? 0)}`
                  }
                  return `${row.fixture_display} | ${row.date_label} | Result ${row.result}`
                }}
              />
              <Bar yAxisId="left" dataKey="value" radius={[4, 4, 0, 0]} barSize={barSize}>
                {data.map((entry) => (
                  <Cell key={`${entry.match_id}-${entry.label}-${entry.venue}`} fill={entry.color} />
                ))}
              </Bar>
              {hasOverlay && overlayScale ? (
                <Line
                  yAxisId="right"
                  type="linear"
                  dataKey="overlayValue"
                  stroke={overlayConfig?.color || '#2d6bff'}
                  strokeWidth={2.2}
                  dot={false}
                  connectNulls
                  activeDot={{ r: 3, strokeWidth: 0, fill: overlayConfig?.color || '#2d6bff' }}
                  isAnimationActive={false}
                />
              ) : null}
              {isLineType ? (
                <ReferenceLine
                  yAxisId="left"
                  y={line}
                  stroke="#f8c629"
                  strokeWidth={2}
                  onClick={onReferenceLineClick}
                  onContextMenu={onReferenceLineContextMenu}
                />
              ) : (!isDoubleChance && !isBtts) ? (
                <ReferenceLine yAxisId="left" y={referenceLineValue} stroke="#f8c629" strokeWidth={2} />
              ) : null}
            </ComposedChart>
          </ResponsiveContainer>
          {isLineType ? (
            <button
              type="button"
              className="ou-line-paddle"
              style={{ top: `${lineY}px`, left: `${linePaddleLeft}px` }}
              onPointerDown={handlePaddlePointerDown}
              aria-label={`${isCorners ? 'Corners' : 'Over under'} line ${line.toFixed(1)}. Drag to adjust.`}
            >
              {line.toFixed(1)}
            </button>
          ) : null}
        </div>
      </div>

      {/* Legend */}
      <div className="chart-legend">
        {isLineType ? (
          <div className="chart-legend-item">
            <div className="chart-legend-swatch" style={{ background: '#f8c629' }} />
            <span>{isCorners ? 'Corners Line' : 'O/U Line'}</span>
          </div>
        ) : null}
        {hasOverlay ? (
          <div className="chart-legend-item">
            <div className="chart-legend-swatch" style={{ background: overlayConfig?.color || '#2d6bff' }} />
            <span>{overlayConfig?.label || 'Overlay'}</span>
          </div>
        ) : null}
      </div>
    </div>
  )
}

export default function ChartArea({
  recentMatches,
  betType = '1X2',
  overUnderLine = 2.5,
  onOverUnderLineDraftChange,
  onOverUnderLineCommit,
  cornersLine = 8.5,
  onCornersLineDraftChange,
  onCornersLineCommit,
  teamView,
  isFiltersOpen = false,
  onToggleFilters,
  activeOverlayFilter,
}) {
  const activeTeamView = teamView ?? VIEW_BOTH

  const normalizedBetType = String(betType).toLowerCase()
  const isOverUnder = isOverUnderFamilyBetType(normalizedBetType)
  const isCorners = isCornersBetType(normalizedBetType)
  const isLineType = isOverUnder || isCorners
  const isDoubleChance = normalizedBetType === BET_TYPE_DOUBLE_CHANCE
  const isBtts = normalizedBetType === BET_TYPE_BTTS
  const line = isCorners ? normalizeCornersLineValue(cornersLine) : normalizeLineValue(overUnderLine)
  const layoutVersion = isFiltersOpen ? 'open' : 'closed'

  // Get overlay configuration
  const overlayConfig = activeOverlayFilter ? OVERLAY_FIELD_MAP[activeOverlayFilter] || null : null

  const updateLineDraft = (nextValue) => {
    if (!isLineType) return
    if (isCorners) {
      if (typeof onCornersLineDraftChange === 'function') onCornersLineDraftChange(normalizeCornersLineValue(nextValue))
    } else {
      if (typeof onOverUnderLineDraftChange === 'function') onOverUnderLineDraftChange(normalizeLineValue(nextValue))
    }
  }

  const commitLine = (nextValue) => {
    if (!isLineType) return
    if (isCorners) {
      if (typeof onCornersLineCommit === 'function') onCornersLineCommit(normalizeCornersLineValue(nextValue))
    } else {
      if (typeof onOverUnderLineCommit === 'function') onOverUnderLineCommit(normalizeLineValue(nextValue))
    }
  }

  const lineStep = isCorners ? CORNERS_LINE_STEP : OVER_UNDER_LINE_STEP
  const normalizeFn = isCorners ? normalizeCornersLineValue : normalizeLineValue

  const handleReferenceLineClick = (event) => {
    if (!isLineType) return
    const rect = event?.currentTarget?.getBoundingClientRect?.()
    if (!rect) return
    const clickX = event.clientX - rect.left
    const next = normalizeFn(clickX < rect.width / 2 ? line - lineStep : line + lineStep)
    updateLineDraft(next)
    commitLine(next)
  }

  const handleReferenceLineContextMenu = (event) => {
    if (!isLineType) return
    event.preventDefault()
    const next = normalizeFn(line - lineStep)
    updateLineDraft(next)
    commitLine(next)
  }

  const homeData = useMemo(() => {
    let bars
    if (isCorners) {
      bars = buildCornerBars(recentMatches?.home || [], line)
    } else if (isOverUnder) {
      bars = buildOverUnderBars(recentMatches?.home || [], line, normalizedBetType)
    } else if (isBtts) {
      bars = buildBttsBars(recentMatches?.home || [])
    } else if (isDoubleChance) {
      bars = buildDoubleChanceBars(recentMatches?.home || [])
    } else {
      bars = buildOneXTwoBars(recentMatches?.home || [])
    }
    return enrichWithOverlay(bars, overlayConfig)
  }, [recentMatches, isCorners, isOverUnder, isBtts, isDoubleChance, line, normalizedBetType, overlayConfig])

  const awayData = useMemo(() => {
    let bars
    if (isCorners) {
      bars = buildCornerBars(recentMatches?.away || [], line)
    } else if (isOverUnder) {
      bars = buildOverUnderBars(recentMatches?.away || [], line, normalizedBetType)
    } else if (isBtts) {
      bars = buildBttsBars(recentMatches?.away || [])
    } else if (isDoubleChance) {
      bars = buildDoubleChanceBars(recentMatches?.away || [])
    } else {
      bars = buildOneXTwoBars(recentMatches?.away || [])
    }
    return enrichWithOverlay(bars, overlayConfig)
  }, [recentMatches, isCorners, isOverUnder, isBtts, isDoubleChance, line, normalizedBetType, overlayConfig])

  return (
    <div className="chart-area">

      <div className={`chart-split ${activeTeamView === VIEW_BOTH ? 'both' : 'single'}`}>
        {activeTeamView !== VIEW_AWAY ? (
          <TeamBarChart
            title="Home Team Matches"
            data={homeData}
            betType={betType}
            line={line}
            onReferenceLineClick={handleReferenceLineClick}
            onReferenceLineContextMenu={handleReferenceLineContextMenu}
            onReferenceLineDragChange={updateLineDraft}
            onReferenceLineDragCommit={commitLine}
            layoutVersion={layoutVersion}
            overlayConfig={overlayConfig}
            isFiltersOpen={isFiltersOpen}
            onToggleFilters={onToggleFilters}
          />
        ) : null}
        {activeTeamView !== VIEW_HOME ? (
          <TeamBarChart
            title="Away Team Matches"
            data={awayData}
            betType={betType}
            line={line}
            onReferenceLineClick={handleReferenceLineClick}
            onReferenceLineContextMenu={handleReferenceLineContextMenu}
            onReferenceLineDragChange={updateLineDraft}
            onReferenceLineDragCommit={commitLine}
            layoutVersion={layoutVersion}
            overlayConfig={overlayConfig}
            isFiltersOpen={isFiltersOpen}
            onToggleFilters={onToggleFilters}
          />
        ) : null}
      </div>
    </div>
  )
}
