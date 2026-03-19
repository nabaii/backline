import React, { useMemo } from 'react'
import {
  ComposedChart, Bar, XAxis, YAxis, ReferenceLine,
  ResponsiveContainer, Cell, Tooltip
} from 'recharts'

const HIT_COLOR = '#2ecc71'
const MISS_COLOR = '#e74c3c'
const DRAW_COLOR = '#f39c12'
const LINE_COLOR = '#f8c629'

function getBarColor(result) {
  if (result === 'O' || result === 'W' || result === 'H') return HIT_COLOR
  if (result === 'D') return DRAW_COLOR
  return MISS_COLOR
}

function fmt(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '–'
  return Number.isInteger(v) ? String(v) : v.toFixed(1)
}

/* ── Tooltip (hover/click on bar) ── */
function ChartTooltip({ active, payload }) {
  if (!active || !payload?.[0]) return null
  const d = payload[0].payload
  const venue = d.venue === 'home' ? 'vs' : '@'
  return (
    <div style={{
      background: '#111114', border: '1px solid #2a2a30', borderRadius: 10,
      padding: '6px 10px', color: '#edeef2', fontSize: '0.8rem',
      fontFamily: "'Aldrich', sans-serif",
    }}>
      <div>{venue} {d.opponent_name}</div>
      <div style={{ color: getBarColor(d.result), fontWeight: 700 }}>{fmt(d.value)}</div>
    </div>
  )
}

export default function ChatMiniChart({ chartData, betType, line, teamName, onNavigateToKitchen }) {
  const data = useMemo(() => {
    if (!chartData?.length) return []
    return chartData.map((m) => ({
      ...m,
      label: m.venue === 'home'
        ? `vs ${(m.opponent_name || '').split(' ').slice(-1)[0]}`
        : `@ ${(m.opponent_name || '').split(' ').slice(-1)[0]}`,
      color: getBarColor(m.result),
    }))
  }, [chartData])

  // Stats
  const hitCount = data.filter(d => d.result === 'O' || d.result === 'W' || d.result === 'H').length
  const hitRate = data.length ? (hitCount / data.length) * 100 : null
  const hitRateClass = hitRate != null ? (hitRate > 50 ? 'hit-rate-positive' : hitRate < 50 ? 'hit-rate-negative' : '') : ''

  const graphAvg = useMemo(() => {
    const vals = data.map(d => d.value).filter(v => Number.isFinite(v))
    return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null
  }, [data])

  const seasonAvg = graphAvg

  if (!data.length) return null

  const miniBarRadius = data.length > 20 ? 2 : 4
  const isLineType = betType === 'over_under' || betType === 'corners'
  const maxVal = Math.max(...data.map(d => d.value ?? 0), isLineType ? (line || 0) + 1 : 1.5)
  const yMax = Math.ceil(maxVal + 0.5)

  return (
    <div className="chat-chart-wrap chat-chart-compressed">
      {/* ── Header ── */}
      <div className="chat-chart-header">
        <div className="chat-chart-identity">
          <span className="chat-chart-team">{teamName}</span>
          {isLineType && <span className="chat-chart-subtitle">{betType === 'corners' ? 'Corners' : 'Over/Under'} {line} Total {betType === 'corners' ? 'Corners' : 'Goals'}</span>}
          {!isLineType && <span className="chat-chart-subtitle">{betType === 'one_x_two' ? '1X2' : betType === 'double_chance' ? 'Double Chance' : betType}</span>}
        </div>

        {hitRate != null && (
          <div className="chat-chart-stat">
            <span className="chat-chart-stat-label">HIT RATE</span>
            <span className={`chat-chart-stat-value ${hitRateClass}`}>{hitRate.toFixed(1)}%</span>
          </div>
        )}

        {seasonAvg != null && (
          <div className="chat-chart-stat">
            <span className="chat-chart-stat-label">SZN</span>
            <span className="chat-chart-stat-value">{seasonAvg.toFixed(2)}</span>
          </div>
        )}

        {graphAvg != null && (
          <div className="chat-chart-stat">
            <span className="chat-chart-stat-label">GRPH</span>
            <span className="chat-chart-stat-value">{graphAvg.toFixed(2)}</span>
          </div>
        )}
      </div>

      {/* ── Chart body ── */}
      <div className="chat-chart-body" style={{ position: 'relative' }}>
        <ResponsiveContainer width="100%" height={120}>
          <ComposedChart
            data={data}
            barCategoryGap="4%"
            margin={{ top: 6, right: 4, left: 0, bottom: 2 }}
          >
            <XAxis dataKey="label" hide />
            <YAxis
              domain={[0, yMax]}
              tickLine={false}
              axisLine={false}
              width={24}
              tick={{ fill: '#9b9ca6', fontSize: 9, fontFamily: "'Aldrich', sans-serif" }}
              tickMargin={1}
            />

            <Tooltip
              cursor={{ fill: 'rgba(255, 255, 255, 0.04)' }}
              content={<ChartTooltip />}
            />

            {/* Reference line for over/under and corners */}
            {isLineType && (
              <ReferenceLine
                y={line}
                stroke={LINE_COLOR}
                strokeWidth={2}
                strokeDasharray="4 3"
              />
            )}

            <Bar
              dataKey="value"
              radius={[miniBarRadius, miniBarRadius, 0, 0]}
              isAnimationActive={false}
            >
              {data.map((d, i) => (
                <Cell key={i} fill={d.color} />
              ))}
            </Bar>
          </ComposedChart>
        </ResponsiveContainer>
        {onNavigateToKitchen && (
          <button
            className="chat-chart-expand-btn"
            onClick={() => onNavigateToKitchen(betType)}
            aria-label="Open in Kitchen"
            title="Open in Kitchen"
          >
            ⊞
          </button>
        )}
      </div>
    </div>
  )
}
