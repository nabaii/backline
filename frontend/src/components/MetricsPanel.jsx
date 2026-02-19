import React from 'react'

const BET_TYPE_OVER_UNDER = 'over_under'
const BET_TYPE_HOME_OU = 'home_ou'
const BET_TYPE_AWAY_OU = 'away_ou'
const BET_TYPE_DOUBLE_CHANCE = 'double_chance'
const BET_TYPE_BTTS = 'btts'
const BET_TYPE_CORNERS = 'corners'

function isOverUnderFamilyBetType(betType) {
  const normalized = String(betType).toLowerCase()
  return normalized === BET_TYPE_OVER_UNDER || normalized === BET_TYPE_HOME_OU || normalized === BET_TYPE_AWAY_OU
}

function renderSampleSize(sampleSize, sampleSizes) {
  const homeSize = sampleSizes?.home_team
  const awaySize = sampleSizes?.away_team
  if (Number.isFinite(homeSize) || Number.isFinite(awaySize)) {
    const homeLabel = Number.isFinite(homeSize) ? homeSize : 0
    const awayLabel = Number.isFinite(awaySize) ? awaySize : 0
    if (homeLabel === awayLabel) {
      return String(homeLabel)
    }
    return `Home ${homeLabel} | Away ${awayLabel}`
  }
  return String(sampleSize ?? 0)
}

function TeamMetricsBlock({ title, metrics, isOverUnder, isDoubleChance, isBtts, isCorners }) {
  if (isCorners) {
    const over = Number(metrics?.over ?? 0)
    const under = Number(metrics?.under ?? 0)
    const total = over + under
    const percent = total > 0 ? Math.round((over / total) * 100) : 0
    const hitRateLabel = `${over}/${total} (${percent}%)`
    const hitRateToneClass = percent > 50 ? 'metrics-hit-rate--positive' : percent < 50 ? 'metrics-hit-rate--negative' : ''
    return (
      <div>
        <strong>{title}</strong>
        <div>Hit Rate: <span className={`metrics-hit-rate ${hitRateToneClass}`}>{hitRateLabel}</span></div>
        <div>Over: {over}</div>
        <div>Under: {under}</div>
        <div>Avg Total Corners: {Number(metrics?.avg_total_corners ?? 0).toFixed(2)}</div>
        <div>Min Total Corners: {Number(metrics?.min_total_corners ?? 0).toFixed(2)}</div>
        <div>Max Total Corners: {Number(metrics?.max_total_corners ?? 0).toFixed(2)}</div>
      </div>
    )
  }

  const total = isOverUnder
    ? Number((metrics?.over ?? 0) + (metrics?.under ?? 0))
    : isDoubleChance
      ? Number((metrics?.hits ?? 0) + (metrics?.misses ?? 0))
      : isBtts
        ? Number((metrics?.hits ?? 0) + (metrics?.misses ?? 0))
        : Number((metrics?.wins ?? 0) + (metrics?.draws ?? 0) + (metrics?.losses ?? 0))
  const hits = isOverUnder
    ? Number(metrics?.over ?? 0)
    : isDoubleChance
      ? Number(metrics?.hits ?? 0)
      : isBtts
        ? Number(metrics?.hits ?? 0)
        : Number(metrics?.wins ?? 0)
  const percent = total > 0 ? Math.round((hits / total) * 100) : 0
  const hitRateLabel = `${hits}/${total} (${percent}%)`
  const hitRateToneClass = percent > 50 ? 'metrics-hit-rate--positive' : percent < 50 ? 'metrics-hit-rate--negative' : ''

  if (isOverUnder) {
    return (
      <div>
        <strong>{title}</strong>
        <div>Hit Rate: <span className={`metrics-hit-rate ${hitRateToneClass}`}>{hitRateLabel}</span></div>
        <div>Over: {metrics?.over ?? 0}</div>
        <div>Under: {metrics?.under ?? 0}</div>
      </div>
    )
  }

  if (isDoubleChance) {
    return (
      <div>
        <strong>{title}</strong>
        <div>Hit Rate: <span className={`metrics-hit-rate ${hitRateToneClass}`}>{hitRateLabel}</span></div>
        <div>Hits (Win/Draw): {metrics?.hits ?? 0}</div>
        <div>Misses (Loss): {metrics?.misses ?? 0}</div>
      </div>
    )
  }

  if (isBtts) {
    return (
      <div>
        <strong>{title}</strong>
        <div>Hit Rate: <span className={`metrics-hit-rate ${hitRateToneClass}`}>{hitRateLabel}</span></div>
        <div>BTTS Yes: {metrics?.hits ?? 0}</div>
        <div>BTTS No: {metrics?.misses ?? 0}</div>
      </div>
    )
  }

  return (
    <div>
      <strong>{title}</strong>
      <div>Hit Rate: <span className={`metrics-hit-rate ${hitRateToneClass}`}>{hitRateLabel}</span></div>
      <div>Wins: {metrics?.wins ?? 0}</div>
      <div>Draws: {metrics?.draws ?? 0}</div>
      <div>Losses: {metrics?.losses ?? 0}</div>
    </div>
  )
}

export default function MetricsPanel({ betType = '1X2', metrics, sampleSize, sampleSizes }) {
  const normalizedBetType = String(betType).toLowerCase()
  const isOverUnder = isOverUnderFamilyBetType(normalizedBetType)
  const isDoubleChance = normalizedBetType === BET_TYPE_DOUBLE_CHANCE
  const isBtts = normalizedBetType === BET_TYPE_BTTS
  const isCorners = normalizedBetType === BET_TYPE_CORNERS

  return (
    <aside className="metrics-panel">
      <h4>Metrics</h4>
      <div>Sample size: {renderSampleSize(sampleSize, sampleSizes)}</div>
      <div className="team-metrics">
        <TeamMetricsBlock
          title="Home"
          metrics={metrics?.home_team}
          isOverUnder={isOverUnder}
          isDoubleChance={isDoubleChance}
          isBtts={isBtts}
          isCorners={isCorners}
        />
        <TeamMetricsBlock
          title="Away"
          metrics={metrics?.away_team}
          isOverUnder={isOverUnder}
          isDoubleChance={isDoubleChance}
          isBtts={isBtts}
          isCorners={isCorners}
        />
      </div>
    </aside>
  )
}
