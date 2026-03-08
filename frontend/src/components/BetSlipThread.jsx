import React from 'react'
import ChatMiniChart from './ChatMiniChart'

/**
 * Renders a thread of bet slip analyses.
 * Each item shows: match header, selection label, mini chart, and explanation.
 */
export default function BetSlipThread({ analyses, extractedCount, isProcessing, onNavigateToKitchen }) {
  if (!analyses?.length && !isProcessing) return null

  return (
    <div className="betslip-thread">
      <div className="betslip-thread-header">
        <span className="betslip-thread-icon">&#9635;</span>
        <span className="betslip-thread-title">Bet Slip Analysis</span>
        <span className="betslip-thread-count">
          {analyses.length}{extractedCount ? ` / ${extractedCount}` : ''} selections
        </span>
      </div>

      <div className="betslip-thread-items">
        {analyses.map((item, i) => (
          <div key={i} className="betslip-thread-item">
            {/* Match header */}
            <div className="betslip-item-header">
              <span className="betslip-item-index">{i + 1}</span>
              <div className="betslip-item-match">
                <span className="betslip-item-teams">
                  {item.home_team} vs {item.away_team}
                </span>
                <span className="betslip-item-selection">{item.selection_label}</span>
              </div>
              {!item.matched && (
                <span className="betslip-item-badge betslip-item-badge--unmatched">
                  No data
                </span>
              )}
            </div>

            {/* Chart */}
            {item.chart_data?.recent_matches?.length > 0 && (
              <div className="betslip-item-chart">
                <ChatMiniChart
                  chartData={item.chart_data.recent_matches}
                  betType={item.chart_data.bet_type}
                  line={item.chart_data.line}
                  teamName={item.chart_data.home_team}
                  onNavigateToKitchen={onNavigateToKitchen}
                />
              </div>
            )}

            {/* Explanation */}
            {item.explanation && (
              <div className="betslip-item-explanation">
                {item.explanation}
              </div>
            )}
          </div>
        ))}

        {/* Loading indicator for next bet */}
        {isProcessing && (
          <div className="betslip-thread-item betslip-thread-item--loading">
            <div className="betslip-item-header">
              <span className="betslip-item-index">{analyses.length + 1}</span>
              <div className="betslip-item-match">
                <span className="betslip-loading-dots">Analyzing</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
