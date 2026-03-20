import React from 'react'

const TABS = [
  { id: '1X2', label: '1X2' },
  { id: 'double_chance', label: 'Double Chance' },
  { id: 'btts', label: 'BTTS' },
  { id: 'over_under', label: 'Over/Under' },
  { id: 'home_ou', label: 'Home O/U' },
  { id: 'away_ou', label: 'Away O/U' },
  { id: 'corners', label: 'Corners' },
]

export default function WorkspaceLoadingPlaceholder({ activeBetType = '1X2' }) {
  return (
    <div className="workspace-loading-shell">
      <div className="workspace-header">
        <div className="bettabs">
          {TABS.map(tab => (
            <button
              key={tab.id}
              className={tab.id === activeBetType ? 'active' : ''}
              type="button"
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div className="workspace-loading-body">
        <div className="workspace-loading-chart">
          <div className="workspace-loading-title">Chart Area</div>
          <div className="workspace-loading-empty" />
        </div>
        <div className="workspace-loading-chart">
          <div className="workspace-loading-title">Chart Area</div>
          <div className="workspace-loading-empty" />
        </div>
      </div>
    </div>
  )
}
