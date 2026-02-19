import React from 'react'
import { getLeagueLogo } from '../utils/premierLeagueLogos'

export default function LeagueSelector({ leagues = [], value, onChange }) {
  if (!leagues.length) return null

  return (
    <div className="league-tabs">
      {leagues.map(l => {
        const logoUrl = getLeagueLogo(l.id)
        return (
          <button
            key={l.id}
            className={`league-tab${value === l.id ? ' league-tab--active' : ''}`}
            onClick={() => onChange(l.id)}
            title={l.name}
          >
            {logoUrl
              ? <img src={logoUrl} alt={l.name} className="league-tab-logo" />
              : <span className="league-tab-flag">{l.flag || ''}</span>
            }
            <span className="league-tab-name">{l.name}</span>
          </button>
        )
      })}
    </div>
  )
}
