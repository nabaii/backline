import React, { useEffect, useMemo, useState } from 'react'
import { getTeamLogo, formatFixtureDate, formatFixtureTime } from '../utils/premierLeagueLogos'

export default function FixtureList({ fixtures = [], selected, onSelect }) {
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    setExpanded(false)
  }, [fixtures])

  const hasMore = fixtures.length > 11
  const visibleFixtures = useMemo(() => {
    if (expanded || !hasMore) return fixtures
    return fixtures.slice(0, 11)
  }, [fixtures, expanded, hasMore])

  return (
    <div className="fixture-list">
      <h3>Fixtures</h3>
      <ul>
        {visibleFixtures.map(f => {
          const homeLogo = getTeamLogo(f.home_team_name, f.home_team_id)
          const awayLogo = getTeamLogo(f.away_team_name, f.away_team_id)

          return (
            <li key={f.match_id} className={selected === f.match_id ? 'selected' : ''} onClick={() => onSelect(f.match_id)}>
              <div className="fixture-item">
                <div className="fixture-team-block">
                  <div className="fixture-team-logo-wrap">
                    {homeLogo ? <img src={homeLogo} alt={`${f.home_team_name} logo`} className="fixture-team-logo" /> : null}
                  </div>
                  <div className="fixture-team-name">{f.home_team_name}</div>
                </div>

                <div className="fixture-meta-block">
                  <div className="fixture-meta-date">{formatFixtureDate(f.kickoff)}</div>
                  <div className="fixture-meta-time">{formatFixtureTime(f.kickoff)}</div>
                </div>

                <div className="fixture-team-block">
                  <div className="fixture-team-logo-wrap">
                    {awayLogo ? <img src={awayLogo} alt={`${f.away_team_name} logo`} className="fixture-team-logo" /> : null}
                  </div>
                  <div className="fixture-team-name">{f.away_team_name}</div>
                </div>
              </div>
            </li>
          )
        })}
      </ul>
      {hasMore ? (
        <button
          type="button"
          className="fixture-view-more"
          onClick={() => setExpanded(current => !current)}
        >
          {expanded ? 'View less' : `View more (${fixtures.length - 11} more)`}
        </button>
      ) : null}
    </div>
  )
}
