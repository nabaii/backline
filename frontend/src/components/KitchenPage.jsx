import React, { useEffect, useRef, useState } from 'react'
import { getLeagues, getFixturesForLeague } from '../api/backendApi'
import { getTeamLogo } from '../utils/premierLeagueLogos'
import LeagueSelector from './LeagueSelector'
import FixtureList from './FixtureList'
import BetTypeWorkspace from './BetTypeWorkspace'
import WorkspaceLoadingPlaceholder from './WorkspaceLoadingPlaceholder'

function FixtureBottomBar({ fixtures = [], selected, onSelect }) {
  const barRef = useRef(null)

  // Auto-scroll to selected card
  useEffect(() => {
    if (!barRef.current) return
    const selectedCard = barRef.current.querySelector('.fixture-bottom-card.selected')
    if (selectedCard) {
      selectedCard.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' })
    }
  }, [selected])

  if (!fixtures.length) return null

  return (
    <div className="fixture-bottom-bar" ref={barRef}>
      {fixtures.map(f => {
        const homeLogo = getTeamLogo(f.home_team_name, f.home_team_id)
        const awayLogo = getTeamLogo(f.away_team_name, f.away_team_id)
        const shortHome = (f.home_team_name || '').split(' ').slice(-1)[0] || '?'
        const shortAway = (f.away_team_name || '').split(' ').slice(-1)[0] || '?'

        return (
          <button
            key={f.match_id}
            type="button"
            className={`fixture-bottom-card${selected === f.match_id ? ' selected' : ''}`}
            onClick={() => onSelect(f.match_id)}
          >
            {homeLogo
              ? <img src={homeLogo} alt={f.home_team_name} className="fixture-bottom-card-logo" />
              : <span className="fixture-bottom-card-name">{shortHome}</span>
            }
            <span className="fixture-bottom-card-vs">@</span>
            {awayLogo
              ? <img src={awayLogo} alt={f.away_team_name} className="fixture-bottom-card-logo" />
              : <span className="fixture-bottom-card-name">{shortAway}</span>
            }
          </button>
        )
      })}
    </div>
  )
}

export default function KitchenPage() {
  const [leagues, setLeagues] = useState([])
  const [selectedLeague, setSelectedLeague] = useState(null)
  const [fixtures, setFixtures] = useState([])
  const [selectedMatch, setSelectedMatch] = useState(null)
  const [error, setError] = useState(null)
  const [isLeagueLoading, setIsLeagueLoading] = useState(false)
  const fixturesCacheRef = useRef(new Map())
  const selectedMatchByLeagueRef = useRef(new Map())
  const fixturesDateRef = useRef(new Date().toISOString().slice(0, 10))
  const selectedFixture = fixtures.find(f => f.match_id === selectedMatch) || null

  const pickSelectedMatch = (leagueId, leagueFixtures) => {
    if (!leagueFixtures.length) return null
    const previousMatch = selectedMatchByLeagueRef.current.get(leagueId)
    if (previousMatch && leagueFixtures.some(f => f.match_id === previousMatch)) {
      return previousMatch
    }
    return leagueFixtures[0].match_id
  }

  useEffect(() => {
    setError(null)
    setIsLeagueLoading(true)
    getLeagues()
      .then(r => {
        const nextLeagues = r.leagues || []
        setLeagues(nextLeagues)
        const firstLeague = nextLeagues?.[0]?.id || null
        setSelectedLeague(firstLeague)

        // Warm cache for faster league switches.
        Promise.allSettled(
          nextLeagues.map(async (league) => {
            if (!league?.id || league.id === firstLeague || fixturesCacheRef.current.has(league.id)) return
            const response = await getFixturesForLeague(league.id, fixturesDateRef.current)
            const nextFixtures = response.fixtures || []
            fixturesCacheRef.current.set(league.id, nextFixtures)
          })
        ).catch(() => { })

        if (!firstLeague) {
          setIsLeagueLoading(false)
        }
      })
      .catch(err => {
        setError(err?.message || 'Failed to load leagues')
        setIsLeagueLoading(false)
      })
  }, [])

  useEffect(() => {
    if (!selectedLeague) return
    let cancelled = false
    setError(null)
    const cachedFixtures = fixturesCacheRef.current.get(selectedLeague)
    if (cachedFixtures) {
      setFixtures(cachedFixtures)
      setSelectedMatch(pickSelectedMatch(selectedLeague, cachedFixtures))
      setIsLeagueLoading(false)
      return () => {
        cancelled = true
      }
    }

    setIsLeagueLoading(true)
    getFixturesForLeague(selectedLeague, fixturesDateRef.current)
      .then(r => {
        if (cancelled) return
        const nextFixtures = r.fixtures || []
        fixturesCacheRef.current.set(selectedLeague, nextFixtures)
        setFixtures(nextFixtures)
        if (nextFixtures.length) {
          setSelectedMatch(pickSelectedMatch(selectedLeague, nextFixtures))
        } else {
          setSelectedMatch(null)
          setError('No upcoming fixtures available from the backend for the selected league.')
        }
      })
      .catch(err => {
        if (cancelled) return
        setFixtures([])
        setSelectedMatch(null)
        setError(err?.message || 'Failed to load fixtures')
      })
      .finally(() => {
        if (!cancelled) {
          setIsLeagueLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [selectedLeague])

  useEffect(() => {
    if (!selectedLeague || !selectedMatch) return
    selectedMatchByLeagueRef.current.set(selectedLeague, selectedMatch)
  }, [selectedLeague, selectedMatch])

  return (
    <div className="kitchen-page">
      <header className="kitchen-header">
        <div className="brand-block">
          <h1>Backline</h1>
          <p>Football Analytics Workspace</p>
        </div>
        <LeagueSelector leagues={leagues} value={selectedLeague} onChange={setSelectedLeague} />
      </header>

      <div className="kitchen-body">
        <aside className="fixtures-pane">
          <FixtureList fixtures={fixtures} selected={selectedMatch} onSelect={setSelectedMatch} />
        </aside>

        <main className="workspace-pane">
          {error ? (
            <div className="workspace-error">
              <strong>Could not load match workspace.</strong>
              <div>{error}</div>
              <div>Ensure the backend is running with: <code>python backend_api.py</code></div>
            </div>
          ) : selectedMatch ? (
            <BetTypeWorkspace
              matchId={selectedMatch}
              homeTeamId={selectedFixture?.home_team_id}
              awayTeamId={selectedFixture?.away_team_id}
            />
          ) : isLeagueLoading ? (
            <WorkspaceLoadingPlaceholder />
          ) : <div className="workspace-error">No match selected.</div>}
        </main>
      </div>

      {/* Mobile fixture bottom bar */}
      <FixtureBottomBar fixtures={fixtures} selected={selectedMatch} onSelect={setSelectedMatch} />
    </div>
  )
}
