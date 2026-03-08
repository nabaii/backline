import React, { useCallback, useEffect, useRef, useState } from 'react'
import { getLeagues, getFixturesForLeague } from '../api/backendApi'
import { getTeamLogo } from '../utils/premierLeagueLogos'
import LeagueSelector from './LeagueSelector'
import FixtureList from './FixtureList'
import BetTypeWorkspace from './BetTypeWorkspace'
import WorkspaceLoadingPlaceholder from './WorkspaceLoadingPlaceholder'
import ChatWindow from './ChatWindow'

function FixtureBottomMenu({ fixtures = [], selected, onSelect }) {
  const menuRef = useRef(null)

  useEffect(() => {
    if (!menuRef.current) return
    const selectedItem = menuRef.current.querySelector('.fixture-bottom-menu-item.selected')
    if (selectedItem) {
      selectedItem.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' })
    }
  }, [selected])

  if (!fixtures.length) return null

  return (
    <nav className="fixture-bottom-menu" ref={menuRef} aria-label="Fixtures">
      {fixtures.map((fixture) => {
        const homeLogo = getTeamLogo(fixture.home_team_name, fixture.home_team_id)
        const awayLogo = getTeamLogo(fixture.away_team_name, fixture.away_team_id)
        const homeShort = (fixture.home_team_name || '').split(' ').slice(-1)[0] || '?'
        const awayShort = (fixture.away_team_name || '').split(' ').slice(-1)[0] || '?'
        return (
          <button
            key={fixture.match_id}
            type="button"
            className={`fixture-bottom-menu-item ${selected === fixture.match_id ? 'selected' : ''}`}
            onClick={() => onSelect(fixture.match_id)}
          >
            <span className="fixture-bottom-team">
              {homeLogo ? <img src={homeLogo} alt={fixture.home_team_name} className="fixture-bottom-logo" /> : null}
              <span>{homeShort}</span>
            </span>
            <span className="fixture-bottom-vs">vs</span>
            <span className="fixture-bottom-team">
              {awayLogo ? <img src={awayLogo} alt={fixture.away_team_name} className="fixture-bottom-logo" /> : null}
              <span>{awayShort}</span>
            </span>
          </button>
        )
      })}
    </nav>
  )
}

export default function KitchenPage() {
  const [activeTab, setActiveTab] = useState('main')
  const [leagues, setLeagues] = useState([])
  const [selectedLeague, setSelectedLeague] = useState(null)
  const [fixtures, setFixtures] = useState([])
  const [selectedMatch, setSelectedMatch] = useState(null)
  const [error, setError] = useState(null)
  const [isLeagueLoading, setIsLeagueLoading] = useState(false)
  const [initialBetType, setInitialBetType] = useState(null)
  const fixturesCacheRef = useRef(new Map())
  const selectedMatchByLeagueRef = useRef(new Map())
  const fixturesDateRef = useRef(new Date().toISOString().slice(0, 10))
  const selectedFixture = fixtures.find(f => f.match_id === selectedMatch) || null

  const handleNavigateToKitchen = useCallback((chatBetType) => {
    // Map chat bet_type values to BetTypeWorkspace constants
    const BET_TYPE_MAP = {
      'over_under': 'over_under',
      'corners': 'corners',
      'one_x_two': '1X2',
      '1X2': '1X2',
      'double_chance': 'double_chance',
      'btts': 'btts',
      'home_ou': 'home_ou',
      'away_ou': 'away_ou',
      'win_either_half': 'win_either_half',
      'win_both_halves': 'win_both_halves',
    }
    setInitialBetType(BET_TYPE_MAP[chatBetType] || '1X2')
    setActiveTab('kitchen')
  }, [])

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
    setSelectedMatch(null)
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
        <div className="kitchen-header-top">
          <div className="brand-block">
            <h1>Backline</h1>
          </div>
          <nav className="page-tabs" aria-label="Pages">
            <button
              className={`page-tab${activeTab === 'main' ? ' page-tab--active' : ''}`}
              onClick={() => setActiveTab('main')}
            >
              Main
            </button>
            <button
              className={`page-tab${activeTab === 'kitchen' ? ' page-tab--active' : ''}`}
              onClick={() => setActiveTab('kitchen')}
            >
              Kitchen
            </button>
          </nav>
          <LeagueSelector leagues={leagues} value={selectedLeague} onChange={setSelectedLeague} />
        </div>
      </header>

      <div className="kitchen-body">
        <aside className="fixtures-pane">
          <FixtureList fixtures={fixtures} selected={selectedMatch} onSelect={setSelectedMatch} />
        </aside>

        <main className="workspace-pane">
          {activeTab === 'main' ? (
            <ChatWindow selectedFixture={selectedFixture} onNavigateToKitchen={handleNavigateToKitchen} />
          ) : error ? (
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
              homeTeamName={selectedFixture?.home_team_name}
              awayTeamName={selectedFixture?.away_team_name}
              leagueId={selectedLeague}
              initialBetType={initialBetType}
            />
          ) : isLeagueLoading ? (
            <WorkspaceLoadingPlaceholder />
          ) : <div className="workspace-error">No match selected.</div>}
        </main>
      </div>

      <FixtureBottomMenu fixtures={fixtures} selected={selectedMatch} onSelect={setSelectedMatch} />
    </div>
  )
}
