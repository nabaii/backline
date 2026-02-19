const API_BASE = import.meta.env.VITE_API_BASE || ''

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  })

  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const message = payload.error || `Request failed (${response.status})`
    throw new Error(message)
  }
  return payload
}

export async function getLeagues() {
  return requestJson('/api/leagues')
}

export async function getFixturesForLeague(league_id, date, gameweek) {
  const params = new URLSearchParams()
  if (league_id) params.set('league_id', league_id)
  if (date) params.set('date', date)
  if (Number.isFinite(gameweek)) params.set('gameweek', String(gameweek))
  const query = params.toString()
  return requestJson(`/api/fixtures${query ? `?${query}` : ''}`)
}

export async function getMatch(match_id) {
  return requestJson(`/api/matches/${match_id}`)
}

export async function getWorkspace1X2({ match_id, home_team_id, away_team_id, limit = 0, filters = {}, evidenceFilters = [] }) {
  return requestJson('/api/workspace/1x2', {
    method: 'POST',
    body: JSON.stringify({
      match_id,
      home_team_id,
      away_team_id,
      limit,
      filters,
      evidenceFilters,
    }),
  })
}

export async function getWorkspaceOverUnder({
  match_id,
  home_team_id,
  away_team_id,
  line = 2.5,
  limit = 0,
  filters = {},
  evidenceFilters = [],
}) {
  return requestJson('/api/workspace/over_under', {
    method: 'POST',
    body: JSON.stringify({
      match_id,
      home_team_id,
      away_team_id,
      line,
      limit,
      filters,
      evidenceFilters,
    }),
  })
}

export async function getWorkspaceDoubleChance({
  match_id,
  home_team_id,
  away_team_id,
  limit = 0,
  filters = {},
  evidenceFilters = [],
}) {
  return requestJson('/api/workspace/double_chance', {
    method: 'POST',
    body: JSON.stringify({
      match_id,
      home_team_id,
      away_team_id,
      limit,
      filters,
      evidenceFilters,
    }),
  })
}

export async function getWorkspaceBtts({
  match_id,
  home_team_id,
  away_team_id,
  limit = 0,
  filters = {},
  evidenceFilters = [],
}) {
  return requestJson('/api/workspace/btts', {
    method: 'POST',
    body: JSON.stringify({
      match_id,
      home_team_id,
      away_team_id,
      limit,
      filters,
      evidenceFilters,
    }),
  })
}

export async function getWorkspaceHomeOu({
  match_id,
  home_team_id,
  away_team_id,
  line = 2.5,
  limit = 0,
  filters = {},
  evidenceFilters = [],
}) {
  return requestJson('/api/workspace/home_ou', {
    method: 'POST',
    body: JSON.stringify({
      match_id,
      home_team_id,
      away_team_id,
      line,
      limit,
      filters,
      evidenceFilters,
    }),
  })
}

export async function getWorkspaceAwayOu({
  match_id,
  home_team_id,
  away_team_id,
  line = 2.5,
  limit = 0,
  filters = {},
  evidenceFilters = [],
}) {
  return requestJson('/api/workspace/away_ou', {
    method: 'POST',
    body: JSON.stringify({
      match_id,
      home_team_id,
      away_team_id,
      line,
      limit,
      filters,
      evidenceFilters,
    }),
  })
}

export async function getWorkspaceCorners({
  match_id,
  home_team_id,
  away_team_id,
  line = 8.5,
  limit = 0,
  filters = {},
  evidenceFilters = [],
}) {
  return requestJson('/api/workspace/corners', {
    method: 'POST',
    body: JSON.stringify({
      match_id,
      home_team_id,
      away_team_id,
      line,
      limit,
      filters,
      evidenceFilters,
    }),
  })
}

export default {
  getLeagues,
  getFixturesForLeague,
  getMatch,
  getWorkspace1X2,
  getWorkspaceOverUnder,
  getWorkspaceDoubleChance,
  getWorkspaceBtts,
  getWorkspaceHomeOu,
  getWorkspaceAwayOu,
  getWorkspaceCorners,
}
