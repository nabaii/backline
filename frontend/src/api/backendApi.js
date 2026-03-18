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

export async function getWorkspace1X2({
  match_id,
  home_team_id,
  away_team_id,
  home_team_name = '',
  away_team_name = '',
  league_id = '',
  limit = 0,
  filters = {},
  evidenceFilters = [],
}) {
  return requestJson('/api/workspace/1x2', {
    method: 'POST',
    body: JSON.stringify({
      match_id,
      home_team_id,
      away_team_id,
      home_team_name,
      away_team_name,
      league_id,
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
  home_team_name = '',
  away_team_name = '',
  league_id = '',
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
      home_team_name,
      away_team_name,
      league_id,
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
  home_team_name = '',
  away_team_name = '',
  league_id = '',
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
      home_team_name,
      away_team_name,
      league_id,
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
  home_team_name = '',
  away_team_name = '',
  league_id = '',
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
      home_team_name,
      away_team_name,
      league_id,
      limit,
      filters,
      evidenceFilters,
    }),
  })
}

export async function getWorkspace1X2Ou({
  match_id,
  home_team_id,
  away_team_id,
  home_team_name = '',
  away_team_name = '',
  league_id = '',
  line = 2.5,
  limit = 0,
  filters = {},
  evidenceFilters = [],
}) {
  return requestJson('/api/workspace/1x2_ou', {
    method: 'POST',
    body: JSON.stringify({
      match_id,
      home_team_id,
      away_team_id,
      home_team_name,
      away_team_name,
      league_id,
      line,
      limit,
      filters,
      evidenceFilters,
    }),
  })
}

export async function getWorkspaceDoubleChanceOu({
  match_id,
  home_team_id,
  away_team_id,
  home_team_name = '',
  away_team_name = '',
  league_id = '',
  line = 2.5,
  limit = 0,
  filters = {},
  evidenceFilters = [],
}) {
  return requestJson('/api/workspace/double_chance_ou', {
    method: 'POST',
    body: JSON.stringify({
      match_id,
      home_team_id,
      away_team_id,
      home_team_name,
      away_team_name,
      league_id,
      line,
      limit,
      filters,
      evidenceFilters,
    }),
  })
}

export async function getWorkspaceBttsOu({
  match_id,
  home_team_id,
  away_team_id,
  home_team_name = '',
  away_team_name = '',
  league_id = '',
  line = 2.5,
  limit = 0,
  filters = {},
  evidenceFilters = [],
}) {
  return requestJson('/api/workspace/btts_ou', {
    method: 'POST',
    body: JSON.stringify({
      match_id,
      home_team_id,
      away_team_id,
      home_team_name,
      away_team_name,
      league_id,
      line,
      limit,
      filters,
      evidenceFilters,
    }),
  })
}

export async function getWorkspaceFirstHalfOu({
  match_id,
  home_team_id,
  away_team_id,
  home_team_name = '',
  away_team_name = '',
  league_id = '',
  line = 1.5,
  limit = 0,
  filters = {},
  evidenceFilters = [],
}) {
  return requestJson('/api/workspace/first_half_ou', {
    method: 'POST',
    body: JSON.stringify({
      match_id,
      home_team_id,
      away_team_id,
      home_team_name,
      away_team_name,
      league_id,
      line,
      limit,
      filters,
      evidenceFilters,
    }),
  })
}

export async function getWorkspaceFirstHalf1X2({
  match_id,
  home_team_id,
  away_team_id,
  home_team_name = '',
  away_team_name = '',
  league_id = '',
  limit = 0,
  filters = {},
  evidenceFilters = [],
}) {
  return requestJson('/api/workspace/first_half_1x2', {
    method: 'POST',
    body: JSON.stringify({
      match_id,
      home_team_id,
      away_team_id,
      home_team_name,
      away_team_name,
      league_id,
      limit,
      filters,
      evidenceFilters,
    }),
  })
}

export async function getWorkspaceWinBothHalves({
  match_id,
  home_team_id,
  away_team_id,
  home_team_name = '',
  away_team_name = '',
  league_id = '',
  limit = 0,
  filters = {},
  evidenceFilters = [],
}) {
  return requestJson('/api/workspace/win_both_halves', {
    method: 'POST',
    body: JSON.stringify({
      match_id,
      home_team_id,
      away_team_id,
      home_team_name,
      away_team_name,
      league_id,
      limit,
      filters,
      evidenceFilters,
    }),
  })
}

export async function getWorkspaceWinEitherHalf({
  match_id,
  home_team_id,
  away_team_id,
  home_team_name = '',
  away_team_name = '',
  league_id = '',
  limit = 0,
  filters = {},
  evidenceFilters = [],
}) {
  return requestJson('/api/workspace/win_either_half', {
    method: 'POST',
    body: JSON.stringify({
      match_id,
      home_team_id,
      away_team_id,
      home_team_name,
      away_team_name,
      league_id,
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
  home_team_name = '',
  away_team_name = '',
  league_id = '',
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
      home_team_name,
      away_team_name,
      league_id,
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
  home_team_name = '',
  away_team_name = '',
  league_id = '',
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
      home_team_name,
      away_team_name,
      league_id,
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
  home_team_name = '',
  away_team_name = '',
  league_id = '',
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
      home_team_name,
      away_team_name,
      league_id,
      line,
      limit,
      filters,
      evidenceFilters,
    }),
  })
}

export async function getWorkspaceHomeCorners({
  match_id,
  home_team_id,
  away_team_id,
  home_team_name = '',
  away_team_name = '',
  league_id = '',
  line = 4.5,
  limit = 0,
  filters = {},
  evidenceFilters = [],
}) {
  return requestJson('/api/workspace/home_corners', {
    method: 'POST',
    body: JSON.stringify({
      match_id,
      home_team_id,
      away_team_id,
      home_team_name,
      away_team_name,
      league_id,
      line,
      limit,
      filters,
      evidenceFilters,
    }),
  })
}

export async function getWorkspaceAwayCorners({
  match_id,
  home_team_id,
  away_team_id,
  home_team_name = '',
  away_team_name = '',
  league_id = '',
  line = 4.5,
  limit = 0,
  filters = {},
  evidenceFilters = [],
}) {
  return requestJson('/api/workspace/away_corners', {
    method: 'POST',
    body: JSON.stringify({
      match_id,
      home_team_id,
      away_team_id,
      home_team_name,
      away_team_name,
      league_id,
      line,
      limit,
      filters,
      evidenceFilters,
    }),
  })
}

/**
 * Stream a RAG query response from the backend.
 * @param {object} params - { query, home_team_id, away_team_id, extra_context }
 * @param {function} onChunk - called with each text chunk as it arrives
 * @returns {Promise<void>} resolves when the stream is complete
 */
export async function ragStream({ query, home_team_id, away_team_id, extra_context = '' }, onChunk) {
  const response = await fetch(`${API_BASE}/api/rag/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, home_team_id, away_team_id, extra_context }),
  })

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw new Error(payload.error || `Request failed (${response.status})`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    onChunk(decoder.decode(value, { stream: true }))
  }
}

/**
 * Stream a chat response that queries workspaces internally.
 * @param {object} params - { query, home_team_id, away_team_id, home_team_name, away_team_name, league_id }
 * @param {function} onChunk - called with each text chunk as it arrives
 * @returns {Promise<void>} resolves when the stream is complete
 */
export async function chatStream({ query, home_team_id, away_team_id, home_team_name = '', away_team_name = '', league_id = '' }, onChunk) {
  const response = await fetch(`${API_BASE}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, home_team_id, away_team_id, home_team_name, away_team_name, league_id }),
  })

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw new Error(payload.error || `Request failed (${response.status})`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    onChunk(decoder.decode(value, { stream: true }))
  }
}

/**
 * Upload a bet slip image for OCR + analysis.
 * Streams newline-delimited JSON events.
 * @param {File} imageFile - the image file to upload
 * @param {string} leagueId - optional league_id for team matching
 * @param {function} onEvent - called with each parsed JSON event
 * @returns {Promise<void>}
 */
export async function analyzeBetSlip(imageFile, leagueId = '', onEvent) {
  const formData = new FormData()
  formData.append('image', imageFile)
  if (leagueId) formData.append('league_id', leagueId)

  const response = await fetch(`${API_BASE}/api/betslip/analyze`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw new Error(payload.error || `Request failed (${response.status})`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // Split on newlines to parse NDJSON
    const lines = buffer.split('\n')
    buffer = lines.pop() // keep incomplete line in buffer
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed) continue
      try {
        onEvent(JSON.parse(trimmed))
      } catch {
        // skip unparseable lines
      }
    }
  }

  // Flush remaining buffer
  if (buffer.trim()) {
    try {
      onEvent(JSON.parse(buffer.trim()))
    } catch {
      // skip
    }
  }
}

export default {
  getLeagues,
  getFixturesForLeague,
  getMatch,
  getWorkspace1X2,
  getWorkspaceOverUnder,
  getWorkspaceDoubleChance,
  getWorkspaceBtts,
  getWorkspace1X2Ou,
  getWorkspaceDoubleChanceOu,
  getWorkspaceBttsOu,
  getWorkspaceFirstHalfOu,
  getWorkspaceFirstHalf1X2,
  getWorkspaceWinEitherHalf,
  getWorkspaceWinBothHalves,
  getWorkspaceHomeOu,
  getWorkspaceAwayOu,
  getWorkspaceCorners,
  getWorkspaceHomeCorners,
  getWorkspaceAwayCorners,
  ragStream,
  chatStream,
  analyzeBetSlip,
}
