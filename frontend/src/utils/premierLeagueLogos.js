/**
 * Team and league logo helpers.
 *
 * Local logo assets are preferred for all supported leagues.
 * Fallback is Sofascore CDN by team id.
 */

function localLogoPath(folder, encodedFileName) {
  return `/logos/${folder}/${encodedFileName}`
}

const TEAM_LOGOS = {
  // Premier League
  arsenal: '/logos/pl/arsenal.svg',
  astonvilla: '/logos/pl/aston-villa.svg',
  bournemouth: '/logos/pl/bournemouth.svg',
  brentford: '/logos/pl/brentford.svg',
  brighton: '/logos/pl/brighton.svg',
  burnley: '/logos/pl/burnley.svg',
  chelsea: '/logos/pl/chelsea.svg',
  crystalpalace: '/logos/pl/crystal-palace.svg',
  everton: '/logos/pl/everton.svg',
  fulham: '/logos/pl/fulham.svg',
  leedsunited: '/logos/pl/leeds-united.svg',
  liverpool: '/logos/pl/liverpool.svg',
  manchestercity: '/logos/pl/manchester-city.svg',
  manchesterunited: '/logos/pl/manchester-united.svg',
  newcastleunited: '/logos/pl/newcastle-united.svg',
  nottinghamforest: '/logos/pl/nottingham-forest.svg',
  sunderland: '/logos/pl/sunderland.svg',
  tottenhamhotspur: '/logos/pl/tottenham-hotspur.svg',
  westhamunited: '/logos/pl/west-ham-united.svg',
  wolverhamptonwanderers: '/logos/pl/wolverhampton-wanderers.svg',
  ipswich: '/logos/pl/ipswich.svg',
  leicester: '/logos/pl/leicester.svg',
  southampton: '/logos/pl/southampton.svg',

  // La Liga
  alaves: localLogoPath('la-liga', 'Alaves.svg'),
  athleticbilbao: localLogoPath('la-liga', 'Athletic%20Bilbao.svg'),
  atleticomadrid: localLogoPath('la-liga', 'Atl%C3%A9tico%20Madrid.svg'),
  barcelona: localLogoPath('la-liga', 'Barcelona.svg'),
  celtavigo: localLogoPath('la-liga', 'Celta%20de%20Vigo.svg'),
  elche: localLogoPath('la-liga', 'Elche.svg'),
  espanyol: localLogoPath('la-liga', 'Espanyol.svg'),
  getafe: localLogoPath('la-liga', 'Getafe%20CF.svg'),
  girona: localLogoPath('la-liga', 'Girona.svg'),
  levante: localLogoPath('la-liga', 'Levante.svg'),
  mallorca: localLogoPath('la-liga', 'Mallorca.svg'),
  osasuna: localLogoPath('la-liga', 'Osasuna.svg'),
  rayovallecano: localLogoPath('la-liga', 'Rayo%20Vallecano.svg'),
  realbetis: localLogoPath('la-liga', 'Real%20Betis.svg'),
  realmadrid: localLogoPath('la-liga', 'Real%20Madrid.svg'),
  realoviedo: localLogoPath('la-liga', 'Real%20Oviedo.svg'),
  realsociedad: localLogoPath('la-liga', 'Real%20Sociedad.svg'),
  sevilla: localLogoPath('la-liga', 'Sevilla%20FC.svg'),
  valencia: localLogoPath('la-liga', 'Valencia%20CF.svg'),
  villarreal: localLogoPath('la-liga', 'Villarreal%20CF.svg'),

  // Bundesliga
  augsburg: localLogoPath('bundesliga', 'Augsburg.svg'),
  bayerleverkusen: localLogoPath('bundesliga', 'Bayer%20Leverkusen.svg'),
  bayernmunich: localLogoPath('bundesliga', 'Bayern%20Munich.svg'),
  dortmund: localLogoPath('bundesliga', 'Dortmund.svg'),
  eintracht: localLogoPath('bundesliga', 'Eintracht.svg'),
  fcstpauli: localLogoPath('bundesliga', 'FC%20St.%20Pauli.svg'),
  hamburgersv: localLogoPath('bundesliga', 'Hamburger%20SV.svg'),
  heidenheim: localLogoPath('bundesliga', 'Heidenheim.svg'),
  koln: localLogoPath('bundesliga', 'K%C3%B6ln.svg'),
  mainz05: localLogoPath('bundesliga', 'Mainz%2005.svg'),
  rbleipzig: localLogoPath('bundesliga', 'RB%20Leipzig.svg'),
  scfreiburg: localLogoPath('bundesliga', 'SC%20Freiburg.svg'),
  tsghoffenheim: localLogoPath('bundesliga', 'TSG%201899%20Hoffenheim.svg'),
  unionberlin: localLogoPath('bundesliga', 'Union.svg'),
  stuttgart: localLogoPath('bundesliga', 'VFB%20Stuttgart.svg'),
  wolfsburg: localLogoPath('bundesliga', 'VfL%20Wolfsburg.svg'),
  werderbremen: localLogoPath('bundesliga', 'Werder.svg'),
  gladbach: localLogoPath('bundesliga', 'Borussia%20Monchengladbach.svg'),
  bochum: 'https://api.sofascore.app/api/v1/team/2538/image',
  hertha: 'https://api.sofascore.app/api/v1/team/2512/image',
  schalke: 'https://api.sofascore.app/api/v1/team/2528/image',

  // Serie A
  atalanta: localLogoPath('serie-a', 'Atalanta.svg'),
  bologna: localLogoPath('serie-a', 'Bologna.svg'),
  cagliari: localLogoPath('serie-a', 'Cagliari.svg'),
  como: localLogoPath('serie-a', 'Como.svg'),
  cremonese: localLogoPath('serie-a', 'Cremonese.svg'),
  fiorentina: localLogoPath('serie-a', 'Fiorentina.svg'),
  genoa: localLogoPath('serie-a', 'Genoa.svg'),
  hellasverona: localLogoPath('serie-a', 'Hellas%20Verona.svg'),
  inter: localLogoPath('serie-a', 'Inter.svg'),
  juventus: localLogoPath('serie-a', 'Juventus.svg'),
  lazio: localLogoPath('serie-a', 'Lazio.svg'),
  lecce: localLogoPath('serie-a', 'Lecce.svg'),
  milan: localLogoPath('serie-a', 'Milan.svg'),
  napoli: localLogoPath('serie-a', 'Napoli.svg'),
  parma: localLogoPath('serie-a', 'Parma.svg'),
  pisa: localLogoPath('serie-a', 'Pisa.svg'),
  roma: localLogoPath('serie-a', 'Roma.svg'),
  sassuolo: localLogoPath('serie-a', 'Sassuolo.svg'),
  torino: localLogoPath('serie-a', 'Torino.svg'),
  udinese: localLogoPath('serie-a', 'Udinese.svg'),

  // Ligue 1
  auxerre: localLogoPath('ligue-1', 'AJ_Auxerre-O1GzU9ZnJ_brandlogos.net.svg'),
  angers: localLogoPath('ligue-1', 'Angers_SCO-OmW0PIkbJ_brandlogos.net.svg'),
  asmonaco: localLogoPath('ligue-1', 'as-monaco-fc-logo-DB03F7SM_brandlogos.net.svg'),
  lorient: localLogoPath('ligue-1', 'FC_Lorient-OczEiMhMU_brandlogos.net.svg'),
  nantes: localLogoPath('ligue-1', 'FC_Nantes-O40KP04jQ_brandlogos.net.svg'),
  metz: localLogoPath('ligue-1', 'fc-metz-logo-brandlogos.net_r3b5pdfge.svg'),
  lehavre: localLogoPath('ligue-1', 'le-havre-ac-logo-brandlogos.net_z2lpewsrr.svg'),
  lille: localLogoPath('ligue-1', 'lille-osc-logo-402D6RgQ_brandlogos.net.svg'),
  nice: localLogoPath('ligue-1', 'OGC_Nice-OMlRYqcS4_brandlogos.net.svg'),
  olympiquelyonnais: localLogoPath('ligue-1', 'Olympique_Lyonnais-OJIZp0W43_brandlogos.net.svg'),
  olympiquedemarseille: localLogoPath('ligue-1', 'olympique-de-marseille-logo-brandlogos.net_hhg2rfa2f.svg'),
  parissaintgermain: localLogoPath('ligue-1', 'paris-saint-germain-logo-5B8D1w4P_brandlogos.net.svg'),
  parisfc: localLogoPath('ligue-1', 'Paris%20FC.svg'),
  rcstrasbourg: localLogoPath('ligue-1', 'RC_Strasbourg_Alsace-O1z8c5aEq_brandlogos.net.svg'),
  rclens: localLogoPath('ligue-1', 'rc-lens-logo-25602qz3_brandlogos.net.svg'),
  stadebrestois: localLogoPath('ligue-1', 'Stade_Brestois_29-OMU9U6TjL_brandlogos.net.svg'),
  staderennais: localLogoPath('ligue-1', 'Stade_Rennais_FC-ON3GFQ2QC_brandlogos.net.svg'),
  toulouse: localLogoPath('ligue-1', 'Toulouse_FC-OlJ6zglRa_brandlogos.net.svg'),
}

const TEAM_ALIASES = {
  // Common aliases
  city: 'manchestercity',
  manc: 'manchestercity',
  mancity: 'manchestercity',
  manchestercityfc: 'manchestercity',
  manutd: 'manchesterunited',
  manutdfc: 'manchesterunited',
  united: 'manchesterunited',
  newcastle: 'newcastleunited',
  newcastleutd: 'newcastleunited',
  nufc: 'newcastleunited',
  nottmforest: 'nottinghamforest',
  forest: 'nottinghamforest',
  spurs: 'tottenhamhotspur',
  tottenham: 'tottenhamhotspur',
  westham: 'westhamunited',
  wolves: 'wolverhamptonwanderers',
  wolverhampton: 'wolverhamptonwanderers',
  brightonhovealbion: 'brighton',
  brightonandhovealbion: 'brighton',
  leeds: 'leedsunited',
  leedsunitedfc: 'leedsunited',
  astonvillafc: 'astonvilla',
  ipswitchtownfc: 'ipswich',
  ipswichtown: 'ipswich',
  leicestercity: 'leicester',
  leicestercityfc: 'leicester',
  southamptonfc: 'southampton',

  // La Liga aliases
  deportivoalaves: 'alaves',
  athleticclub: 'athleticbilbao',
  atletico: 'atleticomadrid',
  atleticodemadrid: 'atleticomadrid',
  celtadevigo: 'celtavigo',
  getafecf: 'getafe',
  gironafc: 'girona',
  levanteud: 'levante',
  sevillafc: 'sevilla',
  valenciacf: 'valencia',
  villarrealcf: 'villarreal',

  // Bundesliga aliases
  fcaugsburg: 'augsburg',
  fca: 'augsburg',
  bayer04leverkusen: 'bayerleverkusen',
  bayerleverkusenfc: 'bayerleverkusen',
  bayern: 'bayernmunich',
  fcbayernmunchen: 'bayernmunich',
  fcbayernmunich: 'bayernmunich',
  bayernmunchen: 'bayernmunich',
  borussiadortmund: 'dortmund',
  bvb: 'dortmund',
  eintrachtfrankfurt: 'eintracht',
  stpauli: 'fcstpauli',
  hsv: 'hamburgersv',
  '1fcheidenheim': 'heidenheim',
  heidenheimfc: 'heidenheim',
  '1fckoln': 'koln',
  fckoln: 'koln',
  mainz: 'mainz05',
  '1fsvmainz05': 'mainz05',
  leipzig: 'rbleipzig',
  borussialeipzig: 'rbleipzig',
  rbleipzigfc: 'rbleipzig',
  freiburg: 'scfreiburg',
  scfreiburgfc: 'scfreiburg',
  hoffenheim: 'tsghoffenheim',
  tsg1899hoffenheim: 'tsghoffenheim',
  hoffenheimtsg: 'tsghoffenheim',
  union: 'unionberlin',
  '1fcunionberlin': 'unionberlin',
  unionberlinfc: 'unionberlin',
  vfbstuttgart: 'stuttgart',
  stuttgartvfb: 'stuttgart',
  vflwolfsburg: 'wolfsburg',
  wolfsburgvfl: 'wolfsburg',
  werder: 'werderbremen',
  svwerderbremen: 'werderbremen',
  werderbremensv: 'werderbremen',
  borussiamgladbach: 'gladbach',
  gladbach: 'gladbach',
  borussiamonchengladbach: 'gladbach',
  vflbochum: 'bochum',
  herthabsc: 'hertha',
  fcschalke04: 'schalke',

  // Serie A aliases
  hellas: 'hellasverona',
  hellasveronafc: 'hellasverona',

  // Ligue 1 aliases
  ajauxerre: 'auxerre',
  angerssco: 'angers',
  monaco: 'asmonaco',
  fclorient: 'lorient',
  fcnantes: 'nantes',
  fcmetz: 'metz',
  lehavreac: 'lehavre',
  lilleosc: 'lille',
  ogcnice: 'nice',
  lyon: 'olympiquelyonnais',
  ol: 'olympiquelyonnais',
  marseille: 'olympiquedemarseille',
  om: 'olympiquedemarseille',
  psg: 'parissaintgermain',
  rclensfc: 'rclens',
  strasbourg: 'rcstrasbourg',
  rcstrasbourgalsace: 'rcstrasbourg',
  brest: 'stadebrestois',
  stadebrestois29: 'stadebrestois',
  rennes: 'staderennais',
  staderennaisfc: 'staderennais',
  toulousefc: 'toulouse',
}

const LEAGUE_LOGO_LOCAL = {
  england_premier_league: localLogoPath('pl', 'Premier%20League.svg'),
  spain_la_liga: localLogoPath('la-liga', 'La%20Liga.svg'),
  germany_bundesliga: localLogoPath('bundesliga', 'Bundesliga.svg'),
  italy_serie_a: localLogoPath('serie-a', 'Seria%20A.svg'),
}

const LEAGUE_LOGO_MAP = {
  england_premier_league: 17,
  spain_la_liga: 8,
  germany_bundesliga: 35,
  italy_serie_a: 23,
  france_ligue_1: 34,
}

function normalizeTeamName(value) {
  return String(value || '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/&/g, 'and')
    .replace(/\bf\.?c\.?\b/g, '')
    .replace(/[^a-z0-9]/g, '')
}

function parseDate(value) {
  if (!value) return null
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return null
  return parsed
}

export function getTeamLogo(teamName, teamId) {
  const normalized = normalizeTeamName(teamName)
  const canonical = TEAM_ALIASES[normalized] || normalized
  const local = TEAM_LOGOS[canonical]
  if (local) return local

  if (teamId && teamId > 0) {
    return `https://api.sofascore.app/api/v1/team/${teamId}/image`
  }

  return null
}

export function getPremierLeagueLogo(teamName) {
  return getTeamLogo(teamName, null)
}

export function getLeagueLogo(leagueId) {
  const localLogo = LEAGUE_LOGO_LOCAL[leagueId]
  if (localLogo) return localLogo

  const tournamentId = LEAGUE_LOGO_MAP[leagueId]
  if (!tournamentId) return null
  return `https://api.sofascore.app/api/v1/unique-tournament/${tournamentId}/image`
}

export function formatPropsMadnessMatchDateParts(rawDate) {
  const parsed = parseDate(rawDate)
  if (!parsed) {
    return { month: '--', day: '--' }
  }
  return {
    month: parsed.toLocaleDateString(undefined, { month: 'short' }),
    day: parsed.toLocaleDateString(undefined, { day: '2-digit' }),
  }
}

export function formatShortMatchDate(rawDate) {
  const parsed = parseDate(rawDate)
  if (!parsed) return '--'
  return parsed.toLocaleDateString(undefined, {
    month: 'short',
    day: '2-digit',
  })
}

export function formatFixtureDate(rawDate) {
  const parsed = parseDate(rawDate)
  if (!parsed) return '--'
  return parsed.toLocaleDateString(undefined, {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
  })
}

export function formatFixtureTime(rawDate) {
  const parsed = parseDate(rawDate)
  if (!parsed) return '--:--'
  return parsed.toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit',
  })
}
