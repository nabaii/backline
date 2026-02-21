from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd
from flask import Flask, jsonify, request

from backend.bet_type.double_chance.double_chance import DoubleChanceWorkspace
from backend.bet_type.corners.corners import CornerWorkspace
from backend.bet_type.over_under.over_under import OverUnderWorkspace
from backend.bet_type.one_x_two.one_x_two import OneXTwoWorkspace
from backend.builders.match_analytics_builder import MatchAnalyticsBuilder
from backend.contracts.evidence import EvidenceRequest
from backend.contracts.filter_spec import FilterSpec
from backend.filters.filters import (
    FieldTiltFilter,
    GoalsConceded,
    GoalsScored,
    OpponentPossessionFilter,
    OpponentMomentumFilter,
    OpponentXG,
    TeamShotXGFilter,
    TeamPossessionFilter,
    TeamMomentumFilter,
    TeamXG,
    TotalMatchGoals,
    VenueFilter,
)
from backend.store.analytics_store import AnalyticsStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
LEAGUES_DIR = DATA_DIR / "leagues"
FIXTURE_CSV_PATH = DATA_DIR / "EPL_fixture.csv"
SEASON_DATE_FALLBACK_PATH = DATA_DIR / "season_df.csv"
PRIMARY_SEASON_DATA_PATH = DATA_DIR / "season_df.csv"
LEGACY_SEASON_DATA_PATH = DATA_DIR / "season_df_v1.csv"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
CORNER_OVERRIDES_PATH = DATA_DIR / "corner_overrides.csv"

# â”€â”€ Multi-league registry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LEAGUE_REGISTRY: list[dict[str, str]] = [
    {"id": "england_premier_league",  "name": "Premier League",  "country": "ENG", "flag": "\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F"},
    {"id": "spain_la_liga",           "name": "La Liga",         "country": "ESP", "flag": "\U0001F1EA\U0001F1F8"},
    {"id": "germany_bundesliga",      "name": "Bundesliga",      "country": "GER", "flag": "\U0001F1E9\U0001F1EA"},
    {"id": "italy_serie_a",           "name": "Serie A",         "country": "ITA", "flag": "\U0001F1EE\U0001F1F9"},
    {"id": "france_ligue_1",          "name": "Ligue 1",         "country": "FRA", "flag": "\U0001F1EB\U0001F1F7"},
]
LEAGUE_ID_SET = {lg["id"] for lg in LEAGUE_REGISTRY}

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
FPL_REQUEST_TIMEOUT_SECONDS = 30
FPL_CACHE_TTL_SECONDS = 60
SOFASCORE_FIXTURE_CACHE_TTL_SECONDS = 600
SOFASCORE_LEAGUE_NAME_BY_ID: dict[str, str] = {
    "spain_la_liga": "Spain La Liga",
    "germany_bundesliga": "Germany Bundesliga",
    "italy_serie_a": "Italy Serie A",
    "france_ligue_1": "France Ligue 1",
}

TEAM_NAME_ALIASES = {
    "mancity": "manchestercity",
    "manutd": "manchesterunited",
    "leeds": "leedsunited",
    "newcastle": "newcastleunited",
    "nottmforest": "nottinghamforest",
    "spurs": "tottenhamhotspur",
    "westham": "westhamunited",
    "wolves": "wolverhampton",
    "brighton": "brightonhovealbion",
}

OPPONENT_RANK_FILTER_CONFIG: dict[str, dict[str, Any]] = {
    "opponent_rank_xgd_range": {
        "label": "xGD",
        "table_column": "expected_goal_difference",
        "ascending": False,
    },
    "opponent_rank_xgf_range": {
        "label": "xGF",
        "table_column": "expected_goals_for",
        "ascending": False,
    },
    "opponent_rank_xga_range": {
        "label": "xGA",
        "table_column": "expected_goals_against",
        "ascending": True,
    },
    "opponent_rank_position_range": {
        "label": "Position",
        "table_column": "rank",
        "ascending": True,
    },
    "opponent_rank_corners_range": {
        "label": "Corners",
        "table_column": "average_corners",
        "ascending": False,
    },
    "opponent_rank_momentum_range": {
        "label": "Momentum",
        "table_column": "momentum",
        "ascending": False,
    },
    "opponent_rank_possession_range": {
        "label": "Possession",
        "table_column": "average_possession",
        "ascending": False,
    },
}

DATE_COLUMN_CANDIDATES = (
    "match_datetime",
    "datetime",
    "date",
    "kickoff",
    "kickoff_time",
    "start_time",
    "startTimestamp",
    "start_timestamp",
)


@dataclass(frozen=True)
class FixtureRow:
    match_id: int
    league_id: str
    gameweek: int
    kickoff: str
    home_team_id: int
    away_team_id: int
    home_team_name: str
    away_team_name: str
    home_team_score: int | None
    away_team_score: int | None
    finished: bool


@dataclass(frozen=True)
class FixtureSnapshot:
    fixtures: list[FixtureRow]
    current_gameweek: int
    updated_at: str
    source: str


_snapshot_cache: dict[str, Any] = {
    "snapshot": None,
    "expires_at": datetime(1970, 1, 1, tzinfo=timezone.utc),
}
_sofascore_fixture_cache: dict[str, dict[str, Any]] = {}
_sofascore_season_cache: dict[str, dict[str, Any]] = {}


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_opt_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_utc_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        number = float(value)
        unit = "ms" if abs(number) >= 1e12 else "s"
        parsed = pd.to_datetime(number, unit=unit, utc=True, errors="coerce")
    else:
        text = str(value).strip()
        if not text:
            return None
        parsed = pd.NaT
        try:
            number = float(text)
            if pd.notna(number):
                unit = "ms" if abs(number) >= 1e12 else "s"
                parsed = pd.to_datetime(number, unit=unit, utc=True, errors="coerce")
        except (TypeError, ValueError):
            pass
        if pd.isna(parsed):
            parsed = pd.to_datetime(text, utc=True, errors="coerce")

    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _current_season_label(reference_time: datetime | None = None) -> str:
    now = reference_time or datetime.now(timezone.utc)
    start_year = now.year if now.month >= 7 else now.year - 1
    end_year = start_year + 1
    return f"{start_year % 100:02d}/{end_year % 100:02d}"


def _is_sofascore_event_finished(event: dict[str, Any]) -> bool:
    status = event.get("status")
    if isinstance(status, dict):
        status_type = str(status.get("type", "")).strip().lower()
        if status_type in {"notstarted", "inprogress"}:
            return False
        if status_type in {"finished", "canceled", "postponed"}:
            return True

    home_score = event.get("homeScore")
    away_score = event.get("awayScore")
    home_normaltime = _safe_opt_int((home_score or {}).get("normaltime") if isinstance(home_score, dict) else None)
    away_normaltime = _safe_opt_int((away_score or {}).get("normaltime") if isinstance(away_score, dict) else None)
    return home_normaltime is not None and away_normaltime is not None


def _is_sofascore_event_inprogress(event: dict[str, Any]) -> bool:
    status = event.get("status")
    if not isinstance(status, dict):
        return False
    return str(status.get("type", "")).strip().lower() == "inprogress"


def _sofascore_event_to_fixture_row(event: dict[str, Any], league_id: str) -> FixtureRow | None:
    match_id = _safe_int(event.get("id"), 0)
    if match_id <= 0:
        return None

    round_info = event.get("roundInfo")
    if not isinstance(round_info, dict):
        round_info = {}

    kickoff_dt = _parse_utc_datetime(event.get("startTimestamp"))
    if kickoff_dt is None:
        kickoff_dt = _parse_utc_datetime(event.get("startDate") or event.get("date") or event.get("time"))
    kickoff = kickoff_dt.isoformat().replace("+00:00", "Z") if kickoff_dt else ""

    home_team = event.get("homeTeam")
    away_team = event.get("awayTeam")
    if not isinstance(home_team, dict):
        home_team = {}
    if not isinstance(away_team, dict):
        away_team = {}

    home_score = event.get("homeScore")
    away_score = event.get("awayScore")
    if not isinstance(home_score, dict):
        home_score = {}
    if not isinstance(away_score, dict):
        away_score = {}

    return FixtureRow(
        match_id=match_id,
        league_id=league_id,
        gameweek=_safe_int(round_info.get("round"), 0),
        kickoff=kickoff,
        home_team_id=_safe_int(home_team.get("id"), 0),
        away_team_id=_safe_int(away_team.get("id"), 0),
        home_team_name=_format_team_name(str(home_team.get("name", ""))),
        away_team_name=_format_team_name(str(away_team.get("name", ""))),
        home_team_score=_safe_opt_int(home_score.get("normaltime")),
        away_team_score=_safe_opt_int(away_score.get("normaltime")),
        finished=_is_sofascore_event_finished(event),
    )


def _resolve_sofascore_season_context(league_name: str, season_label: str) -> tuple[int, int] | None:
    try:
        import ScraperFC as sfc
        from ScraperFC.sofascore import comps
    except ImportError:
        return None

    cache_key = f"{league_name}:{season_label}"
    cached = _sofascore_season_cache.get(cache_key)
    now = datetime.now(timezone.utc)
    if isinstance(cached, dict) and now < cached.get("expires_at", datetime(1970, 1, 1, tzinfo=timezone.utc)):
        tournament_id = _safe_int(cached.get("tournament_id"), 0)
        season_id = _safe_int(cached.get("season_id"), 0)
        if tournament_id > 0 and season_id > 0:
            return tournament_id, season_id

    if league_name not in comps:
        return None

    tournament_id = _safe_int(comps[league_name].get("SOFASCORE"), 0)
    if tournament_id <= 0:
        return None

    ss = sfc.Sofascore()
    valid_seasons = ss.get_valid_seasons(league_name)
    season_id = _safe_int(valid_seasons.get(season_label), 0)
    if season_id <= 0:
        if not valid_seasons:
            return None
        latest_label = sorted(valid_seasons.keys())[-1]
        season_id = _safe_int(valid_seasons[latest_label], 0)
    if season_id <= 0:
        return None

    _sofascore_season_cache[cache_key] = {
        "tournament_id": tournament_id,
        "season_id": season_id,
        "expires_at": now + timedelta(hours=6),
    }
    return tournament_id, season_id


def _fetch_sofascore_events(
    tournament_id: int,
    season_id: int,
    direction: str,
    max_pages: int = 1,
) -> list[dict[str, Any]]:
    try:
        from ScraperFC.sofascore import API_PREFIX
        from ScraperFC.utils import botasaurus_browser_get_json
    except ImportError:
        return []

    if direction not in {"last", "next"}:
        return []
    if tournament_id <= 0 or season_id <= 0:
        return []

    events: list[dict[str, Any]] = []
    for page in range(max(1, int(max_pages))):
        url = (
            f"{API_PREFIX}/unique-tournament/{tournament_id}/"
            f"season/{season_id}/events/{direction}/{page}"
        )
        payload = botasaurus_browser_get_json(url)
        page_events = payload.get("events") if isinstance(payload, dict) else None
        if not page_events:
            break
        events.extend(e for e in page_events if isinstance(e, dict))

    return events


def _get_live_sofascore_league_fixtures(
    league_id: str,
    reference_time: datetime | None = None,
) -> list[FixtureRow]:
    league_name = SOFASCORE_LEAGUE_NAME_BY_ID.get(league_id)
    if not league_name:
        return []

    try:
        now = datetime.now(timezone.utc)
        cached = _sofascore_fixture_cache.get(league_id)
        cached_expires_at = (
            cached.get("expires_at", datetime(1970, 1, 1, tzinfo=timezone.utc))
            if isinstance(cached, dict)
            else datetime(1970, 1, 1, tzinfo=timezone.utc)
        )
        if isinstance(cached, dict) and now < cached_expires_at:
            return list(cached.get("fixtures", []))

        season_label = _current_season_label(reference_time or now)
        season_context = _resolve_sofascore_season_context(
            league_name=league_name,
            season_label=season_label,
        )
        if season_context is None:
            return []
        tournament_id, season_id = season_context

        try:
            next_events = _fetch_sofascore_events(
                tournament_id=tournament_id,
                season_id=season_id,
                direction="next",
                max_pages=1,
            )
            last_events = _fetch_sofascore_events(
                tournament_id=tournament_id,
                season_id=season_id,
                direction="last",
                max_pages=1,
            )
        except Exception:
            return []

        # Keep upcoming fixtures plus currently in-progress events from the latest completed page.
        by_match_id: dict[int, dict[str, Any]] = {}
        for event in last_events:
            if not _is_sofascore_event_inprogress(event):
                continue
            match_id = _safe_int(event.get("id"), 0)
            if match_id > 0:
                by_match_id[match_id] = event
        for event in next_events:
            match_id = _safe_int(event.get("id"), 0)
            if match_id > 0:
                by_match_id[match_id] = event

        rows = [
            row
            for row in (
                _sofascore_event_to_fixture_row(event, league_id=league_id)
                for event in by_match_id.values()
            )
            if row is not None
        ]

        rows.sort(
            key=lambda row: (
                _parse_utc_datetime(row.kickoff) is None,
                _parse_utc_datetime(row.kickoff) or datetime.max.replace(tzinfo=timezone.utc),
                row.gameweek,
                row.home_team_name,
            )
        )

        _sofascore_fixture_cache[league_id] = {
            "fixtures": rows,
            "expires_at": now + timedelta(seconds=SOFASCORE_FIXTURE_CACHE_TTL_SECONDS),
        }
        return rows
    except Exception:
        # Never let third-party fixture lookups crash league fixture responses.
        return []


def _lookup_cached_live_fixture(match_id: int) -> FixtureRow | None:
    for entry in _sofascore_fixture_cache.values():
        fixtures = entry.get("fixtures", []) if isinstance(entry, dict) else []
        for fixture in fixtures:
            if fixture.match_id == match_id:
                return fixture
    return None


def _normalize_range(value: Any, default_low: float = 0.0, default_high: float = 10.0) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return (default_low, default_high)
    low = float(value[0])
    high = float(value[1])
    if low > high:
        low, high = high, low
    return (low, high)


@lru_cache(maxsize=1)
def _load_raw_df() -> pd.DataFrame:
    """Load and concatenate season data from all league directories."""
    frames: list[pd.DataFrame] = []

    if LEAGUES_DIR.exists():
        for league_dir in sorted(LEAGUES_DIR.iterdir()):
            csv_path = league_dir / "season_df.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["league_slug"] = league_dir.name
                frames.append(df)

    # Fallback: try legacy single-file path if no league folders found
    if not frames:
        data_path = PRIMARY_SEASON_DATA_PATH if PRIMARY_SEASON_DATA_PATH.exists() else LEGACY_SEASON_DATA_PATH
        if not data_path.exists():
            raise FileNotFoundError(
                f"No league data found in {LEAGUES_DIR} and no fallback at {PRIMARY_SEASON_DATA_PATH}"
            )
        df = pd.read_csv(data_path)
        df["league_slug"] = "england_premier_league"
        frames.append(df)

    raw_df = pd.concat(frames, ignore_index=True)
    if "match_id" not in raw_df.columns and "game_id" in raw_df.columns:
        raw_df = raw_df.rename(columns={"game_id": "match_id"})
    return raw_df


@lru_cache(maxsize=1)
def _build_store() -> AnalyticsStore:
    raw_df = _load_raw_df()
    builder = MatchAnalyticsBuilder()
    analytics = []
    for match_id in raw_df["match_id"].unique():
        match_df = raw_df[raw_df["match_id"] == match_id].copy()
        analytics.append(builder.build(match_df))

    store = AnalyticsStore()
    store.ingest(analytics)
    return store


def _format_team_name(raw: str) -> str:
    return raw.replace("_", " ").title()


def _normalize_team_name(value: str) -> str:
    compact = re.sub(r"[^a-z0-9]", "", str(value).lower())
    return TEAM_NAME_ALIASES.get(compact, compact)


def _http_json(url: str) -> Any:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=FPL_REQUEST_TIMEOUT_SECONDS) as response:
        return json.load(response)


def _current_gameweek_from_events(events: list[dict[str, Any]]) -> int:
    current = next((e for e in events if e.get("is_current")), None)
    if current:
        return int(current["id"])

    nxt = next((e for e in events if e.get("is_next")), None)
    if nxt:
        return int(nxt["id"])

    finished = [int(e["id"]) for e in events if e.get("finished")]
    if finished:
        return max(finished)
    return 1


def _fetch_live_fixture_snapshot() -> FixtureSnapshot:
    bootstrap = _http_json(FPL_BOOTSTRAP_URL)
    fixtures = _http_json(FPL_FIXTURES_URL)

    team_map = {int(t["id"]): t["name"] for t in bootstrap.get("teams", [])}
    current_gameweek = _current_gameweek_from_events(bootstrap.get("events", []))

    rows: list[FixtureRow] = []
    for item in fixtures:
        gameweek = _safe_int(item.get("event"), 0)
        if gameweek <= 0:
            continue

        home_team_id = _safe_int(item.get("team_h"), 0)
        away_team_id = _safe_int(item.get("team_a"), 0)
        rows.append(
            FixtureRow(
                match_id=_safe_int(item.get("id"), 0),
                league_id="england_premier_league",
                gameweek=gameweek,
                kickoff=item.get("kickoff_time") or "",
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                home_team_name=team_map.get(home_team_id, f"Team {home_team_id}"),
                away_team_name=team_map.get(away_team_id, f"Team {away_team_id}"),
                home_team_score=_safe_opt_int(item.get("team_h_score")),
                away_team_score=_safe_opt_int(item.get("team_a_score")),
                finished=bool(item.get("finished", False)),
            )
        )

    rows.sort(key=lambda r: (r.gameweek, r.kickoff or "9999-12-31T00:00:00Z", r.home_team_name))
    snapshot = FixtureSnapshot(
        fixtures=rows,
        current_gameweek=current_gameweek,
        updated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        source="live_fpl",
    )
    _write_fixture_csv(rows)
    return snapshot


def _write_fixture_csv(rows: list[FixtureRow]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "gameweek",
        "kickoff_time",
        "home_team",
        "away_team",
        "home_team_score",
        "away_team_score",
        "finished",
        "fixture_id",
        "league_id",
        "home_team_id",
        "away_team_id",
    ]
    with FIXTURE_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "gameweek": r.gameweek,
                    "kickoff_time": r.kickoff,
                    "home_team": r.home_team_name,
                    "away_team": r.away_team_name,
                    "home_team_score": "" if r.home_team_score is None else r.home_team_score,
                    "away_team_score": "" if r.away_team_score is None else r.away_team_score,
                    "finished": r.finished,
                    "fixture_id": r.match_id,
                    "league_id": r.league_id,
                    "home_team_id": r.home_team_id,
                    "away_team_id": r.away_team_id,
                }
            )


def _load_fixture_csv_snapshot() -> FixtureSnapshot | None:
    if not FIXTURE_CSV_PATH.exists():
        return None

    rows: list[FixtureRow] = []
    with FIXTURE_CSV_PATH.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            rows.append(
                FixtureRow(
                    match_id=_safe_int(item.get("fixture_id"), 0),
                    league_id=item.get("league_id") or "england_premier_league",
                    gameweek=_safe_int(item.get("gameweek"), 0),
                    kickoff=item.get("kickoff_time") or "",
                    home_team_id=_safe_int(item.get("home_team_id"), 0),
                    away_team_id=_safe_int(item.get("away_team_id"), 0),
                    home_team_name=item.get("home_team") or "",
                    away_team_name=item.get("away_team") or "",
                    home_team_score=_safe_opt_int(item.get("home_team_score")),
                    away_team_score=_safe_opt_int(item.get("away_team_score")),
                    finished=str(item.get("finished", "False")).lower() == "true",
                )
            )

    if not rows:
        return None

    return FixtureSnapshot(
        fixtures=rows,
        current_gameweek=_infer_gameweek(rows),
        updated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        source="csv_cache",
    )


@lru_cache(maxsize=1)
def _historical_fixture_snapshot() -> FixtureSnapshot:
    raw_df = _load_raw_df()
    needed_cols = ["match_id", "home_team_id", "away_team_id", "home_team", "away_team"]
    if "league_slug" in raw_df.columns:
        needed_cols.append("league_slug")
    if "match_datetime" in raw_df.columns:
        needed_cols.append("match_datetime")

    fixture_df = (
        raw_df[needed_cols]
        .drop_duplicates(subset=["match_id"])
    )

    # Sort by match_datetime descending (most recent first)
    if "match_datetime" in fixture_df.columns:
        fixture_df["_sort_dt"] = pd.to_datetime(fixture_df["match_datetime"], utc=True, errors="coerce")
        fixture_df = fixture_df.sort_values(by="_sort_dt", ascending=False, na_position="last")
        fixture_df = fixture_df.drop(columns=["_sort_dt"])
    else:
        fixture_df = fixture_df.sort_values(by="match_id", ascending=False)
    fixture_df = fixture_df.reset_index(drop=True)

    rows: list[FixtureRow] = []
    for idx, row in fixture_df.iterrows():
        league_id = str(row.get("league_slug", "england_premier_league"))

        # Try to use real kickoff datetime if available
        kickoff_str = None
        if "match_datetime" in row.index:
            try:
                kickoff_str = pd.Timestamp(row["match_datetime"]).isoformat().replace("+00:00", "Z")
            except Exception:
                pass
        if not kickoff_str:
            kickoff_str = ""

        rows.append(
            FixtureRow(
                match_id=int(row["match_id"]),
                league_id=league_id,
                gameweek=0,
                kickoff=kickoff_str,
                home_team_id=int(row["home_team_id"]),
                away_team_id=int(row["away_team_id"]),
                home_team_name=_format_team_name(str(row["home_team"])),
                away_team_name=_format_team_name(str(row["away_team"])),
                home_team_score=None,
                away_team_score=None,
                finished=True,
            )
        )

    return FixtureSnapshot(
        fixtures=rows,
        current_gameweek=0,
        updated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        source="historical_fallback",
    )


@lru_cache(maxsize=1)
def _historical_fixture_index() -> dict[int, FixtureRow]:
    return {fixture.match_id: fixture for fixture in _historical_fixture_snapshot().fixtures}


def _resolve_fixture_for_match(snapshot: FixtureSnapshot | None, match_id: int) -> FixtureRow | None:
    if match_id <= 0:
        return None

    # Prefer live non-EPL cache first so workspace requests stay aligned with
    # fixtures returned by `/api/fixtures` for non-Premier League leagues.
    fixture = _lookup_cached_live_fixture(match_id)
    if fixture is not None:
        return fixture

    if snapshot is not None:
        fixture = next((f for f in snapshot.fixtures if f.match_id == match_id), None)
        if fixture is not None:
            return fixture

    return _historical_fixture_index().get(match_id)


def _infer_gameweek(rows: list[FixtureRow]) -> int:
    active = sorted({r.gameweek for r in rows if r.gameweek > 0 and not r.finished})
    if active:
        return active[0]
    known = [r.gameweek for r in rows if r.gameweek > 0]
    if known:
        return max(known)
    return 1


@lru_cache(maxsize=1)
def _raw_team_name_index() -> dict[str, tuple[int, ...]]:
    raw_df = _load_raw_df()
    mapping: dict[str, set[int]] = {}

    for _, row in raw_df.iterrows():
        home_key = _normalize_team_name(str(row["home_team"]))
        away_key = _normalize_team_name(str(row["away_team"]))
        mapping.setdefault(home_key, set()).add(int(row["home_team_id"]))
        mapping.setdefault(away_key, set()).add(int(row["away_team_id"]))

    return {k: tuple(sorted(v)) for k, v in mapping.items()}


def _candidate_team_ids(team_id: int, team_name: str | None) -> list[int]:
    # Live fixture provider IDs can differ from local analytics IDs.
    # Prefer name-mapped local IDs first, then fallback to provided numeric ID.
    mapped_ids: list[int] = []
    if team_name:
        team_key = _normalize_team_name(team_name)
        mapped_ids = list(_raw_team_name_index().get(team_key, ()))

    if mapped_ids:
        return mapped_ids

    if team_id > 0:
        return [team_id]

    return []


def _merge_historical_for_other_leagues(snapshot: FixtureSnapshot) -> FixtureSnapshot:
    """Merge historical fixtures for non-EPL leagues into a snapshot that only has EPL."""
    existing_ids = {f.match_id for f in snapshot.fixtures}
    historical = _historical_fixture_snapshot()
    extra = [
        f for f in historical.fixtures
        if f.league_id != "england_premier_league" and f.match_id not in existing_ids
    ]
    if extra:
        merged_fixtures = list(snapshot.fixtures) + extra
        return FixtureSnapshot(
            fixtures=merged_fixtures,
            current_gameweek=snapshot.current_gameweek,
            updated_at=snapshot.updated_at,
            source=snapshot.source,
        )
    return snapshot


def _get_fixture_snapshot(force_refresh: bool = False) -> FixtureSnapshot:
    now = datetime.now(timezone.utc)
    cached = _snapshot_cache.get("snapshot")
    expires_at = _snapshot_cache.get("expires_at", datetime(1970, 1, 1, tzinfo=timezone.utc))

    if cached and (not force_refresh or now < expires_at):
        return cached

    if force_refresh:
        try:
            snapshot = _fetch_live_fixture_snapshot()
            snapshot = _merge_historical_for_other_leagues(snapshot)
            _snapshot_cache["snapshot"] = snapshot
            _snapshot_cache["expires_at"] = now + timedelta(seconds=FPL_CACHE_TTL_SECONDS)
            return snapshot
        except Exception:
            pass

    if cached:
        # Non-refresh calls should not block on network.
        return cached

    csv_snapshot = _load_fixture_csv_snapshot()
    if csv_snapshot:
        csv_snapshot = _merge_historical_for_other_leagues(csv_snapshot)
        _snapshot_cache["snapshot"] = csv_snapshot
        _snapshot_cache["expires_at"] = now + timedelta(seconds=FPL_CACHE_TTL_SECONDS)
        return csv_snapshot

    historical = _historical_fixture_snapshot()
    _snapshot_cache["snapshot"] = historical
    _snapshot_cache["expires_at"] = now + timedelta(seconds=FPL_CACHE_TTL_SECONDS)
    return historical


def _get_nonblocking_fixture_snapshot() -> FixtureSnapshot:
    """
    Returns the best available snapshot without forcing a live network fetch.
    Used by latency-sensitive endpoints like workspace filtering.
    """
    try:
        return _get_fixture_snapshot(force_refresh=False)
    except Exception:
        csv_snapshot = _load_fixture_csv_snapshot()
        if csv_snapshot:
            return csv_snapshot
        try:
            return _historical_fixture_snapshot()
        except Exception:
            return FixtureSnapshot(
                fixtures=[],
                current_gameweek=0,
                updated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                source="empty_fallback",
            )


def _get_live_refreshed_fixture_snapshot() -> FixtureSnapshot:
    """
    Forces a live refresh attempt; falls back safely to cached/local snapshot.
    """
    try:
        return _get_fixture_snapshot(force_refresh=True)
    except Exception:
        return _get_nonblocking_fixture_snapshot()


def _select_next_up_fixtures(
    league_fixtures: list[FixtureRow], reference_time: datetime, gameweeks_to_include: int = 2
) -> tuple[list[FixtureRow], int, list[int], bool]:
    if not league_fixtures:
        return [], 0, [], False

    unfinished = [f for f in league_fixtures if not f.finished]
    if not unfinished:
        return [], 0, [], False

    kickoff_lookup = {f.match_id: _parse_utc_datetime(f.kickoff) for f in unfinished}
    gameweeks_to_include = max(1, int(gameweeks_to_include))

    def _next_gameweeks_from_anchor(anchor_gameweek: int) -> list[int]:
        positive_gameweeks = sorted({f.gameweek for f in unfinished if f.gameweek > 0})
        if not positive_gameweeks:
            return [anchor_gameweek]

        if anchor_gameweek not in positive_gameweeks:
            positive_gameweeks.append(anchor_gameweek)
            positive_gameweeks = sorted(set(positive_gameweeks))

        anchor_index = positive_gameweeks.index(anchor_gameweek)
        return positive_gameweeks[anchor_index : anchor_index + gameweeks_to_include]

    def _select_by_gameweeks(gameweeks: list[int]) -> list[FixtureRow]:
        gameweek_set = set(gameweeks)
        selected = [f for f in unfinished if f.gameweek in gameweek_set]
        selected.sort(
            key=lambda row: (
                kickoff_lookup.get(row.match_id) is None,
                kickoff_lookup.get(row.match_id) or datetime.max.replace(tzinfo=timezone.utc),
                row.gameweek,
                row.home_team_name,
            )
        )
        return selected

    # 1) If matches are currently in progress, prioritize that gameweek.
    ongoing = [
        f
        for f in unfinished
        if kickoff_lookup.get(f.match_id) is not None and kickoff_lookup[f.match_id] <= reference_time
    ]
    if ongoing:
        positive_gameweeks = [f.gameweek for f in ongoing if f.gameweek > 0]
        target_gameweek = min(positive_gameweeks) if positive_gameweeks else ongoing[0].gameweek
        selected_gameweeks = _next_gameweeks_from_anchor(target_gameweek)
        selected = _select_by_gameweeks(selected_gameweeks)
        return selected, target_gameweek, selected_gameweeks, False

    # 2) Otherwise pick the earliest future kickoff and return that gameweek.
    future = [
        f
        for f in unfinished
        if kickoff_lookup.get(f.match_id) is not None and kickoff_lookup[f.match_id] >= reference_time
    ]
    if future:
        first_future = min(future, key=lambda row: kickoff_lookup[row.match_id])
        target_gameweek = first_future.gameweek
        selected_gameweeks = _next_gameweeks_from_anchor(target_gameweek)
        selected = _select_by_gameweeks(selected_gameweeks)
        return selected, target_gameweek, selected_gameweeks, False

    # 3) Fallback for fixtures without kickoff timestamps.
    positive_gameweeks = [f.gameweek for f in unfinished if f.gameweek > 0]
    target_gameweek = min(positive_gameweeks) if positive_gameweeks else unfinished[0].gameweek
    selected_gameweeks = _next_gameweeks_from_anchor(target_gameweek)
    selected = _select_by_gameweeks(selected_gameweeks)
    return selected, target_gameweek, selected_gameweeks, True


def _sort_fixtures_by_kickoff(rows: list[FixtureRow]) -> list[FixtureRow]:
    return sorted(
        rows,
        key=lambda row: (
            _parse_utc_datetime(row.kickoff) is None,
            _parse_utc_datetime(row.kickoff) or datetime.max.replace(tzinfo=timezone.utc),
            row.gameweek,
            row.home_team_name,
        ),
    )


def _upcoming_or_live_fixtures(
    rows: list[FixtureRow],
    reference_time: datetime,
    stale_grace_hours: int = 6,
) -> list[FixtureRow]:
    stale_cutoff = reference_time - timedelta(hours=max(0, int(stale_grace_hours)))
    filtered: list[FixtureRow] = []
    for row in rows:
        if row.finished:
            continue
        kickoff_dt = _parse_utc_datetime(row.kickoff)
        if kickoff_dt is not None and kickoff_dt < stale_cutoff:
            continue
        filtered.append(row)
    return _sort_fixtures_by_kickoff(filtered)


def _default_evidence_filters(filters_payload: dict[str, Any]) -> list[FilterSpec]:
    specs: list[FilterSpec] = []

    venue = filters_payload.get("home_away")
    if venue in {"home", "away"}:
        specs.append(VenueFilter.build(operator="==", value=venue))

    team_range = _normalize_range(filters_payload.get("team_momentum_range"))
    specs.append(TeamMomentumFilter.build(operator="between", value=team_range))

    opp_range = _normalize_range(filters_payload.get("opponent_momentum_range"))
    specs.append(OpponentMomentumFilter.build(operator="between", value=opp_range))

    total_goals_range = _normalize_range(filters_payload.get("total_match_goals_range"), default_low=0.0, default_high=10.0)
    specs.append(TotalMatchGoals.build(operator="between", value=total_goals_range))

    team_goals_range = _normalize_range(filters_payload.get("team_goals_range"), default_low=0.0, default_high=10.0)
    specs.append(GoalsScored.build(operator="between", value=team_goals_range))

    opposition_goals_range = _normalize_range(filters_payload.get("opposition_goals_range"), default_low=0.0, default_high=10.0)
    specs.append(GoalsConceded.build(operator="between", value=opposition_goals_range))

    team_xg_range = _normalize_range(filters_payload.get("team_xg_range"), default_low=0.0, default_high=5.0)
    specs.append(TeamXG.build(operator="between", value=team_xg_range))

    opposition_xg_range = _normalize_range(filters_payload.get("opposition_xg_range"), default_low=0.0, default_high=5.0)
    specs.append(OpponentXG.build(operator="between", value=opposition_xg_range))

    team_possession_range = _normalize_range(
        filters_payload.get("team_possession_range"),
        default_low=0.0,
        default_high=100.0,
    )
    specs.append(TeamPossessionFilter.build(operator="between", value=team_possession_range))

    opponent_possession_range = _normalize_range(
        filters_payload.get("opposition_possession_range"),
        default_low=0.0,
        default_high=100.0,
    )
    specs.append(OpponentPossessionFilter.build(operator="between", value=opponent_possession_range))

    field_tilt_range = _normalize_range(filters_payload.get("field_tilt_range"), default_low=0.0, default_high=1.0)
    specs.append(FieldTiltFilter.build(operator="between", value=field_tilt_range))

    shot_xg_threshold = _safe_float(filters_payload.get("shot_xg_threshold"), 0.0)
    shot_xg_min_shots = _safe_int(filters_payload.get("shot_xg_min_shots"), 0)
    if shot_xg_min_shots > 0:
        specs.append(
            TeamShotXGFilter.build(
                operator=">=",
                value={"min_xg": shot_xg_threshold, "min_shots": shot_xg_min_shots},
            )
        )

    return specs


def _parse_filters(payload: dict[str, Any]) -> list[FilterSpec]:
    evidence_filters = payload.get("evidenceFilters")
    if isinstance(evidence_filters, list):
        return [FilterSpec.from_dict(f) for f in evidence_filters if isinstance(f, dict)]

    filters_payload = payload.get("filters")
    if isinstance(filters_payload, dict):
        return _default_evidence_filters(filters_payload)

    return _default_evidence_filters({})


@lru_cache(maxsize=8)
def _load_league_table(league_id: str) -> pd.DataFrame:
    if not league_id:
        return pd.DataFrame()
    table_path = LEAGUES_DIR / str(league_id) / "league_table.csv"
    if not table_path.exists():
        return pd.DataFrame()
    try:
        table_df = pd.read_csv(table_path)
    except Exception:
        return pd.DataFrame()
    if table_df.empty or "team" not in table_df.columns:
        return pd.DataFrame()
    return table_df


def _normalize_rank_range(raw_value: Any, max_rank: int) -> tuple[int, int] | None:
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 2:
        return None
    try:
        low = int(float(raw_value[0]))
        high = int(float(raw_value[1]))
    except (TypeError, ValueError):
        return None
    if low > high:
        low, high = high, low
    low = max(1, min(max_rank, low))
    high = max(1, min(max_rank, high))
    if low > high:
        return None
    return (low, high)


def _parse_opponent_rank_filters(filters_payload: dict[str, Any], max_rank: int) -> dict[str, tuple[int, int]]:
    parsed: dict[str, tuple[int, int]] = {}
    for key in OPPONENT_RANK_FILTER_CONFIG:
        normalized_range = _normalize_rank_range(filters_payload.get(key), max_rank=max_rank)
        if normalized_range is not None and normalized_range != (1, max_rank):
            parsed[key] = normalized_range
    return parsed


def _build_team_rank_map(table_df: pd.DataFrame, filter_key: str) -> dict[str, int]:
    config = OPPONENT_RANK_FILTER_CONFIG.get(filter_key)
    if not config:
        return {}

    column = str(config["table_column"])
    if column not in table_df.columns:
        return {}

    team_series = table_df["team"].astype(str)
    if filter_key == "opponent_rank_position_range":
        rank_values = pd.to_numeric(table_df["rank"], errors="coerce")
    else:
        metric_values = pd.to_numeric(table_df[column], errors="coerce")
        rank_values = metric_values.rank(method="min", ascending=bool(config["ascending"]))

    mapping: dict[str, int] = {}
    for team_name, rank_value in zip(team_series.tolist(), rank_values.tolist()):
        if pd.isna(rank_value):
            continue
        normalized_team = _normalize_team_name(team_name)
        if normalized_team:
            mapping[normalized_team] = int(rank_value)
    return mapping


def _resolve_opponent_team_name_from_row(row: pd.Series, match_lookup: dict[int, tuple[str, str]]) -> str:
    match_id = _safe_int(row.get("match_id"), -1)
    home_name, away_name = match_lookup.get(match_id, ("", ""))
    venue = str(row.get("venue", "")).strip().lower()
    if venue == "home":
        return away_name
    if venue == "away":
        return home_name
    return ""


def _apply_opponent_rank_filters(
    df: pd.DataFrame,
    filters_payload: dict[str, Any],
    league_id: str | None,
    notes: list[str] | None = None,
    side_label: str = "team",
) -> pd.DataFrame:
    if df.empty or not isinstance(filters_payload, dict):
        return df

    resolved_league_id = str(league_id or "")
    if not resolved_league_id:
        return df

    table_df = _load_league_table(resolved_league_id)
    if table_df.empty:
        if notes is not None:
            notes.append(f"{side_label}_opponent_rank_filters_skipped=no_league_table")
        return df

    max_rank = int(len(table_df))
    if max_rank <= 0:
        return df

    rank_filters = _parse_opponent_rank_filters(filters_payload, max_rank=max_rank)
    if not rank_filters:
        return df

    match_lookup = _match_name_index()
    opponent_names = df.apply(lambda row: _resolve_opponent_team_name_from_row(row, match_lookup), axis=1)
    opponent_norm = opponent_names.map(_normalize_team_name)
    filtered_df = df.copy()
    filtered_opponent_norm = opponent_norm.copy()

    for filter_key, (low, high) in rank_filters.items():
        rank_map = _build_team_rank_map(table_df, filter_key)
        if not rank_map:
            if notes is not None:
                notes.append(f"{side_label}_{filter_key}_skipped=no_rank_map")
            continue

        before_count = int(len(filtered_df))
        rank_series = filtered_opponent_norm.map(rank_map)
        keep_mask = rank_series.between(low, high, inclusive="both").fillna(False)
        filtered_df = filtered_df.loc[keep_mask]
        filtered_opponent_norm = filtered_opponent_norm.loc[filtered_df.index]
        after_count = int(len(filtered_df))
        if notes is not None:
            notes.append(f"{side_label}_{filter_key}={low}-{high}:{before_count}->{after_count}")

    return filtered_df


def _build_match_datetime_index_from_df(df: pd.DataFrame) -> dict[int, pd.Timestamp]:
    if df.empty:
        return {}

    match_id_col = None
    for candidate in ("match_id", "game_id", "id"):
        if candidate in df.columns:
            match_id_col = candidate
            break
    if match_id_col is None:
        return {}

    date_col = next((c for c in DATE_COLUMN_CANDIDATES if c in df.columns), None)
    if date_col is None:
        return {}

    parsed = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    enriched = pd.DataFrame(
        {
            "match_id": pd.to_numeric(df[match_id_col], errors="coerce"),
            "match_datetime": parsed,
        }
    ).dropna(subset=["match_id", "match_datetime"])

    if enriched.empty:
        return {}

    enriched["match_id"] = enriched["match_id"].astype(int)
    enriched = enriched.sort_values("match_datetime").drop_duplicates(subset=["match_id"], keep="last")
    return {int(row["match_id"]): row["match_datetime"] for _, row in enriched.iterrows()}


@lru_cache(maxsize=1)
def _match_datetime_index() -> dict[int, pd.Timestamp]:
    raw_df = _load_raw_df()
    mapping = _build_match_datetime_index_from_df(raw_df)
    if mapping:
        return mapping

    # Local fallback if the primary analytics file lacks date columns.
    if SEASON_DATE_FALLBACK_PATH.exists():
        fallback_df = pd.read_csv(SEASON_DATE_FALLBACK_PATH)
        mapping = _build_match_datetime_index_from_df(fallback_df)
        if mapping:
            return mapping

    return {}


def _sort_team_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    sorted_df = df.copy()
    if "match_datetime" in sorted_df.columns:
        sorted_df["_sort_datetime"] = pd.to_datetime(sorted_df["match_datetime"], utc=True, errors="coerce")
    else:
        datetime_lookup = _match_datetime_index()
        sorted_df["_sort_datetime"] = sorted_df["match_id"].map(
            lambda value: datetime_lookup.get(_safe_int(value, -1))
        )

    sorted_df["_sort_match_id"] = pd.to_numeric(sorted_df.get("match_id"), errors="coerce")
    sorted_df = sorted_df.sort_values(
        by=["_sort_datetime", "_sort_match_id"],
        ascending=[True, True],
        na_position="last",
        kind="mergesort",
    )
    return sorted_df.drop(columns=["_sort_datetime", "_sort_match_id"], errors="ignore")


def _split_result_counts(df: pd.DataFrame) -> dict[str, int]:
    if "one_x_two_result" not in df.columns:
        return {"wins": 0, "draws": 0, "losses": 0}

    result_series = df["one_x_two_result"]
    wins = int((result_series == 1.0).sum())
    draws = int((result_series == 0.5).sum())
    losses = int((result_series == 0.1).sum())
    return {"wins": wins, "draws": draws, "losses": losses}


def _split_over_under_counts(df: pd.DataFrame, line: float) -> dict[str, int]:
    if "total_goals" not in df.columns:
        return {"over": 0, "under": 0}

    total_goals = pd.to_numeric(df["total_goals"], errors="coerce").fillna(0.0)
    over = int((total_goals > line).sum())
    under = int((total_goals <= line).sum())
    return {"over": over, "under": under}


def _split_over_under_counts_for_column(df: pd.DataFrame, line: float, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {"over": 0, "under": 0}

    goal_values = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    over = int((goal_values > line).sum())
    under = int((goal_values <= line).sum())
    return {"over": over, "under": under}


def _split_double_chance_counts(df: pd.DataFrame) -> dict[str, int]:
    if "double_chance_outcome" in df.columns:
        outcomes = pd.to_numeric(df["double_chance_outcome"], errors="coerce").fillna(0)
        hits = int((outcomes == 1).sum())
        misses = int((outcomes == 0).sum())
        return {"hits": hits, "misses": misses}

    if {"goals_scored", "opponent_goals"}.issubset(df.columns):
        goals_scored = pd.to_numeric(df["goals_scored"], errors="coerce").fillna(0)
        opponent_goals = pd.to_numeric(df["opponent_goals"], errors="coerce").fillna(0)
        hits = int((goals_scored >= opponent_goals).sum())
        misses = int((goals_scored < opponent_goals).sum())
        return {"hits": hits, "misses": misses}

    return {"hits": 0, "misses": 0}


def _split_btts_counts(df: pd.DataFrame) -> dict[str, int]:
    if {"goals_scored", "opponent_goals"}.issubset(df.columns):
        goals_scored = pd.to_numeric(df["goals_scored"], errors="coerce").fillna(0)
        opponent_goals = pd.to_numeric(df["opponent_goals"], errors="coerce").fillna(0)
        hits = int(((goals_scored > 0) & (opponent_goals > 0)).sum())
        misses = int(((goals_scored <= 0) | (opponent_goals <= 0)).sum())
        return {"hits": hits, "misses": misses}

    return {"hits": 0, "misses": 0}


def _coerce_numeric_series(df: pd.DataFrame, column_candidates: list[str]) -> pd.Series:
    for column in column_candidates:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype="float64")


@lru_cache(maxsize=1)
def _corner_total_override_by_match_id() -> dict[int, float]:
    if not CORNER_OVERRIDES_PATH.exists():
        return {}

    try:
        overrides_df = pd.read_csv(CORNER_OVERRIDES_PATH)
    except Exception:
        return {}

    if overrides_df.empty:
        return {}

    match_id_col = "match_id" if "match_id" in overrides_df.columns else None
    total_col = "total_corners" if "total_corners" in overrides_df.columns else None
    if match_id_col is None or total_col is None:
        return {}

    match_ids = pd.to_numeric(overrides_df[match_id_col], errors="coerce")
    totals = pd.to_numeric(overrides_df[total_col], errors="coerce")
    cleaned = pd.DataFrame({"match_id": match_ids, "total_corners": totals}).dropna(subset=["match_id", "total_corners"])
    if cleaned.empty:
        return {}

    cleaned["match_id"] = cleaned["match_id"].astype(int)
    return {
        int(row["match_id"]): float(row["total_corners"])
        for _, row in cleaned.iterrows()
    }


def _resolve_total_corners_series(df: pd.DataFrame) -> pd.Series:
    if "total_corners" in df.columns:
        resolved = pd.to_numeric(df["total_corners"], errors="coerce").fillna(0.0)
    else:
        home_series = _coerce_numeric_series(df, ["home_corners", "corners_home", "corner_kicks_home"])
        away_series = _coerce_numeric_series(df, ["away_corners", "corners_away", "corner_kicks_away"])
        resolved = (home_series + away_series).fillna(0.0)

    overrides = _corner_total_override_by_match_id()
    if overrides and "match_id" in df.columns:
        match_ids = pd.to_numeric(df["match_id"], errors="coerce")
        override_values = match_ids.map(lambda value: overrides.get(int(value)) if pd.notna(value) else None)
        override_mask = override_values.notna()
        if override_mask.any():
            resolved = resolved.copy()
            resolved.loc[override_mask] = override_values.loc[override_mask].astype(float)

    return resolved


def _resolve_total_corners_from_row(row: pd.Series) -> float:
    match_id = _safe_int(row.get("match_id"), -1)
    override = _corner_total_override_by_match_id().get(match_id)
    if override is not None:
        return float(override)

    total_corners = pd.to_numeric(row.get("total_corners"), errors="coerce")
    if pd.notna(total_corners):
        return float(total_corners)

    home_corners = 0.0
    for column in ("home_corners", "corners_home", "corner_kicks_home"):
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(value):
            home_corners = float(value)
            break

    away_corners = 0.0
    for column in ("away_corners", "corners_away", "corner_kicks_away"):
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(value):
            away_corners = float(value)
            break

    return home_corners + away_corners


def _split_corner_metrics(df: pd.DataFrame, line: float = 8.5) -> dict[str, float]:
    if df.empty:
        return {
            "matches": 0,
            "avg_total_corners": 0.0,
            "min_total_corners": 0.0,
            "max_total_corners": 0.0,
            "over": 0,
            "under": 0,
        }

    corners = _resolve_total_corners_series(df)
    if corners.empty:
        return {
            "matches": 0,
            "avg_total_corners": 0.0,
            "min_total_corners": 0.0,
            "max_total_corners": 0.0,
            "over": 0,
            "under": 0,
        }

    over = int((corners > line).sum())
    under = int((corners <= line).sum())

    return {
        "matches": int(len(corners)),
        "avg_total_corners": round(float(corners.mean()), 2),
        "min_total_corners": round(float(corners.min()), 2),
        "max_total_corners": round(float(corners.max()), 2),
        "over": over,
        "under": under,
    }


@lru_cache(maxsize=1)
def _match_name_index() -> dict[int, tuple[str, str]]:
    raw_df = _load_raw_df()
    match_df = raw_df[["match_id", "home_team", "away_team"]].drop_duplicates(subset=["match_id"])
    mapping: dict[int, tuple[str, str]] = {}
    for _, row in match_df.iterrows():
        match_id = int(row["match_id"])
        home_name = _format_team_name(str(row["home_team"]))
        away_name = _format_team_name(str(row["away_team"]))
        mapping[match_id] = (home_name, away_name)
    return mapping


def _safe_optional_float(value: Any) -> float | None:
    numeric = _safe_float(value, float("nan"))
    if pd.isna(numeric):
        return None
    return float(numeric)


def _resolve_team_opponent_values(
    row: pd.Series,
    venue: str,
    home_key: str,
    away_key: str,
) -> tuple[float | None, float | None]:
    home_value = _safe_optional_float(row.get(home_key))
    away_value = _safe_optional_float(row.get(away_key))
    if venue == "home":
        return home_value, away_value
    if venue == "away":
        return away_value, home_value
    return home_value, away_value


@lru_cache(maxsize=16)
def _opponent_rank_maps_for_league(league_id: str) -> dict[str, dict[str, int]]:
    resolved_league_id = str(league_id or "").strip()
    if not resolved_league_id:
        return {}

    table_df = _load_league_table(resolved_league_id)
    if table_df.empty:
        return {}

    return {
        key: _build_team_rank_map(table_df, key)
        for key in OPPONENT_RANK_FILTER_CONFIG
    }


def _build_overlay_metrics_from_row(
    row: pd.Series,
    venue: str,
    opponent_name: str,
    league_id: str | None = None,
) -> dict[str, Any]:
    normalized_venue = str(venue).strip().lower()
    team_momentum, opponent_momentum = _resolve_team_opponent_values(
        row, normalized_venue, "home_momentum", "away_momentum"
    )
    team_xg, opponent_xg = _resolve_team_opponent_values(
        row, normalized_venue, "expected_goals_home", "expected_goals_away"
    )
    team_possession, opponent_possession = _resolve_team_opponent_values(
        row, normalized_venue, "ball_possession_home", "ball_possession_away"
    )
    team_field_tilt, _ = _resolve_team_opponent_values(
        row, normalized_venue, "field_tilt_home", "field_tilt_away"
    )

    team_goals = _safe_optional_float(row.get("goals_scored"))
    opponent_goals = _safe_optional_float(row.get("opponent_goals"))
    total_goals = _safe_optional_float(row.get("total_goals"))
    if total_goals is None and team_goals is not None and opponent_goals is not None:
        total_goals = round(team_goals + opponent_goals, 2)

    overlay_payload: dict[str, Any] = {
        "team_momentum_range": team_momentum,
        "opponent_momentum_range": opponent_momentum,
        "total_match_goals_range": total_goals,
        "team_goals_range": team_goals,
        "opposition_goals_range": opponent_goals,
        "team_xg": team_xg,
        "opponent_xg": opponent_xg,
        "team_xg_range": team_xg,
        "opposition_xg_range": opponent_xg,
        "team_possession_range": team_possession,
        "opposition_possession_range": opponent_possession,
        "field_tilt_range": team_field_tilt,
    }

    resolved_league_id = str(league_id or row.get("league_id", "") or "").strip()
    opponent_norm = _normalize_team_name(opponent_name)
    if resolved_league_id and opponent_norm:
        rank_maps = _opponent_rank_maps_for_league(resolved_league_id)
        for filter_key, rank_map in rank_maps.items():
            rank_value = rank_map.get(opponent_norm)
            if rank_value is not None:
                overlay_payload[filter_key] = int(rank_value)

    return overlay_payload


def _build_recent_matches(df: pd.DataFrame, league_id: str | None = None) -> list[dict[str, Any]]:
    recent_df = df.copy()
    rows: list[dict[str, Any]] = []
    match_lookup = _match_name_index()
    match_datetime_lookup = _match_datetime_index()

    for _, row in recent_df.iterrows():
        result_value = float(row.get("one_x_two_result", 0.5))
        if result_value == 1.0:
            result = "W"
        elif result_value == 0.5:
            result = "D"
        else:
            result = "L"

        venue = str(row.get("venue", ""))
        home_momentum = float(row.get("home_momentum", 0.0))
        away_momentum = float(row.get("away_momentum", 0.0))
        match_id = int(row.get("match_id", 0))
        home_name, away_name = match_lookup.get(match_id, ("Home", "Away"))

        if venue == "home":
            team_name = home_name
            opponent_name = away_name
            chart_label = f"vs {opponent_name}"
            fixture_display = f"{team_name} vs {opponent_name}"
        else:
            team_name = away_name
            opponent_name = home_name
            chart_label = f"@ {opponent_name}"
            fixture_display = f"{opponent_name} vs {team_name}"

        match_datetime = match_datetime_lookup.get(match_id)
        match_date = ""
        if match_datetime is not None:
            try:
                ts = pd.Timestamp(match_datetime)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                match_date = ts.isoformat().replace("+00:00", "Z")
            except Exception:
                match_date = ""

        rows.append(
            {
                "match_id": match_id,
                "venue": venue,
                "opponent_id": int(row.get("opponent_id", 0)),
                "team_name": team_name,
                "opponent_name": opponent_name,
                "chart_label": chart_label,
                "fixture_display": fixture_display,
                "match_date": match_date,
                "result": result,
                "team_momentum": home_momentum if venue == "home" else away_momentum,
                "opponent_momentum": away_momentum if venue == "home" else home_momentum,
                **_build_overlay_metrics_from_row(
                    row,
                    venue=venue,
                    opponent_name=opponent_name,
                    league_id=league_id,
                ),
            }
        )
    return rows


def _build_recent_matches_over_under(
    df: pd.DataFrame,
    line: float,
    league_id: str | None = None,
) -> list[dict[str, Any]]:
    recent_df = df.copy()
    rows: list[dict[str, Any]] = []
    match_lookup = _match_name_index()
    match_datetime_lookup = _match_datetime_index()

    for _, row in recent_df.iterrows():
        venue = str(row.get("venue", ""))
        total_goals = float(row.get("total_goals", 0.0))
        match_id = int(row.get("match_id", 0))
        home_name, away_name = match_lookup.get(match_id, ("Home", "Away"))

        if venue == "home":
            team_name = home_name
            opponent_name = away_name
            chart_label = f"vs {opponent_name}"
            fixture_display = f"{team_name} vs {opponent_name}"
        else:
            team_name = away_name
            opponent_name = home_name
            chart_label = f"@ {opponent_name}"
            fixture_display = f"{opponent_name} vs {team_name}"

        match_datetime = match_datetime_lookup.get(match_id)
        match_date = ""
        if match_datetime is not None:
            try:
                ts = pd.Timestamp(match_datetime)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                match_date = ts.isoformat().replace("+00:00", "Z")
            except Exception:
                match_date = ""

        over_under_result = "O" if total_goals > line else "U"
        rows.append(
            {
                "match_id": match_id,
                "venue": venue,
                "opponent_id": int(row.get("opponent_id", 0)),
                "team_name": team_name,
                "opponent_name": opponent_name,
                "chart_label": chart_label,
                "fixture_display": fixture_display,
                "match_date": match_date,
                "total_goals": total_goals,
                "over_under_result": over_under_result,
                **_build_overlay_metrics_from_row(
                    row,
                    venue=venue,
                    opponent_name=opponent_name,
                    league_id=league_id,
                ),
            }
        )

    return rows


def _build_recent_matches_team_goals_over_under(
    df: pd.DataFrame,
    line: float,
    goals_column: str = "goals_scored",
    league_id: str | None = None,
) -> list[dict[str, Any]]:
    recent_df = df.copy()
    rows: list[dict[str, Any]] = []
    match_lookup = _match_name_index()
    match_datetime_lookup = _match_datetime_index()

    for _, row in recent_df.iterrows():
        venue = str(row.get("venue", ""))
        team_goals = float(row.get(goals_column, 0.0))
        match_id = int(row.get("match_id", 0))
        home_name, away_name = match_lookup.get(match_id, ("Home", "Away"))

        if venue == "home":
            team_name = home_name
            opponent_name = away_name
            chart_label = f"vs {opponent_name}"
            fixture_display = f"{team_name} vs {opponent_name}"
        else:
            team_name = away_name
            opponent_name = home_name
            chart_label = f"@ {opponent_name}"
            fixture_display = f"{opponent_name} vs {team_name}"

        match_datetime = match_datetime_lookup.get(match_id)
        match_date = ""
        if match_datetime is not None:
            try:
                ts = pd.Timestamp(match_datetime)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                match_date = ts.isoformat().replace("+00:00", "Z")
            except Exception:
                match_date = ""

        over_under_result = "O" if team_goals > line else "U"
        rows.append(
            {
                "match_id": match_id,
                "venue": venue,
                "opponent_id": int(row.get("opponent_id", 0)),
                "team_name": team_name,
                "opponent_name": opponent_name,
                "chart_label": chart_label,
                "fixture_display": fixture_display,
                "match_date": match_date,
                "team_goals": team_goals,
                "over_under_result": over_under_result,
                **_build_overlay_metrics_from_row(
                    row,
                    venue=venue,
                    opponent_name=opponent_name,
                    league_id=league_id,
                ),
            }
        )

    return rows


def _build_recent_matches_corners(
    df: pd.DataFrame,
    line: float = 8.5,
    league_id: str | None = None,
) -> list[dict[str, Any]]:
    recent_df = df.copy()
    rows: list[dict[str, Any]] = []
    match_lookup = _match_name_index()
    match_datetime_lookup = _match_datetime_index()

    for _, row in recent_df.iterrows():
        venue = str(row.get("venue", ""))
        match_id = int(row.get("match_id", 0))
        home_name, away_name = match_lookup.get(match_id, ("Home", "Away"))
        total_corners = _resolve_total_corners_from_row(row)

        if venue == "home":
            team_name = home_name
            opponent_name = away_name
            chart_label = f"vs {opponent_name}"
            fixture_display = f"{team_name} vs {opponent_name}"
        else:
            team_name = away_name
            opponent_name = home_name
            chart_label = f"@ {opponent_name}"
            fixture_display = f"{opponent_name} vs {team_name}"

        match_datetime = match_datetime_lookup.get(match_id)
        match_date = ""
        if match_datetime is not None:
            try:
                ts = pd.Timestamp(match_datetime)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                match_date = ts.isoformat().replace("+00:00", "Z")
            except Exception:
                match_date = ""

        corners_over_under_result = "O" if total_corners > line else "U"
        rows.append(
            {
                "match_id": match_id,
                "venue": venue,
                "opponent_id": int(row.get("opponent_id", 0)),
                "team_name": team_name,
                "opponent_name": opponent_name,
                "chart_label": chart_label,
                "fixture_display": fixture_display,
                "match_date": match_date,
                "total_corners": total_corners,
                "corners_over_under_result": corners_over_under_result,
                **_build_overlay_metrics_from_row(
                    row,
                    venue=venue,
                    opponent_name=opponent_name,
                    league_id=league_id,
                ),
            }
        )

    return rows


def _build_recent_matches_double_chance(
    df: pd.DataFrame,
    league_id: str | None = None,
) -> list[dict[str, Any]]:
    recent_df = df.copy()
    rows: list[dict[str, Any]] = []
    match_lookup = _match_name_index()
    match_datetime_lookup = _match_datetime_index()

    for _, row in recent_df.iterrows():
        goals_scored = float(row.get("goals_scored", 0.0))
        opponent_goals = float(row.get("opponent_goals", 0.0))
        if goals_scored > opponent_goals:
            result = "W"
        elif goals_scored == opponent_goals:
            result = "D"
        else:
            result = "L"

        venue = str(row.get("venue", ""))
        match_id = int(row.get("match_id", 0))
        home_name, away_name = match_lookup.get(match_id, ("Home", "Away"))

        if venue == "home":
            team_name = home_name
            opponent_name = away_name
            chart_label = f"vs {opponent_name}"
            fixture_display = f"{team_name} vs {opponent_name}"
        else:
            team_name = away_name
            opponent_name = home_name
            chart_label = f"@ {opponent_name}"
            fixture_display = f"{opponent_name} vs {team_name}"

        match_datetime = match_datetime_lookup.get(match_id)
        match_date = ""
        if match_datetime is not None:
            try:
                ts = pd.Timestamp(match_datetime)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                match_date = ts.isoformat().replace("+00:00", "Z")
            except Exception:
                match_date = ""

        is_hit = result in {"W", "D"}
        rows.append(
            {
                "match_id": match_id,
                "venue": venue,
                "opponent_id": int(row.get("opponent_id", 0)),
                "team_name": team_name,
                "opponent_name": opponent_name,
                "chart_label": chart_label,
                "fixture_display": fixture_display,
                "match_date": match_date,
                "result": result,
                "double_chance_result": "H" if is_hit else "M",
                "double_chance_value": 1.0 if is_hit else 0.1,
                **_build_overlay_metrics_from_row(
                    row,
                    venue=venue,
                    opponent_name=opponent_name,
                    league_id=league_id,
                ),
            }
        )

    return rows


def _build_recent_matches_btts(
    df: pd.DataFrame,
    league_id: str | None = None,
) -> list[dict[str, Any]]:
    recent_df = df.copy()
    rows: list[dict[str, Any]] = []
    match_lookup = _match_name_index()
    match_datetime_lookup = _match_datetime_index()

    for _, row in recent_df.iterrows():
        goals_scored = float(row.get("goals_scored", 0.0))
        opponent_goals = float(row.get("opponent_goals", 0.0))
        is_btts = goals_scored > 0 and opponent_goals > 0

        venue = str(row.get("venue", ""))
        match_id = int(row.get("match_id", 0))
        home_name, away_name = match_lookup.get(match_id, ("Home", "Away"))

        if venue == "home":
            team_name = home_name
            opponent_name = away_name
            chart_label = f"vs {opponent_name}"
            fixture_display = f"{team_name} vs {opponent_name}"
        else:
            team_name = away_name
            opponent_name = home_name
            chart_label = f"@ {opponent_name}"
            fixture_display = f"{opponent_name} vs {team_name}"

        match_datetime = match_datetime_lookup.get(match_id)
        match_date = ""
        if match_datetime is not None:
            try:
                ts = pd.Timestamp(match_datetime)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                match_date = ts.isoformat().replace("+00:00", "Z")
            except Exception:
                match_date = ""

        rows.append(
            {
                "match_id": match_id,
                "venue": venue,
                "opponent_id": int(row.get("opponent_id", 0)),
                "team_name": team_name,
                "opponent_name": opponent_name,
                "chart_label": chart_label,
                "fixture_display": fixture_display,
                "match_date": match_date,
                "goals_scored": goals_scored,
                "opponent_goals": opponent_goals,
                "btts_result": "Y" if is_btts else "N",
                **_build_overlay_metrics_from_row(
                    row,
                    venue=venue,
                    opponent_name=opponent_name,
                    league_id=league_id,
                ),
            }
        )

    return rows


def _resolve_team_anchor_match(
    raw_df: pd.DataFrame,
    team_id: int,
    team_name: str | None,
    preferred_perspective: str,
    league_id: str | None = None,
) -> tuple[int, str] | None:
    scoped_df = raw_df
    resolved_league_id = str(league_id or "").strip()
    if resolved_league_id and "league_slug" in raw_df.columns:
        league_matches = raw_df[raw_df["league_slug"].astype(str) == resolved_league_id]
        if not league_matches.empty:
            scoped_df = league_matches

    for candidate_id in _candidate_team_ids(team_id, team_name):
        if preferred_perspective == "home":
            preferred = scoped_df[scoped_df["home_team_id"] == candidate_id]
        else:
            preferred = scoped_df[scoped_df["away_team_id"] == candidate_id]

        if not preferred.empty:
            row = preferred.sort_values("match_id", ascending=False).iloc[0]
            return int(row["match_id"]), preferred_perspective

        fallback = scoped_df[
            (scoped_df["home_team_id"] == candidate_id) | (scoped_df["away_team_id"] == candidate_id)
        ]
        if not fallback.empty:
            row = fallback.sort_values("match_id", ascending=False).iloc[0]
            perspective = "home" if int(row["home_team_id"]) == candidate_id else "away"
            return int(row["match_id"]), perspective

    return None


def create_app() -> Flask:
    app = Flask(__name__)

    cors_origin = os.environ.get("CORS_ORIGIN", "*")

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = cors_origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return response

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/api/leagues", methods=["GET"])
    def get_leagues():
        snapshot = _get_nonblocking_fixture_snapshot()
        leagues_out = []
        for lg in LEAGUE_REGISTRY:
            leagues_out.append(
                {
                    "id": lg["id"],
                    "name": lg["name"],
                    "country": lg["country"],
                    "flag": lg["flag"],
                    "current_gameweek": snapshot.current_gameweek,
                }
            )
        return jsonify(
            {
                "leagues": leagues_out,
                "updated_at": snapshot.updated_at,
                "source": snapshot.source,
            }
        )

    @app.route("/api/fixtures", methods=["GET"])
    def get_fixtures():
        snapshot = _get_nonblocking_fixture_snapshot()
        league_id = request.args.get("league_id", "england_premier_league")
        gameweek_param = request.args.get("gameweek")
        requested_gameweek = _safe_int(gameweek_param, -1) if gameweek_param is not None else None
        requested_date = request.args.get("date")
        now_reference = datetime.now(timezone.utc)
        requested_reference = _parse_utc_datetime(requested_date)
        # Never use a reference time behind "now" to avoid surfacing stale fixtures.
        reference_time = max(now_reference, requested_reference) if requested_reference else now_reference
        response_source = snapshot.source
        response_updated_at = snapshot.updated_at

        league_fixtures = [f for f in snapshot.fixtures if f.league_id == league_id]
        if league_id != "england_premier_league":
            live_non_epl_fixtures = _get_live_sofascore_league_fixtures(
                league_id=league_id,
                reference_time=reference_time,
            )
            if live_non_epl_fixtures:
                league_fixtures = live_non_epl_fixtures
                response_source = "live_sofascore"
                response_updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        fixtures = league_fixtures
        fallback_used = False
        selected_gameweeks: list[int] = []

        if requested_gameweek is not None and requested_gameweek > 0:
            gameweek = requested_gameweek
            fixtures = [f for f in fixtures if f.gameweek == gameweek]
            fixtures = _upcoming_or_live_fixtures(fixtures, reference_time)
            selected_gameweeks = [gameweek] if fixtures else []
            if not fixtures:
                fixtures = _upcoming_or_live_fixtures(league_fixtures, reference_time)
                selected_gameweeks = sorted({f.gameweek for f in fixtures if f.gameweek > 0})
                if selected_gameweeks:
                    gameweek = selected_gameweeks[0]
                fallback_used = True
        else:
            fixtures, gameweek, selected_gameweeks, fallback_used = _select_next_up_fixtures(
                league_fixtures,
                reference_time,
                gameweeks_to_include=2,
            )
            if not fixtures:
                fixtures = _upcoming_or_live_fixtures(league_fixtures, reference_time)
                selected_gameweeks = sorted({f.gameweek for f in fixtures if f.gameweek > 0})
                gameweek = selected_gameweeks[0] if selected_gameweeks else snapshot.current_gameweek
                fallback_used = True

        payload = [
            {
                "match_id": f.match_id,
                "kickoff": f.kickoff,
                "gameweek": f.gameweek,
                "home_team_id": f.home_team_id,
                "home_team_name": f.home_team_name,
                "away_team_id": f.away_team_id,
                "away_team_name": f.away_team_name,
                "home_team_score": f.home_team_score,
                "away_team_score": f.away_team_score,
                "finished": f.finished,
            }
            for f in fixtures
        ]
        return jsonify(
            {
                "fixtures": payload,
                "gameweek": gameweek,
                "gameweeks": selected_gameweeks,
                "updated_at": response_updated_at,
                "source": response_source,
                "fallback_used": fallback_used,
            }
        )

    @app.route("/api/matches/<int:match_id>", methods=["GET"])
    def get_match(match_id: int):
        snapshot = _get_nonblocking_fixture_snapshot()
        fixture = _resolve_fixture_for_match(snapshot, match_id)
        if fixture is None:
            return jsonify({"error": f"Match {match_id} not found"}), 404

        return jsonify(
            {
                "match_id": fixture.match_id,
                "league_id": fixture.league_id,
                "gameweek": fixture.gameweek,
                "kickoff": fixture.kickoff,
                "home_team": {"team_id": fixture.home_team_id, "name": fixture.home_team_name},
                "away_team": {"team_id": fixture.away_team_id, "name": fixture.away_team_name},
            }
        )

    @app.route("/api/workspace/1x2", methods=["POST", "OPTIONS"])
    def get_workspace_1x2():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        requested_match_id = _safe_int(payload.get("match_id"), -1)
        if requested_match_id < 0:
            return jsonify({"error": "match_id is required"}), 400

        snapshot = _get_nonblocking_fixture_snapshot()
        fixture = _resolve_fixture_for_match(snapshot, requested_match_id)

        home_team_id = _safe_int(payload.get("home_team_id"), fixture.home_team_id if fixture else -1)
        away_team_id = _safe_int(payload.get("away_team_id"), fixture.away_team_id if fixture else -1)
        home_team_name = payload.get("home_team_name") or (fixture.home_team_name if fixture else "")
        away_team_name = payload.get("away_team_name") or (fixture.away_team_name if fixture else "")
        requested_league_id = str(payload.get("league_id", "") or "").strip()
        league_id_for_ranking = requested_league_id or (fixture.league_id if fixture is not None else "")
        if home_team_id < 0 or away_team_id < 0:
            return jsonify({"error": "home_team_id and away_team_id are required"}), 400

        home_df = pd.DataFrame()
        away_df = pd.DataFrame()
        notes: list[str] = ["real backend evidence request", f"fixtures_source={snapshot.source}"]

        try:
            filters = _parse_filters(payload)
            raw_df = _load_raw_df()
            home_anchor = _resolve_team_anchor_match(
                raw_df,
                home_team_id,
                home_team_name,
                "home",
                league_id=league_id_for_ranking,
            )
            away_anchor = _resolve_team_anchor_match(
                raw_df,
                away_team_id,
                away_team_name,
                "away",
                league_id=league_id_for_ranking,
            )

            store = _build_store()
            workspace = OneXTwoWorkspace(store, outcome_type="1")

            if home_anchor is not None:
                home_anchor_match_id, home_perspective = home_anchor
                home_evidence = workspace.get_evidence(
                    match_id=home_anchor_match_id,
                    bet_type="one_x_two",
                    filters=filters,
                    perspective=home_perspective,
                )
                home_df = home_evidence.df
                notes.append(f"home_anchor={home_anchor_match_id}:{home_perspective}")
            else:
                notes.append(f"home_team_unavailable={home_team_name or home_team_id}")

            if away_anchor is not None:
                away_anchor_match_id, away_perspective = away_anchor
                away_evidence = workspace.get_evidence(
                    match_id=away_anchor_match_id,
                    bet_type="one_x_two",
                    filters=filters,
                    perspective=away_perspective,
                )
                away_df = away_evidence.df
                notes.append(f"away_anchor={away_anchor_match_id}:{away_perspective}")
            else:
                notes.append(f"away_team_unavailable={away_team_name or away_team_id}")
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        home_df = _sort_team_history(home_df)
        away_df = _sort_team_history(away_df)
        ranking_filters_payload = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
        home_df = _apply_opponent_rank_filters(
            home_df,
            ranking_filters_payload,
            league_id_for_ranking,
            notes=notes,
            side_label="home",
        )
        away_df = _apply_opponent_rank_filters(
            away_df,
            ranking_filters_payload,
            league_id_for_ranking,
            notes=notes,
            side_label="away",
        )

        home_metrics = _split_result_counts(home_df)
        away_metrics = _split_result_counts(away_df)

        chart_series = [
            {
                "label": "Win",
                "home_count": home_metrics["wins"],
                "away_count": away_metrics["wins"],
                "total": home_metrics["wins"] + away_metrics["wins"],
            },
            {
                "label": "Draw",
                "home_count": home_metrics["draws"],
                "away_count": away_metrics["draws"],
                "total": home_metrics["draws"] + away_metrics["draws"],
            },
            {
                "label": "Loss",
                "home_count": home_metrics["losses"],
                "away_count": away_metrics["losses"],
                "total": home_metrics["losses"] + away_metrics["losses"],
            },
        ]

        home_sample_size = int(len(home_df))
        away_sample_size = int(len(away_df))
        primary_sample_size = home_sample_size if home_sample_size > 0 else away_sample_size

        response = {
            "workspace": {
                "bet_type": "1X2",
                "match_id": requested_match_id,
                "as_of": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
            "sample_size": primary_sample_size,
            "sample_sizes": {
                "home_team": home_sample_size,
                "away_team": away_sample_size,
            },
            "metrics": {
                "home_team": home_metrics,
                "away_team": away_metrics,
                "head_to_head": {"home_win": 0, "draw": 0, "away_win": 0},
            },
            "chartSeries": chart_series,
            "recent_matches": {
                "home": _build_recent_matches(home_df, league_id=league_id_for_ranking),
                "away": _build_recent_matches(away_df, league_id=league_id_for_ranking),
            },
            "notes": notes,
        }
        return jsonify(response)

    @app.route("/api/workspace/over_under", methods=["POST", "OPTIONS"])
    def get_workspace_over_under():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        requested_match_id = _safe_int(payload.get("match_id"), -1)
        if requested_match_id < 0:
            return jsonify({"error": "match_id is required"}), 400

        line = _safe_float(payload.get("line"), 2.5)

        snapshot = _get_nonblocking_fixture_snapshot()
        fixture = _resolve_fixture_for_match(snapshot, requested_match_id)

        home_team_id = _safe_int(payload.get("home_team_id"), fixture.home_team_id if fixture else -1)
        away_team_id = _safe_int(payload.get("away_team_id"), fixture.away_team_id if fixture else -1)
        home_team_name = payload.get("home_team_name") or (fixture.home_team_name if fixture else "")
        away_team_name = payload.get("away_team_name") or (fixture.away_team_name if fixture else "")
        requested_league_id = str(payload.get("league_id", "") or "").strip()
        league_id_for_ranking = requested_league_id or (fixture.league_id if fixture is not None else "")
        if home_team_id < 0 or away_team_id < 0:
            return jsonify({"error": "home_team_id and away_team_id are required"}), 400

        home_df = pd.DataFrame()
        away_df = pd.DataFrame()
        notes: list[str] = [
            "real backend evidence request",
            f"fixtures_source={snapshot.source}",
            f"line={line}",
        ]

        try:
            filters = _parse_filters(payload)
            raw_df = _load_raw_df()
            home_anchor = _resolve_team_anchor_match(
                raw_df,
                home_team_id,
                home_team_name,
                "home",
                league_id=league_id_for_ranking,
            )
            away_anchor = _resolve_team_anchor_match(
                raw_df,
                away_team_id,
                away_team_name,
                "away",
                league_id=league_id_for_ranking,
            )

            store = _build_store()
            workspace = OverUnderWorkspace(store)

            if home_anchor is not None:
                home_anchor_match_id, home_perspective = home_anchor
                home_evidence = workspace.get_evidence(
                    match_id=home_anchor_match_id,
                    bet_type="over_under",
                    filters=filters,
                    perspective=home_perspective,
                    line=line,
                )
                home_df = home_evidence.df
                notes.append(f"home_anchor={home_anchor_match_id}:{home_perspective}")
            else:
                notes.append(f"home_team_unavailable={home_team_name or home_team_id}")

            if away_anchor is not None:
                away_anchor_match_id, away_perspective = away_anchor
                away_evidence = workspace.get_evidence(
                    match_id=away_anchor_match_id,
                    bet_type="over_under",
                    filters=filters,
                    perspective=away_perspective,
                    line=line,
                )
                away_df = away_evidence.df
                notes.append(f"away_anchor={away_anchor_match_id}:{away_perspective}")
            else:
                notes.append(f"away_team_unavailable={away_team_name or away_team_id}")
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        home_df = _sort_team_history(home_df)
        away_df = _sort_team_history(away_df)
        ranking_filters_payload = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
        home_df = _apply_opponent_rank_filters(
            home_df,
            ranking_filters_payload,
            league_id_for_ranking,
            notes=notes,
            side_label="home",
        )
        away_df = _apply_opponent_rank_filters(
            away_df,
            ranking_filters_payload,
            league_id_for_ranking,
            notes=notes,
            side_label="away",
        )

        home_metrics = _split_over_under_counts(home_df, line)
        away_metrics = _split_over_under_counts(away_df, line)

        chart_series = [
            {
                "label": "Over",
                "home_count": home_metrics["over"],
                "away_count": away_metrics["over"],
                "total": home_metrics["over"] + away_metrics["over"],
            },
            {
                "label": "Under",
                "home_count": home_metrics["under"],
                "away_count": away_metrics["under"],
                "total": home_metrics["under"] + away_metrics["under"],
            },
        ]

        home_sample_size = int(len(home_df))
        away_sample_size = int(len(away_df))
        primary_sample_size = home_sample_size if home_sample_size > 0 else away_sample_size

        response = {
            "workspace": {
                "bet_type": "over_under",
                "match_id": requested_match_id,
                "line": line,
                "as_of": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
            "sample_size": primary_sample_size,
            "sample_sizes": {
                "home_team": home_sample_size,
                "away_team": away_sample_size,
            },
            "metrics": {
                "home_team": home_metrics,
                "away_team": away_metrics,
            },
            "chartSeries": chart_series,
            "recent_matches": {
                "home": _build_recent_matches_over_under(home_df, line, league_id=league_id_for_ranking),
                "away": _build_recent_matches_over_under(away_df, line, league_id=league_id_for_ranking),
            },
            "notes": notes,
        }
        return jsonify(response)

    @app.route("/api/workspace/double_chance", methods=["POST", "OPTIONS"])
    def get_workspace_double_chance():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        requested_match_id = _safe_int(payload.get("match_id"), -1)
        if requested_match_id < 0:
            return jsonify({"error": "match_id is required"}), 400

        snapshot = _get_nonblocking_fixture_snapshot()
        fixture = _resolve_fixture_for_match(snapshot, requested_match_id)

        home_team_id = _safe_int(payload.get("home_team_id"), fixture.home_team_id if fixture else -1)
        away_team_id = _safe_int(payload.get("away_team_id"), fixture.away_team_id if fixture else -1)
        home_team_name = payload.get("home_team_name") or (fixture.home_team_name if fixture else "")
        away_team_name = payload.get("away_team_name") or (fixture.away_team_name if fixture else "")
        requested_league_id = str(payload.get("league_id", "") or "").strip()
        league_id_for_ranking = requested_league_id or (fixture.league_id if fixture is not None else "")
        if home_team_id < 0 or away_team_id < 0:
            return jsonify({"error": "home_team_id and away_team_id are required"}), 400

        home_df = pd.DataFrame()
        away_df = pd.DataFrame()
        notes: list[str] = ["real backend evidence request", f"fixtures_source={snapshot.source}"]

        try:
            filters = _parse_filters(payload)
            raw_df = _load_raw_df()
            home_anchor = _resolve_team_anchor_match(
                raw_df,
                home_team_id,
                home_team_name,
                "home",
                league_id=league_id_for_ranking,
            )
            away_anchor = _resolve_team_anchor_match(
                raw_df,
                away_team_id,
                away_team_name,
                "away",
                league_id=league_id_for_ranking,
            )

            store = _build_store()
            workspace = DoubleChanceWorkspace(store)

            if home_anchor is not None:
                home_anchor_match_id, home_perspective = home_anchor
                home_evidence = workspace.get_evidence(
                    match_id=home_anchor_match_id,
                    bet_type="double_chance",
                    filters=filters,
                    perspective=home_perspective,
                )
                home_df = home_evidence.df
                notes.append(f"home_anchor={home_anchor_match_id}:{home_perspective}")
            else:
                notes.append(f"home_team_unavailable={home_team_name or home_team_id}")

            if away_anchor is not None:
                away_anchor_match_id, away_perspective = away_anchor
                away_evidence = workspace.get_evidence(
                    match_id=away_anchor_match_id,
                    bet_type="double_chance",
                    filters=filters,
                    perspective=away_perspective,
                )
                away_df = away_evidence.df
                notes.append(f"away_anchor={away_anchor_match_id}:{away_perspective}")
            else:
                notes.append(f"away_team_unavailable={away_team_name or away_team_id}")
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        home_df = _sort_team_history(home_df)
        away_df = _sort_team_history(away_df)
        ranking_filters_payload = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
        home_df = _apply_opponent_rank_filters(
            home_df,
            ranking_filters_payload,
            league_id_for_ranking,
            notes=notes,
            side_label="home",
        )
        away_df = _apply_opponent_rank_filters(
            away_df,
            ranking_filters_payload,
            league_id_for_ranking,
            notes=notes,
            side_label="away",
        )

        home_metrics = _split_double_chance_counts(home_df)
        away_metrics = _split_double_chance_counts(away_df)

        chart_series = [
            {
                "label": "Hit (Win/Draw)",
                "home_count": home_metrics["hits"],
                "away_count": away_metrics["hits"],
                "total": home_metrics["hits"] + away_metrics["hits"],
            },
            {
                "label": "Miss (Loss)",
                "home_count": home_metrics["misses"],
                "away_count": away_metrics["misses"],
                "total": home_metrics["misses"] + away_metrics["misses"],
            },
        ]

        home_sample_size = int(len(home_df))
        away_sample_size = int(len(away_df))
        primary_sample_size = home_sample_size if home_sample_size > 0 else away_sample_size

        response = {
            "workspace": {
                "bet_type": "double_chance",
                "match_id": requested_match_id,
                "as_of": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
            "sample_size": primary_sample_size,
            "sample_sizes": {
                "home_team": home_sample_size,
                "away_team": away_sample_size,
            },
            "metrics": {
                "home_team": home_metrics,
                "away_team": away_metrics,
            },
            "chartSeries": chart_series,
            "recent_matches": {
                "home": _build_recent_matches_double_chance(home_df, league_id=league_id_for_ranking),
                "away": _build_recent_matches_double_chance(away_df, league_id=league_id_for_ranking),
            },
            "notes": notes,
        }
        return jsonify(response)

    @app.route("/api/workspace/btts", methods=["POST", "OPTIONS"])
    def get_workspace_btts():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        requested_match_id = _safe_int(payload.get("match_id"), -1)
        if requested_match_id < 0:
            return jsonify({"error": "match_id is required"}), 400

        snapshot = _get_nonblocking_fixture_snapshot()
        fixture = _resolve_fixture_for_match(snapshot, requested_match_id)

        home_team_id = _safe_int(payload.get("home_team_id"), fixture.home_team_id if fixture else -1)
        away_team_id = _safe_int(payload.get("away_team_id"), fixture.away_team_id if fixture else -1)
        home_team_name = payload.get("home_team_name") or (fixture.home_team_name if fixture else "")
        away_team_name = payload.get("away_team_name") or (fixture.away_team_name if fixture else "")
        requested_league_id = str(payload.get("league_id", "") or "").strip()
        league_id_for_ranking = requested_league_id or (fixture.league_id if fixture is not None else "")
        if home_team_id < 0 or away_team_id < 0:
            return jsonify({"error": "home_team_id and away_team_id are required"}), 400

        home_df = pd.DataFrame()
        away_df = pd.DataFrame()
        notes: list[str] = ["real backend evidence request", f"fixtures_source={snapshot.source}"]

        try:
            filters = _parse_filters(payload)
            raw_df = _load_raw_df()
            home_anchor = _resolve_team_anchor_match(
                raw_df,
                home_team_id,
                home_team_name,
                "home",
                league_id=league_id_for_ranking,
            )
            away_anchor = _resolve_team_anchor_match(
                raw_df,
                away_team_id,
                away_team_name,
                "away",
                league_id=league_id_for_ranking,
            )

            store = _build_store()
            filter_validator = DoubleChanceWorkspace(store)
            filter_validator.validate_filters(filters)

            if home_anchor is not None:
                home_anchor_match_id, home_perspective = home_anchor
                home_request = EvidenceRequest(
                    match_id=home_anchor_match_id,
                    bet_type="btts",
                    perspective=home_perspective,
                    filters=filters,
                    required_features=["goals_scored", "opponent_goals"],
                )
                home_df = store.query(home_request).df
                notes.append(f"home_anchor={home_anchor_match_id}:{home_perspective}")
            else:
                notes.append(f"home_team_unavailable={home_team_name or home_team_id}")

            if away_anchor is not None:
                away_anchor_match_id, away_perspective = away_anchor
                away_request = EvidenceRequest(
                    match_id=away_anchor_match_id,
                    bet_type="btts",
                    perspective=away_perspective,
                    filters=filters,
                    required_features=["goals_scored", "opponent_goals"],
                )
                away_df = store.query(away_request).df
                notes.append(f"away_anchor={away_anchor_match_id}:{away_perspective}")
            else:
                notes.append(f"away_team_unavailable={away_team_name or away_team_id}")
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        home_df = _sort_team_history(home_df)
        away_df = _sort_team_history(away_df)
        ranking_filters_payload = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
        home_df = _apply_opponent_rank_filters(
            home_df,
            ranking_filters_payload,
            league_id_for_ranking,
            notes=notes,
            side_label="home",
        )
        away_df = _apply_opponent_rank_filters(
            away_df,
            ranking_filters_payload,
            league_id_for_ranking,
            notes=notes,
            side_label="away",
        )

        home_metrics = _split_btts_counts(home_df)
        away_metrics = _split_btts_counts(away_df)

        chart_series = [
            {
                "label": "BTTS Yes",
                "home_count": home_metrics["hits"],
                "away_count": away_metrics["hits"],
                "total": home_metrics["hits"] + away_metrics["hits"],
            },
            {
                "label": "BTTS No",
                "home_count": home_metrics["misses"],
                "away_count": away_metrics["misses"],
                "total": home_metrics["misses"] + away_metrics["misses"],
            },
        ]

        home_sample_size = int(len(home_df))
        away_sample_size = int(len(away_df))
        primary_sample_size = home_sample_size if home_sample_size > 0 else away_sample_size

        response = {
            "workspace": {
                "bet_type": "btts",
                "match_id": requested_match_id,
                "as_of": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
            "sample_size": primary_sample_size,
            "sample_sizes": {
                "home_team": home_sample_size,
                "away_team": away_sample_size,
            },
            "metrics": {
                "home_team": home_metrics,
                "away_team": away_metrics,
            },
            "chartSeries": chart_series,
            "recent_matches": {
                "home": _build_recent_matches_btts(home_df, league_id=league_id_for_ranking),
                "away": _build_recent_matches_btts(away_df, league_id=league_id_for_ranking),
            },
            "notes": notes,
        }
        return jsonify(response)

    @app.route("/api/workspace/home_ou", methods=["POST", "OPTIONS"])
    def get_workspace_home_ou():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        requested_match_id = _safe_int(payload.get("match_id"), -1)
        if requested_match_id < 0:
            return jsonify({"error": "match_id is required"}), 400

        line = _safe_float(payload.get("line"), 2.5)
        snapshot = _get_nonblocking_fixture_snapshot()
        fixture = _resolve_fixture_for_match(snapshot, requested_match_id)

        home_team_id = _safe_int(payload.get("home_team_id"), fixture.home_team_id if fixture else -1)
        away_team_id = _safe_int(payload.get("away_team_id"), fixture.away_team_id if fixture else -1)
        home_team_name = payload.get("home_team_name") or (fixture.home_team_name if fixture else "")
        requested_league_id = str(payload.get("league_id", "") or "").strip()
        league_id_for_ranking = requested_league_id or (fixture.league_id if fixture is not None else "")
        if home_team_id < 0 or away_team_id < 0:
            return jsonify({"error": "home_team_id and away_team_id are required"}), 400

        home_df = pd.DataFrame()
        notes: list[str] = [
            "real backend evidence request",
            f"fixtures_source={snapshot.source}",
            f"line={line}",
        ]

        try:
            filters = _parse_filters(payload)
            raw_df = _load_raw_df()
            home_anchor = _resolve_team_anchor_match(
                raw_df,
                home_team_id,
                home_team_name,
                "home",
                league_id=league_id_for_ranking,
            )

            store = _build_store()
            filter_validator = OverUnderWorkspace(store)
            filter_validator.validate_filters(filters)

            if home_anchor is not None:
                home_anchor_match_id, home_perspective = home_anchor
                home_request = EvidenceRequest(
                    match_id=home_anchor_match_id,
                    bet_type="home_ou",
                    perspective=home_perspective,
                    filters=filters,
                    required_features=["goals_scored", "opponent_goals"],
                )
                home_df = store.query(home_request).df
                notes.append(f"home_anchor={home_anchor_match_id}:{home_perspective}")
            else:
                notes.append(f"home_team_unavailable={home_team_name or home_team_id}")
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        home_df = _sort_team_history(home_df)
        away_df = pd.DataFrame()
        ranking_filters_payload = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
        home_df = _apply_opponent_rank_filters(
            home_df,
            ranking_filters_payload,
            league_id_for_ranking,
            notes=notes,
            side_label="home",
        )

        home_metrics = _split_over_under_counts_for_column(home_df, line, "goals_scored")
        away_metrics = {"over": 0, "under": 0}

        chart_series = [
            {
                "label": "Over",
                "home_count": home_metrics["over"],
                "away_count": away_metrics["over"],
                "total": home_metrics["over"],
            },
            {
                "label": "Under",
                "home_count": home_metrics["under"],
                "away_count": away_metrics["under"],
                "total": home_metrics["under"],
            },
        ]

        home_sample_size = int(len(home_df))
        away_sample_size = 0
        response = {
            "workspace": {
                "bet_type": "home_ou",
                "match_id": requested_match_id,
                "line": line,
                "as_of": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
            "sample_size": home_sample_size,
            "sample_sizes": {
                "home_team": home_sample_size,
                "away_team": away_sample_size,
            },
            "metrics": {
                "home_team": home_metrics,
                "away_team": away_metrics,
            },
            "chartSeries": chart_series,
            "recent_matches": {
                "home": _build_recent_matches_team_goals_over_under(
                    home_df,
                    line,
                    goals_column="goals_scored",
                    league_id=league_id_for_ranking,
                ),
                "away": [],
            },
            "notes": notes,
        }
        return jsonify(response)

    @app.route("/api/workspace/away_ou", methods=["POST", "OPTIONS"])
    def get_workspace_away_ou():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        requested_match_id = _safe_int(payload.get("match_id"), -1)
        if requested_match_id < 0:
            return jsonify({"error": "match_id is required"}), 400

        line = _safe_float(payload.get("line"), 2.5)
        snapshot = _get_nonblocking_fixture_snapshot()
        fixture = _resolve_fixture_for_match(snapshot, requested_match_id)

        home_team_id = _safe_int(payload.get("home_team_id"), fixture.home_team_id if fixture else -1)
        away_team_id = _safe_int(payload.get("away_team_id"), fixture.away_team_id if fixture else -1)
        away_team_name = payload.get("away_team_name") or (fixture.away_team_name if fixture else "")
        requested_league_id = str(payload.get("league_id", "") or "").strip()
        league_id_for_ranking = requested_league_id or (fixture.league_id if fixture is not None else "")
        if home_team_id < 0 or away_team_id < 0:
            return jsonify({"error": "home_team_id and away_team_id are required"}), 400

        away_df = pd.DataFrame()
        notes: list[str] = [
            "real backend evidence request",
            f"fixtures_source={snapshot.source}",
            f"line={line}",
        ]

        try:
            filters = _parse_filters(payload)
            raw_df = _load_raw_df()
            away_anchor = _resolve_team_anchor_match(
                raw_df,
                away_team_id,
                away_team_name,
                "away",
                league_id=league_id_for_ranking,
            )

            store = _build_store()
            filter_validator = OverUnderWorkspace(store)
            filter_validator.validate_filters(filters)

            if away_anchor is not None:
                away_anchor_match_id, away_perspective = away_anchor
                away_request = EvidenceRequest(
                    match_id=away_anchor_match_id,
                    bet_type="away_ou",
                    perspective=away_perspective,
                    filters=filters,
                    required_features=["goals_scored", "opponent_goals"],
                )
                away_df = store.query(away_request).df
                notes.append(f"away_anchor={away_anchor_match_id}:{away_perspective}")
            else:
                notes.append(f"away_team_unavailable={away_team_name or away_team_id}")
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        away_df = _sort_team_history(away_df)
        home_df = pd.DataFrame()
        ranking_filters_payload = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
        away_df = _apply_opponent_rank_filters(
            away_df,
            ranking_filters_payload,
            league_id_for_ranking,
            notes=notes,
            side_label="away",
        )

        home_metrics = {"over": 0, "under": 0}
        away_metrics = _split_over_under_counts_for_column(away_df, line, "goals_scored")

        chart_series = [
            {
                "label": "Over",
                "home_count": home_metrics["over"],
                "away_count": away_metrics["over"],
                "total": away_metrics["over"],
            },
            {
                "label": "Under",
                "home_count": home_metrics["under"],
                "away_count": away_metrics["under"],
                "total": away_metrics["under"],
            },
        ]

        home_sample_size = 0
        away_sample_size = int(len(away_df))
        response = {
            "workspace": {
                "bet_type": "away_ou",
                "match_id": requested_match_id,
                "line": line,
                "as_of": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
            "sample_size": away_sample_size,
            "sample_sizes": {
                "home_team": home_sample_size,
                "away_team": away_sample_size,
            },
            "metrics": {
                "home_team": home_metrics,
                "away_team": away_metrics,
            },
            "chartSeries": chart_series,
            "recent_matches": {
                "home": [],
                "away": _build_recent_matches_team_goals_over_under(
                    away_df,
                    line,
                    goals_column="goals_scored",
                    league_id=league_id_for_ranking,
                ),
            },
            "notes": notes,
        }
        return jsonify(response)

    @app.route("/api/workspace/corners", methods=["POST", "OPTIONS"])
    def get_workspace_corners():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        requested_match_id = _safe_int(payload.get("match_id"), -1)
        if requested_match_id < 0:
            return jsonify({"error": "match_id is required"}), 400

        line = _safe_float(payload.get("line"), 8.5)

        snapshot = _get_nonblocking_fixture_snapshot()
        fixture = _resolve_fixture_for_match(snapshot, requested_match_id)

        home_team_id = _safe_int(payload.get("home_team_id"), fixture.home_team_id if fixture else -1)
        away_team_id = _safe_int(payload.get("away_team_id"), fixture.away_team_id if fixture else -1)
        home_team_name = payload.get("home_team_name") or (fixture.home_team_name if fixture else "")
        away_team_name = payload.get("away_team_name") or (fixture.away_team_name if fixture else "")
        requested_league_id = str(payload.get("league_id", "") or "").strip()
        league_id_for_ranking = requested_league_id or (fixture.league_id if fixture is not None else "")
        if home_team_id < 0 or away_team_id < 0:
            return jsonify({"error": "home_team_id and away_team_id are required"}), 400

        home_df = pd.DataFrame()
        away_df = pd.DataFrame()
        notes: list[str] = ["real backend evidence request", f"fixtures_source={snapshot.source}", f"line={line}"]

        try:
            filters = _parse_filters(payload)
            raw_df = _load_raw_df()
            home_anchor = _resolve_team_anchor_match(
                raw_df,
                home_team_id,
                home_team_name,
                "home",
                league_id=league_id_for_ranking,
            )
            away_anchor = _resolve_team_anchor_match(
                raw_df,
                away_team_id,
                away_team_name,
                "away",
                league_id=league_id_for_ranking,
            )

            store = _build_store()
            workspace = CornerWorkspace(store)

            if home_anchor is not None:
                home_anchor_match_id, home_perspective = home_anchor
                home_evidence = workspace.get_evidence(
                    match_id=home_anchor_match_id,
                    bet_type="corners",
                    filters=filters,
                    perspective=home_perspective,
                    line=line,
                )
                home_df = home_evidence.df
                notes.append(f"home_anchor={home_anchor_match_id}:{home_perspective}")
            else:
                notes.append(f"home_team_unavailable={home_team_name or home_team_id}")

            if away_anchor is not None:
                away_anchor_match_id, away_perspective = away_anchor
                away_evidence = workspace.get_evidence(
                    match_id=away_anchor_match_id,
                    bet_type="corners",
                    filters=filters,
                    perspective=away_perspective,
                    line=line,
                )
                away_df = away_evidence.df
                notes.append(f"away_anchor={away_anchor_match_id}:{away_perspective}")
            else:
                notes.append(f"away_team_unavailable={away_team_name or away_team_id}")
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        home_df = _sort_team_history(home_df)
        away_df = _sort_team_history(away_df)
        ranking_filters_payload = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
        home_df = _apply_opponent_rank_filters(
            home_df,
            ranking_filters_payload,
            league_id_for_ranking,
            notes=notes,
            side_label="home",
        )
        away_df = _apply_opponent_rank_filters(
            away_df,
            ranking_filters_payload,
            league_id_for_ranking,
            notes=notes,
            side_label="away",
        )

        home_metrics = _split_corner_metrics(home_df, line)
        away_metrics = _split_corner_metrics(away_df, line)

        home_sample_size = int(len(home_df))
        away_sample_size = int(len(away_df))
        primary_sample_size = home_sample_size if home_sample_size > 0 else away_sample_size

        response = {
            "workspace": {
                "bet_type": "corners",
                "match_id": requested_match_id,
                "line": line,
                "as_of": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
            "sample_size": primary_sample_size,
            "sample_sizes": {
                "home_team": home_sample_size,
                "away_team": away_sample_size,
            },
            "metrics": {
                "home_team": home_metrics,
                "away_team": away_metrics,
            },
            "chartSeries": [
                {
                    "label": "Over",
                    "home_count": home_metrics["over"],
                    "away_count": away_metrics["over"],
                    "total": home_metrics["over"] + away_metrics["over"],
                },
                {
                    "label": "Under",
                    "home_count": home_metrics["under"],
                    "away_count": away_metrics["under"],
                    "total": home_metrics["under"] + away_metrics["under"],
                },
            ],
            "recent_matches": {
                "home": _build_recent_matches_corners(home_df, line, league_id=league_id_for_ranking),
                "away": _build_recent_matches_corners(away_df, line, league_id=league_id_for_ranking),
            },
            "notes": notes,
        }
        return jsonify(response)

    # â”€â”€ Serve frontend SPA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        from flask import send_from_directory
        if path and (FRONTEND_DIST / path).is_file():
            return send_from_directory(str(FRONTEND_DIST), path)
        index_path = FRONTEND_DIST / "index.html"
        if index_path.is_file():
            return send_from_directory(str(FRONTEND_DIST), "index.html")
        return "Frontend not built. Run: cd frontend && npm install && npm run build", 404

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

