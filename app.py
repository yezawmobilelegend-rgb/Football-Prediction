from __future__ import annotations

"""Professional Football Prediction Dashboard V8.5

Providers:
- football-data.org v4: fixtures, recent finished results, standings
- The Odds API v4: multi-region h2h and totals odds (on-demand per match; auto on Top 10)
- Live Football API v1: match injuries, lineups, and available match statistics (on-demand)

Model:
- chronological Elo ratings
- league-relative venue-specific attack/defence goal-rate xG proxy (attacking power = 0.75)
- Dixon-Coles-adjusted Poisson score matrix + Top 2 & Top 3 exact scores + Goal Difference probabilities
- short-rest, injuries, lineup status, manual tactical/motivation, and optional odds blend
"""

import math
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Football Prediction Dashboard V8.5",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

UTC = timezone.utc
LOCAL_TZ = ZoneInfo("Asia/Yangon")
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
LIVE_FOOTBALL_API_BASES = [
    "https://live-football-api.com/api/v1",
    "https://api.live-football-api.com/v1",
]
REQUEST_TIMEOUT = 18
MODEL_VERSION = "elo-dixon-coles-v8.5"
DIXON_COLES_RHO = -0.04
HOME_ELO_ADVANTAGE = 55.0
DEFAULT_ATTACK_POWER = 0.75
MAX_GOALS = 8

TEAM_ALIASES: Dict[str, str] = {
    "1fckoln": "koln",
    "fckoln": "koln",
    "koln": "koln",
    "cologne": "koln",
    "fccologne": "koln",
    "1fccologne": "koln",
    "fcbayernmunchen": "bayernmunich",
    "bayernmunchen": "bayernmunich",
    "fcbayernmunich": "bayernmunich",
    "bayernmunich": "bayernmunich",
    "bayern": "bayernmunich",
    "borussiamonchengladbach": "monchengladbach",
    "monchengladbach": "monchengladbach",
    "gladbach": "monchengladbach",
    "bvb": "borussiadortmund",
    "borussiadortmund": "borussiadortmund",
    "rbleipzig": "leipzig",
    "rasenballsportleipzig": "leipzig",
    "leipzig": "leipzig",
    "bayer04leverkusen": "bayerleverkusen",
    "bayerleverkusen": "bayerleverkusen",
    "tsg1899hoffenheim": "hoffenheim",
    "tsghoffenheim": "hoffenheim",
    "hoffenheim": "hoffenheim",
    "vfbstuttgart": "stuttgart",
    "stuttgart": "stuttgart",
    "vflwolfsburg": "wolfsburg",
    "wolfsburg": "wolfsburg",
    "svwerderbremen": "werderbremen",
    "werderbremen": "werderbremen",
    "1fsvmainz05": "mainz05",
    "mainz05": "mainz05",
    "mainz": "mainz05",
    "fcaugsburg": "augsburg",
    "augsburg": "augsburg",
    "1fcunionberlin": "unionberlin",
    "unionberlin": "unionberlin",
    "fcschalke04": "schalke04",
    "schalke04": "schalke04",
    "schalke": "schalke04",
    "hamburgersv": "hamburg",
    "hamburger": "hamburg",
    "hamburg": "hamburg",
    "manchestercity": "mancity",
    "mancity": "mancity",
    "manchesterunited": "manutd",
    "manutd": "manutd",
    "tottenhamhotspur": "tottenham",
    "spurs": "tottenham",
    "tottenham": "tottenham",
    "wolverhamptonwanderers": "wolves",
    "wolves": "wolves",
    "brightonandhovealbion": "brighton",
    "brighton": "brighton",
    "westhamunited": "westham",
    "westham": "westham",
    "newcastleunited": "newcastle",
    "newcastle": "newcastle",
    "nottinghamforest": "nottinghamforest",
    "nottmforest": "nottinghamforest",
    "leicestercity": "leicester",
    "leicester": "leicester",
    "intermilan": "inter",
    "internazionale": "inter",
    "inter": "inter",
    "acmilan": "milan",
    "milan": "milan",
    "asroma": "roma",
    "roma": "roma",
    "sscnapoli": "napoli",
    "napoli": "napoli",
    "atleticomadrid": "atleticomadrid",
    "atleticodemadrid": "atleticomadrid",
    "athleticbilbao": "athleticclub",
    "athleticclub": "athleticclub",
    "realbetis": "betis",
    "betis": "betis",
    "realsociedad": "sociedad",
    "sociedad": "sociedad",
    "parissaintgermain": "psg",
    "psg": "psg",
    "olympiquelyonnais": "lyon",
    "olympiquelyon": "lyon",
    "lyon": "lyon",
    "olympiquedemarseille": "marseille",
    "marseille": "marseille",
    "asmonaco": "monaco",
    "monaco": "monaco",
}

LEAGUES: Dict[str, Dict[str, Any]] = {
    "England Premier League": {"football_data": "PL", "odds": "soccer_epl", "lfa_league": "premier-league"},
    "Spain LaLiga": {"football_data": "PD", "odds": "soccer_spain_la_liga", "lfa_league": "la-liga"},
    "Germany Bundesliga": {"football_data": "BL1", "odds": "soccer_germany_bundesliga", "lfa_league": "bundesliga"},
    "Italy Serie A": {"football_data": "SA", "odds": "soccer_italy_serie_a", "lfa_league": "serie-a"},
    "France Ligue 1": {"football_data": "FL1", "odds": "soccer_france_ligue_one", "lfa_league": "ligue-1"},
    "UEFA Champions League": {"football_data": "CL", "odds": "soccer_uefa_champs_league", "lfa_league": "champions-league"},
    "UEFA Europa League": {"football_data": "EL", "odds": "soccer_uefa_europa_league", "lfa_league": "europa-league"},
}


# -----------------------------
# Generic helpers
# -----------------------------


def safe_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return str(value or "")


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except (TypeError, ValueError):
        return None


def local_datetime(value: Any) -> str:
    parsed = parse_datetime(value)
    return parsed.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M MMT") if parsed else "—"


def normalize_team(name: Any) -> str:
    text = unicodedata.normalize("NFKD", str(name or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\b(fc|afc|cf|ssc|sc|club|1\.|04|05)\b", " ", text)
    cleaned = re.sub(r"[^a-z0-9]", "", text)
    return TEAM_ALIASES.get(cleaned, cleaned)


def same_team(left: Any, right: Any) -> bool:
    a, b = normalize_team(left), normalize_team(right)
    return bool(a and b and (a == b or (len(a) > 5 and a in b) or (len(b) > 5 and b in a)))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def error_message(response: requests.Response) -> str:
    messages = {
        401: "API key is invalid or missing.",
        403: "API key does not have permission for this resource.",
        404: "API endpoint or competition was not found.",
        429: "Provider rate limit or quota reached. Wait or check your plan.",
    }
    try:
        detail = response.json()
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("errors") or detail.get("error")
        else:
            detail = None
    except ValueError:
        detail = None
    base = messages.get(response.status_code, f"HTTP {response.status_code}")
    return f"{base} {detail}" if detail else base


def http_json(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Any], Optional[str], Dict[str, str]]:
    try:
        response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        response_headers = {k.lower(): v for k, v in response.headers.items()}
        if response.status_code != 200:
            return None, error_message(response), response_headers
        return response.json(), None, response_headers
    except requests.Timeout:
        return None, "API request timed out.", {}
    except (requests.RequestException, ValueError) as exc:
        return None, f"API/network error: {exc}", {}


# -----------------------------
# Provider 1: football-data.org v4
# -----------------------------


@st.cache_data(ttl=300, show_spinner=False)
def fetch_fixtures(competition_code: str, api_key: str, date_from: str, date_to: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not api_key:
        return [], "Football-data.org API key is not configured."
    payload, error, _ = http_json(
        f"{FOOTBALL_DATA_BASE}/competitions/{competition_code}/matches",
        headers={"X-Auth-Token": api_key},
        params={"dateFrom": date_from, "dateTo": date_to},
    )
    if error:
        return [], error
    return (payload or {}).get("matches", []), None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_finished_matches(competition_code: str, api_key: str, date_from: str, date_to: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not api_key:
        return [], "Football-data.org API key is not configured."
    payload, error, _ = http_json(
        f"{FOOTBALL_DATA_BASE}/competitions/{competition_code}/matches",
        headers={"X-Auth-Token": api_key},
        params={"dateFrom": date_from, "dateTo": date_to, "status": "FINISHED"},
    )
    if error:
        return [], error
    matches = (payload or {}).get("matches", [])
    return [m for m in matches if m.get("status") == "FINISHED"], None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_standings(competition_code: str, api_key: str) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    if not api_key:
        return {}, "Football-data.org API key is not configured."
    payload, error, _ = http_json(
        f"{FOOTBALL_DATA_BASE}/competitions/{competition_code}/standings",
        headers={"X-Auth-Token": api_key},
    )
    if error:
        return {}, error
    standings: Dict[str, Dict[str, Any]] = {}
    for block in (payload or {}).get("standings", []):
        if block.get("type") != "TOTAL":
            continue
        for row in block.get("table", []):
            name = row.get("team", {}).get("name")
            if name:
                standings[normalize_team(name)] = {
                    "name": name,
                    "position": row.get("position", 0),
                    "points": row.get("points", 0),
                    "played": row.get("playedGames", 0),
                    "goal_difference": row.get("goalDifference", 0),
                    "goals_for": row.get("goalsFor", 0),
                    "goals_against": row.get("goalsAgainst", 0),
                }
    return standings, None


def fixture_window(days_ahead: int = 7) -> Tuple[str, str, datetime, datetime]:
    now = utc_now()
    start = now - timedelta(hours=12)
    end = now + timedelta(days=days_ahead)
    return start.date().isoformat(), (end.date() + timedelta(days=1)).isoformat(), start, end


def finished_window(days_back: int = 210) -> Tuple[str, str]:
    today = utc_now().date()
    return (today - timedelta(days=days_back)).isoformat(), (today + timedelta(days=1)).isoformat()


def upcoming_only(matches: Iterable[Dict[str, Any]], start: datetime, end: datetime) -> List[Dict[str, Any]]:
    found = []
    for match in matches:
        kickoff = parse_datetime(match.get("utcDate"))
        if kickoff and start <= kickoff <= end and match.get("status") in {"SCHEDULED", "TIMED"}:
            found.append(match)
    return sorted(found, key=lambda m: m.get("utcDate", ""))


# -----------------------------
# Provider 2: The Odds API v4
# -----------------------------


@st.cache_data(ttl=600, show_spinner=False)
def fetch_global_odds(sport_key: str, api_key: str, regions: str) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    if not api_key:
        return [], "The Odds API key is not configured.", None
    base_params = {
        "regions": regions,
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    payload, error, response_headers = http_json(
        f"{ODDS_API_BASE}/sports/{sport_key}/odds",
        headers={"x-api-key": api_key},
        params=base_params,
    )
    if error and ("401" in error or "403" in error or "invalid" in error.lower()):
        query_params = dict(base_params)
        query_params["apiKey"] = api_key
        payload2, error2, response_headers2 = http_json(
            f"{ODDS_API_BASE}/sports/{sport_key}/odds",
            params=query_params,
        )
        if not error2:
            payload, error, response_headers = payload2, error2, response_headers2

    remaining = response_headers.get("x-requests-remaining")
    if error:
        return [], error, remaining
    return (payload if isinstance(payload, list) else []), None, remaining


def match_odds_event(home: str, away: str, odds_events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for event in odds_events:
        if same_team(home, event.get("home_team")) and same_team(away, event.get("away_team")):
            return event
    return None


def _outcome_side(name: str, home: str, away: str) -> Optional[str]:
    value = str(name or "").strip().lower()
    if value in {"draw", "tie"}:
        return "draw"
    if same_team(name, home):
        return "home"
    if same_team(name, away):
        return "away"
    return None


def parse_global_odds(event: Optional[Dict[str, Any]], home: str, away: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "h2h_best": {},
        "h2h_market_probability": {},
        "totals25_best": {},
        "totals25_market_probability": {},
        "bookmakers": 0,
        "last_update": None,
    }
    if not event:
        return result
    h2h_by_book: List[Dict[str, float]] = []
    totals_by_book: List[Dict[str, float]] = []
    for bookmaker in event.get("bookmakers", []):
        book_h2h: Dict[str, float] = {}
        book_total: Dict[str, float] = {}
        result["last_update"] = bookmaker.get("last_update") or result["last_update"]
        for market in bookmaker.get("markets", []):
            key = str(market.get("key", "")).lower()
            outcomes = market.get("outcomes", [])
            if key == "h2h":
                for outcome in outcomes:
                    side = _outcome_side(outcome.get("name"), home, away)
                    price = safe_float(outcome.get("price"))
                    if side and price and price > 1:
                        result["h2h_best"][side] = max(result["h2h_best"].get(side, 0), price)
                        book_h2h[side] = 1.0 / price
            elif key == "totals":
                for outcome in outcomes:
                    point = safe_float(outcome.get("point"))
                    price = safe_float(outcome.get("price"))
                    name = str(outcome.get("name", "")).lower()
                    side = "over" if "over" in name else "under" if "under" in name else None
                    if point == 2.5 and side and price and price > 1:
                        result["totals25_best"][side] = max(result["totals25_best"].get(side, 0), price)
                        book_total[side] = 1.0 / price
        if len(book_h2h) >= 2:
            overround = sum(book_h2h.values())
            if overround:
                h2h_by_book.append({k: v / overround for k, v in book_h2h.items()})
        if len(book_total) == 2:
            overround = sum(book_total.values())
            if overround:
                totals_by_book.append({k: v / overround for k, v in book_total.items()})
    result["bookmakers"] = len(event.get("bookmakers", []))
    for key in ("home", "draw", "away"):
        values = [book[key] for book in h2h_by_book if key in book]
        if values:
            result["h2h_market_probability"][key] = float(np.mean(values))
    for key in ("over", "under"):
        values = [book[key] for book in totals_by_book if key in book]
        if values:
            result["totals25_market_probability"][key] = float(np.mean(values))
    return result


def build_value_bets(model: Dict[str, Any], odds: Dict[str, Any]) -> List[Dict[str, Any]]:
    p = model["probabilities"]
    rows: List[Dict[str, Any]] = []
    candidates = [
        ("Home win", "home", p["home_win"], odds["h2h_best"].get("home"), odds["h2h_market_probability"].get("home")),
        ("Draw", "draw", p["draw"], odds["h2h_best"].get("draw"), odds["h2h_market_probability"].get("draw")),
        ("Away win", "away", p["away_win"], odds["h2h_best"].get("away"), odds["h2h_market_probability"].get("away")),
        ("Over 2.5 goals", "over25", p["over25"], odds["totals25_best"].get("over"), odds["totals25_market_probability"].get("over")),
        ("Under 2.5 goals", "under25", p["under25"], odds["totals25_best"].get("under"), odds["totals25_market_probability"].get("under")),
    ]
    for label, key, model_probability, best_odds, market_probability in candidates:
        if best_odds is None or best_odds <= 1:
            continue
        rows.append({
            "Market": label,
            "key": key,
            "Best odds": best_odds,
            "Model probability": model_probability,
            "No-vig market probability": market_probability,
            "Edge": (model_probability - market_probability) if market_probability is not None else None,
            "Expected value": model_probability * best_odds - 1.0,
        })
    return rows


# -----------------------------
# Provider 3: Live Football API v1 team news
# -----------------------------


def live_football_get(
    api_key: str,
    endpoints: Iterable[str],
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    if not api_key:
        return None, "Live Football API key is not configured."
    query_params = dict(params or {})
    query_params["api_key"] = api_key
    last_error = "No response from Live Football API."

    for base in LIVE_FOOTBALL_API_BASES:
        for ep in endpoints:
            clean_ep = ep.lstrip("/")
            payload, err, _ = http_json(
                f"{base}/{clean_ep}",
                headers={"Accept": "application/json"},
                params=query_params,
            )
            if not err and payload is not None:
                if isinstance(payload, dict) and payload.get("error"):
                    last_error = str(payload["error"])
                    continue
                return payload, None
            if err:
                last_error = err
    return None, last_error


def _extract_items(payload: Any, keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for k in keys:
            val = payload.get(k)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
            if isinstance(val, dict):
                return [val]
        data = payload.get("data")
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            for k in keys:
                val = data.get(k)
                if isinstance(val, list):
                    return [x for x in val if isinstance(x, dict)]
    return []


@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_football_team_news(
    api_key: str,
    league_slug: str,
    kickoff_date: str,
    home_name: str,
    away_name: str,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "match_id": None,
        "fixture": {},
        "injuries": [],
        "lineups": [],
        "statistics": [],
        "errors": [],
    }
    if not api_key:
        result["errors"].append("Live Football API key is not configured.")
        return result

    fixtures_payload, fixture_err = live_football_get(
        api_key,
        endpoints=("fixtures", "matches", "events"),
        params={"league": league_slug, "date": kickoff_date},
    )
    if fixture_err:
        result["errors"].append(f"Fixtures lookup: {fixture_err}")

    matched_fixture: Dict[str, Any] = {}
    for item in _extract_items(fixtures_payload, ("fixtures", "matches", "events", "results")):
        f_home = (
            item.get("home_team")
            or item.get("home")
            or (item.get("teams", {}).get("home", {}).get("name") if isinstance(item.get("teams"), dict) else None)
        )
        f_away = (
            item.get("away_team")
            or item.get("away")
            or (item.get("teams", {}).get("away", {}).get("name") if isinstance(item.get("teams"), dict) else None)
        )
        if same_team(home_name, f_home) and same_team(away_name, f_away):
            matched_fixture = item
            break

    result["fixture"] = matched_fixture
    match_id = (
        matched_fixture.get("id")
        or matched_fixture.get("match_id")
        or matched_fixture.get("fixture_id")
    )
    result["match_id"] = match_id

    detail_params: Dict[str, Any] = {"league": league_slug, "date": kickoff_date}
    if match_id is not None:
        detail_params["match_id"] = match_id
        detail_params["id"] = match_id

    inj_payload, inj_err = live_football_get(
        api_key,
        endpoints=("injuries", "match/injuries", "fixture/injuries"),
        params=detail_params,
    )
    if inj_err:
        result["errors"].append(f"Injuries: {inj_err}")
    else:
        result["injuries"] = _extract_items(inj_payload, ("injuries", "sidelined", "absences"))

    lineup_payload, lineup_err = live_football_get(
        api_key,
        endpoints=("lineups", "match/lineups", "fixture/lineups"),
        params=detail_params,
    )
    if lineup_err:
        result["errors"].append(f"Lineups: {lineup_err}")
    else:
        result["lineups"] = _extract_items(lineup_payload, ("lineups", "lineup", "starting_xi"))

    stats_payload, stats_err = live_football_get(
        api_key,
        endpoints=("statistics", "stats", "match/statistics", "fixture/statistics"),
        params=detail_params,
    )
    if stats_err:
        result["errors"].append(f"Statistics: {stats_err}")
    else:
        result["statistics"] = _extract_items(stats_payload, ("statistics", "stats"))

    return result


def summarize_team_news(team_news: Dict[str, Any], home_name: str, away_name: str) -> Dict[str, Any]:
    home_injuries: List[str] = []
    away_injuries: List[str] = []
    other_injuries: List[str] = []

    for row in team_news.get("injuries", []):
        player = (
            row.get("player_name")
            or row.get("player")
            or (row.get("player", {}).get("name") if isinstance(row.get("player"), dict) else None)
            or "Unknown Player"
        )
        reason = row.get("reason") or row.get("type") or row.get("status") or "Injured/Out"
        t_name = (
            row.get("team_name")
            or row.get("team")
            or (row.get("team", {}).get("name") if isinstance(row.get("team"), dict) else "")
        )
        label = f"{player} ({reason})"
        if same_team(t_name, home_name):
            home_injuries.append(label)
        elif same_team(t_name, away_name):
            away_injuries.append(label)
        else:
            other_injuries.append(f"{t_name}: {label}".strip(": "))

    home_lineup: List[str] = []
    away_lineup: List[str] = []
    home_formation: Optional[str] = None
    away_formation: Optional[str] = None

    for row in team_news.get("lineups", []):
        t_name = (
            row.get("team_name")
            or row.get("team")
            or (row.get("team", {}).get("name") if isinstance(row.get("team"), dict) else "")
        )
        formation = row.get("formation") or row.get("system")
        players_raw = row.get("startXI") or row.get("starting_xi") or row.get("players") or row.get("lineup") or []
        names: List[str] = []
        if isinstance(players_raw, list):
            for p in players_raw:
                if isinstance(p, dict):
                    p_obj = p.get("player") if isinstance(p.get("player"), dict) else p
                    p_name = p_obj.get("name") or p_obj.get("player_name")
                    if p_name:
                        names.append(str(p_name))
                elif isinstance(p, str) and p.strip():
                    names.append(p.strip())
        elif isinstance(players_raw, str) and players_raw.strip():
            names = [x.strip() for x in players_raw.split(",") if x.strip()]

        if same_team(t_name, home_name):
            home_formation = formation or home_formation
            home_lineup.extend(names)
        elif same_team(t_name, away_name):
            away_formation = formation or away_formation
            away_lineup.extend(names)

    has_data = bool(
        home_injuries or away_injuries or other_injuries or home_lineup or away_lineup or team_news.get("fixture")
    )
    return {
        "has_data": has_data,
        "home_injuries": home_injuries,
        "away_injuries": away_injuries,
        "other_injuries": other_injuries,
        "home_lineup": home_lineup,
        "away_lineup": away_lineup,
        "home_formation": home_formation,
        "away_formation": away_formation,
    }


# -----------------------------
# Model: Elo + Venue xG + Dixon-Coles
# -----------------------------


def _dixon_coles_tau(x: int, y: int, lam_x: float, lam_y: float, rho: float) -> float:
    if x == 0 and y == 0:
        return max(0.05, 1.0 - lam_x * lam_y * rho)
    if x == 0 and y == 1:
        return max(0.05, 1.0 + lam_x * rho)
    if x == 1 and y == 0:
        return max(0.05, 1.0 + lam_y * rho)
    if x == 1 and y == 1:
        return max(0.05, 1.0 - rho)
    return 1.0


def build_ratings_and_xg_stats(
    finished_matches: List[Dict[str, Any]],
    attack_power: float = DEFAULT_ATTACK_POWER,
) -> Dict[str, Any]:
    elo: Dict[str, float] = {}
    last_played: Dict[str, datetime] = {}

    home_gf: Dict[str, List[float]] = {}
    home_ga: Dict[str, List[float]] = {}
    away_gf: Dict[str, List[float]] = {}
    away_ga: Dict[str, List[float]] = {}
    all_home_goals: List[float] = []
    all_away_goals: List[float] = []

    sorted_matches = sorted(finished_matches, key=lambda m: m.get("utcDate", ""))
    for m in sorted_matches:
        h_raw = m.get("homeTeam", {}).get("name")
        a_raw = m.get("awayTeam", {}).get("name")
        h_key, a_key = normalize_team(h_raw), normalize_team(a_raw)
        ft = (m.get("score") or {}).get("fullTime") or {}
        hg, ag = ft.get("home"), ft.get("away")
        if not h_key or not a_key or hg is None or ag is None:
            continue

        hg_f, ag_f = float(hg), float(ag)
        elo.setdefault(h_key, 1500.0)
        elo.setdefault(a_key, 1500.0)

        h_elo = elo[h_key]
        a_elo = elo[a_key]
        exp_h = 1.0 / (1.0 + 10.0 ** ((a_elo - (h_elo + HOME_ELO_ADVANTAGE)) / 400.0))
        exp_a = 1.0 - exp_h

        if hg_f > ag_f:
            act_h, act_a = 1.0, 0.0
        elif hg_f < ag_f:
            act_h, act_a = 0.0, 1.0
        else:
            act_h, act_a = 0.5, 0.5

        goal_diff_mult = math.log(abs(hg_f - ag_f) + 1.0) + 1.0
        k_factor = 24.0 * goal_diff_mult
        elo[h_key] = h_elo + k_factor * (act_h - exp_h)
        elo[a_key] = a_elo + k_factor * (act_a - exp_a)

        dt = parse_datetime(m.get("utcDate"))
        if dt:
            last_played[h_key] = dt
            last_played[a_key] = dt

        home_gf.setdefault(h_key, []).append(hg_f)
        home_ga.setdefault(h_key, []).append(ag_f)
        away_gf.setdefault(a_key, []).append(ag_f)
        away_ga.setdefault(a_key, []).append(hg_f)
        all_home_goals.append(hg_f)
        all_away_goals.append(ag_f)

    league_home_avg = float(np.mean(all_home_goals)) if all_home_goals else 1.50
    league_away_avg = float(np.mean(all_away_goals)) if all_away_goals else 1.18

    return {
        "elo": elo,
        "last_played": last_played,
        "home_gf": home_gf,
        "home_ga": home_ga,
        "away_gf": away_gf,
        "away_ga": away_ga,
        "league_home_avg": max(0.6, league_home_avg),
        "league_away_avg": max(0.5, league_away_avg),
        "attack_power": attack_power,
    }


def _team_venue_rate(values: List[float], fallback: float, window: int = 10) -> float:
    if not values:
        return fallback
    recent = values[-window:]
    weights = np.linspace(0.6, 1.0, len(recent))
    return float(np.average(recent, weights=weights))


def predict_match(
    home_name: str,
    away_name: str,
    kickoff_dt: Optional[datetime],
    ratings_data: Dict[str, Any],
    *,
    home_tactical_adj: float = 0.0,
    away_tactical_adj: float = 0.0,
    odds_blend_weight: float = 0.0,
    odds_parsed: Optional[Dict[str, Any]] = None,
    injury_penalty_home: float = 0.0,
    injury_penalty_away: float = 0.0,
) -> Dict[str, Any]:
    h_key, a_key = normalize_team(home_name), normalize_team(away_name)
    elo_map = ratings_data["elo"]
    last_played = ratings_data["last_played"]
    h_elo = elo_map.get(h_key, 1500.0)
    a_elo = elo_map.get(a_key, 1500.0)

    league_h_avg = ratings_data["league_home_avg"]
    league_a_avg = ratings_data["league_away_avg"]
    atk_pow = ratings_data.get("attack_power", DEFAULT_ATTACK_POWER)

    h_att_rate = _team_venue_rate(ratings_data["home_gf"].get(h_key, []), league_h_avg)
    h_def_rate = _team_venue_rate(ratings_data["home_ga"].get(h_key, []), league_a_avg)
    a_att_rate = _team_venue_rate(ratings_data["away_gf"].get(a_key, []), league_a_avg)
    a_def_rate = _team_venue_rate(ratings_data["away_ga"].get(a_key, []), league_h_avg)

    h_att_idx = clamp((h_att_rate / league_h_avg) ** atk_pow, 0.55, 1.65)
    a_def_idx = clamp((a_def_rate / league_h_avg) ** atk_pow, 0.55, 1.65)
    a_att_idx = clamp((a_att_rate / league_a_avg) ** atk_pow, 0.55, 1.65)
    h_def_idx = clamp((h_def_rate / league_a_avg) ** atk_pow, 0.55, 1.65)

    elo_diff = (h_elo + HOME_ELO_ADVANTAGE) - a_elo
    elo_h_factor = clamp(1.0 + elo_diff / 1800.0, 0.72, 1.38)
    elo_a_factor = clamp(1.0 - elo_diff / 1800.0, 0.72, 1.38)

    rest_days_h: Optional[float] = None
    rest_days_a: Optional[float] = None
    rest_mult_h = 1.0
    rest_mult_a = 1.0
    if kickoff_dt:
        if h_key in last_played:
            rest_days_h = (kickoff_dt - last_played[h_key]).total_seconds() / 86400.0
            if rest_days_h < 3.5:
                rest_mult_h = 0.93
        if a_key in last_played:
            rest_days_a = (kickoff_dt - last_played[a_key]).total_seconds() / 86400.0
            if rest_days_a < 3.5:
                rest_mult_a = 0.93

    tac_mult_h = math.exp(home_tactical_adj * 0.08)
    tac_mult_a = math.exp(away_tactical_adj * 0.08)
    inj_mult_h = clamp(1.0 - injury_penalty_home, 0.65, 1.0)
    inj_mult_a = clamp(1.0 - injury_penalty_away, 0.65, 1.0)

    lam_home = clamp(
        league_h_avg * h_att_idx * a_def_idx * elo_h_factor * rest_mult_h * tac_mult_h * inj_mult_h,
        0.25,
        4.80,
    )
    lam_away = clamp(
        league_a_avg * a_att_idx * h_def_idx * elo_a_factor * rest_mult_a * tac_mult_a * inj_mult_a,
        0.20,
        4.50,
    )

    # Dixon-Coles score matrix
    matrix = np.zeros((MAX_GOALS, MAX_GOALS), dtype=float)
    pmf_h = [math.exp(-lam_home) * (lam_home ** i) / math.factorial(i) for i in range(MAX_GOALS)]
    pmf_a = [math.exp(-lam_away) * (lam_away ** j) / math.factorial(j) for j in range(MAX_GOALS)]

    for i in range(MAX_GOALS):
        for j in range(MAX_GOALS):
            tau = _dixon_coles_tau(i, j, lam_home, lam_away, DIXON_COLES_RHO)
            matrix[i, j] = pmf_h[i] * pmf_a[j] * tau

    total_mass = float(np.sum(matrix))
    if total_mass > 0:
        matrix /= total_mass

    home_win = float(np.sum(np.tril(matrix, -1)))
    draw = float(np.sum(np.diag(matrix)))
    away_win = float(np.sum(np.triu(matrix, 1)))

    over25 = 0.0
    under25 = 0.0
    home_2plus_goals = 0.0
    away_2plus_goals = 0.0
    score_list: List[Tuple[str, float]] = []

    for i in range(MAX_GOALS):
        for j in range(MAX_GOALS):
            prob = float(matrix[i, j])
            score_list.append((f"{i}-{j}", prob))
            if i + j >= 3:
                over25 += prob
            else:
                under25 += prob
            if i >= 2:
                home_2plus_goals += prob
            if j >= 2:
                away_2plus_goals += prob

    score_list.sort(key=lambda x: x[1], reverse=True)

    # Optional blend with no-vig market probabilities
    mkt_h2h = (odds_parsed or {}).get("h2h_market_probability") or {}
    if odds_blend_weight > 0 and all(k in mkt_h2h for k in ("home", "draw", "away")):
        w = clamp(odds_blend_weight, 0.0, 0.85)
        home_win = (1.0 - w) * home_win + w * float(mkt_h2h["home"])
        draw = (1.0 - w) * draw + w * float(mkt_h2h["draw"])
        away_win = (1.0 - w) * away_win + w * float(mkt_h2h["away"])
        s = home_win + draw + away_win
        home_win, draw, away_win = home_win / s, draw / s, away_win / s

    mkt_tot = (odds_parsed or {}).get("totals25_market_probability") or {}
    if odds_blend_weight > 0 and "over" in mkt_tot and "under" in mkt_tot:
        w = clamp(odds_blend_weight, 0.0, 0.85)
        over25 = (1.0 - w) * over25 + w * float(mkt_tot["over"])
        under25 = (1.0 - w) * under25 + w * float(mkt_tot["under"])
        ts = over25 + under25
        over25, under25 = over25 / ts, under25 / ts

    expected_gd = lam_home - lam_away
    expected_total_goals = lam_home + lam_away

    return {
        "home_elo": h_elo,
        "away_elo": a_elo,
        "lam_home": lam_home,
        "lam_away": lam_away,
        "expected_gd": expected_gd,
        "expected_total_goals": expected_total_goals,
        "rest_days_h": rest_days_h,
        "rest_days_a": rest_days_a,
        "top_scores": score_list[:5],
        "probabilities": {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
            "over25": over25,
            "under25": under25,
            "home_2plus_goals": home_2plus_goals,
            "away_2plus_goals": away_2plus_goals,
        },
    }


# -----------------------------
# Burmese Detailed Analysis Engine
# -----------------------------


def build_detailed_burmese_analysis(
    home_name: str,
    away_name: str,
    model: Dict[str, Any],
    odds_parsed: Dict[str, Any],
    value_bets: List[Dict[str, Any]],
    team_news_summary: Dict[str, Any],
    odds_fetched: bool,
    live_fetched: bool,
) -> str:
    p = model["probabilities"]
    gd = model["expected_gd"]
    lam_h = model["lam_home"]
    lam_a = model["lam_away"]

    vb_by_key = {row["key"]: row for row in value_bets}

    # Evaluate candidate picks across Home, Away, Over 2.5, Under 2.5
    candidates_meta = [
        ("home", f"အိမ်ကွင်းအသင်း ({home_name})", p["home_win"]),
        ("away", f"အဝေးကွင်းအသင်း ({away_name})", p["away_win"]),
        ("over25", "ဂိုးပေါင်းအပေါ် (Over 2.5 goals)", p["over25"]),
        ("under25", "ဂိုးပေါင်းအောက် (Under 2.5 goals)", p["under25"]),
    ]

    def _score_candidate(key: str, prob: float) -> float:
        vb = vb_by_key.get(key)
        score = prob * 100.0
        if vb:
            ev = vb.get("Expected value") or 0.0
            edge = vb.get("Edge") or 0.0
            if ev > 0:
                score += ev * 45.0 + max(0.0, edge) * 60.0
            elif ev < -0.05:
                score += ev * 25.0
        return score

    best_key, best_label, best_prob = max(
        candidates_meta,
        key=lambda item: _score_candidate(item[0], item[2]),
    )
    best_vb = vb_by_key.get(best_key)

    lines: List[str] = ["### အသေးစိတ် သုံးသပ်ချက်", ""]

    # 1. Best Overall Pick (Goals & Value focus like image 1)
    best_line = f"၁။ **အရွေးချယ်သင့်ဆုံး - {best_label}:**"
    lines.append(best_line)

    detail_1 = f"ရွေးချယ်မှု၏ Model ဖြစ်နိုင်ခြေမှာ **{best_prob * 100:.1f}%** အထိ ရှိနေပြီး ခန့်မှန်းဂိုးစုစုပေါင်းမှာ **{lam_h + lam_a:.2f} ဂိုး** (အိမ်ကွင်း **{lam_h:.2f}** / အဝေးကွင်း **{lam_a:.2f}**) ဖြစ်ပါသည်။"
    if odds_fetched and best_vb:
        odds_val = best_vb["Best odds"]
        ev_pct = (best_vb["Expected value"] or 0.0) * 100.0
        edge_pp = (best_vb["Edge"] or 0.0) * 100.0
        mkt_prob = best_vb.get("No-vig market probability")
        mkt_str = f" (No-vig ဈေးကွက် {mkt_prob * 100:.1f}%)" if mkt_prob is not None else ""
        detail_1 += f" အကောင်းဆုံး ပေါက်ကြေးမှာ **{odds_val:.2f}** ရှိနေပါတယ်။"
        lines.append(f"  * {detail_1}")
        ev_sign = "+" if ev_pct >= 0 else ""
        edge_sign = "+" if edge_pp >= 0 else ""
        lines.append(
            f"  * Value Bet Finder အရ Model နှင့် ဈေးကွက်ကွာဟချက် (**Edge {edge_sign}{edge_pp:.1f} pp**{mkt_str}) ရှိပြီး "
            f"**Expected Value ({ev_sign}{ev_pct:.1f}%)** ဖြစ်သောကြောင့် နှိုင်းယှဉ်ရွေးချယ်မှုများထဲတွင် **Value** အကိုက်ညီဆုံး ဖြစ်ပါသည်။"
        )
    else:
        lines.append(f"  * {detail_1}")
        if not odds_fetched:
            lines.append("  * Odds API မခေါ်ရသေးသဖြင့် ပေါက်ကြေးနှင့် EV တွက်ချက်မှု မပါဝင်သေးပါ (Odds API Fetch လုပ်ပါက Value Bet နှိုင်းယှဉ်ချက် ထည့်ပေးပါမည်)။")

    lines.append("")

    # 2. Team Selection: Home vs Away (like image 2)
    home_2p = p["home_2plus_goals"] * 100.0
    away_2p = p["away_2plus_goals"] * 100.0
    gd_sign = "+" if gd >= 0 else ""
    fav_team = f"အိမ်ကွင်းအသင်း ({home_name})" if p["home_win"] >= p["away_win"] else f"အဝေးကွင်းအသင်း ({away_name})"
    fav_prob = max(p["home_win"], p["away_win"]) * 100.0

    lines.append(f"၂။ **အသင်းရွေးချယ်မှု - အိမ်ကွင်း ({home_name}) vs အဝေးကွင်း ({away_name}):**")
    lines.append(
        f"  * အသင်းဘက်က ရွေးချယ်မည်ဆိုလျှင် **{fav_team}** က နိုင်ခြေ **{fav_prob:.1f}%** (အိမ်ကွင်း **{p['home_win']*100:.1f}%** / သရေ **{p['draw']*100:.1f}%** / အဝေးကွင်း **{p['away_win']*100:.1f}%**) ရှိပြီး "
        f"ခန့်မှန်းဂိုးကွာခြားချက် (**Expected Goal Difference**) ကလည်း **{gd_sign}{gd:.2f} ဂိုး** ရှိပါသည်။"
    )
    lines.append(
        f"  * ထို့အပြင် **{home_name}** က ၂ ဂိုးပြတ်နိုင်ခြေ **{home_2p:.1f}%** ရှိပြီး **{away_name}** က ၂ ဂိုးပြတ်နိုင်ခြေ **{away_2p:.1f}%** ရှိသဖြင့် "
        f"နှစ်သင်း၏ တိုက်စစ်/ခံစစ် ဟန်ချက်ကို ရှင်းရှင်းလင်းလင်း နှိုင်းယှဉ်နိုင်ပါသည်။"
    )

    if odds_fetched and value_bets:
        h_vb = vb_by_key.get("home")
        a_vb = vb_by_key.get("away")
        vb_notes: List[str] = []
        if h_vb:
            vb_notes.append(
                f"အိမ်ကွင်း Odds {h_vb['Best odds']:.2f} / EV {(h_vb['Expected value'] or 0)*100:+.1f}%"
            )
        if a_vb:
            vb_notes.append(
                f"အဝေးကွင်း Odds {a_vb['Best odds']:.2f} / EV {(a_vb['Expected value'] or 0)*100:+.1f}%"
            )
        if vb_notes:
            lines.append(f"  * **Value Bet Finder (1X2):** {' | '.join(vb_notes)} ဖြစ်ပါသည်။")

    lines.append("")

    # 3. Over 2.5 vs Under 2.5 Comparison
    lines.append("၃။ **ဂိုးပေါင်းရွေးချယ်မှု - ဂိုးပေါင်းအပေါ် (Over 2.5) vs ဂိုးပေါင်းအောက် (Under 2.5):**")
    over_note = f"ဂိုးပေါင်းအပေါ် (Over 2.5) ဖြစ်နိုင်ခြေ **{p['over25']*100:.1f}%**"
    under_note = f"ဂိုးပေါင်းအောက် (Under 2.5) ဖြစ်နိုင်ခြေ **{p['under25']*100:.1f}%**"
    if odds_fetched:
        o_vb = vb_by_key.get("over25")
        u_vb = vb_by_key.get("under25")
        if o_vb:
            over_note += f" (Best Odds {o_vb['Best odds']:.2f}, EV {(o_vb['Expected value'] or 0)*100:+.1f}%)"
        if u_vb:
            under_note += f" (Best Odds {u_vb['Best odds']:.2f}, EV {(u_vb['Expected value'] or 0)*100:+.1f}%)"
    lines.append(f"  * {over_note} နှင့် {under_note} ရှိပြီး မျှော်မှန်းဂိုးစုစုပေါင်းမှာ **{lam_h + lam_a:.2f} ဂိုး** ဖြစ်ပါသည်။")

    lines.append("")

    # 4. Live Football API: Lineups & Injury List
    lines.append("၄။ **Lineup နှင့် Injury List သုံးသပ်ချက် (Live Football API):**")
    if live_fetched and team_news_summary.get("has_data"):
        h_inj = team_news_summary.get("home_injuries") or []
        a_inj = team_news_summary.get("away_injuries") or []
        h_lu = team_news_summary.get("home_lineup") or []
        a_lu = team_news_summary.get("away_lineup") or []
        h_form = team_news_summary.get("home_formation")
        a_form = team_news_summary.get("away_formation")

        h_inj_txt = ", ".join(h_inj) if h_inj else "ဒဏ်ရာ/ပွဲပယ် စာရင်း မတွေ့ရှိပါ"
        a_inj_txt = ", ".join(a_inj) if a_inj else "ဒဏ်ရာ/ပွဲပယ် စာရင်း မတွေ့ရှိပါ"
        lines.append(f"  * **ဒဏ်ရာ/ပွဲပယ်စာရင်း (Injuries):** {home_name} — {h_inj_txt} | {away_name} — {a_inj_txt}။")

        if h_lu or a_lu or h_form or a_form:
            h_lu_txt = f"({h_form}) " + ", ".join(h_lu[:11]) if h_lu else (h_form or "Lineup မထွက်သေးပါ")
            a_lu_txt = f"({a_form}) " + ", ".join(a_lu[:11]) if a_lu else (a_form or "Lineup မထွက်သေးပါ")
            lines.append(f"  * **လူစာရင်း (Lineups):** {home_name} — {h_lu_txt} | {away_name} — {a_lu_txt}။")
        else:
            lines.append("  * **လူစာရင်း (Lineups):** ပွဲမစမီ Starting XI စာရင်း မထွက်ရှိသေးပါ (သို့မဟုတ် Provider ဘက်မှ မပေးသေးပါ)။")
    elif live_fetched:
        lines.append("  * Live Football API ကို Fetch လုပ်ခဲ့သော်လည်း ဤပွဲအတွက် Lineup သို့မဟုတ် Injury ဒေတာ မတွေ့ရှိသေးပါ။")
    else:
        lines.append("  * Live Football API ကို Fetch မလုပ်ရသေးပါ။ ပွဲမတိုင်မီ Lineup နှင့် Injury စာရင်းများကို ကြည့်ရန် **Fetch Live Football API** ကို နှိပ်ပါ။")

    lines.append("")
    lines.append("*(မှတ်ချက် - ဤကိန်းဂဏန်းများသည် သင်္ချာမော်ဒယ် ခန့်မှန်းချက်များသာ ဖြစ်ပြီး ရာနှုန်းပြည့် သေချာမှု မဟုတ်ပါ။)*")
    return "\n".join(lines)


# -----------------------------
# Streamlit UI
# -----------------------------


def main() -> None:
    st.title("⚽ Football Prediction Dashboard V8.5")
    st.caption(f"Model: `{MODEL_VERSION}` • Elo + Venue xG + Dixon-Coles + Odds Value Finder + Live Team News")

    with st.sidebar:
        st.header("🔑 API Keys")
        fd_key = st.text_input("football-data.org API Key", value=safe_secret("FOOTBALL_DATA_API_KEY"), type="password")
        odds_key = st.text_input("The Odds API Key", value=safe_secret("ODDS_API_KEY"), type="password")
        lfa_key = st.text_input("Live Football API Key", value=safe_secret("LIVE_FOOTBALL_API_KEY"), type="password")

        st.header("🏆 Competition & Window")
        league_name = st.selectbox("League", list(LEAGUES.keys()), index=0)
        days_ahead = st.slider("Upcoming window (days)", min_value=1, max_value=14, value=7)
        days_back = st.slider("Historical lookback (days)", min_value=60, max_value=365, value=210)
        odds_regions = st.text_input("Odds regions", value="uk,eu,us")

        st.header("⚙️ Model & Adjustments")
        attack_power = st.slider("Attacking power exponent", 0.40, 1.10, DEFAULT_ATTACK_POWER, 0.05)
        odds_blend = st.slider("Blend with no-vig market prob", 0.0, 0.80, 0.0, 0.05)
        home_tac = st.slider("Home tactical/motivation adj", -3.0, 3.0, 0.0, 0.5)
        away_tac = st.slider("Away tactical/motivation adj", -3.0, 3.0, 0.0, 0.5)
        inj_pen_h = st.slider("Home injury impact penalty", 0.0, 0.35, 0.0, 0.02)
        inj_pen_a = st.slider("Away injury impact penalty", 0.0, 0.35, 0.0, 0.02)

    league_cfg = LEAGUES[league_name]
    comp_code = league_cfg["football_data"]
    sport_key = league_cfg["odds"]
    lfa_slug = league_cfg["lfa_league"]

    d_from, d_to, win_start, win_end = fixture_window(days_ahead)
    hist_from, hist_to = finished_window(days_back)

    fixtures_raw, fix_err = fetch_fixtures(comp_code, fd_key, d_from, d_to)
    finished_raw, fin_err = fetch_finished_matches(comp_code, fd_key, hist_from, hist_to)
    standings, std_err = fetch_standings(comp_code, fd_key)

    if fix_err:
        st.error(f"Fixtures error: {fix_err}")
    if fin_err:
        st.warning(f"Finished matches warning: {fin_err}")
    if std_err:
        st.info(f"Standings info: {std_err}")

    upcoming = upcoming_only(fixtures_raw, win_start, win_end)
    if not upcoming:
        st.info("Selected window တွင် SCHEDULED / TIMED ပွဲစဉ်များ မတွေ့ရှိပါ။")
        return

    fixture_labels: List[str] = []
    fixture_map: Dict[str, Dict[str, Any]] = {}
    for m in upcoming:
        h_n = m.get("homeTeam", {}).get("name", "Home")
        a_n = m.get("awayTeam", {}).get("name", "Away")
        dt_str = local_datetime(m.get("utcDate"))
        lbl = f"{h_n} vs {a_n} ({dt_str})"
        fixture_labels.append(lbl)
        fixture_map[lbl] = m

    selected_label = st.selectbox("Select Match", fixture_labels)
    match = fixture_map[selected_label]
    home_name = match.get("homeTeam", {}).get("name", "Home")
    away_name = match.get("awayTeam", {}).get("name", "Away")
    kickoff_dt = parse_datetime(match.get("utcDate"))
    kickoff_date_str = kickoff_dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d") if kickoff_dt else utc_now().date().isoformat()

    # Session state for on-demand API fetches
    odds_state_key = f"odds_fetched_{comp_code}"
    live_state_key = f"live_fetched_{comp_code}_{normalize_team(home_name)}_{normalize_team(away_name)}_{kickoff_date_str}"

    st.session_state.setdefault(odds_state_key, False)
    st.session_state.setdefault(live_state_key, False)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        if st.button("💹 Fetch Odds API", use_container_width=True):
            st.session_state[odds_state_key] = True
    with col_btn2:
        if st.button("🩺 Fetch Live Football API", use_container_width=True):
            st.session_state[live_state_key] = True
    with col_btn3:
        st.caption(f"Kickoff: **{local_datetime(match.get('utcDate'))}** | Standings teams loaded: **{len(standings)}**")

    odds_parsed: Dict[str, Any] = {
        "h2h_best": {},
        "h2h_market_probability": {},
        "totals25_best": {},
        "totals25_market_probability": {},
        "bookmakers": 0,
        "last_update": None,
    }
    odds_fetched = bool(st.session_state.get(odds_state_key))
    if odds_fetched:
        odds_events, odds_err, remaining_req = fetch_global_odds(sport_key, odds_key, odds_regions)
        if odds_err:
            st.warning(f"Odds API: {odds_err}")
        else:
            ev = match_odds_event(home_name, away_name, odds_events)
            odds_parsed = parse_global_odds(ev, home_name, away_name)
            if remaining_req is not None:
                st.caption(f"Odds API requests remaining: `{remaining_req}` | Bookmakers: `{odds_parsed['bookmakers']}`")

    live_fetched = bool(st.session_state.get(live_state_key))
    team_news_raw: Dict[str, Any] = {"injuries": [], "lineups": [], "statistics": [], "errors": []}
    team_news_summary: Dict[str, Any] = {"has_data": False}
    if live_fetched:
        team_news_raw = fetch_live_football_team_news(
            lfa_key,
            lfa_slug,
            kickoff_date_str,
            home_name,
            away_name,
        )
        team_news_summary = summarize_team_news(team_news_raw, home_name, away_name)
        for err_msg in team_news_raw.get("errors", []):
            st.caption(f"Live Football API note: {err_msg}")

    ratings_data = build_ratings_and_xg_stats(finished_raw, attack_power=attack_power)
    model = predict_match(
        home_name,
        away_name,
        kickoff_dt,
        ratings_data,
        home_tactical_adj=home_tac,
        away_tactical_adj=away_tac,
        odds_blend_weight=odds_blend,
        odds_parsed=odds_parsed if odds_fetched else None,
        injury_penalty_home=inj_pen_h,
        injury_penalty_away=inj_pen_a,
    )
    value_bets = build_value_bets(model, odds_parsed) if odds_fetched else []

    # Top KPIs
    p = model["probabilities"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"{home_name} Win", f"{p['home_win']*100:.1f}%", f"Elo {model['home_elo']:.0f}")
    m2.metric("Draw", f"{p['draw']*100:.1f}%")
    m3.metric(f"{away_name} Win", f"{p['away_win']*100:.1f}%", f"Elo {model['away_elo']:.0f}")
    m4.metric("Over 2.5 Goals", f"{p['over25']*100:.1f}%", f"xG {model['expected_total_goals']:.2f}")
    m5.metric("Under 2.5 Goals", f"{p['under25']*100:.1f}%", f"Exp GD {model['expected_gd']:+.2f}")

    # Detailed Burmese Analysis
    analysis_md = build_detailed_burmese_analysis(
        home_name=home_name,
        away_name=away_name,
        model=model,
        odds_parsed=odds_parsed,
        value_bets=value_bets,
        team_news_summary=team_news_summary,
        odds_fetched=odds_fetched,
        live_fetched=live_fetched,
    )
    st.markdown(analysis_md)

    st.divider()

    # Supporting tables: Value Bet Finder + Top Scores + Team News
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("🎯 Value Bet Finder")
        if odds_fetched and value_bets:
            df_vb = pd.DataFrame(value_bets).drop(columns=["key"])
            for col in ("Model probability", "No-vig market probability", "Edge", "Expected value"):
                df_vb[col] = df_vb[col].apply(lambda v: f"{v*100:+.1f}%" if pd.notna(v) else "—")
            df_vb["Best odds"] = df_vb["Best odds"].apply(lambda v: f"{v:.2f}")
            st.dataframe(df_vb, use_container_width=True, hide_index=True)
        else:
            st.info("Value Bet table ကြည့်ရန် **Fetch Odds API** ကို နှိပ်ပါ။")

        st.subheader("🔢 Top Exact Scores (Dixon-Coles)")
        df_scores = pd.DataFrame(
            [{"Score": sc, "Probability": f"{pr*100:.1f}%"} for sc, pr in model["top_scores"]]
        )
        st.dataframe(df_scores, use_container_width=True, hide_index=True)

    with c_right:
        st.subheader("🩺 Lineups & Injury List")
        if live_fetched and team_news_summary.get("has_data"):
            st.markdown(f"**{home_name} Injuries:** {', '.join(team_news_summary['home_injuries']) or 'None'}")
            st.markdown(f"**{away_name} Injuries:** {', '.join(team_news_summary['away_injuries']) or 'None'}")
            st.markdown(
                f"**{home_name} Lineup ({team_news_summary.get('home_formation') or '—'}):** "
                f"{', '.join(team_news_summary['home_lineup']) or 'Not available'}"
            )
            st.markdown(
                f"**{away_name} Lineup ({team_news_summary.get('away_formation') or '—'}):** "
                f"{', '.join(team_news_summary['away_lineup']) or 'Not available'}"
            )
        else:
            st.info("Lineup နှင့် Injury စာရင်း ကြည့်ရန် **Fetch Live Football API** ကို နှိပ်ပါ။")


if __name__ == "__main__":
    main()
