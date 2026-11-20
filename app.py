from __future__ import annotations

"""Professional Football Prediction Dashboard V8.3

Providers:
- football-data.org v4: fixtures, recent finished results, standings
- The Odds API v4: multi-region h2h and totals odds (on-demand per match; auto on Top 10)
- Live Football API v1: match injuries, lineups, and available match statistics (on-demand)

Model:
- chronological Elo ratings
- league-relative venue-specific attack/defence goal-rate xG proxy
- Dixon-Coles-adjusted Poisson score matrix + Goal Difference probabilities
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
    page_title="Football Prediction Dashboard V8",
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
MODEL_VERSION = "elo-dixon-coles-v8.3"
DIXON_COLES_RHO = -0.06
HOME_ELO_ADVANTAGE = 55.0

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
    text = re.sub(r"\b(fc|afc|cf|ssc|sc|club)\b", " ", text)
    return re.sub(r"[^a-z0-9]", "", text)


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
    payload, error, response_headers = http_json(
        f"{ODDS_API_BASE}/sports/{sport_key}/odds",
        headers={"x-api-key": api_key},
        params={"regions": regions, "markets": "h2h,totals", "oddsFormat": "decimal", "dateFormat": "iso"},
    )
    if error:
        return [], error, response_headers.get("x-requests-remaining")
    remaining = response_headers.get("x-requests-remaining")
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
        "h2h_best": {}, "h2h_market_probability": {},
        "totals25_best": {}, "totals25_market_probability": {},
        "bookmakers": 0, "last_update": None,
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
        ("Home win", p["home_win"], odds["h2h_best"].get("home"), odds["h2h_market_probability"].get("home")),
        ("Draw", p["draw"], odds["h2h_best"].get("draw"), odds["h2h_market_probability"].get("draw")),
        ("Away win", p["away_win"], odds["h2h_best"].get("away"), odds["h2h_market_probability"].get("away")),
        ("Over 2.5 goals", p["over25"], odds["totals25_best"].get("over"), odds["totals25_market_probability"].get("over")),
        ("Under 2.5 goals", p["under25"], odds["totals25_best"].get("under"), odds["totals25_market_probability"].get("under")),
    ]
    for label, model_probability, best_odds, market_probability in candidates:
        if best_odds is None or best_odds <= 1:
            continue
        rows.append({
            "Market": label,
            "Best odds": best_odds,
            "Model probability": model_probability,
            "No-vig market probability": market_probability,
            "Edge": model_probability - market_probability if market_probability is not None else None,
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
        params={"date": kickoff_date, "league": league_slug},
    )
    if fixture_err:
        result["errors"].append(f"Fixture lookup: {fixture_err}")
        return result

    candidates = _extract_items(fixtures_payload, ("fixtures", "matches", "response", "events"))
    matched: Optional[Dict[str, Any]] = None
    for item in candidates:
        h_cand = (
            item.get("home_team")
            or item.get("homeTeam")
            or (item.get("teams", {}).get("home", {}).get("name") if isinstance(item.get("teams"), dict) else None)
            or item.get("home")
        )
        a_cand = (
            item.get("away_team")
            or item.get("awayTeam")
            or (item.get("teams", {}).get("away", {}).get("name") if isinstance(item.get("teams"), dict) else None)
            or item.get("away")
        )
        if isinstance(h_cand, dict):
            h_cand = h_cand.get("name")
        if isinstance(a_cand, dict):
            a_cand = a_cand.get("name")
        if same_team(home_name, h_cand) and same_team(away_name, a_cand):
            matched = item
            break

    if not matched:
        result["errors"].append("Matching fixture not found in Live Football API for this date.")
        return result

    result["fixture"] = matched
    match_id = matched.get("id") or matched.get("match_id") or matched.get("fixture_id")
    result["match_id"] = match_id

    lookup_params: Dict[str, Any] = {"date": kickoff_date, "league": league_slug}
    if match_id:
        lookup_params["match_id"] = match_id
        lookup_params["fixture_id"] = match_id
        lookup_params["id"] = match_id

    inj_payload, inj_err = live_football_get(api_key, ("injuries", "match/injuries", "fixture/injuries"), lookup_params)
    if inj_err:
        result["errors"].append(f"injuries: {inj_err}")
    else:
        result["injuries"] = _extract_items(inj_payload, ("injuries", "response", "missing"))

    lineup_payload, lineup_err = live_football_get(api_key, ("lineups", "match/lineups", "fixture/lineups"), lookup_params)
    if lineup_err:
        result["errors"].append(f"lineups: {lineup_err}")
    else:
        result["lineups"] = _extract_items(lineup_payload, ("lineups", "response", "teams"))

    stats_payload, stats_err = live_football_get(api_key, ("statistics", "match/statistics", "fixture/statistics"), lookup_params)
    if not stats_err and stats_payload:
        result["statistics"] = _extract_items(stats_payload, ("statistics", "response", "stats"))

    return result


def summarize_team_news(bundle: Dict[str, Any], home_name: str, away_name: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "fixture_id": bundle.get("match_id"),
        "home_injuries": [],
        "away_injuries": [],
        "home_lineup_confirmed": False,
        "away_lineup_confirmed": False,
        "home_starters": None,
        "away_starters": None,
        "statistics_available": False,
        "home_xg": None,
        "away_xg": None,
        "home_shots_on_target": None,
        "away_shots_on_target": None,
    }

    for item in bundle.get("injuries", []):
        player = item.get("player_name") or (item.get("player", {}).get("name") if isinstance(item.get("player"), dict) else None) or item.get("name") or "Unknown player"
        reason = item.get("reason") or item.get("status") or item.get("type") or "Unavailable"
        team_label = item.get("team_name") or item.get("team") or (item.get("team", {}).get("name") if isinstance(item.get("team"), dict) else "")
        side = str(item.get("side") or item.get("team_side") or "").lower()
        entry = f"{player} ({reason})"
        if side in {"home", "h"} or same_team(team_label, home_name):
            summary["home_injuries"].append(entry)
        elif side in {"away", "a"} or same_team(team_label, away_name):
            summary["away_injuries"].append(entry)

    for lineup in bundle.get("lineups", []):
        team_label = lineup.get("team_name") or lineup.get("team") or (lineup.get("team", {}).get("name") if isinstance(lineup.get("team"), dict) else "")
        side = str(lineup.get("side") or lineup.get("team_side") or "").lower()
        starters = lineup.get("startXI") or lineup.get("starting_xi") or lineup.get("starters") or lineup.get("lineup") or []
        count = len(starters) if isinstance(starters, list) and starters else None
        if side in {"home", "h"} or same_team(team_label, home_name):
            summary["home_lineup_confirmed"] = bool(count)
            summary["home_starters"] = count
        elif side in {"away", "a"} or same_team(team_label, away_name):
            summary["away_lineup_confirmed"] = bool(count)
            summary["away_starters"] = count

    for stat in bundle.get("statistics", []):
        key = re.sub(r"[^a-z]", "", str(stat.get("type") or stat.get("name") or "").lower())
        if key in {"expectedgoals", "xg"}:
            summary["home_xg"] = safe_float(stat.get("home"))
            summary["away_xg"] = safe_float(stat.get("away"))
            summary["statistics_available"] = True
        elif key in {"shotsongoal", "shotsontarget"}:
            summary["home_shots_on_target"] = safe_float(stat.get("home"))
            summary["away_shots_on_target"] = safe_float(stat.get("away"))
            summary["statistics_available"] = True

    return summary


# -----------------------------
# Rating and goal probability model
# -----------------------------


def valid_finished_score(match: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    score = match.get("score", {}).get("fullTime", {})
    home_goals, away_goals = safe_float(score.get("home")), safe_float(score.get("away"))
    if home_goals is None or away_goals is None:
        return None
    return int(home_goals), int(away_goals)


def build_team_context(
    finished_matches: List[Dict[str, Any]],
    standings: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    matches = []
    for match in finished_matches:
        score = valid_finished_score(match)
        home = match.get("homeTeam", {}).get("name")
        away = match.get("awayTeam", {}).get("name")
        dt = parse_datetime(match.get("utcDate"))
        if score and home and away and dt:
            matches.append((dt, match, home, away, score[0], score[1]))
    matches.sort(key=lambda row: row[0])

    team_history: Dict[str, List[Dict[str, Any]]] = {}
    team_names: Dict[str, str] = {}
    elo: Dict[str, float] = {}
    sum_home = sum_away = 0.0
    league_count = 0

    for dt, match, home_name, away_name, hg, ag in matches:
        hk, ak = normalize_team(home_name), normalize_team(away_name)
        team_names[hk], team_names[ak] = home_name, away_name
        home_elo, away_elo = elo.get(hk, 1500.0), elo.get(ak, 1500.0)
        expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo - HOME_ELO_ADVANTAGE) / 400.0))
        actual_home = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        margin = abs(hg - ag)
        k_factor = 20.0 * min(1.8, 1.0 + math.log1p(margin) * 0.25)
        delta = k_factor * (actual_home - expected_home)
        elo[hk], elo[ak] = home_elo + delta, away_elo - delta

        team_history.setdefault(hk, []).append({"date": dt, "gf": hg, "ga": ag, "home": True, "points": 3 if hg > ag else 1 if hg == ag else 0})
        team_history.setdefault(ak, []).append({"date": dt, "gf": ag, "ga": hg, "home": False, "points": 3 if ag > hg else 1 if ag == hg else 0})
        sum_home += hg
        sum_away += ag
        league_count += 1

    base_home = sum_home / league_count if league_count else 1.48
    base_away = sum_away / league_count if league_count else 1.15
    profiles: Dict[str, Dict[str, Any]] = {}
    for key, games in team_history.items():
        recent = games[-10:]
        home_games = [x for x in recent if x["home"]][-6:]
        away_games = [x for x in recent if not x["home"]][-6:]
        standings_row = standings.get(key, {})
        profiles[key] = {
            "name": team_names.get(key, key),
            "elo": elo.get(key, 1500.0),
            "recent": recent,
            "home_games": home_games,
            "away_games": away_games,
            "recent_points": sum(x["points"] for x in recent),
            "recent_played": len(recent),
            "last_match": max((x["date"] for x in games), default=None),
            "standing": standings_row,
        }
    for key, row in standings.items():
        profiles.setdefault(key, {
            "name": row.get("name", key), "elo": 1500.0, "recent": [],
            "home_games": [], "away_games": [], "recent_points": 0,
            "recent_played": 0, "last_match": None, "standing": row,
        })
    return {
        "teams": profiles,
        "league_baseline": {"home_goals": max(0.7, base_home), "away_goals": max(0.6, base_away), "matches": league_count},
        "standings": standings,
        "finished_count": league_count,
    }


def _venue_rate(games: List[Dict[str, Any]], field: str, prior: float, prior_n: float = 4.0) -> float:
    values = [safe_float(g.get(field)) for g in games]
    values = [x for x in values if x is not None and x >= 0]
    return (sum(values) + prior * prior_n) / (len(values) + prior_n)


def poisson_pmf(lam: float, goals: int) -> float:
    return math.exp(-lam) * lam ** goals / math.factorial(goals)


def dixon_coles_matrix(home_lambda: float, away_lambda: float, rho: float = DIXON_COLES_RHO, max_goals: int = 10) -> np.ndarray:
    matrix = np.array([
        [poisson_pmf(home_lambda, h) * poisson_pmf(away_lambda, a) for a in range(max_goals + 1)]
        for h in range(max_goals + 1)
    ], dtype=float)
    matrix[0, 0] *= max(0.05, 1.0 - home_lambda * away_lambda * rho)
    matrix[0, 1] *= max(0.05, 1.0 + home_lambda * rho)
    matrix[1, 0] *= max(0.05, 1.0 + away_lambda * rho)
    matrix[1, 1] *= max(0.05, 1.0 - rho)
    total = float(matrix.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Invalid Dixon-Coles probability matrix")
    return matrix / total


def calculate_probabilities(matrix: np.ndarray) -> Dict[str, Any]:
    size = matrix.shape[0]
    home_win = float(sum(matrix[h, a] for h in range(size) for a in range(size) if h > a))
    draw = float(sum(matrix[h, a] for h in range(size) for a in range(size) if h == a))
    away_win = float(sum(matrix[h, a] for h in range(size) for a in range(size) if h < a))

    home_gd_2plus = float(sum(matrix[h, a] for h in range(size) for a in range(size) if (h - a) >= 2))
    home_gd_3plus = float(sum(matrix[h, a] for h in range(size) for a in range(size) if (h - a) >= 3))
    away_gd_2plus = float(sum(matrix[h, a] for h in range(size) for a in range(size) if (a - h) >= 2))
    away_gd_3plus = float(sum(matrix[h, a] for h in range(size) for a in range(size) if (a - h) >= 3))
    any_gd_2plus = float(sum(matrix[h, a] for h in range(size) for a in range(size) if abs(h - a) >= 2))
    expected_gd = float(sum((h - a) * matrix[h, a] for h in range(size) for a in range(size)))

    total_probs = {
        total: float(sum(matrix[h, total - h] for h in range(size) if 0 <= total - h < size))
        for total in range(2 * size - 1)
    }
    over15 = float(sum(p for goals, p in total_probs.items() if goals >= 2))
    over25 = float(sum(p for goals, p in total_probs.items() if goals >= 3))
    over35 = float(sum(p for goals, p in total_probs.items() if goals >= 4))
    over45 = float(sum(p for goals, p in total_probs.items() if goals >= 5))
    home_scores = float(sum(matrix[h, a] for h in range(1, size) for a in range(size)))
    away_scores = float(sum(matrix[h, a] for h in range(size) for a in range(1, size)))
    btts_yes = float(sum(matrix[h, a] for h in range(1, size) for a in range(1, size)))
    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "home_gd_2plus": home_gd_2plus,
        "home_gd_3plus": home_gd_3plus,
        "away_gd_2plus": away_gd_2plus,
        "away_gd_3plus": away_gd_3plus,
        "any_gd_2plus": any_gd_2plus,
        "expected_gd": expected_gd,
        "over15": over15,
        "over25": over25,
        "over35": over35,
        "over45": over45,
        "under25": 1.0 - over25,
        "home_scores_1plus": home_scores,
        "away_scores_1plus": away_scores,
        "btts_yes": btts_yes,
        "btts_no": 1.0 - btts_yes,
        "total_probs": total_probs,
    }


def _get_profile(context: Dict[str, Any], team_name: str) -> Dict[str, Any]:
    key = normalize_team(team_name)
    found = context.get("teams", {}).get(key)
    if found:
        return found
    for profile_key, profile in context.get("teams", {}).items():
        if same_team(team_name, profile.get("name", profile_key)):
            return profile
    return {"name": team_name, "elo": 1500.0, "recent": [], "home_games": [], "away_games": [], "recent_points": 0, "recent_played": 0, "last_match": None, "standing": {}}


def predict_match(
    match: Dict[str, Any],
    league_context: Dict[str, Any],
    *,
    manual_context: Optional[Dict[str, float]] = None,
    team_news: Optional[Dict[str, Any]] = None,
    parsed_odds: Optional[Dict[str, Any]] = None,
    odds_weight: float = 0.25,
) -> Dict[str, Any]:
    home_name = match.get("homeTeam", {}).get("name", "Home")
    away_name = match.get("awayTeam", {}).get("name", "Away")
    home = _get_profile(league_context, home_name)
    away = _get_profile(league_context, away_name)
    baseline = league_context.get("league_baseline", {"home_goals": 1.48, "away_goals": 1.15})
    base_home = clamp(float(baseline.get("home_goals", 1.48)), 0.70, 2.50)
    base_away = clamp(float(baseline.get("away_goals", 1.15)), 0.60, 2.20)

    home_attack = _venue_rate(home.get("home_games", []), "gf", base_home)
    home_conceded = _venue_rate(home.get("home_games", []), "ga", base_away)
    away_attack = _venue_rate(away.get("away_games", []), "gf", base_away)
    away_conceded = _venue_rate(away.get("away_games", []), "ga", base_home)
    home_lambda = base_home * math.sqrt((home_attack / base_home) * (away_conceded / base_home))
    away_lambda = base_away * math.sqrt((away_attack / base_away) * (home_conceded / base_away))

    elo_difference = float(home.get("elo", 1500.0)) - float(away.get("elo", 1500.0))
    elo_shift = clamp(elo_difference / 2200.0, -0.22, 0.22)
    home_lambda *= math.exp(elo_shift)
    away_lambda *= math.exp(-elo_shift * 0.80)

    kickoff = parse_datetime(match.get("utcDate")) or utc_now()
    rest_days: Dict[str, Optional[int]] = {}
    rest_factor: Dict[str, float] = {}
    for side, profile in (("home", home), ("away", away)):
        last_match = profile.get("last_match")
        days = max(0, (kickoff.date() - last_match.date()).days) if last_match else None
        rest_days[side] = days
        rest_factor[side] = 0.92 if days is not None and days <= 3 else 0.97 if days is not None and days <= 5 else 1.0
    home_lambda *= rest_factor["home"]
    away_lambda *= rest_factor["away"]

    manual = manual_context or {}
    home_delta = 0.018 * (manual.get("home_motivation", 0.0) - manual.get("away_motivation", 0.0))
    home_delta += 0.015 * (manual.get("home_tactical", 0.0) - manual.get("away_tactical", 0.0))
    home_lambda *= math.exp(clamp(home_delta, -0.14, 0.14))
    away_lambda *= math.exp(clamp(-home_delta * 0.70, -0.12, 0.12))

    news = team_news or {}
    home_injuries = len(news.get("home_injuries", []))
    away_injuries = len(news.get("away_injuries", []))
    home_lineup_ok = bool(news.get("home_lineup_confirmed"))
    away_lineup_ok = bool(news.get("away_lineup_confirmed"))

    home_lineup_mod = 0.015 if home_lineup_ok else -0.01 if news else 0.0
    away_lineup_mod = 0.015 if away_lineup_ok else -0.01 if news else 0.0

    home_lambda *= math.exp(clamp(-0.020 * home_injuries + home_lineup_mod, -0.14, 0.03))
    away_lambda *= math.exp(clamp(-0.020 * away_injuries + away_lineup_mod, -0.14, 0.03))

    odds_used = False
    mkt_h2h = (parsed_odds or {}).get("h2h_market_probability") or {}
    mkt_tot = (parsed_odds or {}).get("totals25_market_probability") or {}
    w = clamp(float(odds_weight), 0.0, 0.60)
    if w > 0 and all(k in mkt_h2h for k in ("home", "draw", "away")):
        odds_used = True
        mkt_home = clamp(float(mkt_h2h["home"]), 0.03, 0.92)
        mkt_away = clamp(float(mkt_h2h["away"]), 0.03, 0.92)
        ratio = clamp(mkt_home / max(0.03, mkt_away), 0.25, 4.0)
        home_lambda *= math.exp(w * 0.25 * math.log(ratio))
        away_lambda *= math.exp(-w * 0.25 * math.log(ratio))
        if "over" in mkt_tot and "under" in mkt_tot:
            total_tilt = clamp((float(mkt_tot["over"]) - 0.50) * 0.35 * w, -0.08, 0.08)
            home_lambda *= math.exp(total_tilt)
            away_lambda *= math.exp(total_tilt)

    home_lambda = clamp(home_lambda, 0.20, 4.50)
    away_lambda = clamp(away_lambda, 0.15, 4.20)

    matrix = dixon_coles_matrix(home_lambda, away_lambda)
    flat_index = int(np.argmax(matrix))
    best_score = tuple(int(x) for x in np.unravel_index(flat_index, matrix.shape))
    probs = calculate_probabilities(matrix)

    blended_1x2 = {
        "home_win": probs["home_win"],
        "draw": probs["draw"],
        "away_win": probs["away_win"],
    }
    if odds_used:
        bh = (1.0 - w) * probs["home_win"] + w * float(mkt_h2h["home"])
        bd = (1.0 - w) * probs["draw"] + w * float(mkt_h2h["draw"])
        ba = (1.0 - w) * probs["away_win"] + w * float(mkt_h2h["away"])
        s = bh + bd + ba
        if s > 0:
            blended_1x2 = {"home_win": bh / s, "draw": bd / s, "away_win": ba / s}

    top_outcome = max(blended_1x2["home_win"], blended_1x2["draw"], blended_1x2["away_win"])
    played_mean = (home.get("recent_played", 0) + away.get("recent_played", 0)) / 2
    quality = clamp(
        0.20
        + min(0.40, played_mean / 22.0)
        + (0.15 if home.get("standing") else 0.0)
        + (0.15 if away.get("standing") else 0.0)
        + (0.05 if odds_used else 0.0)
        + (0.05 if news else 0.0),
        0.10,
        1.0,
    )

    return {
        "home_name": home_name,
        "away_name": away_name,
        "home_lambda": home_lambda,
        "away_lambda": away_lambda,
        "expected_total_goals": home_lambda + away_lambda,
        "best_score": best_score,
        "best_score_probability": float(matrix[best_score]),
        "matrix": matrix,
        "probabilities": probs,
        "blended_1x2": blended_1x2,
        "odds_used": odds_used,
        "home_elo": float(home.get("elo", 1500.0)),
        "away_elo": float(away.get("elo", 1500.0)),
        "elo_difference": elo_difference,
        "home_attack_index": home_attack / base_home,
        "away_attack_index": away_attack / base_away,
        "home_rest_days": rest_days["home"],
        "away_rest_days": rest_days["away"],
        "home_injury_count": home_injuries,
        "away_injury_count": away_injuries,
        "home_lineup_confirmed": home_lineup_ok,
        "away_lineup_confirmed": away_lineup_ok,
        "top_outcome_probability": top_outcome,
        "data_quality": quality,
        "xg_source": "league-relative goals-for/against proxy + context/odds adjustments",
        "model_version": MODEL_VERSION,
    }


# -----------------------------
# Rendering
# -----------------------------


def render_value_finder(model: Dict[str, Any], odds: Dict[str, Any]) -> None:
    with st.expander("Value Bet Finder — compare model probabilities with global odds", expanded=False):
        rows = build_value_bets(model, odds)
        if not rows:
            st.info("No matching h2h or 2.5 totals odds are available for this fixture.")
            return
        table = pd.DataFrame(rows)
        for column in ("Model probability", "No-vig market probability"):
            table[column] = table[column].map(lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%")
        table["Edge"] = table["Edge"].map(lambda x: "—" if pd.isna(x) else f"{x * 100:+.1f} pp")
        table["Expected value"] = table["Expected value"].map(lambda x: f"{x * 100:+.1f}%")
        table["Best odds"] = table["Best odds"].map(lambda x: f"{x:.2f}")
        st.dataframe(table, hide_index=True, use_container_width=True)
        candidates = [row for row in rows if row["Expected value"] > 0 and row["Edge"] is not None and row["Edge"] > 0]
        if candidates:
            best = max(candidates, key=lambda row: row["Expected value"])
            st.success(f"Positive model-vs-price signal: {best['Market']} | EV {best['Expected value'] * 100:+.1f}% (not a guarantee).")
        else:
            st.caption("No positive model-vs-market edge was found at the returned best prices.")
        st.caption("Market probabilities are normalized across each bookmaker's outcomes before averaging. EV uses the best listed decimal price and the model probability; it is not a profit guarantee.")


def render_fixture_card(
    match: Dict[str, Any],
    league_name: str,
    league_context: Dict[str, Any],
    odds_key: str,
    odds_sport_key: str,
    regions: str,
    odds_weight: float,
    lfa_key: str,
    lfa_league: str,
    preloaded_odds_events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    home_name = match.get("homeTeam", {}).get("name", "Home")
    away_name = match.get("awayTeam", {}).get("name", "Away")
    match_id = str(match.get("id", f"{home_name}-{away_name}"))
    odds_state_key = f"odds_fetch_{match_id}_{odds_sport_key}_{regions}"
    lfa_state_key = f"lfa_news_{match_id}"

    with st.expander(f"⚽ {home_name} vs {away_name} | {local_datetime(match.get('utcDate'))}", expanded=False):
        st.markdown(f"### {home_name}  **vs**  {away_name}")
        st.caption(f"{league_name} | Kickoff {local_datetime(match.get('utcDate'))} | {MODEL_VERSION}")

        b1, b2 = st.columns(2)
        with b1:
            fetch_odds_clicked = st.button(
                "Fetch Odds API",
                key=f"btn_odds_{match_id}",
                disabled=not bool(odds_key),
            )
        with b2:
            fetch_lfa_clicked = st.button(
                "Fetch Live Football (Lineup/Injury)",
                key=f"btn_lfa_{match_id}",
                disabled=not bool(lfa_key),
            )

        if fetch_odds_clicked:
            with st.spinner("Fetching Odds API data…"):
                ev_list, od_err, rem = fetch_global_odds(odds_sport_key, odds_key, regions)
                st.session_state[odds_state_key] = {
                    "events": ev_list,
                    "error": od_err,
                    "remaining": rem,
                }

        if fetch_lfa_clicked:
            kickoff_date = (parse_datetime(match.get("utcDate")) or utc_now()).date().isoformat()
            with st.spinner("Fetching Live Football injuries, lineups & stats…"):
                st.session_state[lfa_state_key] = fetch_live_football_team_news(
                    lfa_key, lfa_league, kickoff_date, home_name, away_name
                )

        odds_bundle = st.session_state.get(odds_state_key)
        if odds_bundle is not None:
            odds_events = odds_bundle.get("events", [])
            if odds_bundle.get("error"):
                st.warning(f"Odds API: {odds_bundle['error']}")
            if odds_bundle.get("remaining") is not None:
                st.caption(f"The Odds API requests remaining: {odds_bundle['remaining']}")
        elif preloaded_odds_events is not None:
            odds_events = preloaded_odds_events
        else:
            odds_events = []
            st.caption("Odds API: Click 'Fetch Odds API' to load market prices and blend into prediction.")

        odds_event = match_odds_event(home_name, away_name, odds_events)
        odds = parse_global_odds(odds_event, home_name, away_name)

        lfa_bundle = st.session_state.get(lfa_state_key, {})
        summary = summarize_team_news(lfa_bundle, home_name, away_name) if lfa_bundle else {}
        if lfa_bundle:
            for err in lfa_bundle.get("errors", []):
                st.warning(f"Live Football API: {err}")
            if summary:
                st.write({
                    "Live Football match ID": summary["fixture_id"],
                    f"{home_name} injuries": summary["home_injuries"] or "No reported entries / unavailable",
                    f"{away_name} injuries": summary["away_injuries"] or "No reported entries / unavailable",
                    f"{home_name} lineup confirmed": summary["home_lineup_confirmed"],
                    f"{away_name} lineup confirmed": summary["away_lineup_confirmed"],
                })
                if summary.get("statistics_available"):
                    st.caption(
                        f"Fixture stats: {home_name} xG={summary['home_xg']} / SoT={summary['home_shots_on_target']}; "
                        f"{away_name} xG={summary['away_xg']} / SoT={summary['away_shots_on_target']}."
                    )

        with st.expander("Manual tactical and motivation inputs", expanded=False):
            st.caption("Analyst inputs: zero means neutral.")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                home_motivation = st.slider("Home motivation", -2.0, 2.0, 0.0, 0.25, key=f"mot_h_{match_id}")
            with c2:
                away_motivation = st.slider("Away motivation", -2.0, 2.0, 0.0, 0.25, key=f"mot_a_{match_id}")
            with c3:
                home_tactical = st.slider("Home tactical adjustment", -2.0, 2.0, 0.0, 0.25, key=f"tac_h_{match_id}")
            with c4:
                away_tactical = st.slider("Away tactical adjustment", -2.0, 2.0, 0.0, 0.25, key=f"tac_a_{match_id}")

        manual_context = {
            "home_motivation": home_motivation,
            "away_motivation": away_motivation,
            "home_tactical": home_tactical,
            "away_tactical": away_tactical,
        }
        prediction = predict_match(
            match,
            league_context,
            manual_context=manual_context,
            team_news=summary,
            parsed_odds=odds if odds_event else None,
            odds_weight=odds_weight,
        )
        probs = prediction["probabilities"]
        b1x2 = prediction["blended_1x2"]

        st.markdown("#### Dashboard prediction")
        home_goals, away_goals = prediction["best_score"]
        st.markdown(f"## **{home_goals} – {away_goals}**")
        st.caption(
            f"1X2 ({'Odds+Model blended' if prediction['odds_used'] else 'Model'}): "
            f"Home {b1x2['home_win']*100:.1f}% | Draw {b1x2['draw']*100:.1f}% | Away {b1x2['away_win']*100:.1f}% | Over 2.5: {probs['over25']*100:.1f}%"
        )

        st.markdown("#### Goal Difference (ဂိုးပြတ်နိုင်ခြေ)")
        gd_df = pd.DataFrame([
            {"Goal Diff Market": "Home win by 2+ goals (GD >= +2)", "Probability": f"{probs['home_gd_2plus']*100:.1f}%"},
            {"Goal Diff Market": "Home win by 3+ goals (GD >= +3)", "Probability": f"{probs['home_gd_3plus']*100:.1f}%"},
            {"Goal Diff Market": "Away win by 2+ goals (GD <= -2)", "Probability": f"{probs['away_gd_2plus']*100:.1f}%"},
            {"Goal Diff Market": "Away win by 3+ goals (GD <= -3)", "Probability": f"{probs['away_gd_3plus']*100:.1f}%"},
            {"Goal Diff Market": "Any 2+ goal margin (|GD| >= 2)", "Probability": f"{probs['any_gd_2plus']*100:.1f}%"},
            {"Goal Diff Market": "Expected Goal Difference (Home - Away)", "Probability": f"{probs['expected_gd']:+.2f} goals"},
        ])
        st.dataframe(gd_df, hide_index=True, use_container_width=True)

        with st.expander("Model components and data quality", expanded=False):
            st.write(f"Expected goals: home {prediction['home_lambda']:.2f}, away {prediction['away_lambda']:.2f} (Total {prediction['expected_total_goals']:.2f})")
            st.write(f"Elo: home {prediction['home_elo']:.0f}, away {prediction['away_elo']:.0f}; gap {prediction['elo_difference']:+.0f}.")
            st.write(f"Recent rest days: home {prediction['home_rest_days'] if prediction['home_rest_days'] is not None else 'unknown'}, away {prediction['away_rest_days'] if prediction['away_rest_days'] is not None else 'unknown'}.")
            st.write(f"Injuries: home {prediction['home_injury_count']}, away {prediction['away_injury_count']} | Lineup confirmed: home={prediction['home_lineup_confirmed']}, away={prediction['away_lineup_confirmed']}.")
            st.write(f"Odds blended into prediction: {'Yes' if prediction['odds_used'] else 'No (fetch odds first)'} | Data quality: {prediction['data_quality']*100:.0f}%.")

        if odds_event:
            st.caption(f"Global odds: {odds['bookmakers']} bookmakers | last update: {odds['last_update'] or 'not provided'}")
            render_value_finder(prediction, odds)
        elif odds_bundle is not None:
            st.info("No matching global odds were returned for this fixture.")
        return prediction

    cached_odds = st.session_state.get(odds_state_key, {})
    cached_lfa = st.session_state.get(lfa_state_key, {})
    ev_list = cached_odds.get("events", preloaded_odds_events or [])
    ev_match = match_odds_event(home_name, away_name, ev_list)
    parsed_ev = parse_global_odds(ev_match, home_name, away_name)
    lfa_sum = summarize_team_news(cached_lfa, home_name, away_name) if cached_lfa else {}
    return predict_match(
        match,
        league_context,
        team_news=lfa_sum,
        parsed_odds=parsed_ev if ev_match else None,
        odds_weight=odds_weight,
    )


def make_league_context(league_name: str, football_key: str, days_back: int) -> Tuple[Dict[str, Any], List[str]]:
    league_code = LEAGUES[league_name]["football_data"]
    start, end = finished_window(days_back)
    results, result_error = fetch_finished_matches(league_code, football_key, start, end)
    standings, standings_error = fetch_standings(league_code, football_key)
    context = build_team_context(results, standings)
    errors = [err for err in (result_error, standings_error) if err]
    return context, errors


def render_selected_league(
    league_name: str,
    football_key: str,
    odds_key: str,
    lfa_key: str,
    regions: str,
    odds_weight: float,
    days_back: int,
) -> None:
    cfg = LEAGUES[league_name]
    date_from, date_to, start, end = fixture_window(7)
    fixtures_raw, fixtures_error = fetch_fixtures(cfg["football_data"], football_key, date_from, date_to)
    fixtures = upcoming_only(fixtures_raw, start, end)
    context, context_errors = make_league_context(league_name, football_key, days_back)

    st.subheader(f"{league_name} — upcoming fixtures")
    for err in context_errors:
        st.warning(err)
    if fixtures_error:
        st.warning(fixtures_error)
    if not fixtures:
        st.info("No scheduled matches in the next 7 days, or football-data.org did not return fixture access for this competition.")
        return
    st.success(f"{len(fixtures)} fixture(s) found. Odds & Live Football calls run on demand per match.")
    for match in fixtures:
        render_fixture_card(
            match,
            league_name,
            context,
            odds_key,
            cfg["odds"],
            regions,
            odds_weight,
            lfa_key,
            cfg["lfa_league"],
            preloaded_odds_events=None,
        )


def render_top10(
    football_key: str,
    odds_key: str,
    lfa_key: str,
    regions: str,
    odds_weight: float,
    days_back: int,
) -> None:
    st.subheader("Daily Best Prediction Matches — Top 10 (Auto Odds Fetch)")
    st.caption("Opening Top 10 automatically fetches Odds API for leagues present in the upcoming window.")
    date_from, date_to, start, end = fixture_window(7)
    entries: List[Dict[str, Any]] = []
    errors: List[str] = []
    active_leagues: Dict[str, Dict[str, Any]] = {}

    for league_name, cfg in LEAGUES.items():
        raw, fixture_error = fetch_fixtures(cfg["football_data"], football_key, date_from, date_to)
        fixtures = upcoming_only(raw, start, end)
        if fixture_error:
            errors.append(f"{league_name}: {fixture_error}")
        if not fixtures:
            continue
        active_leagues[league_name] = cfg
        context, context_errors = make_league_context(league_name, football_key, days_back)
        errors.extend(f"{league_name}: {err}" for err in context_errors)
        for match in fixtures:
            prediction = predict_match(match, context)
            entries.append({
                "Match": f"{match.get('homeTeam', {}).get('name')} vs {match.get('awayTeam', {}).get('name')}",
                "League": league_name,
                "Kickoff (MMT)": local_datetime(match.get("utcDate")),
                "Prediction": f"{prediction['best_score'][0]}–{prediction['best_score'][1]}",
                "_rank": prediction["top_outcome_probability"] * (0.75 + 0.25 * prediction["data_quality"]),
                "_fixture": match,
                "_context": context,
            })

    for err in errors:
        st.warning(err)
    if not entries:
        st.info("No upcoming fixtures were returned. Check provider keys, plan access, or fixture availability.")
        return

    odds_by_league: Dict[str, List[Dict[str, Any]]] = {}
    if odds_key:
        for lg_name, cfg in active_leagues.items():
            ev_list, od_err, rem = fetch_global_odds(cfg["odds"], odds_key, regions)
            odds_by_league[lg_name] = ev_list
            if od_err:
                st.caption(f"{lg_name} Odds: {od_err}")
            elif rem is not None:
                st.caption(f"{lg_name} Odds remaining quota: {rem}")

    entries.sort(key=lambda row: row["_rank"], reverse=True)
    top = entries[:10]
    table = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in top])
    st.dataframe(table, hide_index=True, use_container_width=True)

    for rank, row in enumerate(top, 1):
        st.markdown(f"#### #{rank} {row['Match']} — {row['League']}")
        cfg = LEAGUES[row["League"]]
        render_fixture_card(
            row["_fixture"],
            row["League"],
            row["_context"],
            odds_key,
            cfg["odds"],
            regions,
            odds_weight,
            lfa_key,
            cfg["lfa_league"],
            preloaded_odds_events=odds_by_league.get(row["League"], []),
        )


# -----------------------------
# Main application
# -----------------------------

st.title("⚽ Professional Football Prediction Dashboard V8")
st.markdown("On-demand Odds & Live Football API (Auto Odds on Top 10) + Goal Difference + Elo / Dixon–Coles")

with st.sidebar:
    st.header("API connection settings")
    football_key = st.text_input("Football-data.org API token", value=safe_secret("FOOTBALL_DATA_API_KEY"), type="password")
    odds_key = st.text_input("The Odds API key", value=safe_secret("THE_ODDS_API_KEY"), type="password")
    lfa_key = st.text_input("Live Football API key", value=safe_secret("LIVE_FOOTBALL_API_KEY"), type="password")
    regions_choice = st.multiselect("Global odds regions", ["eu", "uk", "us", "au"], default=["eu"])
    odds_weight = st.slider("Odds weight in prediction", min_value=0.0, max_value=0.50, value=0.25, step=0.05)
    days_back = st.slider("Recent match history used", min_value=60, max_value=300, value=210, step=30)
    nav = st.radio("Navigation", ["Selected league", "Daily Best Top 10", "About the model"])

regions = ",".join(regions_choice) if regions_choice else "eu"

if nav == "Selected league":
    selected_league = st.sidebar.selectbox("League", list(LEAGUES.keys()))
    render_selected_league(selected_league, football_key, odds_key, lfa_key, regions, odds_weight, days_back)
elif nav == "Daily Best Top 10":
    render_top10(football_key, odds_key, lfa_key, regions, odds_weight, days_back)
else:
    st.subheader("Model layers and provider limitations")
    st.markdown(
        """
- **On-demand buttons:** Odds API and Live Football API buttons fetch data only when clicked in Selected League mode.
- **Daily Best Top 10:** Automatically fetches Odds API for active leagues in the upcoming window.
- **Prediction integration:** Fetched no-vig odds, injury counts, and lineup confirmation status are incorporated into match probabilities and goal rates.
- **Goal Difference:** Displays probabilities for 2+ and 3+ goal margins for home and away teams alongside expected goal difference.
        """
    )
