from __future__ import annotations

"""Professional Football Prediction Dashboard V8.0 Providers: - football-data.org v4: fixtures, recent finished results, standings - The Odds API v4: multi-region h2h and totals odds - API-Football v3: fixture injuries, lineups, and available match statistics Model: - chronological Elo ratings - league-relative venue-specific attack/defence goal-rate xG proxy - Dixon-Coles-adjusted Poisson score matrix - short-rest, injuries, and user-entered tactical/motivation adjustments Important: this is a transparent baseline model, not a trained/calibrated guarantee. The expected-goals values are goal-rate proxies unless a provider explicitly returns xG. """

import math
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Official API references:
# football-data.org v4: https://docs.football-data.org/general/v4/index.html
# The Odds API v4: https://the-odds-api.com/liveapi/guides/v4/
# API-Football v3: https://www.api-football.com/documentation-v3

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
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
REQUEST_TIMEOUT = 18
MODEL_VERSION = "elo-dixon-coles-v8.0"
DIXON_COLES_RHO = -0.06  # transparent starting prior; should be estimated by backtest
HOME_ELO_ADVANTAGE = 55.0

LEAGUES: Dict[str, Dict[str, Any]] = {
    "England Premier League": {"football_data": "PL", "odds": "soccer_epl", "api_football": 39},
    "Spain LaLiga": {"football_data": "PD", "odds": "soccer_spain_la_liga", "api_football": 140},
    "Germany Bundesliga": {"football_data": "BL1", "odds": "soccer_germany_bundesliga", "api_football": 78},
    "Italy Serie A": {"football_data": "SA", "odds": "soccer_italy_serie_a", "api_football": 135},
    "France Ligue 1": {"football_data": "FL1", "odds": "soccer_france_ligue_one", "api_football": 61},
    "UEFA Champions League": {"football_data": "CL", "odds": "soccer_uefa_champs_league", "api_football": 2},
    "UEFA Europa League": {"football_data": "EL", "odds": "soccer_uefa_europa_league", "api_football": 3},
}

# -----------------------------
# Generic helpers
# -----------------------------


def safe_secret(name: str, default: str = "") -> str:
    """Read an optional Streamlit secret without failing when no secrets file exists."""
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


def http_json( url: str, *, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None, ) -> Tuple[Optional[Any], Optional[str], Dict[str, str]]:
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


def result_code(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals == away_goals:
        return "D"
    return "A"


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
    # football-data.org dateTo is exclusive; include one extra calendar date.
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
# Provider 3: API-Football v3 team news
# -----------------------------


def api_football_get(api_key: str, endpoint: str, params: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not api_key:
        return None, "API-Football key is not configured."
    return_data, error, _ = http_json(
        f"{API_FOOTBALL_BASE}/{endpoint.lstrip('/')}",
        headers={"x-apisports-key": api_key},
        params=params,
    )
    if error:
        return None, error
    if isinstance(return_data, dict) and return_data.get("errors"):
        return None, str(return_data["errors"])
    return return_data, None


def season_start_year(kickoff: Any) -> int:
    dt = parse_datetime(kickoff)
    if not dt:
        return utc_now().year if utc_now().month >= 7 else utc_now().year - 1
    return dt.year if dt.month >= 7 else dt.year - 1


@st.cache_data(ttl=900, show_spinner=False)
def fetch_api_football_team_news( api_key: str, api_league_id: int, season: int, kickoff_date: str, home_name: str, away_name: str, ) -> Dict[str, Any]:
    result: Dict[str, Any] = {"fixture": {}, "injuries": [], "lineups": [], "statistics": [], "errors": []}
    if not api_key:
        result["errors"].append("API-Football key is not configured.")
        return result
    fixtures, error = api_football_get(api_key, "fixtures", {"league": api_league_id, "season": season, "date": kickoff_date})
    if error:
        result["errors"].append(f"Fixture lookup: {error}")
        return result
    candidates = (fixtures or {}).get("response", [])
    matched = None
    for item in candidates:
        teams = item.get("teams", {})
        if same_team(home_name, teams.get("home", {}).get("name")) and same_team(away_name, teams.get("away", {}).get("name")):
            matched = item
            break
    fixture_id = (matched or {}).get("fixture", {}).get("id")
    if not fixture_id:
        result["errors"].append("Matching fixture not found in API-Football for this date. Team naming or competition coverage may differ.")
        return result
    result["fixture"] = matched
    for endpoint, key in (("injuries", "injuries"), ("fixtures/lineups", "lineups"), ("fixtures/statistics", "statistics")):
        payload, endpoint_error = api_football_get(api_key, endpoint, {"fixture": fixture_id})
        if endpoint_error:
            result["errors"].append(f"{endpoint}: {endpoint_error}")
        else:
            result[key] = (payload or {}).get("response", [])
    return result


def summarize_team_news(bundle: Dict[str, Any], home_name: str, away_name: str) -> Dict[str, Any]:
    summary = {
        "fixture_id": None,
        "home_injuries": [], "away_injuries": [],
        "home_lineup_confirmed": False, "away_lineup_confirmed": False,
        "home_starters": None, "away_starters": None,
        "statistics_available": False,
        "home_xg": None, "away_xg": None,
        "home_shots_on_target": None, "away_shots_on_target": None,
        "lineup_note": "Lineup data is provider/competition dependent and may only appear shortly before kickoff.",
    }
    fixture = bundle.get("fixture", {})
    summary["fixture_id"] = fixture.get("fixture", {}).get("id")
    team_ids = fixture.get("teams", {})
    home_id = team_ids.get("home", {}).get("id")
    away_id = team_ids.get("away", {}).get("id")
    for item in bundle.get("injuries", []):
        team_id = item.get("team", {}).get("id")
        player = item.get("player", {}).get("name", "Unknown player")
        reason = item.get("player", {}).get("reason") or item.get("player", {}).get("type") or "Unavailable"
        if team_id == home_id:
            summary["home_injuries"].append(f"{player} ({reason})")
        elif team_id == away_id:
            summary["away_injuries"].append(f"{player} ({reason})")
    for lineup in bundle.get("lineups", []):
        team_id = lineup.get("team", {}).get("id")
        starters = lineup.get("startXI", [])
        if team_id == home_id:
            summary["home_lineup_confirmed"] = bool(starters)
            summary["home_starters"] = len(starters) if starters else None
        elif team_id == away_id:
            summary["away_lineup_confirmed"] = bool(starters)
            summary["away_starters"] = len(starters) if starters else None
    for team_stats in bundle.get("statistics", []):
        team_id = team_stats.get("team", {}).get("id")
        side = "home" if team_id == home_id else "away" if team_id == away_id else None
        if not side:
            continue
        for stat in team_stats.get("statistics", []):
            key = re.sub(r"[^a-z]", "", str(stat.get("type", "")).lower())
            value = stat.get("value")
            if key in {"expectedgoals", "xg"}:
                summary[f"{side}_xg"] = safe_float(value)
            if key in {"shotsongoal", "shotsontarget"}:
                summary[f"{side}_shots_on_target"] = safe_float(value)
            if key:
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


def build_team_context( finished_matches: List[Dict[str, Any]], standings: Dict[str, Dict[str, Any]], ) -> Dict[str, Any]:
    """Build chronological Elo ratings, venue-specific recent rates, and league priors."""
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
    # Include teams with standings but no finished scores in the rolling window.
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
    """Independent Poisson score matrix with a low-score Dixon-Coles correction."""
    matrix = np.array([
        [poisson_pmf(home_lambda, h) * poisson_pmf(away_lambda, a) for a in range(max_goals + 1)]
        for h in range(max_goals + 1)
    ], dtype=float)
    # Dixon-Coles tau adjustments for 0-0, 0-1, 1-0 and 1-1.
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
        "home_win": home_win, "draw": draw, "away_win": away_win,
        "over15": over15, "over25": over25, "over35": over35, "over45": over45,
        "under25": 1.0 - over25,
        "home_scores_1plus": home_scores, "away_scores_1plus": away_scores,
        "btts_yes": btts_yes, "btts_no": 1.0 - btts_yes,
        "total_probs": total_probs,
    }


def _get_profile(context: Dict[str, Any], team_name: str) -> Dict[str, Any]:
    key = normalize_team(team_name)
    found = context.get("teams", {}).get(key)
    if found:
        return found
    # Alias/partial matching for provider naming differences.
    for profile_key, profile in context.get("teams", {}).items():
        if same_team(team_name, profile.get("name", profile_key)):
            return profile
    return {"name": team_name, "elo": 1500.0, "recent": [], "home_games": [], "away_games": [], "recent_points": 0, "recent_played": 0, "last_match": None, "standing": {}}


def predict_match( match: Dict[str, Any], league_context: Dict[str, Any], *, manual_context: Optional[Dict[str, float]] = None, team_news: Optional[Dict[str, Any]] = None, ) -> Dict[str, Any]:
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

    # Elo adjusts the goal rates modestly; league home scoring already contains home advantage.
    elo_difference = float(home.get("elo", 1500.0)) - float(away.get("elo", 1500.0))
    elo_shift = clamp(elo_difference / 2200.0, -0.22, 0.22)
    home_lambda *= math.exp(elo_shift)
    away_lambda *= math.exp(-elo_shift * 0.80)

    # Schedule congestion: conservatively lower attacking rates after short rest.
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
    # Injury counts are not player-value ratings; apply only a very small capped team-level adjustment.
    home_lambda *= math.exp(-min(0.12, 0.018 * home_injuries))
    away_lambda *= math.exp(-min(0.12, 0.018 * away_injuries))
    home_lambda = clamp(home_lambda, 0.20, 4.20)
    away_lambda = clamp(away_lambda, 0.15, 3.80)

    matrix = dixon_coles_matrix(home_lambda, away_lambda)
    flat_index = int(np.argmax(matrix))
    best_score = tuple(int(x) for x in np.unravel_index(flat_index, matrix.shape))
    probs = calculate_probabilities(matrix)
    top_outcome = max(probs["home_win"], probs["draw"], probs["away_win"])
    played_mean = (home.get("recent_played", 0) + away.get("recent_played", 0)) / 2
    quality = clamp(0.20 + min(0.45, played_mean / 20.0) + (0.20 if home.get("standing") else 0.0) + (0.15 if away.get("standing") else 0.0), 0.10, 1.0)
    return {
        "home_name": home_name, "away_name": away_name,
        "home_lambda": home_lambda, "away_lambda": away_lambda,
        "expected_total_goals": home_lambda + away_lambda,
        "best_score": best_score,
        "best_score_probability": float(matrix[best_score]),
        "matrix": matrix,
        "probabilities": probs,
        "home_elo": float(home.get("elo", 1500.0)),
        "away_elo": float(away.get("elo", 1500.0)),
        "elo_difference": elo_difference,
        "home_attack_index": home_attack / base_home,
        "away_attack_index": away_attack / base_away,
        "home_rest_days": rest_days["home"],
        "away_rest_days": rest_days["away"],
        "home_rest_factor": rest_factor["home"],
        "away_rest_factor": rest_factor["away"],
        "home_injury_count": home_injuries,
        "away_injury_count": away_injuries,
        "top_outcome_probability": top_outcome,
        "data_quality": quality,
        "xg_source": "league-relative goals-for/against proxy; not provider shot-based xG",
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


def render_fixture_card( match: Dict[str, Any], league_name: str, league_context: Dict[str, Any], api_football_key: str, api_league_id: int, odds_data: List[Dict[str, Any]], ) -> Dict[str, Any]:
    home_name = match.get("homeTeam", {}).get("name", "Home")
    away_name = match.get("awayTeam", {}).get("name", "Away")
    match_id = str(match.get("id", f"{home_name}-{away_name}"))
    odds_event = match_odds_event(home_name, away_name, odds_data)
    odds = parse_global_odds(odds_event, home_name, away_name)
    saved_key = f"api_football_news_{match_id}"
    with st.expander(f"⚽ {home_name} vs {away_name} | {local_datetime(match.get('utcDate'))}", expanded=False):
        st.markdown(f"### {home_name} **vs** {away_name}")
        st.caption(f"{league_name} | Kickoff {local_datetime(match.get('utcDate'))} | {MODEL_VERSION}")

        with st.expander("Manual tactical and motivation inputs", expanded=False):
            st.caption("These are analyst inputs, not official API fields. Zero means neutral; use only when you have a documented reason.")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                home_motivation = st.slider("Home motivation", -2.0, 2.0, 0.0, 0.25, key=f"mot_h_{match_id}")
            with c2:
                away_motivation = st.slider("Away motivation", -2.0, 2.0, 0.0, 0.25, key=f"mot_a_{match_id}")
            with c3:
                home_tactical = st.slider("Home tactical adjustment", -2.0, 2.0, 0.0, 0.25, key=f"tac_h_{match_id}")
            with c4:
                away_tactical = st.slider("Away tactical adjustment", -2.0, 2.0, 0.0, 0.25, key=f"tac_a_{match_id}")

        season = season_start_year(match.get("utcDate"))
        kickoff_date = (parse_datetime(match.get("utcDate")) or utc_now()).date().isoformat()
        fetch_col, note_col = st.columns([1, 2])
        with fetch_col:
            fetch_clicked = st.button("Fetch API-Football injuries/lineup", key=f"fetch_team_news_{match_id}", disabled=not bool(api_football_key))
        with note_col:
            st.caption("Lineups/injuries depend on API coverage; lineups can appear only shortly before kickoff.")
        if fetch_clicked:
            with st.spinner("Looking up fixture, injuries, lineups and available statistics…"):
                st.session_state[saved_key] = fetch_api_football_team_news(
                    api_football_key, api_league_id, season, kickoff_date, home_name, away_name
                )
        bundle = st.session_state.get(saved_key, {})
        summary = summarize_team_news(bundle, home_name, away_name) if bundle else {}
        if bundle:
            for err in bundle.get("errors", []):
                st.warning(err)
            if summary:
                st.write({
                    "API-Football fixture ID": summary["fixture_id"],
                    f"{home_name} injuries": summary["home_injuries"] or "No reported entries / unavailable",
                    f"{away_name} injuries": summary["away_injuries"] or "No reported entries / unavailable",
                    f"{home_name} lineup confirmed": summary["home_lineup_confirmed"],
                    f"{away_name} lineup confirmed": summary["away_lineup_confirmed"],
                })
                if summary.get("statistics_available"):
                    st.caption(f"Available fixture statistics: {home_name} xG={summary['home_xg']} / shots on target={summary['home_shots_on_target']}; {away_name} xG={summary['away_xg']} / shots on target={summary['away_shots_on_target']}. These in-match stats are not treated as pre-match xG forecasts.")

        manual_context = {
            "home_motivation": home_motivation, "away_motivation": away_motivation,
            "home_tactical": home_tactical, "away_tactical": away_tactical,
        }
        prediction = predict_match(match, league_context, manual_context=manual_context, team_news=summary)
        st.markdown("#### Dashboard prediction")
        home_goals, away_goals = prediction["best_score"]
        st.markdown(f"## **{home_goals} – {away_goals}**")
        st.caption("One most-likely exact score only. Score is a probability mode, not a guaranteed result.")

        with st.expander("Model components and data quality", expanded=False):
            st.write(f"Expected-goals proxy: home {prediction['home_lambda']:.2f}, away {prediction['away_lambda']:.2f} — {prediction['xg_source']}")
            st.write(f"Elo: home {prediction['home_elo']:.0f}, away {prediction['away_elo']:.0f}; rating gap {prediction['elo_difference']:+.0f}.")
            st.write(f"Recent rest days: home {prediction['home_rest_days'] if prediction['home_rest_days'] is not None else 'unknown'}, away {prediction['away_rest_days'] if prediction['away_rest_days'] is not None else 'unknown'}.")
            st.write(f"Reported injury counts used for small adjustment: home {prediction['home_injury_count']}, away {prediction['away_injury_count']}.")
            st.write(f"League finished-match history used: {league_context.get('finished_count', 0)}; data coverage indicator: {prediction['data_quality'] * 100:.0f}%.")
            st.caption(f"Dixon-Coles rho={DIXON_COLES_RHO:.2f} is a documented starting value, not a fitted league parameter. Backtest and tune by league/season before operational or betting use.")

        if odds_event:
            st.caption(f"Global odds: {odds['bookmakers']} bookmakers | last update: {odds['last_update'] or 'not provided'}")
            render_value_finder(prediction, odds)
        else:
            st.info("No matching global odds were returned for this fixture.")
        return prediction
    return predict_match(match, league_context)


def make_league_context(league_name: str, football_key: str, days_back: int) -> Tuple[Dict[str, Any], List[str]]:
    league_code = LEAGUES[league_name]["football_data"]
    start, end = finished_window(days_back)
    results, result_error = fetch_finished_matches(league_code, football_key, start, end)
    standings, standings_error = fetch_standings(league_code, football_key)
    context = build_team_context(results, standings)
    errors = [err for err in (result_error, standings_error) if err]
    return context, errors


def render_selected_league( league_name: str, football_key: str, odds_key: str, api_football_key: str, regions: str, days_back: int, ) -> None:
    cfg = LEAGUES[league_name]
    date_from, date_to, start, end = fixture_window(7)
    fixtures_raw, fixtures_error = fetch_fixtures(cfg["football_data"], football_key, date_from, date_to)
    fixtures = upcoming_only(fixtures_raw, start, end)
    context, context_errors = make_league_context(league_name, football_key, days_back)
    odds_data, odds_error, remaining = fetch_global_odds(cfg["odds"], odds_key, regions)
    st.subheader(f"{league_name} — upcoming fixtures")
    for err in context_errors:
        st.warning(err)
    if fixtures_error:
        st.warning(fixtures_error)
    if odds_error:
        st.info(odds_error)
    if remaining is not None:
        st.caption(f"The Odds API requests remaining: {remaining}")
    if not fixtures:
        st.info("No scheduled matches in the next 7 days, or football-data.org did not return fixture access for this competition.")
        return
    st.success(f"{len(fixtures)} fixture(s) found.")
    for match in fixtures:
        render_fixture_card(match, league_name, context, api_football_key, cfg["api_football"], odds_data)


def render_top10( football_key: str, odds_key: str, api_football_key: str, regions: str, days_back: int, ) -> None:
    st.subheader("Daily Best Prediction Matches — Top 10")
    st.caption("Ranks upcoming fixtures in the next 7 days by the strongest 1X2 model outcome, then by data coverage. This is a ranking, not a confidence guarantee.")
    st.caption("The first uncached scan may use up to 21 football-data.org calls and up to 7 Odds API calls across the leagues; cached data reduces repeat requests.")
    date_from, date_to, start, end = fixture_window(7)
    entries: List[Dict[str, Any]] = []
    errors: List[str] = []
    for league_name, cfg in LEAGUES.items():
        raw, fixture_error = fetch_fixtures(cfg["football_data"], football_key, date_from, date_to)
        fixtures = upcoming_only(raw, start, end)
        if fixture_error:
            errors.append(f"{league_name}: {fixture_error}")
        if not fixtures:
            continue
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
                "_prediction": prediction,
            })
    for err in errors:
        st.warning(err)
    if not entries:
        st.info("No upcoming fixtures were returned. Check provider keys, plan access, or fixture availability.")
        return
    entries.sort(key=lambda row: row["_rank"], reverse=True)
    top = entries[:10]
    table = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in top])
    st.dataframe(table, hide_index=True, use_container_width=True)
    for rank, row in enumerate(top, 1):
        st.markdown(f"#### #{rank} {row['Match']} — {row['League']}")
        cfg = LEAGUES[row["League"]]
        odds_data, odds_error, _ = fetch_global_odds(cfg["odds"], odds_key, regions)
        if odds_error:
            st.caption(odds_error)
        render_fixture_card(row["_fixture"], row["League"], row["_context"], api_football_key, cfg["api_football"], odds_data)


# -----------------------------
# Main application
# -----------------------------

st.title("⚽ Professional Football Prediction Dashboard V8")
st.markdown("Football-data.org fixtures + The Odds API global prices + API-Football team news | Elo + Dixon–Coles/Poisson")

with st.sidebar:
    st.header("API connection settings")
    football_key = st.text_input("Football-data.org API token", value=safe_secret("FOOTBALL_DATA_API_KEY"), type="password", help="Used for competition fixtures, recent results and standings.")
    odds_key = st.text_input("The Odds API key", value=safe_secret("THE_ODDS_API_KEY"), type="password", help="Uses v4 h2h and totals markets; global regions consume more quota.")
    api_football_key = st.text_input("API-Football v3 key", value=safe_secret("API_FOOTBALL_KEY"), type="password", help="Used on demand for injuries, lineups and fixture statistics.")
    regions_choice = st.multiselect("Global odds regions", ["eu", "uk", "us", "au"], default=["eu", "uk"])
    st.caption("Odds quota is typically charged per market × region for each sport request. Cached for 10 minutes.")
    days_back = st.slider("Recent match history used", min_value=60, max_value=300, value=210, step=30)
    nav = st.radio("Navigation", ["Selected league", "Daily Best Top 10", "About the model"])

regions = ",".join(regions_choice) if regions_choice else "eu"

if nav == "Selected league":
    selected_league = st.sidebar.selectbox("League", list(LEAGUES.keys()))
    render_selected_league(selected_league, football_key, odds_key, api_football_key, regions, days_back)
elif nav == "Daily Best Top 10":
    render_top10(football_key, odds_key, api_football_key, regions, days_back)
else:
    st.subheader("Model layers and provider limitations")
    st.markdown(
        """ - **Fixtures, recent results and standings:** football-data.org v4. Competition access depends on the provider plan and account. - **Global odds:** The Odds API v4 `h2h` and `totals` markets. Returned odds can vary by region, bookmaker, and fixture availability. - **Injuries, lineups and available match statistics:** API-Football v3. Lineups and statistics are not available for every competition or before every kickoff. - **Elo layer:** ratings are updated chronologically from finished match results, with a modest home-field rating prior. - **Goal model:** venue-specific recent goals-for/goals-against rates are shrunk toward league averages. The output is a goal-rate xG proxy, not shot-based xG unless an API explicitly supplies xG. - **Score probabilities:** Dixon-Coles low-score adjustment is applied to a Poisson score matrix. The rho value is an untrained starting prior and should be tuned by historical backtesting. - **Context adjustments:** short rest and reported injury counts have small capped effects. Motivation and tactical inputs are manual analyst assumptions; they are not objective provider measurements. - **Value Bet Finder:** compares model probabilities with no-vig averages from bookmaker markets and computes expected value at the best returned price. A positive edge is not a guaranteed profit. This dashboard provides model estimates, not guaranteed outcomes or financial advice. Validate the model with out-of-sample results, proper scoring rules, and closing odds before relying on it. """
    )