from __future__ import annotations

import math
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============================================================
# Professional Football Predictor V7.1.1 (High-Score Enabled)
# - Selected league view
# - Top 10 cross-league view
# - Prediction history and multi-metric accuracy
# - Robust API/error handling
# - Data-quality and model-confidence disclosure
# - Reproducible Poisson prediction with High-Score (3+) tuning
# ============================================================

st.set_page_config(
    page_title="Professional Football Predictor V7.1",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "football_history.db"
LOCAL_TZ = ZoneInfo("Asia/Yangon")
UTC = timezone.utc

LEAGUES: Dict[str, str] = {
    "English Premier League": "PL",
    "Italian Serie A": "SA",
    "Spanish La Liga": "PD",
    "German Bundesliga": "BL1",
    "French Ligue 1": "FL1",
    "UEFA Champions League": "CL",
    "UEFA Europa League": "EL",
}

MODEL_VERSION = "poisson-v7.1-highscore"
REQUEST_TIMEOUT = 20

# -----------------------------
# Generic helpers
# -----------------------------

def now_utc() -> datetime:
    return datetime.now(UTC)


def iso_now() -> str:
    return now_utc().isoformat()


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def local_time(value: str) -> str:
    return parse_utc(value).astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M MMT")


def normalize_name(name: Any) -> str:
    text = str(name or "").lower()
    replacements = {
        "&": "and",
        "fc": "",
        "afc": "",
        "cf": "",
        "ssc": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"[^a-z0-9]", "", text)


def team_match(left: Any, right: Any) -> bool:
    a, b = normalize_name(left), normalize_name(right)
    if not a or not b:
        return False
    return a == b or (len(a) >= 6 and a in b) or (len(b) >= 6 and b in a)


def result_code(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals == away_goals:
        return "D"
    return "A"

# -----------------------------
# Database
# -----------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT UNIQUE NOT NULL,
    league_code TEXT,
    league_name TEXT,
    match_date TEXT,
    home_team TEXT,
    away_team TEXT,
    predicted_home INTEGER,
    predicted_away INTEGER,
    home_lambda REAL,
    away_lambda REAL,
    home_prob REAL,
    draw_prob REAL,
    away_prob REAL,
    over25_prob REAL,
    under25_prob REAL,
    btts_yes_prob REAL,
    confidence REAL,
    data_quality REAL,
    model_version TEXT,
    actual_home INTEGER,
    actual_away INTEGER,
    result_correct INTEGER,
    exact_score_correct INTEGER,
    over25_correct INTEGER,
    btts_correct INTEGER,
    created_at TEXT,
    actual_updated_at TEXT
)
"""


def db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    with db_connection() as conn:
        conn.execute(SCHEMA)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
        additions = {
            "league_code": "TEXT",
            "league_name": "TEXT",
            "home_prob": "REAL",
            "draw_prob": "REAL",
            "away_prob": "REAL",
            "over25_prob": "REAL",
            "under25_prob": "REAL",
            "btts_yes_prob": "REAL",
            "confidence": "REAL",
            "data_quality": "REAL",
            "model_version": "TEXT",
            "result_correct": "INTEGER",
            "exact_score_correct": "INTEGER",
            "over25_correct": "INTEGER",
            "btts_correct": "INTEGER",
            "actual_updated_at": "TEXT",
        }
        for column, kind in additions.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {column} {kind}")
        conn.commit()


def save_prediction(match: Dict[str, Any], prediction: Dict[str, Any], league_code: str, league_name: str) -> None:
    with db_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                match_id, league_code, league_name, match_date, home_team, away_team,
                predicted_home, predicted_away, home_lambda, away_lambda,
                home_prob, draw_prob, away_prob, over25_prob, under25_prob,
                btts_yes_prob, confidence, data_quality, model_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id) DO UPDATE SET
                league_code=excluded.league_code,
                league_name=excluded.league_name,
                home_lambda=excluded.home_lambda,
                away_lambda=excluded.away_lambda,
                home_prob=excluded.home_prob,
                draw_prob=excluded.draw_prob,
                away_prob=excluded.away_prob,
                over25_prob=excluded.over25_prob,
                under25_prob=excluded.under25_prob,
                btts_yes_prob=excluded.btts_yes_prob,
                confidence=excluded.confidence,
                data_quality=excluded.data_quality,
                model_version=excluded.model_version
            """,
            (
                str(match["id"]), league_code, league_name, match["utcDate"],
                match["homeTeam"]["name"], match["awayTeam"]["name"],
                int(prediction["best_score"][0]), int(prediction["best_score"][1]),
                prediction["home_lambda"], prediction["away_lambda"],
                prediction["probabilities"]["home_win"], prediction["probabilities"]["draw"],
                prediction["probabilities"]["away_win"], prediction["probabilities"]["over25"],
                prediction["probabilities"]["under25"], prediction["probabilities"]["btts_yes"],
                prediction["confidence"], prediction["data_quality"], MODEL_VERSION, iso_now(),
            ),
        )
        conn.commit()


def load_history() -> pd.DataFrame:
    with db_connection() as conn:
        return pd.read_sql_query("SELECT * FROM predictions ORDER BY match_date DESC", conn)

# -----------------------------
# API layer
# -----------------------------

def api_get(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[str]]:
    try:
        response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            try:
                return response.json(), None
            except ValueError:
                return None, "API returned invalid JSON."
        messages = {
            401: "API key is invalid or missing.",
            403: "API key does not have permission for this endpoint.",
            404: "Requested API endpoint was not found.",
            429: "API rate limit reached. Please wait before refreshing.",
        }
        return None, messages.get(response.status_code, f"API error: HTTP {response.status_code}")
    except requests.Timeout:
        return None, "API request timed out."
    except requests.RequestException as exc:
        return None, f"Network error: {exc}"


@st.cache_data(ttl=900, show_spinner=False)
def fetch_fixtures(league_code: str, api_key: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not api_key:
        return [], "Football-data.org API key is missing."
    data, error = api_get(
        f"https://api.football-data.org/v4/competitions/{league_code}/matches",
        headers={"X-Auth-Token": api_key},
        params={"status": "SCHEDULED"},
    )
    if error or not data:
        return [], error or "No fixture data returned."
    start, end = now_utc(), now_utc() + timedelta(days=7)
    fixtures = []
    for match in data.get("matches", []):
        try:
            kickoff = parse_utc(match["utcDate"])
            if start <= kickoff <= end:
                fixtures.append(match)
        except (KeyError, ValueError):
            continue
    return sorted(fixtures, key=lambda x: x.get("utcDate", "")), None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_finished(league_code: str, api_key: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not api_key:
        return [], "Football-data.org API key is missing."
    data, error = api_get(
        f"https://api.football-data.org/v4/competitions/{league_code}/matches",
        headers={"X-Auth-Token": api_key},
        params={"status": "FINISHED"},
    )
    if error or not data:
        return [], error or "No finished-match data returned."
    return data.get("matches", []), None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_standings(league_code: str, api_key: str) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    if not api_key:
        return {}, "Football-data.org API key is missing."
    data, error = api_get(
        f"https://api.football-data.org/v4/competitions/{league_code}/standings",
        headers={"X-Auth-Token": api_key},
    )
    if error or not data:
        return {}, error or "No standings data returned."
    stats: Dict[str, Dict[str, Any]] = {}
    for table in data.get("standings", []):
        if table.get("type") != "TOTAL":
            continue
        for row in table.get("table", []):
            team = row.get("team", {}).get("name")
            if not team:
                continue
            stats[team] = {
                "position": row.get("position"),
                "points": row.get("points", 0),
                "played": row.get("playedGames", 0),
                "gd": row.get("goalDifference", 0),
                "gf": row.get("goalsFor", 0),
                "ga": row.get("goalsAgainst", 0),
            }
    return stats, None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_odds(api_key: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not api_key:
        return [], "Odds API key is missing."
    data, error = api_get(
        "https://api.odds-api.io/v1/odds",
        params={"apiKey": api_key, "sport": "soccer", "region": "eu"},
    )
    if error or data is None:
        return [], error or "No odds data returned."
    if isinstance(data, list):
        return data, None
    if isinstance(data, dict):
        return data.get("data", []), None
    return [], "Unexpected odds API response format."


def update_actual_results(league_code: str, api_key: str) -> Tuple[int, Optional[str]]:
    matches, error = fetch_finished(league_code, api_key)
    if error:
        return 0, error
    updated = 0
    with db_connection() as conn:
        for match in matches:
            score = match.get("score", {}).get("fullTime", {})
            home_goals, away_goals = score.get("home"), score.get("away")
            match_id = str(match.get("id"))
            if home_goals is None or away_goals is None:
                continue
            row = conn.execute(
                "SELECT predicted_home, predicted_away, home_prob, draw_prob, away_prob, over25_prob, btts_yes_prob FROM predictions WHERE match_id=?",
                (match_id,),
            ).fetchone()
            if row is None:
                continue
            predicted_result = result_code(row[0], row[1])
            actual_result = result_code(home_goals, away_goals)
            expected_result = max(
                [(row[2] or 0, "H"), (row[3] or 0, "D"), (row[4] or 0, "A")],
                key=lambda x: x[0],
            )[1]
            actual_over25 = int(home_goals + away_goals > 2.5)
            actual_btts = int(home_goals > 0 and away_goals > 0)
            over_prediction = int((row[5] or 0) >= 0.5)
            btts_prediction = int((row[6] or 0) >= 0.5)
            conn.execute(
                """
                UPDATE predictions SET actual_home=?, actual_away=?,
                result_correct=?, exact_score_correct=?, over25_correct=?, btts_correct=?, actual_updated_at=?
                WHERE match_id=?
                """,
                (
                    home_goals, away_goals,
                    int(expected_result == actual_result),
                    int(row[0] == home_goals and row[1] == away_goals),
                    int(over_prediction == actual_over25),
                    int(btts_prediction == actual_btts),
                    iso_now(), match_id,
                ),
            )
            updated += 1
        conn.commit()
    fetch_finished.clear()
    return updated, None

# -----------------------------
# Form and model
# -----------------------------

def build_form(finished: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_team: Dict[str, List[Dict[str, Any]]] = {}
    for match in finished:
        score = match.get("score", {}).get("fullTime", {})
        hg, ag = score.get("home"), score.get("away")
        if hg is None or ag is None:
            continue
        home = match.get("homeTeam", {}).get("name")
        away = match.get("awayTeam", {}).get("name")
        date = match.get("utcDate", "")
        if not home or not away:
            continue
        by_team.setdefault(home, []).append({"date": date, "points": 3 if hg > ag else 1 if hg == ag else 0, "gf": hg, "ga": ag, "home": True})
        by_team.setdefault(away, []).append({"date": date, "points": 3 if ag > hg else 1 if ag == hg else 0, "gf": ag, "ga": hg, "home": False})
    result = {}
    for team, games in by_team.items():
        games = sorted(games, key=lambda x: x["date"], reverse=True)[:10]
        result[team] = {
            "played": len(games),
            "points": sum(x["points"] for x in games),
            "gf": sum(x["gf"] for x in games),
            "ga": sum(x["ga"] for x in games),
            "form": "".join("W" if x["points"] == 3 else "D" if x["points"] == 1 else "L" for x in games),
            "home_games": [x for x in games if x["home"]],
            "away_games": [x for x in games if not x["home"]],
        }
    return result


def find_team_data(team: str, standings: Dict[str, Dict[str, Any]], form: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], int]:
    standing = next((v for k, v in standings.items() if team_match(team, k)), None)
    recent = next((v for k, v in form.items() if team_match(team, k)), None)
    data = recent or {"played": 0, "points": 0, "gf": 0, "ga": 0, "form": "N/A", "home_games": [], "away_games": []}
    quality = 0
    if standing:
        quality += 30
    if data["played"] >= 5:
        quality += 30
    elif data["played"] > 0:
        quality += 15
    if data["played"] >= 8:
        quality += 20
    if standing and standing.get("played", 0) > 0:
        quality += 20
    points_per_game = (standing.get("points", 0) / standing.get("played", 1)) if standing else 1.2
    gd_per_game = (standing.get("gd", 0) / standing.get("played", 1)) if standing else 0
    data = dict(data)
    data["strength"] = 1500 + clamp(points_per_game, 0, 3) * 45 + clamp(gd_per_game, -3, 3) * 15
    data["standing"] = standing or {}
    return data, min(100, quality)


def poisson_pmf(lam: float, goals: int) -> float:
    return math.exp(-lam) * lam ** goals / math.factorial(goals)


def score_matrix(home_lambda: float, away_lambda: float, max_goals: int = 10) -> np.ndarray:
    matrix = np.array(
        [[poisson_pmf(home_lambda, h) * poisson_pmf(away_lambda, a) for a in range(max_goals + 1)] for h in range(max_goals + 1)],
        dtype=float,
    )
    total = matrix.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Invalid probability matrix")
    return matrix / total


def top_scorelines(matrix: np.ndarray, limit: int = 8, min_total_goals: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            total_goals = home_goals + away_goals
            if min_total_goals is not None and total_goals < min_total_goals:
                continue
            rows.append({
                "score": f"{home_goals}-{away_goals}",
                "home_goals": home_goals,
                "away_goals": away_goals,
                "total_goals": total_goals,
                "probability": float(matrix[home_goals, away_goals]),
            })
    return sorted(rows, key=lambda row: row["probability"], reverse=True)[:limit]


def calculate_probabilities(matrix: np.ndarray) -> Dict[str, Any]:
    size = matrix.shape[0]
    home_win = float(sum(matrix[h, a] for h in range(size) for a in range(size) if h > a))
    draw = float(sum(matrix[h, a] for h in range(size) for a in range(size) if h == a))
    away_win = float(sum(matrix[h, a] for h in range(size) for a in range(size) if h < a))
    total_probs = {total: float(sum(matrix[h, total - h] for h in range(size) if 0 <= total - h < size)) for total in range(2 * size - 1)}
    over25 = sum(p for total, p in total_probs.items() if total > 2.5)
    under25 = sum(p for total, p in total_probs.items() if total < 2.5)
    btts_yes = float(sum(matrix[h, a] for h in range(1, size) for a in range(1, size)))
    return {
        "home_win": home_win, "draw": draw, "away_win": away_win,
        "over25": over25, "under25": under25,
        "btts_yes": btts_yes, "btts_no": 1 - btts_yes,
        "total_probs": total_probs,
    }


def predict_match(home: Dict[str, Any], away: Dict[str, Any]) -> Dict[str, Any]:
    # High-score tuned baseline (Increased base lambda and attacking scale)
    base_home, base_away = 1.65, 1.25
    strength_diff = (home["strength"] - away["strength"]) / 180
    
    home_form_ppg = home["points"] / max(home["played"], 1)
    away_form_ppg = away["points"] / max(away["played"], 1)
    form_diff = (home_form_ppg - away_form_ppg) / 2.5

    # Attacking strength scaling based on goal scoring averages
    home_gf_avg = (home["gf"] / max(home["played"], 1)) if home["played"] > 0 else 1.5
    away_gf_avg = (away["gf"] / max(away["played"], 1)) if away["played"] > 0 else 1.2
    
    home_lambda = base_home + (0.40 * strength_diff) + (0.25 * form_diff) + (0.15 * (home_gf_avg - 1.4))
    away_lambda = base_away - (0.30 * strength_diff) - (0.15 * form_diff) + (0.15 * (away_gf_avg - 1.2))

    home_games = home.get("home_games", [])
    away_games = away.get("away_games", [])
    if home_games:
        home_lambda = 0.55 * home_lambda + 0.45 * (sum(x["gf"] for x in home_games) / len(home_games))
    if away_games:
        away_lambda = 0.55 * away_lambda + 0.45 * (sum(x["gf"] for x in away_games) / len(away_games))

    # Extended clamps to support high-scoring team predictions
    home_lambda = clamp(home_lambda, 0.40, 4.5)
    away_lambda = clamp(away_lambda, 0.30, 4.0)
    
    matrix = score_matrix(home_lambda, away_lambda)
    best_score = tuple(int(x) for x in np.unravel_index(np.argmax(matrix), matrix.shape))
    probabilities = calculate_probabilities(matrix)
    result_probs = [probabilities["home_win"], probabilities["draw"], probabilities["away_win"]]
    confidence = clamp(max(result_probs), 0, 1)
    data_quality = (home.get("quality", 0) + away.get("quality", 0)) / 2 / 100
    adjusted_confidence = confidence * (0.65 + 0.35 * data_quality)
    
    return {
        "home_lambda": home_lambda,
        "away_lambda": away_lambda,
        "matrix": matrix,
        "best_score": best_score,
        "best_probability": float(matrix[best_score]),
        "top_scorelines": top_scorelines(matrix, limit=8),
        "high_scoring_scorelines": top_scorelines(matrix, limit=5, min_total_goals=3),
        "probabilities": probabilities,
        "confidence": adjusted_confidence,
        "data_quality": data_quality,
    }

# -----------------------------
# Odds parsing and value checks
# -----------------------------

def extract_match_odds(home: str, away: str, odds_data: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {
        "home_win": None, "draw": None, "away_win": None,
        "over25": None, "under25": None,
        "home_handicap": None, "away_handicap": None,
        "home_line": None, "away_line": None,
    }
    matching_events = [
        event for event in odds_data
        if team_match(home, event.get("home_team")) and team_match(away, event.get("away_team"))
    ]
    for event in matching_events:
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                key = str(market.get("key", "")).lower()
                for outcome in market.get("outcomes", []):
                    name = str(outcome.get("name", ""))
                    price = safe_float(outcome.get("price"))
                    point = safe_float(outcome.get("point"))
                    if price is None or price <= 1:
                        continue
                    if key in {"h2h", "1x2", "match_winner"}:
                        if team_match(home, name): result["home_win"] = max(result["home_win"] or 0, price)
                        elif team_match(away, name): result["away_win"] = max(result["away_win"] or 0, price)
                        elif "draw" in name.lower() or "tie" in name.lower(): result["draw"] = max(result["draw"] or 0, price)
                    elif key in {"totals", "over_under"} and point == 2.5:
                        if "over" in name.lower(): result["over25"] = max(result["over25"] or 0, price)
                        elif "under" in name.lower(): result["under25"] = max(result["under25"] or 0, price)
                    elif key in {"spreads", "handicap"}:
                        if team_match(home, name): result["home_handicap"], result["home_line"] = max(result["home_handicap"] or 0, price), point
                        elif team_match(away, name): result["away_handicap"], result["away_line"] = max(result["away_handicap"] or 0, price), point
    return result


def implied_probability(decimal_odds: Optional[float]) -> Optional[float]:
    return None if not decimal_odds or decimal_odds <= 1 else 1 / decimal_odds


def value_edge(model_probability: float, decimal_odds: Optional[float]) -> Optional[float]:
    implied = implied_probability(decimal_odds)
    return None if implied is None else model_probability - implied


def parse_local_price(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    digits = str.maketrans("၀၁၂၃၄၅၆၇၈၉", "0123456789")
    text = text.translate(digits).replace(",", "")
    if not text:
        return None
    try:
        raw = float(text)
    except (TypeError, ValueError):
        return None
    magnitude = abs(raw)
    if magnitude == 0:
        return None
    if magnitude >= 10:
        magnitude /= 100.0
    if magnitude >= 5:
        return None
    return 1.0 + magnitude


def parse_line(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().translate(str.maketrans("၀၁၂၃၄၅၆၇၈၉", "0123456789"))
    try:
        line = float(text)
        return line if -5 <= line <= 10 else None
    except (TypeError, ValueError):
        return None


def parse_myanmar_quote(value: Any) -> Tuple[Optional[float], Optional[float]]:
    if value is None:
        return None, None
    text = str(value).strip().translate(str.maketrans("၀၁၂၃၄၅၆၇၈၉", "0123456789"))
    match = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([+-]\s*\d+(?:\.\d+)?)\s*$", text)
    if not match:
        return None, None
    line = parse_line(match.group(1))
    price = parse_local_price(match.group(2).replace(" ", ""))
    return line, price


def split_quarter_line(line: float) -> List[float]:
    if abs(line * 2 - round(line * 2)) > 1e-9:
        return [line - 0.25, line + 0.25]
    return [line]


def asian_market_result(matrix: np.ndarray, line: float, market: str, decimal_odds: float) -> Dict[str, float]:
    outcomes = []
    for half_line in split_quarter_line(line):
        win = push = lose = 0.0
        size = matrix.shape[0]
        for home_goals in range(size):
            for away_goals in range(size):
                p = float(matrix[home_goals, away_goals])
                if market == "home_handicap":
                    margin = home_goals - away_goals + half_line
                elif market == "away_handicap":
                    margin = away_goals - home_goals + half_line
                elif market == "over":
                    margin = home_goals + away_goals - half_line
                else:
                    margin = half_line - home_goals - away_goals
                if margin > 0:
                    win += p
                elif abs(margin) < 1e-9:
                    push += p
                else:
                    lose += p
        outcomes.append((win, push, lose))
    win = sum(x[0] for x in outcomes) / len(outcomes)
    push = sum(x[1] for x in outcomes) / len(outcomes)
    lose = sum(x[2] for x in outcomes) / len(outcomes)
    ev = win * (decimal_odds - 1.0) - lose
    return {"win": win, "push": push, "lose": lose, "ev": ev}


def render_local_value_finder(match: Dict[str, Any], prediction: Dict[str, Any]) -> None:
    probs = prediction["probabilities"]
    match_id = str(match["id"])
    st.markdown("### 🧾 ဒေသတွင်း မြန်မာကြေး Manual Input")
    st.caption("Price ဥပမာ: 90, -90, +90, 0.90, -0.90 | Line ဥပမာ: -0.5, +0.25, 2.5")

    with st.expander("မြန်မာကြေးထည့်ရန် / Value Bet Finder", expanded=True):
        st.markdown("**မြန်မာကြေး compact format** — ဥပမာ `1+60` = အပေါ်သင်း handicap 1၊ price +60; `3+40` = Over 3၊ price +40")
        q1, q2, q3 = st.columns(3)
        with q1:
            top_side = st.selectbox("အပေါ်သင်း", ["Home team", "Away team"], key=f"top_side_{match_id}")
            top_quote = st.text_input("အပေါ်သင်း ကြေး", key=f"top_quote_{match_id}", placeholder="1+60")
        with q2:
            bottom_quote = st.text_input("အောက်သင်း ကြေး", key=f"bottom_quote_{match_id}", placeholder="1-80")
            st.caption("အောက်သင်း quote တွင် line ကို ထပ်မထည့်လည်းရသည် — compact input ထည့်ပါက ပထမနံပါတ်သည် line ဖြစ်သည်။")
        with q3:
            over_quote = st.text_input("ဂိုးပေါင်း Over", key=f"over_quote_{match_id}", placeholder="3+40")
            under_quote = st.text_input("ဂိုးပေါင်း Under", key=f"under_quote_{match_id}", placeholder="3-60")

        h1, h2, h3 = st.columns(3)
        with h1:
            handicap_line = st.text_input("Handicap line", key=f"local_hdp_line_{match_id}", placeholder="-0.5")
            home_price = st.text_input("အိမ်ရှင်ကြေး", key=f"local_home_price_{match_id}", placeholder="-0.90")
            away_price = st.text_input("ဧည့်သည်ကြေး", key=f"local_away_price_{match_id}", placeholder="+0.90")
        with h2:
            ou_line = st.text_input("O/U line", key=f"local_ou_line_{match_id}", placeholder="2.5")
            over_price = st.text_input("Over ကြေး", key=f"local_over_price_{match_id}", placeholder="-0.90")
            under_price = st.text_input("Under ကြေး", key=f"local_under_price_{match_id}", placeholder="+0.80")
        with h3:
            st.markdown("**Optional 1X2 decimal odds**")
            one_x_two_home = st.text_input("Home 1X2", key=f"local_1x2_home_{match_id}", placeholder="1.90")
            one_x_two_draw = st.text_input("Draw 1X2", key=f"local_1x2_draw_{match_id}", placeholder="3.40")
            one_x_two_away = st.text_input("Away 1X2", key=f"local_1x2_away_{match_id}", placeholder="3.80")

        matrix = prediction["matrix"]
        findings: List[Dict[str, Any]] = []
        parsed_hdp = parse_line(handicap_line)
        parsed_ou = parse_line(ou_line)

        compact_top_line, compact_top_price = parse_myanmar_quote(top_quote)
        compact_bottom_line, compact_bottom_price = parse_myanmar_quote(bottom_quote)
        compact_over_line, compact_over_price = parse_myanmar_quote(over_quote)
        compact_under_line, compact_under_price = parse_myanmar_quote(under_quote)
        if compact_top_line is not None and compact_top_price is not None:
            top_line = -abs(compact_top_line)
            bottom_line = abs(compact_bottom_line if compact_bottom_line is not None else compact_top_line)
            if top_side == "Home team":
                compact_home_line, compact_away_line = top_line, bottom_line
                compact_home_price, compact_away_price = compact_top_price, compact_bottom_price
            else:
                compact_home_line, compact_away_line = bottom_line, top_line
                compact_home_price, compact_away_price = compact_bottom_price, compact_top_price
            parsed_hdp = compact_home_line
            home_price = compact_home_price
            away_price = compact_away_price
        if compact_over_line is not None and compact_over_price is not None:
            parsed_ou, over_price = compact_over_line, compact_over_price
        if compact_under_line is not None and compact_under_price is not None:
            parsed_ou = compact_under_line if parsed_ou is None else parsed_ou
            under_price = compact_under_price

        local_markets = [
            ("Home Handicap", parsed_hdp, home_price, "home_handicap", probs["home_win"]),
            ("Away Handicap", -parsed_hdp if parsed_hdp is not None else None, away_price, "away_handicap", probs["away_win"]),
            ("Over " + (str(parsed_ou) if parsed_ou is not None else ""), parsed_ou, over_price, "over", probs["over25"]),
            ("Under " + (str(parsed_ou) if parsed_ou is not None else ""), parsed_ou, under_price, "under", probs["under25"]),
        ]
        for label, line, raw_price, market, model_prob in local_markets:
            decimal = parse_local_price(raw_price)
            if line is None or decimal is None:
                continue
            result = asian_market_result(matrix, line, market, decimal)
            findings.append({"Market": label, "Myanmar price": raw_price, "Decimal odds": decimal, "Win": result["win"], "Push": result["push"], "Lose": result["lose"], "EV": result["ev"], "Model probability": model_prob})

        for label, raw_price, model_prob in [("Home 1X2", one_x_two_home, probs["home_win"]), ("Draw 1X2", one_x_two_draw, probs["draw"]), ("Away 1X2", one_x_two_away, probs["away_win"])]:
            decimal = safe_float(raw_price)
            if decimal is not None and decimal > 1:
                findings.append({"Market": label, "Myanmar price": raw_price, "Decimal odds": decimal, "Win": model_prob, "Push": 0.0, "Lose": 1 - model_prob, "EV": model_prob * (decimal - 1) - (1 - model_prob), "Model probability": model_prob})

        if not findings:
            st.info("ကြေးနှင့် line ထည့်ပြီးမှ Value Bet Finder က တွက်ပေးပါမည်။")
            return

        table = pd.DataFrame(findings)
        table["Win"] = table["Win"].map(lambda x: f"{x * 100:.1f}%")
        table["Push"] = table["Push"].map(lambda x: f"{x * 100:.1f}%")
        table["Lose"] = table["Lose"].map(lambda x: f"{x * 100:.1f}%")
        table["Model probability"] = table["Model probability"].map(lambda x: f"{x * 100:.1f}%")
        table["Decimal odds"] = table["Decimal odds"].map(lambda x: f"{x:.3f}")
        table["EV"] = table["EV"].map(lambda x: f"{x * 100:+.2f}%")
        st.dataframe(table, hide_index=True, use_container_width=True)

        best = max(findings, key=lambda x: x["EV"])
        if best["EV"] >= 0.02 and prediction["data_quality"] >= 0.50:
            st.success(f"✅ VALUE BET candidate: {best['Market']} | EV {best['EV'] * 100:+.2f}% | Model probability {best['Model probability'] * 100:.1f}%")
        elif best["EV"] > 0:
            st.warning(f"⚠️ Small positive edge only: {best['Market']} | EV {best['EV'] * 100:+.2f}%. Data quality or model uncertainty is high.")
        else:
            st.error(f"❌ No positive-value market found. Best EV: {best['EV'] * 100:+.2f}%")
        st.caption("EV သည် model probability နှင့် manual price ကိုသာ အခြေခံထားသည်။")

# -----------------------------
# Context & API Integration
# -----------------------------

def api_football_request(endpoint: str, params: Dict[str, Any], api_key: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not api_key:
        return None, "API-Football key မထည့်ရသေးပါ။"
    try:
        response = requests.get(
            f"https://v3.football.api-sports.io/{endpoint}",
            params=params,
            headers={"x-apisports-key": api_key},
            timeout=20,
        )
        if response.status_code != 200:
            return None, f"API-Football HTTP {response.status_code}"
        data = response.json()
        if data.get("errors"):
            return None, str(data["errors"])
        return data, None
    except requests.RequestException as exc:
        return None, f"API-Football network error: {exc}"


def rapidapi_request(
    endpoint: str,
    params: Dict[str, Any],
    rapidapi_key: str,
    rapidapi_host: str,
    base_path: str = "/v3",
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not rapidapi_key:
        return None, "X-RapidAPI-Key မထည့်ရသေးပါ။"
    if not rapidapi_host:
        return None, "X-RapidAPI-Host မထည့်ရသေးပါ။"
    path = "/" + endpoint.strip("/")
    url = f"https://{rapidapi_host.strip().strip('/')}{base_path.rstrip('/')}{path}"
    try:
        response = requests.get(
            url,
            params=params,
            headers={
                "X-RapidAPI-Key": rapidapi_key,
                "X-RapidAPI-Host": rapidapi_host.strip(),
            },
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            return None, f"RapidAPI HTTP {response.status_code}: {url}"
        payload = response.json()
        if isinstance(payload, dict) and payload.get("errors"):
            return None, str(payload["errors"])
        return payload, None
    except requests.Timeout:
        return None, "RapidAPI request timed out."
    except (requests.RequestException, ValueError) as exc:
        return None, f"RapidAPI error: {exc}"


def rapidapi_response_list(payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not payload:
        return []
    value = payload.get("response", payload.get("data", payload.get("results", [])))
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def fetch_v71_rapidapi_context(
    fixture_id: int,
    rapidapi_key: str,
    rapidapi_host: str,
    base_path: str = "/v3",
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "fixture": {}, "teams": [], "squads": [], "injuries": [],
        "lineups": [], "statistics": [], "errors": [],
        "fetched_at": now_utc().isoformat(), "source": "RapidAPI",
    }
    if not fixture_id:
        result["errors"].append("RapidAPI fixture ID မထည့်ရသေးပါ။")
        return result
    calls = [
        ("fixtures", {"id": fixture_id}, "fixture"),
        ("fixtures/lineups", {"fixture": fixture_id}, "lineups"),
        ("fixtures/statistics", {"fixture": fixture_id}, "statistics"),
        ("injuries", {"fixture": fixture_id}, "injuries"),
    ]
    for endpoint, params, key in calls:
        payload, error = rapidapi_request(endpoint, params, rapidapi_key, rapidapi_host, base_path)
        if error:
            result["errors"].append(f"{endpoint}: {error}")
        else:
            values = rapidapi_response_list(payload)
            result[key] = values if key != "fixture" else (values[0] if values else {})
    if result["fixture"]:
        fixture = result["fixture"]
        result["teams"] = [fixture.get("teams", {}).get("home", {}), fixture.get("teams", {}).get("away", {})]
        for team in result["teams"]:
            team_id = team.get("id")
            if not team_id:
                continue
            payload, error = rapidapi_request("players/squads", {"team": team_id}, rapidapi_key, rapidapi_host, base_path)
            if error:
                result["errors"].append(f"players/squads({team_id}): {error}")
            else:
                result["squads"].append({"team": team, "players": rapidapi_response_list(payload)})
    return result


def resolve_v71_rapidapi_fixture_id(
    match: Dict[str, Any],
    rapidapi_key: str,
    rapidapi_host: str,
    base_path: str = "/v3",
) -> Tuple[Optional[int], Optional[str]]:
    try:
        kickoff = parse_utc(match["utcDate"])
        date_text = kickoff.strftime("%Y-%m-%d")
    except (KeyError, ValueError):
        return None, "Fixture date မဖတ်နိုင်ပါ။"
    payload, error = rapidapi_request("fixtures", {"date": date_text}, rapidapi_key, rapidapi_host, base_path)
    if error:
        return None, error
    candidates = rapidapi_response_list(payload)
    home = match.get("homeTeam", {}).get("name", "")
    away = match.get("awayTeam", {}).get("name", "")
    for item in candidates:
        teams = item.get("teams", {})
        candidate_home = teams.get("home", {}).get("name", item.get("home_team", ""))
        candidate_away = teams.get("away", {}).get("name", item.get("away_team", ""))
        if team_match(home, candidate_home) and team_match(away, candidate_away):
            fixture_id = item.get("fixture", {}).get("id", item.get("id"))
            if fixture_id is not None:
                return int(fixture_id), None
    return None, f"RapidAPI တွင် {home} vs {away} fixture ကို {date_text} ရက်အတွက် မတွေ့ပါ။"


def fetch_v71_team_context(api_fixture_id: int, api_key: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "fixture": {}, "teams": [], "squads": [], "injuries": [], "lineups": [],
        "statistics": [], "errors": [], "fetched_at": now_utc().isoformat(),
    }
    if not api_fixture_id:
        result["errors"].append("API-Football fixture ID မထည့်ရသေးပါ။")
        return result
    calls = [
        ("fixtures", {"id": api_fixture_id}, "fixture"),
        ("fixtures/lineups", {"fixture": api_fixture_id}, "lineups"),
        ("fixtures/statistics", {"fixture": api_fixture_id}, "statistics"),
        ("injuries", {"fixture": api_fixture_id}, "injuries"),
    ]
    for endpoint, params, key in calls:
        data, error = api_football_request(endpoint, params, api_key)
        if error:
            result["errors"].append(f"{endpoint}: {error}")
        elif data:
            result[key] = data.get("response", [])
    if result["fixture"] and isinstance(result["fixture"], list):
        result["fixture"] = result["fixture"][0] if result["fixture"] else {}
    if result["fixture"]:
        result["teams"] = [result["fixture"].get("teams", {}).get("home", {}), result["fixture"].get("teams", {}).get("away", {})]
        for team in result["teams"]:
            team_id = team.get("id")
            if not team_id:
                continue
            data, error = api_football_request("players/squads", {"team": team_id}, api_key)
            if error:
                result["errors"].append(f"players/squads({team_id}): {error}")
            elif data:
                result["squads"].append({"team": team, "players": data.get("response", [])})
    return result


def resolve_v71_api_football_fixture_id(
    match: Dict[str, Any], api_key: str
) -> Tuple[Optional[int], Optional[str]]:
    try:
        date_text = parse_utc(match["utcDate"]).strftime("%Y-%m-%d")
    except (KeyError, ValueError):
        return None, "Fixture date မဖတ်နိုင်ပါ။"
    payload, error = api_football_request("fixtures", {"date": date_text}, api_key)
    if error:
        return None, error
    candidates = payload.get("response", []) if payload else []
    home = match.get("homeTeam", {}).get("name", "")
    away = match.get("awayTeam", {}).get("name", "")
    for item in candidates:
        teams = item.get("teams", {})
        candidate_home = teams.get("home", {}).get("name", "")
        candidate_away = teams.get("away", {}).get("name", "")
        if team_match(home, candidate_home) and team_match(away, candidate_away):
            fixture_id = item.get("fixture", {}).get("id")
            if fixture_id is not None:
                return int(fixture_id), None
    return None, f"API-Football.com တွင် {home} vs {away} fixture မတွေ့ပါ။"


def summarize_v71_context(bundle: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "home_injuries": 0, "away_injuries": 0,
        "home_suspensions": 0, "away_suspensions": 0,
        "home_lineup_confirmed": 0, "away_lineup_confirmed": 0,
        "home_lineup_strength": 1.0, "away_lineup_strength": 1.0,
        "home_xg": None, "away_xg": None,
    }
    teams = bundle.get("teams", [])
    home_id = teams[0].get("id") if len(teams) > 0 else None
    for item in bundle.get("injuries", []):
        team_id = item.get("team", {}).get("id")
        key = "home" if home_id is not None and team_id == home_id else "away"
        summary[f"{key}_injuries"] += 1
        reason = str(item.get("player", {}).get("reason", "")).lower()
        if "susp" in reason:
            summary[f"{key}_suspensions"] += 1
    for lineup in bundle.get("lineups", []):
        team_id = lineup.get("team", {}).get("id")
        key = "home" if home_id is not None and team_id == home_id else "away"
        if lineup.get("startXI"):
            summary[f"{key}_lineup_confirmed"] = 1
        summary[f"{key}_lineup_strength"] = round(min(1.0, len(lineup.get("startXI", [])) / 11), 3)
    for item in bundle.get("statistics", []):
        team_id = item.get("team", {}).get("id")
        key = "home" if home_id is not None and team_id == home_id else "away"
        for stat in item.get("statistics", []):
            name = str(stat.get("type", "")).lower()
            if name in {"expected goals", "xg"}:
                summary[f"{key}_xg"] = safe_float(stat.get("value"))
    return summary


def apply_v71_context_adjustment(prediction: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    p = np.array([prediction["probabilities"]["home_win"], prediction["probabilities"]["draw"], prediction["probabilities"]["away_win"]], dtype=float)
    home_factor = (context.get("home_motivation", 0) - context.get("away_motivation", 0)) * 0.025
    home_factor += (context.get("home_tactical_change", 0) - context.get("away_tactical_change", 0)) * 0.015
    home_factor += (context.get("away_injuries", 0) - context.get("home_injuries", 0)) * 0.008
    home_factor += (context.get("away_suspensions", 0) - context.get("home_suspensions", 0)) * 0.012
    home_factor += (context.get("home_lineup_strength", 1.0) - context.get("away_lineup_strength", 1.0)) * 0.10
    p[0] += home_factor
    p[2] -= home_factor
    p[1] = max(0.01, p[1])
    p = np.clip(p, 0.01, 0.98); p /= p.sum()
    adjusted = dict(prediction)
    adjusted["probabilities"] = dict(prediction["probabilities"])
    adjusted["probabilities"].update({"home_win": float(p[0]), "draw": float(p[1]), "away_win": float(p[2])})
    adjusted["context_adjustment"] = float(home_factor)
    return adjusted


def render_v71_context_panel(
    match: Dict[str, Any],
    prediction: Dict[str, Any],
    api_key: str,
    rapidapi_key: str = "",
    rapidapi_host: str = "",
    rapidapi_base_path: str = "/v3",
) -> Dict[str, Any]:
    match_id = str(match["id"])
    with st.expander("👥 လူစာရင်း / ဒဏ်ရာ / Motivation / Tactical change", expanded=False):
        st.caption("Motivation နဲ့ Tactical change ကို API မရလျှင် manual score -2 မှ +2 ထည့်ပါ။ 0 = neutral")
        api_fixture_id = st.number_input("API-Football fixture ID (optional)", min_value=0, step=1, value=0, key=f"v71_api_fixture_{match_id}")
        a, b, c, d = st.columns(4)
        with a:
            home_motivation = st.slider("Home motivation", -2.0, 2.0, 0.0, 0.25, key=f"v71_hmot_{match_id}")
            home_tactical = st.slider("Home tactical change", -2.0, 2.0, 0.0, 0.25, key=f"v71_htac_{match_id}")
        with b:
            away_motivation = st.slider("Away motivation", -2.0, 2.0, 0.0, 0.25, key=f"v71_amot_{match_id}")
            away_tactical = st.slider("Away tactical change", -2.0, 2.0, 0.0, 0.25, key=f"v71_atac_{match_id}")
        with c:
            home_inj = st.number_input("Home injured players", 0, 30, 0, key=f"v71_hinj_{match_id}")
            home_susp = st.number_input("Home suspensions", 0, 15, 0, key=f"v71_hsus_{match_id}")
            home_strength = st.slider("Home lineup strength", 0.70, 1.00, 1.00, 0.01, key=f"v71_hstr_{match_id}")
        with d:
            away_inj = st.number_input("Away injured players", 0, 30, 0, key=f"v71_ainj_{match_id}")
            away_susp = st.number_input("Away suspensions", 0, 15, 0, key=f"v71_asus_{match_id}")
            away_strength = st.slider("Away lineup strength", 0.70, 1.00, 1.00, 0.01, key=f"v71_astr_{match_id}")
        context = {"home_motivation": home_motivation, "away_motivation": away_motivation, "home_tactical_change": home_tactical, "away_tactical_change": away_tactical, "home_injuries": home_inj, "away_injuries": away_inj, "home_suspensions": home_susp, "away_suspensions": away_susp, "home_lineup_strength": home_strength, "away_lineup_strength": away_strength}
        fetch_a, fetch_b, fetch_c, fetch_d = st.columns(4)
        with fetch_a:
            if api_fixture_id and api_key and st.button("Fetch API-Football context", key=f"v71_fetch_{match_id}"):
                st.session_state[f"v71_bundle_{match_id}"] = fetch_v71_team_context(int(api_fixture_id), api_key)
        with fetch_b:
            if api_fixture_id and rapidapi_key and rapidapi_host and st.button("Fetch RapidAPI context", key=f"v71_rapid_fetch_{match_id}"):
                st.session_state[f"v71_bundle_{match_id}"] = fetch_v71_rapidapi_context(
                    int(api_fixture_id), rapidapi_key, rapidapi_host, rapidapi_base_path
                )
        with fetch_c:
            if rapidapi_key and rapidapi_host and st.button("Auto-fetch current match", key=f"v71_auto_fetch_{match_id}"):
                resolved_id, resolve_error = resolve_v71_rapidapi_fixture_id(
                    match, rapidapi_key, rapidapi_host, rapidapi_base_path
                )
                if resolve_error:
                    st.warning(resolve_error)
                else:
                    st.session_state[f"v71_resolved_fixture_{match_id}"] = resolved_id
                    st.session_state[f"v71_bundle_{match_id}"] = fetch_v71_rapidapi_context(
                        int(resolved_id), rapidapi_key, rapidapi_host, rapidapi_base_path
                    )
                    st.success(f"RapidAPI fixture ID resolved: {resolved_id}")
        with fetch_d:
            if api_key and st.button("Auto-fetch API-Football", key=f"v71_direct_auto_{match_id}"):
                resolved_id, resolve_error = resolve_v71_api_football_fixture_id(match, api_key)
                if resolve_error:
                    st.warning(resolve_error)
                else:
                    st.session_state[f"v71_resolved_fixture_{match_id}"] = resolved_id
                    st.session_state[f"v71_bundle_{match_id}"] = fetch_v71_team_context(int(resolved_id), api_key)
                    st.success(f"API-Football fixture ID resolved: {resolved_id}")
        bundle = st.session_state.get(f"v71_bundle_{match_id}")
        if bundle:
            for error in bundle.get("errors", []): st.warning(error)
            api_summary = summarize_v71_context(bundle)
            context.update({k: v for k, v in api_summary.items() if v is not None})
            squads = bundle.get("squads", [])
            if squads:
                st.markdown("**လက်ရှိ squad roster**")
                for squad in squads:
                    team_name = squad.get("team", {}).get("name", "Team")
                    players = squad.get("players", [])
                    names = [p.get("player", {}).get("name", "Unknown") for p in players]
                    st.write(f"**{team_name}** ({len(names)} players): " + ", ".join(names))
            st.json(api_summary)
        adjusted = apply_v71_context_adjustment(prediction, context)
        st.info(f"Context adjustment: Home {adjusted['context_adjustment']:+.3f} probability shift")
        st.write(f"Adjusted 1X2 — Home: **{adjusted['probabilities']['home_win'] * 100:.1f}%**, Draw: **{adjusted['probabilities']['draw'] * 100:.1f}%**, Away: **{adjusted['probabilities']['away_win'] * 100:.1f}%**")
        return adjusted
    return prediction

# -----------------------------
# UI components
# -----------------------------

def render_prediction_card(
    match: Dict[str, Any],
    prediction: Dict[str, Any],
    odds: Dict[str, Optional[float]],
    show_details: bool = True,
    api_key: str = "",
    rapidapi_key: str = "",
    rapidapi_host: str = "",
    rapidapi_base_path: str = "/v3",
) -> None:
    home = match["homeTeam"]["name"]
    away = match["awayTeam"]["name"]
    probs = prediction["probabilities"]
    score = prediction["best_score"]
    st.markdown(f"### {home}  **{score[0]}–{score[1]}**  {away}")
    st.caption(f"Kickoff: {local_time(match['utcDate'])}  |  Model: {MODEL_VERSION}")
    
    score_a, score_b = st.columns(2)
    with score_a:
        st.markdown("**Most likely scorelines**")
        score_table = pd.DataFrame(prediction.get("top_scorelines", []))
        if not score_table.empty:
            score_table["Probability"] = score_table.pop("probability").map(lambda x: f"{x * 100:.2f}%")
            st.dataframe(score_table[["score", "Probability"]], hide_index=True, use_container_width=True)
    with score_b:
        st.markdown("**High-scoring alternatives (3+ goals)**")
        high_table = pd.DataFrame(prediction.get("high_scoring_scorelines", []))
        if not high_table.empty:
            high_table["Probability"] = high_table.pop("probability").map(lambda x: f"{x * 100:.2f}%")
            st.dataframe(high_table[["score", "Probability"]], hide_index=True, use_container_width=True)
            
    # Updated metrics including High Score (3+)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1X2", f"{max(probs['home_win'], probs['draw'], probs['away_win']) * 100:.1f}%")
    c2.metric("Over 2.5", pct(probs["over25"]))
    c3.metric("High Score (3+)", pct(probs["over25"]))
    c4.metric("Data quality", f"{prediction['data_quality'] * 100:.0f}%")
    
    st.progress(float(prediction["confidence"]), text=f"Adjusted confidence: {prediction['confidence'] * 100:.1f}%")

    result_table = pd.DataFrame({
        "Market": ["Home win", "Draw", "Away win", "Over 2.5", "Under 2.5", "BTTS Yes", "BTTS No"],
        "Model probability": [probs["home_win"], probs["draw"], probs["away_win"], probs["over25"], probs["under25"], probs["btts_yes"], probs["btts_no"]],
    })
    result_table["Model probability"] = result_table["Model probability"].map(lambda x: f"{x * 100:.1f}%")
    st.dataframe(result_table, hide_index=True, use_container_width=True)

    if any(value is not None for value in odds.values()):
        st.caption("Best available decimal odds found from the returned odds feed")
        odds_table = pd.DataFrame({
            "Market": ["Home win", "Draw", "Away win", "Over 2.5", "Under 2.5"],
            "Odds": [odds["home_win"], odds["draw"], odds["away_win"], odds["over25"], odds["under25"]],
            "Model edge": [value_edge(probs["home_win"], odds["home_win"]), value_edge(probs["draw"], odds["draw"]), value_edge(probs["away_win"], odds["away_win"]), value_edge(probs["over25"], odds["over25"]), value_edge(probs["under25"], odds["under25"])],
        })
        odds_table["Odds"] = odds_table["Odds"].map(lambda x: "—" if x is None else f"{x:.2f}")
        odds_table["Model edge"] = odds_table["Model edge"].map(lambda x: "—" if x is None else f"{x * 100:+.1f}%")
        st.dataframe(odds_table, hide_index=True, use_container_width=True)
    else:
        st.info("Odds မရရှိသေးပါ။ Value edge မတွက်နိုင်ပါ။")

    prediction = render_v71_context_panel(
        match, prediction, api_key, rapidapi_key, rapidapi_host, rapidapi_base_path
    )
    render_local_value_finder(match, prediction)

    if show_details:
        with st.expander("Model details and limitations"):
            st.write(f"Expected goals: home **{prediction['home_lambda']:.2f}**, away **{prediction['away_lambda']:.2f}**")


def build_league_context(league_code: str, api_key: str) -> Dict[str, Any]:
    finished, finished_error = fetch_finished(league_code, api_key)
    standings, standings_error = fetch_standings(league_code, api_key)
    form = build_form(finished)
    return {"finished": finished, "standings": standings, "form": form, "errors": [x for x in [finished_error, standings_error] if x]}


def create_prediction(match: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    home, home_quality = find_team_data(match["homeTeam"]["name"], context["standings"], context["form"])
    away, away_quality = find_team_data(match["awayTeam"]["name"], context["standings"], context["form"])
    home["quality"], away["quality"] = home_quality, away_quality
    return predict_match(home, away)


def render_selected_league(league_name: str, league_code: str, fb_key: str, odds_key: str) -> None:
    fixtures, fixture_error = fetch_fixtures(league_code, fb_key)
    context = build_league_context(league_code, fb_key)
    odds_data, odds_error = fetch_odds(odds_key)
    if fixture_error:
        st.warning(fixture_error)
    for error in context["errors"]:
        st.warning(error)
    if odds_error:
        st.info(odds_error)
    st.subheader(f"📅 {league_name} — next 7 days")
    if not fixtures:
        st.info("Scheduled fixture မရှိပါ၊ သို့မဟုတ် API data မရရှိပါ။")
        return
    st.success(f"{len(fixtures)} matches found")
    for match in fixtures:
        prediction = create_prediction(match, context)
        save_prediction(match, prediction, league_code, league_name)
        odds = extract_match_odds(match["homeTeam"]["name"], match["awayTeam"]["name"], odds_data)
        with st.expander(f"⚽ {match['homeTeam']['name']} vs {match['awayTeam']['name']} | {local_time(match['utcDate'])}", expanded=False):
            render_prediction_card(
                match,
                prediction,
                odds,
                api_key=st.session_state.get("v71_api_key", ""),
                rapidapi_key=st.session_state.get("v71_rapid_key", ""),
                rapidapi_host=st.session_state.get("v71_rapid_host", ""),
                rapidapi_base_path=st.session_state.get("v71_rapid_base", "/v3"),
            )


def render_top_matches(fb_key: str, odds_key: str) -> None:
    st.subheader("🔥 Top 10 best matches of the day")
    odds_data, odds_error = fetch_odds(odds_key)
    if odds_error:
        st.info(odds_error)
    rows: List[Dict[str, Any]] = []
    for league_name, league_code in LEAGUES.items():
        fixtures, _ = fetch_fixtures(league_code, fb_key)
        if not fixtures:
            continue
        context = build_league_context(league_code, fb_key)
        for match in fixtures:
            prediction = create_prediction(match, context)
            save_prediction(match, prediction, league_code, league_name)
            probs = prediction["probabilities"]
            rows.append({
                "match": f"{match['homeTeam']['name']} vs {match['awayTeam']['name']}",
                "league": league_name,
                "kickoff": local_time(match["utcDate"]),
                "confidence": prediction["confidence"],
                "quality": prediction["data_quality"],
                "prediction": f"{prediction['best_score'][0]}–{prediction['best_score'][1]}",
                "home_prob": probs["home_win"], "draw_prob": probs["draw"], "away_prob": probs["away_win"],
                "match_obj": match, "prediction_obj": prediction,
            })
    rows.sort(key=lambda x: (x["confidence"], x["quality"]), reverse=True)
    if not rows:
        st.info("API data မရရှိသဖြင့် Top 10 မတည်ဆောက်နိုင်ပါ။")
        return
    top = rows[:10]
    table = pd.DataFrame([{k: v for k, v in row.items() if k not in {"match_obj", "prediction_obj"}} for row in top])
    table["confidence"] = table["confidence"].map(lambda x: f"{x * 100:.1f}%")
    table["quality"] = table["quality"].map(lambda x: f"{x * 100:.0f}%")
    for col in ["home_prob", "draw_prob", "away_prob"]:
        table[col] = table[col].map(lambda x: f"{x * 100:.1f}%")
    st.dataframe(table, hide_index=True, use_container_width=True)
    for index, row in enumerate(top, 1):
        with st.expander(f"#{index} {row['match']} — {row['league']}"):
            odds = extract_match_odds(row["match_obj"]["homeTeam"]["name"], row["match_obj"]["awayTeam"]["name"], odds_data)
            render_prediction_card(
                row["match_obj"],
                row["prediction_obj"],
                odds,
                api_key=st.session_state.get("v71_api_key", ""),
                rapidapi_key=st.session_state.get("v71_rapid_key", ""),
                rapidapi_host=st.session_state.get("v71_rapid_host", ""),
                rapidapi_base_path=st.session_state.get("v71_rapid_base", "/v3"),
            )


def render_history() -> None:
    st.subheader("📚 Match history & accuracy")
    history = load_history()
    if history.empty:
        st.info("Prediction history မရှိသေးပါ။ ပထမဆုံး fixture view ကို ဖွင့်ပါ။")
        return
    completed = history[history["actual_home"].notna() & history["actual_away"].notna()].copy()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saved predictions", len(history))
    c2.metric("Completed", len(completed))
    if completed.empty:
        c3.metric("1X2 accuracy", "—")
        c4.metric("Exact score accuracy", "—")
    else:
        c3.metric("1X2 accuracy", pct(completed["result_correct"].mean()))
        c4.metric("Exact score accuracy", pct(completed["exact_score_correct"].mean()))
        st.metric("Over 2.5 accuracy", pct(completed["over25_correct"].mean()))
        st.metric("BTTS accuracy", pct(completed["btts_correct"].mean()))
    st.caption("Accuracy သည် saved prediction တွေထဲက result update ရပြီးသား match များအပေါ်သာ တွက်ထားပါသည်။")
    display = history.copy()
    display["match"] = display["home_team"] + " vs " + display["away_team"]
    display["prediction"] = display["predicted_home"].astype(str) + "–" + display["predicted_away"].astype(str)
    cols = ["league_name", "match", "match_date", "prediction", "confidence", "data_quality", "actual_home", "actual_away", "result_correct", "exact_score_correct"]
    display = display[[x for x in cols if x in display.columns]]
    if "confidence" in display:
        display["confidence"] = display["confidence"].map(lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%")
    if "data_quality" in display:
        display["data_quality"] = display["data_quality"].map(lambda x: "—" if pd.isna(x) else f"{x * 100:.0f}%")
    st.dataframe(display, hide_index=True, use_container_width=True)

# -----------------------------
# Main app
# -----------------------------

initialize_database()
st.title("⚽ Professional Football Predictor V7.1")
st.markdown("Poisson baseline model with High-Score tuning • multi-league Top 10 • history • data-quality checks")

with st.sidebar:
    st.header("🔑 API settings")
    fb_default = st.secrets.get("FOOTBALL_API_KEY", "")
    odds_default = st.secrets.get("ODDS_API_KEY", "")
    fb_key = st.text_input("Football-data.org API key", value=fb_default, type="password")
    odds_key = st.text_input("Odds-api.io API key", value=odds_default, type="password")
    api_football_key = st.text_input("API-Football.com API key (optional)", value=st.secrets.get("API_FOOTBALL_KEY", ""), type="password")
    st.session_state["v71_api_key"] = api_football_key
    st.markdown("**RapidAPI football enrichment**")
    rapidapi_key = st.text_input("X-RapidAPI-Key (optional)", value=st.secrets.get("X_RAPIDAPI_KEY", ""), type="password")
    rapidapi_host = st.text_input("X-RapidAPI-Host", value=st.secrets.get("X_RAPIDAPI_HOST", "api-football-v1.p.rapidapi.com"))
    rapidapi_base_path = st.text_input("RapidAPI base path", value=st.secrets.get("X_RAPIDAPI_BASE_PATH", "/v3"))
    st.session_state["v71_rapid_key"] = rapidapi_key
    st.session_state["v71_rapid_host"] = rapidapi_host
    st.session_state["v71_rapid_base"] = rapidapi_base_path
    nav_mode = st.radio("Navigation", ["🏟️ Selected league", "🔥 Top 10 best matches", "📚 History & accuracy"])
    if st.button("🔄 Clear cached API data", use_container_width=True):
        fetch_fixtures.clear(); fetch_finished.clear(); fetch_standings.clear(); fetch_odds.clear()
        st.success("Cache cleared. Reloading…")
        st.rerun()
    st.divider()

if nav_mode == "🏟️ Selected league":
    selected_name = st.sidebar.selectbox("League", list(LEAGUES.keys()))
    selected_code = LEAGUES[selected_name]
    if st.sidebar.button("📊 Update actual results", use_container_width=True):
        count, error = update_actual_results(selected_code, fb_key)
        if error:
            st.error(error)
        else:
            st.success(f"{count} saved prediction(s) updated.")
            st.rerun()
    render_selected_league(selected_name, selected_code, fb_key, odds_key)
elif nav_mode == "🔥 Top 10 best matches":
    render_top_matches(fb_key, odds_key)
else:
    render_history()
