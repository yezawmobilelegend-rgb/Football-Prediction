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
# Professional Football Predictor V3.0
# - Selected league view
# - Top 10 cross-league view
# - Prediction history and multi-metric accuracy
# - Robust API/error handling
# - Data-quality and model-confidence disclosure
# - Reproducible Poisson prediction (not random/AI magic)
# ============================================================

st.set_page_config(
    page_title="Professional Football Predictor V3",
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

MODEL_VERSION = "poisson-v3.0"
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

SCHEMA = """ CREATE TABLE IF NOT EXISTS predictions ( id INTEGER PRIMARY KEY AUTOINCREMENT, match_id TEXT UNIQUE NOT NULL, league_code TEXT, league_name TEXT, match_date TEXT, home_team TEXT, away_team TEXT, predicted_home INTEGER, predicted_away INTEGER, home_lambda REAL, away_lambda REAL, home_prob REAL, draw_prob REAL, away_prob REAL, over25_prob REAL, under25_prob REAL, btts_yes_prob REAL, confidence REAL, data_quality REAL, model_version TEXT, actual_home INTEGER, actual_away INTEGER, result_correct INTEGER, exact_score_correct INTEGER, over25_correct INTEGER, btts_correct INTEGER, created_at TEXT, actual_updated_at TEXT ) """


def db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    with db_connection() as conn:
        conn.execute(SCHEMA)
        # Safe migration for databases created by older versions.
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
            """ INSERT INTO predictions ( match_id, league_code, league_name, match_date, home_team, away_team, predicted_home, predicted_away, home_lambda, away_lambda, home_prob, draw_prob, away_prob, over25_prob, under25_prob, btts_yes_prob, confidence, data_quality, model_version, created_at ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(match_id) DO UPDATE SET league_code=excluded.league_code, league_name=excluded.league_name, home_lambda=excluded.home_lambda, away_lambda=excluded.away_lambda, home_prob=excluded.home_prob, draw_prob=excluded.draw_prob, away_prob=excluded.away_prob, over25_prob=excluded.over25_prob, under25_prob=excluded.under25_prob, btts_yes_prob=excluded.btts_yes_prob, confidence=excluded.confidence, data_quality=excluded.data_quality, model_version=excluded.model_version """,
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
                """ UPDATE predictions SET actual_home=?, actual_away=?, result_correct=?, exact_score_correct=?, over25_correct=?, btts_correct=?, actual_updated_at=? WHERE match_id=? """,
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
    # A transparent strength score; it is intentionally not mislabeled as Elo.
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
    # Conservative, reproducible model. It is a baseline and should be backtested before betting use.
    base_home, base_away = 1.48, 1.12
    strength_diff = (home["strength"] - away["strength"]) / 200
    home_form_ppg = home["points"] / max(home["played"], 1)
    away_form_ppg = away["points"] / max(away["played"], 1)
    form_diff = (home_form_ppg - away_form_ppg) / 3

    home_lambda = base_home + 0.35 * strength_diff + 0.20 * form_diff
    away_lambda = base_away - 0.25 * strength_diff - 0.15 * form_diff

    home_games = home.get("home_games", [])
    away_games = away.get("away_games", [])
    if home_games:
        home_lambda = 0.65 * home_lambda + 0.35 * (sum(x["gf"] for x in home_games) / len(home_games))
    if away_games:
        away_lambda = 0.65 * away_lambda + 0.35 * (sum(x["gf"] for x in away_games) / len(away_games))

    home_lambda = clamp(home_lambda, 0.25, 4.0)
    away_lambda = clamp(away_lambda, 0.20, 3.5)
    matrix = score_matrix(home_lambda, away_lambda)
    best_score = tuple(int(x) for x in np.unravel_index(np.argmax(matrix), matrix.shape))
    probabilities = calculate_probabilities(matrix)
    result_probs = [probabilities["home_win"], probabilities["draw"], probabilities["away_win"]]
    confidence = clamp(max(result_probs), 0, 1)
    data_quality = (home.get("quality", 0) + away.get("quality", 0)) / 2 / 100
    # Confidence is reduced when the underlying team data is incomplete.
    adjusted_confidence = confidence * (0.65 + 0.35 * data_quality)
    return {
        "home_lambda": home_lambda,
        "away_lambda": away_lambda,
        "matrix": matrix,
        "best_score": best_score,
        "best_probability": float(matrix[best_score]),
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
    # Prefer the first matching event but aggregate best prices across bookmakers.
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

# -----------------------------
# UI components
# -----------------------------

def render_prediction_card(match: Dict[str, Any], prediction: Dict[str, Any], odds: Dict[str, Optional[float]], show_details: bool = True) -> None:
    home = match["homeTeam"]["name"]
    away = match["awayTeam"]["name"]
    probs = prediction["probabilities"]
    score = prediction["best_score"]
    st.markdown(f"### {home} **{score[0]}–{score[1]}** {away}")
    st.caption(f"Kickoff: {local_time(match['utcDate'])} | Model: {MODEL_VERSION}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1X2", f"{max(probs['home_win'], probs['draw'], probs['away_win']) * 100:.1f}%")
    c2.metric("Over 2.5", pct(probs["over25"]))
    c3.metric("BTTS Yes", pct(probs["btts_yes"]))
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

    if show_details:
        with st.expander("Model details and limitations"):
            st.write(f"Expected goals: home **{prediction['home_lambda']:.2f}**, away **{prediction['away_lambda']:.2f}**")
            st.warning("ဤ model သည် calibrated/backtested production model မဟုတ်သေးပါ။ Injury, lineup, xG နှင့် bookmaker closing-line data မပါသဖြင့် prediction ကို အာမခံချက်အဖြစ် မသုံးပါနှင့်။")


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
            render_prediction_card(match, prediction, odds)


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
            render_prediction_card(row["match_obj"], row["prediction_obj"], odds)


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
st.title("⚽ Professional Football Predictor V3")
st.markdown("Poisson baseline model • multi-league Top 10 • history • calibrated disclosure • data-quality checks")

with st.sidebar:
    st.header("🔑 API settings")
    fb_default = st.secrets.get("FOOTBALL_API_KEY", "")
    odds_default = st.secrets.get("ODDS_API_KEY", "")
    fb_key = st.text_input("Football-data.org API key", value=fb_default, type="password")
    odds_key = st.text_input("Odds-api.io API key", value=odds_default, type="password")
    nav_mode = st.radio("Navigation", ["🏟️ Selected league", "🔥 Top 10 best matches", "📚 History & accuracy"])
    if st.button("🔄 Clear cached API data", use_container_width=True):
        fetch_fixtures.clear(); fetch_finished.clear(); fetch_standings.clear(); fetch_odds.clear()
        st.success("Cache cleared. Reloading…")
        st.rerun()
    st.divider()
    st.caption("Odds မရရှိပါက model probability ကိုသာ ပြပါမည်။ API data မပြည့်စုံလျှင် data quality ကို လျှော့ပြပါမည်။")

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