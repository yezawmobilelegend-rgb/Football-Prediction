
from datetime import datetime, timedelta, timezone
import math
import sqlite3
import requests
import numpy as np
import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo

# =========================================================
# PROFESSIONAL FOOTBALL PREDICTOR V2.4
# =========================================================

st.set_page_config(
    page_title="Professional Football Predictor V2.4",
    page_icon="⚽",
    layout="wide",
)

DB_PATH = "football_history.db"
LOCAL_TZ = ZoneInfo("Asia/Yangon")

LEAGUES = {
    "English Premier League": "PL",
    "Spanish La Liga": "PD",
    "German Bundesliga": "BL1",
    "French Ligue 1": "FL1",
    "UEFA Champions League": "CL",
    "UEFA Europa League": "EL",
    "Italy Serie A": "SA",
}

# =========================================================
# DATABASE
# =========================================================

def db_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with db_connection() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT UNIQUE,
            league TEXT,
            match_date TEXT,
            home_team TEXT,
            away_team TEXT,
            predicted_home INTEGER,
            predicted_away INTEGER,
            home_lambda REAL,
            away_lambda REAL,
            actual_home INTEGER,
            actual_away INTEGER,
            prediction_correct INTEGER,
            created_at TEXT
        )
        """)
        conn.commit()


def save_prediction(
    match_id,
    league,
    match_date,
    home,
    away,
    score,
    hl,
    al,
):
    with db_connection() as conn:
        conn.execute("""
        INSERT INTO predictions (
            match_id, league, match_date,
            home_team, away_team,
            predicted_home, predicted_away,
            home_lambda, away_lambda, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(match_id) DO NOTHING
        """, (
            str(match_id),
            league,
            match_date,
            home,
            away,
            int(score[0]),
            int(score[1]),
            float(hl),
            float(al),
            datetime.now().isoformat(),
        ))
        conn.commit()


def load_history():
    with db_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY match_date DESC",
            conn,
        )


initialize_database()

# =========================================================
# API SETTINGS
# =========================================================

st.sidebar.header("API Settings")

fb_key = st.sidebar.text_input(
    "Football-data.org API Key",
    value=st.secrets.get("FOOTBALL_API_KEY", ""),
    type="password",
)

selected_leagues = st.sidebar.multiselect(
    "Select Leagues",
    list(LEAGUES.keys()),
    default=list(LEAGUES.keys()),
)

days_ahead = st.sidebar.slider(
    "Fixture Days",
    min_value=1,
    max_value=14,
    value=7,
)

# =========================================================
# API HELPERS
# =========================================================

def api_get(url, headers=None, params=None):
    try:
        r = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=25,
        )
        if r.status_code == 200:
            return r.json()
        return None
    except requests.RequestException:
        return None


@st.cache_data(ttl=900)
def fetch_fixtures(league, api_key, days):
    if not api_key:
        return []

    data = api_get(
        f"https://api.football-data.org/v4/"
        f"competitions/{league}/matches",
        headers={"X-Auth-Token": api_key},
        params={"status": "SCHEDULED"},
    )

    if not data:
        return []

    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days)

    result = []

    for m in data.get("matches", []):
        dt = datetime.strptime(
            m["utcDate"],
            "%Y-%m-%dT%H:%M:%SZ",
        ).replace(tzinfo=timezone.utc)

        if now <= dt <= end:
            m["league_code"] = league
            result.append(m)

    return result


@st.cache_data(ttl=3600)
def fetch_standings(league, api_key):
    if not api_key:
        return {}

    data = api_get(
        f"https://api.football-data.org/v4/"
        f"competitions/{league}/standings",
        headers={"X-Auth-Token": api_key},
    )

    if not data:
        return {}

    result = {}

    for table in data.get("standings", []):
        if table.get("type") != "TOTAL":
            continue

        for row in table.get("table", []):
            name = row["team"]["name"]

            result[name] = {
                "points": row.get("points", 0),
                "gd": row.get("goalDifference", 0),
                "elo": (
                    1500
                    + row.get("points", 0) * 15
                    + row.get("goalDifference", 0) * 5
                ),
            }

    return result


@st.cache_data(ttl=1800)
def fetch_recent_form(league, api_key):
    if not api_key:
        return {}

    data = api_get(
        f"https://api.football-data.org/v4/"
        f"competitions/{league}/matches",
        headers={"X-Auth-Token": api_key},
        params={"status": "FINISHED"},
    )

    if not data:
        return {}

    matches = {}

    for m in data.get("matches", []):
        score = m.get("score", {}).get("fullTime", {})
        h = score.get("home")
        a = score.get("away")

        if h is None or a is None:
            continue

        home = m["homeTeam"]["name"]
        away = m["awayTeam"]["name"]
        date = m["utcDate"]

        matches.setdefault(home, []).append({
            "date": date,
            "points": 3 if h > a else 1 if h == a else 0,
            "gf": h,
            "ga": a,
        })

        matches.setdefault(away, []).append({
            "date": date,
            "points": 3 if a > h else 1 if a == h else 0,
            "gf": a,
            "ga": h,
        })

    result = {}

    for team, games in matches.items():
        games = sorted(
            games,
            key=lambda x: x["date"],
            reverse=True,
        )[:5]

        result[team] = {
            "points": sum(x["points"] for x in games),
            "gf": sum(x["gf"] for x in games),
            "ga": sum(x["ga"] for x in games),
            "played": len(games),
            "form": "".join(
                "W" if x["points"] == 3
                else "D" if x["points"] == 1
                else "L"
                for x in games
            ),
        }

    return result


# =========================================================
# MODEL
# =========================================================

def get_team_data(team, standings, form):
    s = standings.get(team, {})
    f = form.get(team, {})

    return {
        "elo": s.get("elo", 1500),
        "points": f.get("points", 0),
        "gf": f.get("gf", 0),
        "ga": f.get("ga", 0),
        "played": f.get("played", 0),
        "form": f.get("form", "N/A"),
    }


def calculate_lambdas(home, away):
    elo_diff = (home["elo"] - away["elo"]) / 200
    form_diff = (home["points"] - away["points"]) / 8

    hl = max(
        0.30,
        1.65 + elo_diff * 0.45 + form_diff * 0.20,
    )

    al = max(
        0.30,
        1.15 - elo_diff * 0.45 - form_diff * 0.20,
    )

    if home["played"] > 0:
        avg = home["gf"] / home["played"]
        hl = hl * 0.60 + avg * 0.40

    if away["played"] > 0:
        avg = away["gf"] / away["played"]
        al = al * 0.60 + avg * 0.40

    return max(0.30, hl), max(0.30, al)


def poisson_pmf(lam, k):
    return math.exp(-lam) * lam**k / math.factorial(k)


def score_matrix(hl, al, max_goals=8):
    matrix = np.zeros((max_goals + 1, max_goals + 1))

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            matrix[h, a] = (
                poisson_pmf(hl, h)
                * poisson_pmf(al, a)
            )

    matrix /= matrix.sum()
    return matrix


def calculate_probabilities(matrix):
    n = matrix.shape[0]

    home_win = 0
    draw = 0
    away_win = 0

    for h in range(n):
        for a in range(n):
            if h > a:
                home_win += matrix[h, a]
            elif h == a:
                draw += matrix[h, a]
            else:
                away_win += matrix[h, a]

    over25 = sum(
        matrix[h, a]
        for h in range(n)
        for a in range(n)
        if h + a > 2.5
    )

    btts = sum(
        matrix[h, a]
        for h in range(1, n)
        for a in range(1, n)
    )

    return {
        "home": home_win,
        "draw": draw,
        "away": away_win,
        "over25": over25,
        "under25": 1 - over25,
        "btts": btts,
    }


def predict_match(home, away):
    hl, al = calculate_lambdas(home, away)
    matrix = score_matrix(hl, al)

    best_score = np.unravel_index(
        np.argmax(matrix),
        matrix.shape,
    )

    probs = calculate_probabilities(matrix)

    return {
        "hl": hl,
        "al": al,
        "score": best_score,
        "score_probability": matrix[best_score],
        "matrix": matrix,
        "probs": probs,
    }


# =========================================================
# LOAD ALL LEAGUES
# =========================================================

st.title("⚽ Professional Football Predictor V2.4")
st.caption("7 Leagues | Daily Fixtures | Statistical Analysis | Top 6")

if not fb_key:
    st.warning("Football-data.org API Key ထည့်ပါ။")
    st.stop()

all_fixtures = []
all_standings = {}
all_forms = {}

progress = st.progress(0)

for i, league_name in enumerate(selected_leagues):
    code = LEAGUES[league_name]

    fixtures = fetch_fixtures(
        code,
        fb_key,
        days_ahead,
    )

    standings = fetch_standings(code, fb_key)
    form = fetch_recent_form(code, fb_key)

    all_fixtures.extend(fixtures)

    for team, value in standings.items():
        all_standings[team] = value

    for team, value in form.items():
        all_forms[team] = value

    progress.progress(
        (i + 1) / max(1, len(selected_leagues))
    )

progress.empty()

st.success(
    f"Leagues: {len(selected_leagues)} | "
    f"Fixtures: {len(all_fixtures)}"
)

# =========================================================
# GENERATE ALL PREDICTIONS
# =========================================================

predictions = []

for match in all_fixtures:
    home = match["homeTeam"]["name"]
    away = match["awayTeam"]["name"]
    match_id = match["id"]

    hd = get_team_data(
        home,
        all_standings,
        all_forms,
    )

    ad = get_team_data(
        away,
        all_standings,
        all_forms,
    )

    pred = predict_match(hd, ad)

    utc_dt = datetime.strptime(
        match["utcDate"],
        "%Y-%m-%dT%H:%M:%SZ",
    ).replace(tzinfo=timezone.utc)

    local_dt = utc_dt.astimezone(LOCAL_TZ)

    probs = pred["probs"]

    # Statistical ranking score
    probability_confidence = max(
        probs["home"],
        probs["draw"],
        probs["away"],
    )

    data_quality = min(
        1.0,
        (hd["played"] + ad["played"]) / 10,
    )

    ranking_score = (
        probability_confidence * 70
        + data_quality * 30
    )

    score = pred["score"]

    row = {
        "Match ID": match_id,
        "Date": local_dt.strftime("%Y-%m-%d"),
        "Kickoff MMT": local_dt.strftime("%H:%M"),
        "League": match["league_code"],
        "Home": home,
        "Away": away,
        "Predicted Score": f"{score[0]} - {score[1]}",
        "Home Win %": round(probs["home"] * 100, 2),
        "Draw %": round(probs["draw"] * 100, 2),
        "Away Win %": round(probs["away"] * 100, 2),
        "Over 2.5 %": round(probs["over25"] * 100, 2),
        "Under 2.5 %": round(probs["under25"] * 100, 2),
        "BTTS %": round(probs["btts"] * 100, 2),
        "Model Confidence %": round(
            probability_confidence * 100,
            2,
        ),
        "Data Quality %": round(data_quality * 100, 2),
        "Ranking Score": round(ranking_score, 2),
        "Home Lambda": round(pred["hl"], 2),
        "Away Lambda": round(pred["al"], 2),
    }

    predictions.append(row)

    save_prediction(
        match_id,
        match["league_code"],
        local_dt.strftime("%Y-%m-%d %H:%M"),
        home,
        away,
        score,
        pred["hl"],
        pred["al"],
    )

df = pd.DataFrame(predictions)

# =========================================================
# DAILY TOP 6 ANALYSIS
# =========================================================

st.divider()
st.header("🏆 Daily Top 6 Statistical Analysis")

if df.empty:
    st.info("Fixture မရှိပါ။")
else:
    selected_date = st.date_input(
        "Select Match Date",
        value=datetime.now(LOCAL_TZ).date(),
    )

    daily = df[
        df["Date"] == selected_date.strftime("%Y-%m-%d")
    ].copy()

    if daily.empty:
        st.warning("ရွေးချယ်ထားသောနေ့တွင် ပွဲမရှိပါ။")
    else:
        top6 = daily.sort_values(
            "Ranking Score",
            ascending=False,
        ).head(6)

        display_columns = [
            "League",
            "Kickoff MMT",
            "Home",
            "Away",
            "Predicted Score",
            "Home Win %",
            "Draw %",
            "Away Win %",
            "Over 2.5 %",
            "Under 2.5 %",
            "Model Confidence %",
            "Data Quality %",
            "Ranking Score",
        ]

        st.dataframe(
            top6[display_columns],
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download Daily Top 6 CSV",
            top6[display_columns].to_csv(index=False).encode(),
            file_name="daily_top6_analysis.csv",
            mime="text/csv",
        )

# =========================================================
# ALL FIXTURES
# =========================================================

st.divider()
st.header("📅 All League Fixtures & Predictions")

if not df.empty:
    st.dataframe(
        df.sort_values(
            ["Date", "Kickoff MMT", "League"]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download All Predictions CSV",
        df.to_csv(index=False).encode(),
        file_name="all_predictions_v24.csv",
        mime="text/csv",
    )

# =========================================================
# HISTORY & ACCURACY
# =========================================================

st.divider()
st.header("📚 Match History & Accuracy")

history = load_history()

if history.empty:
    st.info("Prediction History မရှိသေးပါ။")
else:
    completed = history.dropna(
        subset=["actual_home", "actual_away"]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Predictions",
        len(history),
    )

    col2.metric(
        "Completed Matches",
        len(completed),
    )

    if len(completed) > 0:
        exact_accuracy = (
            completed["prediction_correct"].sum()
            / len(completed)
        ) * 100

        col3.metric(
            "Exact Score Accuracy",
            f"{exact_accuracy:.2f}%",
        )
    else:
        col3.metric(
            "Exact Score Accuracy",
            "N/A",
        )

    st.dataframe(
        history,
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download History CSV",
        history.to_csv(index=False).encode(),
        file_name="football_history_v24.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Professional Football Predictor V2.4 "
    "| 7 Leagues | Daily Top 6 Statistical Analysis"
)
