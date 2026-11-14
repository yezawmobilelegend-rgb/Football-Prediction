
from datetime import datetime, timedelta, timezone
import math
import re
import sqlite3
import requests
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Professional Football Predictor V2",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "football_history.db"

LEAGUES = {
    "English Premier League (PL)": "PL",
    "Spanish La Liga (PD)": "PD",
    "German Bundesliga (BL1)": "BL1",
    "French Ligue 1 (FL1)": "FL1",
    "UEFA Champions League (CL)": "CL",
    "UEFA Europa League (EL)": "EL",
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
    home_team,
    away_team,
    score,
    home_lambda,
    away_lambda,
):

    with db_connection() as conn:

        conn.execute("""
            INSERT INTO predictions (
                match_id,
                league,
                match_date,
                home_team,
                away_team,
                predicted_home,
                predicted_away,
                home_lambda,
                away_lambda,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(match_id) DO NOTHING
        """, (
            str(match_id),
            league,
            match_date,
            home_team,
            away_team,
            score[0],
            score[1],
            home_lambda,
            away_lambda,
            datetime.now().isoformat(),
        ))

        conn.commit()


def update_actual_results(league_code, api_key):

    if not api_key:
        return

    url = (
        f"https://api.football-data.org/v4/"
        f"competitions/{league_code}/matches"
    )

    headers = {"X-Auth-Token": api_key}

    try:

        response = requests.get(
            url,
            headers=headers,
            params={"status": "FINISHED"},
            timeout=20,
        )

        if response.status_code != 200:
            return

        matches = response.json().get("matches", [])

        with db_connection() as conn:

            for match in matches:

                score = match.get("score", {}).get("fullTime", {})

                actual_home = score.get("home")
                actual_away = score.get("away")

                if actual_home is None or actual_away is None:
                    continue

                match_id = str(match["id"])

                row = conn.execute("""
                    SELECT predicted_home, predicted_away
                    FROM predictions
                    WHERE match_id = ?
                """, (match_id,)).fetchone()

                if row is None:
                    continue

                correct = int(
                    row[0] == actual_home
                    and row[1] == actual_away
                )

                conn.execute("""
                    UPDATE predictions
                    SET actual_home = ?,
                        actual_away = ?,
                        prediction_correct = ?
                    WHERE match_id = ?
                """, (
                    actual_home,
                    actual_away,
                    correct,
                    match_id,
                ))

            conn.commit()

    except Exception as e:
        print("Actual Result Update Error:", e)


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

st.sidebar.header("🔑 API Settings")

fb_default = st.secrets.get("FOOTBALL_API_KEY", "")
odds_default = st.secrets.get("ODDS_API_KEY", "")

fb_key = st.sidebar.text_input(
    "Football-data.org API Key",
    value=fb_default,
    type="password",
)

odds_key = st.sidebar.text_input(
    "Odds-api.io API Key",
    value=odds_default,
    type="password",
)

selected_league = st.sidebar.selectbox(
    "League",
    list(LEAGUES.keys()),
)

league_code = LEAGUES[selected_league]

st.sidebar.divider()

st.sidebar.caption(
    "Odds မရရှိပါက Default Odds မထည့်ပါ။"
)

# =========================================================
# API HELPERS
# =========================================================

def api_get(url, headers=None, params=None):

    try:

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=20,
        )

        if response.status_code == 200:
            return response.json()

        return None

    except requests.RequestException:
        return None


@st.cache_data(ttl=900)
def fetch_fixtures(league, api_key):

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
    end = now + timedelta(days=7)

    result = []

    for match in data.get("matches", []):

        match_date = datetime.strptime(
            match["utcDate"],
            "%Y-%m-%dT%H:%M:%SZ",
        ).replace(tzinfo=timezone.utc)

        if now <= match_date <= end:
            result.append(match)

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

    stats = {}

    for table in data.get("standings", []):

        if table.get("type") != "TOTAL":
            continue

        for row in table.get("table", []):

            name = row["team"]["name"]
            points = row.get("points", 0)
            gd = row.get("goalDifference", 0)

            stats[name] = {
                "points": points,
                "gd": gd,
                "elo": 1500 + points * 15 + gd * 5,
            }

    return stats


@st.cache_data(ttl=300)
def fetch_odds(api_key):

    if not api_key:
        return []

    data = api_get(
        "https://api.odds-api.io/v1/odds",
        params={
            "apiKey": api_key,
            "sport": "soccer",
            "region": "eu",
        },
    )

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        return data.get("data", [])

    return []


# =========================================================
# LIVE REFRESH
# =========================================================

if st.sidebar.button(
    "🔄 Live Odds Refresh",
    use_container_width=True,
):

    fetch_odds.clear()

    st.rerun()


# =========================================================
# TEAM NAME MATCH
# =========================================================

def normalize_name(name):

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(name).lower(),
    )


def team_match(a, b):

    a = normalize_name(a)
    b = normalize_name(b)

    return a == b or a in b or b in a


# =========================================================
# ODDS EXTRACTION
# =========================================================

def extract_match_odds(home, away, odds_data):

    result = {
        "home_hdp": None,
        "home_price": None,
        "away_hdp": None,
        "away_price": None,
        "over_line": None,
        "over_price": None,
        "under_line": None,
        "under_price": None,
    }

    for event in odds_data:

        if not (
            team_match(home, event.get("home_team", ""))
            and team_match(away, event.get("away_team", ""))
        ):
            continue

        for bookmaker in event.get("bookmakers", []):

            for market in bookmaker.get("markets", []):

                key = market.get("key", "")

                for outcome in market.get("outcomes", []):

                    name = str(outcome.get("name", "")).lower()
                    point = outcome.get("point")
                    price = outcome.get("price")

                    if key == "spreads":

                        if team_match(home, name):
                            result["home_hdp"] = point
                            result["home_price"] = price

                        elif team_match(away, name):
                            result["away_hdp"] = point
                            result["away_price"] = price

                    elif key == "totals":

                        if "over" in name:
                            result["over_line"] = point
                            result["over_price"] = price

                        elif "under" in name:
                            result["under_line"] = point
                            result["under_price"] = price

        return result

    return result


def odds_price(price):

    if price is None:
        return None

    try:

        price = float(price)

        if price <= 1:
            return None

        profit = round((price - 1) * 100)

        if profit > 0:
            return f"+{profit}"

        if profit < 0:
            return str(profit)

        return "ဆုံ"

    except (ValueError, TypeError):
        return None


def format_handicap(team, line, price):

    if line is None or price is None:
        return None

    p = odds_price(price)

    if p is None:
        return None

    return f"{team} {abs(line)}{p}"


def format_ou(line, price):

    if line is None or price is None:
        return None

    p = odds_price(price)

    if p is None:
        return None

    return f"{line}{p}"


# =========================================================
# RECENT FORM: LAST 5 MATCHES
# =========================================================

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

    team_matches = {}

    for match in data.get("matches", []):

        score = match.get("score", {}).get("fullTime", {})

        h = score.get("home")
        a = score.get("away")

        if h is None or a is None:
            continue

        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]

        date = match["utcDate"]

        team_matches.setdefault(home, []).append({
            "date": date,
            "points": 3 if h > a else 1 if h == a else 0,
            "gf": h,
            "ga": a,
        })

        team_matches.setdefault(away, []).append({
            "date": date,
            "points": 3 if a > h else 1 if a == h else 0,
            "gf": a,
            "ga": h,
        })

    form = {}

    for team, games in team_matches.items():

        games = sorted(
            games,
            key=lambda x: x["date"],
            reverse=True,
        )[:5]

        form[team] = {
            "points": sum(g["points"] for g in games),
            "gf": sum(g["gf"] for g in games),
            "ga": sum(g["ga"] for g in games),
            "played": len(games),
            "form": "".join(
                "W" if g["points"] == 3
                else "D" if g["points"] == 1
                else "L"
                for g in games
            ),
        }

    return form


def get_team_data(team, standings, form):

    s = standings.get(team, {})

    f = form.get(team, {
        "points": 0,
        "gf": 0,
        "ga": 0,
        "played": 0,
        "form": "N/A",
    })

    return {
        "elo": s.get("elo", 1500),
        "league_points": s.get("points", 0),
        "recent_points": f["points"],
        "recent_gf": f["gf"],
        "recent_ga": f["ga"],
        "recent_form": f["form"],
        "recent_played": f["played"],
    }


# =========================================================
# POISSON MODEL
# =========================================================

def poisson_pmf(lam, k):

    return (
        math.exp(-lam)
        * lam**k
        / math.factorial(k)
    )


def calculate_lambdas(home_data, away_data):

    base_home = 1.55
    base_away = 1.05

    elo_diff = (
        home_data["elo"] - away_data["elo"]
    ) / 250

    form_diff = (
        home_data["recent_points"]
        - away_data["recent_points"]
    ) / 10

    home_lambda = max(
        0.20,
        base_home
        + elo_diff * 0.35
        + form_diff * 0.15,
    )

    away_lambda = max(
        0.20,
        base_away
        - elo_diff * 0.35
        - form_diff * 0.15,
    )

    # Recent scoring adjustment
    if home_data["recent_played"] > 0:

        home_avg = (
            home_data["recent_gf"]
            / home_data["recent_played"]
        )

        home_lambda = (
            home_lambda * 0.70
            + home_avg * 0.30
        )

    if away_data["recent_played"] > 0:

        away_avg = (
            away_data["recent_gf"]
            / away_data["recent_played"]
        )

        away_lambda = (
            away_lambda * 0.70
            + away_avg * 0.30
        )

    return (
        max(0.20, home_lambda),
        max(0.20, away_lambda),
    )


# =========================================================
# SCORE PROBABILITY MATRIX
# =========================================================

def score_matrix(home_lambda, away_lambda, max_goals=6):

    matrix = np.zeros(
        (max_goals + 1, max_goals + 1)
    )

    for h in range(max_goals + 1):

        for a in range(max_goals + 1):

            matrix[h, a] = (
                poisson_pmf(home_lambda, h)
                * poisson_pmf(away_lambda, a)
            )

    return matrix


# =========================================================
# STATISTICAL PROBABILITIES
# =========================================================

def calculate_probabilities(matrix):

    max_goals = matrix.shape[0] - 1

    home_win = 0
    draw = 0
    away_win = 0

    for h in range(max_goals + 1):

        for a in range(max_goals + 1):

            if h > a:
                home_win += matrix[h, a]

            elif h == a:
                draw += matrix[h, a]

            else:
                away_win += matrix[h, a]

    # Probability distributions of total goals
    total_probs = {}

    for total in range(2 * max_goals + 1):

        probability = 0

        for h in range(max_goals + 1):

            a = total - h

            if 0 <= a <= max_goals:
                probability += matrix[h, a]

        total_probs[total] = probability

    def over_probability(line):

        return sum(
            p for goals, p in total_probs.items()
            if goals > line
        )

    def under_probability(line):

        return sum(
            p for goals, p in total_probs.items()
            if goals < line
        )

    # BTTS
    btts_yes = sum(
        matrix[h, a]
        for h in range(1, max_goals + 1)
        for a in range(1, max_goals + 1)
    )

    btts_no = 1 - btts_yes

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over15": over_probability(1.5),
        "under15": under_probability(1.5),
        "over25": over_probability(2.5),
        "under25": under_probability(2.5),
        "over35": over_probability(3.5),
        "under35": under_probability(3.5),
        "btts_yes": btts_yes,
        "btts_no": btts_no,
        "total_probs": total_probs,
    }


# =========================================================
# PREDICTION ENGINE
# =========================================================

def predict_match(home_data, away_data):

    home_lambda, away_lambda = calculate_lambdas(
        home_data,
        away_data,
    )

    matrix = score_matrix(
        home_lambda,
        away_lambda,
        max_goals=6,
    )

    best_score = np.unravel_index(
        np.argmax(matrix),
        matrix.shape,
    )

    probs = calculate_probabilities(matrix)

    return {
        "home_lambda": home_lambda,
        "away_lambda": away_lambda,
        "matrix": matrix,
        "best_score": best_score,
        "best_probability": matrix[best_score],
        "probabilities": probs,
    }


# =========================================================
# LOAD DATA
# =========================================================

st.title("⚽ Professional Football Predictor V2")

fixtures = fetch_fixtures(
    league_code,
    fb_key,
)

standings = fetch_standings(
    league_code,
    fb_key,
)

recent_form = fetch_recent_form(
    league_code,
    fb_key,
)

odds_data = fetch_odds(
    odds_key,
)

# =========================================================
# MATCH RESULTS UPDATE
# =========================================================

if st.sidebar.button("📊 Update Actual Results"):

    update_actual_results(
        league_code,
        fb_key,
    )

    st.success("Actual Results Updated")
    st.rerun()


# =========================================================
# FIXTURES DISPLAY
# =========================================================

st.subheader(
    f"📅 {selected_league} | Next 7 Days"
)

if not fixtures:

    st.warning(
        "ပွဲစဉ်မရှိပါ သို့မဟုတ် API Key မရရှိပါ။"
    )

else:

    st.success(
        f"{len(fixtures)} matches found"
    )

for index, match in enumerate(fixtures):

    home = match["homeTeam"]["name"]
    away = match["awayTeam"]["name"]
    match_id = match["id"]
    match_date = match["utcDate"]

    with st.expander(
        f"🏟️ {home} vs {away} | "
        f"{match_date}"
    ):

        left, right = st.columns([1, 1.3])

        # =================================================
        # ODDS
        # =================================================

        with left:

            st.markdown("### 🌐 Live World Odds")

            odds = extract_match_odds(
                home,
                away,
                odds_data,
            )

            home_hdp = format_handicap(
                home,
                odds["home_hdp"],
                odds["home_price"],
            )

            away_hdp = format_handicap(
                away,
                odds["away_hdp"],
                odds["away_price"],
            )

            over = format_ou(
                odds["over_line"],
                odds["over_price"],
            )

            under = format_ou(
                odds["under_line"],
                odds["under_price"],
            )

            if (
                home_hdp is None
                and away_hdp is None
                and over is None
                and under is None
            ):

                st.info(
                    "ကမ္ဘာ့ကြေးများ မထွက်သေးပါ"
                )

            else:

                if home_hdp:
                    st.write("🏠", home_hdp)

                if away_hdp:
                    st.write("✈️", away_hdp)

                if over:
                    st.write("Over:", over)

                if under:
                    st.write("Under:", under)

            st.divider()

            local_hdp = st.text_input(
                "ဒေသတွင်း Handicap",
                key=f"hdp_{match_id}",
            )

            local_ou = st.text_input(
                "ဒေသတွင်း O/U",
                key=f"ou_{match_id}",
            )

        # =================================================
        # AI
        # =================================================

        with right:

            st.markdown("### 🤖 AI Statistical Analysis")

            hd = get_team_data(
                home,
                standings,
                recent_form,
            )

            ad = get_team_data(
                away,
                standings,
                recent_form,
            )

            prediction = predict_match(
                hd,
                ad,
            )

            hl = prediction["home_lambda"]
            al = prediction["away_lambda"]

            score = prediction["best_score"]
            score_prob = prediction["best_probability"]

            matrix = prediction["matrix"]
            probs = prediction["probabilities"]

            st.markdown(
                f"**Elo:** {home} {hd['elo']:.0f} "
                f"| {away} {ad['elo']:.0f}"
            )

            st.markdown(
                f"**Recent Form:** {home} "
                f"`{hd['recent_form']}` "
                f"| {away} `{ad['recent_form']}`"
            )

            st.markdown(
                f"**Expected Goals:** "
                f"{hl:.2f} - {al:.2f}"
            )

            st.success(
                f"🎯 **Predicted Score: "
                f"{home} {score[0]} - {score[1]} {away}**"
            )

            st.write(
                f"Best Score Probability: "
                f"{score_prob * 100:.2f}%"
            )

            # =============================================
            # O/U SCORE-BASED INDICATOR
            # =============================================

            predicted_total = score[0] + score[1]

            st.write(
                f"မှန်းဆဂိုးပေါင်း: {predicted_total} ဂိုး"
            )

            if predicted_total > 2.5:

                st.success("Score-based: Over 2.5")

            else:

                st.info("Score-based: Under 2.5")

            # =============================================
            # O/U PROBABILITY
            # =============================================

            st.divider()

            st.markdown("### 📊 Over / Under Probability")

            ou_df = pd.DataFrame({
                "Market": [
                    "Over 1.5",
                    "Under 1.5",
                    "Over 2.5",
                    "Under 2.5",
                    "Over 3.5",
                    "Under 3.5",
                ],
                "Probability (%)": [
                    probs["over15"] * 100,
                    probs["under15"] * 100,
                    probs["over25"] * 100,
                    probs["under25"] * 100,
                    probs["over35"] * 100,
                    probs["under35"] * 100,
                ],
            })

            st.dataframe(
                ou_df,
                hide_index=True,
                use_container_width=True,
            )

            # =============================================
            # BTTS
            # =============================================

            st.markdown("### ⚽ Both Teams To Score")

            st.write(
                f"BTTS Yes: "
                f"{probs['btts_yes'] * 100:.2f}%"
            )

            st.write(
                f"BTTS No: "
                f"{probs['btts_no'] * 100:.2f}%"
            )

            # =============================================
            # 1X2 PROBABILITY
            # =============================================

            st.markdown("### 🏆 1X2 Probability")

            st.write(
                f"Home Win: "
                f"{probs['home_win'] * 100:.2f}%"
            )

            st.write(
                f"Draw: "
                f"{probs['draw'] * 100:.2f}%"
            )

            st.write(
                f"Away Win: "
                f"{probs['away_win'] * 100:.2f}%"
            )

            # =============================================
            # SCORE MATRIX
            # =============================================

            st.divider()

            st.markdown("### 🔢 Score Probability Matrix")

            matrix_df = pd.DataFrame(
                matrix * 100,
                index=[
                    f"Home {i}"
                    for i in range(matrix.shape[0])
                ],
                columns=[
                    f"Away {i}"
                    for i in range(matrix.shape[1])
                ],
            )

            st.dataframe(
                matrix_df.style.format("{:.2f}%").background_gradient(
                    cmap="RdYlGn"
                ),
                use_container_width=True,
            )

            # =============================================
            # LOCAL O/U COMPARISON
            # =============================================

            if local_ou:

                numbers = re.findall(
                    r"\d+\.?\d*",
                    local_ou,
                )

                if numbers:

                    line = float(numbers[0])

                    if predicted_total > line:

                        st.info(
                            f"Score-based Local O/U: "
                            f"Over {line}"
                        )

                    elif predicted_total < line:

                        st.info(
                            f"Score-based Local O/U: "
                            f"Under {line}"
                        )

                    else:

                        st.info(
                            "Score နှင့် O/U Line တူညီနေသည်"
                        )

            # =============================================
            # SAVE HISTORY
            # =============================================

            save_prediction(
                match_id,
                league_code,
                match_date,
                home,
                away,
                score,
                hl,
                al,
            )


# =========================================================
# HISTORY & ACCURACY
# =========================================================

st.divider()

st.header("📚 Match History & Prediction Accuracy")

history = load_history()

if history.empty:

    st.info(
        "Prediction History မရှိသေးပါ။"
    )

else:

    completed = history.dropna(
        subset=[
            "actual_home",
            "actual_away",
        ]
    )

    total_predictions = len(history)
    completed_count = len(completed)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Predictions",
        total_predictions,
    )

    col2.metric(
        "Completed Matches",
        completed_count,
    )

    if completed_count > 0:

        exact_accuracy = (
            completed["prediction_correct"].sum()
            / completed_count
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
        use_container_width=True,
        hide_index=True,
    )

    csv_data = history.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥 Download Match History CSV",
        data=csv_data,
        file_name="football_prediction_history.csv",
        mime="text/csv",
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Professional Football Predictor Dashboard V2"
)

st.caption(
    "Football-data.org | Odds API | Poisson | SQLite"
)

st.caption(
    "Predictions are statistical estimates, "
    "not guaranteed match outcomes."
)
