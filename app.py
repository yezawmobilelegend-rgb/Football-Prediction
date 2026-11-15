from datetime import datetime, timedelta, timezone
import math
import re
import sqlite3
import requests
import numpy as np
import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Professional Football Predictor V2.4",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "football_history.db"
LOCAL_TZ = ZoneInfo("Asia/Yangon")  # မြန်မာစံတော်ချိန်

LEAGUES = {
    "English Premier League (PL)": "PL",
    "Spanish La Liga (PD)": "PD",
    "German Bundesliga (BL1)": "BL1",
    "French Ligue 1 (FL1)": "FL1",
    "Italy Serie A (SA)": "SA",  # 🇮🇹 Italy Serie A ထည့်သွင်းထားပါပြီ
    "UEFA Champions League (CL)": "CL",
    "UEFA Europa League (EL)": "EL",
}

# =========================================================
# DATABASE
# =========================================================

def db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.text_factory = str
    return conn


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
            ON CONFLICT(match_id) DO UPDATE SET
                predicted_home=excluded.predicted_home,
                predicted_away=excluded.predicted_away,
                home_lambda=excluded.home_lambda,
                away_lambda=excluded.away_lambda
        """, (
            str(match_id),
            str(league),
            str(match_date),
            str(home_team),
            str(away_team),
            int(score[0]),
            int(score[1]),
            float(home_lambda),
            float(away_lambda),
            datetime.now().isoformat(),
        ))
        conn.commit()


def update_actual_results(league_code, api_key):
    if not api_key:
        return

    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
    headers = {"X-Auth-Token": api_key}

    try:
        response = requests.get(url, headers=headers, params={"status": "FINISHED"}, timeout=20)
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
                row = conn.execute("SELECT predicted_home, predicted_away FROM predictions WHERE match_id = ?", (match_id,)).fetchone()
                if row is None:
                    continue

                correct = int(row[0] == actual_home and row[1] == actual_away)
                conn.execute("""
                    UPDATE predictions
                    SET actual_home = ?, actual_away = ?, prediction_correct = ?
                    WHERE match_id = ?
                """, (actual_home, actual_away, correct, match_id))
            conn.commit()
    except Exception as e:
        print("Actual Result Update Error:", e)


def load_history():
    with db_connection() as conn:
        return pd.read_sql_query("SELECT * FROM predictions ORDER BY match_date DESC", conn)


initialize_database()

# =========================================================
# API SETTINGS
# =========================================================

st.sidebar.header("🔑 API Settings")

fb_default = st.secrets.get("FOOTBALL_API_KEY", "")
odds_default = st.secrets.get("ODDS_API_KEY", "")

fb_key = st.sidebar.text_input("Football-data.org API Key", value=fb_default, type="password")
odds_key = st.sidebar.text_input("Odds-api.io API Key", value=odds_default, type="password")

selected_league = st.sidebar.selectbox("League", list(LEAGUES.keys()))
league_code = LEAGUES[selected_league]

st.sidebar.divider()
st.sidebar.caption("Odds မရရှိပါက Default Odds မထည့်ပါ။")

# =========================================================
# API HELPERS
# =========================================================

def api_get(url, headers=None, params=None):
    try:
        response = requests.get(url, headers=headers, params=params, timeout=20)
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
        f"https://api.football-data.org/v4/competitions/{league}/matches",
        headers={"X-Auth-Token": api_key},
        params={"status": "SCHEDULED"},
    )
    if not data:
        return []

    now = datetime.now(timezone.utc)
    end = now + timedelta(days=7)
    result = []

    for match in data.get("matches", []):
        match_date = datetime.strptime(match["utcDate"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        if now <= match_date <= end:
            result.append(match)
    return result


@st.cache_data(ttl=3600)
def fetch_standings(league, api_key):
    if not api_key:
        return {}
    data = api_get(
        f"https://api.football-data.org/v4/competitions/{league}/standings",
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
    data = api_get("https://api.odds-api.io/v1/odds", params={"apiKey": api_key, "sport": "soccer", "region": "eu"})
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("data", [])
    return []


if st.sidebar.button("🔄 Live Odds Refresh", use_container_width=True):
    fetch_odds.clear()
    st.rerun()

# =========================================================
# TEAM NAME MATCH
# =========================================================

def normalize_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def team_match(a, b):
    a = normalize_name(a)
    b = normalize_name(b)
    return a == b or a in b or b in a


# =========================================================
# ODDS EXTRACTION
# =========================================================

def extract_match_odds(home, away, odds_data):
    result = {
        "home_hdp": None, "home_price": None,
        "away_hdp": None, "away_price": None,
        "over_line": None, "over_price": None,
        "under_line": None, "under_price": None,
        "home_win_price": None, "draw_price": None, "away_win_price": None,
    }
    if not odds_data:
        return result

    for event in odds_data:
        if team_match(home, event.get("home_team", "")) and team_match(away, event.get("away_team", "")):
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
                        elif key == "h2h":
                            if team_match(home, name):
                                result["home_win_price"] = price
                            elif team_match(away, name):
                                result["away_win_price"] = price
                            elif "draw" in name or "tie" in name:
                                result["draw_price"] = price
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
# RECENT FORM
# =========================================================

@st.cache_data(ttl=1800)
def fetch_recent_form(league, api_key):
    if not api_key:
        return {}
    data = api_get(
        f"https://api.football-data.org/v4/competitions/{league}/matches",
        headers={"X-Auth-Token": api_key},
        params={"status": "FINISHED"},
    )
    if not data:
        return {}

    team_matches = {}
    for match in data.get("matches", []):
        score = match.get("score", {}).get("fullTime", {})
        h, a = score.get("home"), score.get("away")
        if h is None or a is None:
            continue

        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]
        date = match["utcDate"]

        team_matches.setdefault(home, []).append({"date": date, "points": 3 if h > a else 1 if h == a else 0, "gf": h, "ga": a})
        team_matches.setdefault(away, []).append({"date": date, "points": 3 if a > h else 1 if a == h else 0, "gf": a, "ga": h})

    form = {}
    for team, games in team_matches.items():
        games = sorted(games, key=lambda x: x["date"], reverse=True)[:5]
        form[team] = {
            "points": sum(g["points"] for g in games),
            "gf": sum(g["gf"] for g in games),
            "ga": sum(g["ga"] for g in games),
            "played": len(games),
            "form": "".join("W" if g["points"] == 3 else "D" if g["points"] == 1 else "L" for g in games),
        }
    return form


def get_team_data(team, standings, form):
    s = standings.get(team, {})
    f = form.get(team, {"points": 0, "gf": 0, "ga": 0, "played": 0, "form": "N/A"})
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
# POISSON MODEL (Argmax - Most Probable Score)
# =========================================================

def poisson_pmf(lam, k):
    return math.exp(-lam) * lam**k / math.factorial(k)


def calculate_lambdas(home_data, away_data):
    base_home = 1.65
    base_away = 1.15
    elo_diff = (home_data["elo"] - away_data["elo"]) / 200
    form_diff = (home_data["recent_points"] - away_data["recent_points"]) / 8

    home_lambda = max(0.30, base_home + elo_diff * 0.45 + form_diff * 0.20)
    away_lambda = max(0.30, base_away - elo_diff * 0.45 - form_diff * 0.20)

    if home_data["recent_played"] > 0:
        home_avg = home_data["recent_gf"] / home_data["recent_played"]
        home_lambda = home_lambda * 0.60 + home_avg * 0.40
    if away_data["recent_played"] > 0:
        away_avg = away_data["recent_gf"] / away_data["recent_played"]
        away_lambda = away_lambda * 0.60 + away_avg * 0.40

    return max(0.30, home_lambda), max(0.30, away_lambda)


def score_matrix(home_lambda, away_lambda, max_goals=6):
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            matrix[h, a] = poisson_pmf(home_lambda, h) * poisson_pmf(away_lambda, a)
    return matrix


def calculate_probabilities(matrix):
    max_goals = matrix.shape[0] - 1
    home_win, draw, away_win = 0, 0, 0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            if h > a:
                home_win += matrix[h, a]
            elif h == a:
                draw += matrix[h, a]
            else:
                away_win += matrix[h, a]

    total_probs = {}
    for total in range(2 * max_goals + 1):
        probability = 0
        for h in range(max_goals + 1):
            a = total - h
            if 0 <= a <= max_goals:
                probability += matrix[h, a]
        total_probs[total] = probability

    btts_yes = sum(matrix[h, a] for h in range(1, max_goals + 1) for a in range(1, max_goals + 1))
    
    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over25": sum(p for g, p in total_probs.items() if g > 2.5),
        "under25": sum(p for g, p in total_probs.items() if g < 2.5),
        "btts_yes": btts_yes,
        "total_probs": total_probs,
    }


def predict_match(home_data, away_data):
    home_lambda, away_lambda = calculate_lambdas(home_data, away_data)
    matrix = score_matrix(home_lambda, away_lambda, max_goals=6)
    
    # 🎯 Probability အရ အဖြစ်နိုင်ဆုံး (Argmax) ရလဒ်ကို တိုက်ရိုက်ယူခြင်း
    best_score = np.unravel_index(np.argmax(matrix), matrix.shape)
    best_probability = matrix[best_score]
    probs = calculate_probabilities(matrix)

    return {
        "home_lambda": home_lambda,
        "away_lambda": away_lambda,
        "matrix": matrix,
        "best_score": best_score,
        "best_probability": best_probability,
        "probabilities": probs,
    }


# =========================================================
# UI HEADER & MAIN EXECUTION
# =========================================================

st.title("⚽ Professional Football Predictor V2.4")
st.markdown("အဆင့်မြင့် Poisson AI မော်ဒယ်၊ Italy Serie A နှင့် **Top 10 Most Probable Matches** စနစ်")

fixtures = fetch_fixtures(league_code, fb_key)
standings = fetch_standings(league_code, fb_key)
recent_form = fetch_recent_form(league_code, fb_key)
odds_data = fetch_odds(odds_key)

if st.sidebar.button("📊 Update Actual Results"):
    update_actual_results(league_code, fb_key)
    st.success("Actual Results Updated")
    st.rerun()

# =========================================================
# 🌟 TOP 10 MOST PROBABLE MATCHES OF THE DAY (ALL LEAGUES)
# =========================================================
st.subheader("🔥 The Most Probable Predicted (10) Matches of the Day")

@st.cache_data(ttl=600)
def get_top_10_matches(all_leagues, api_key):
    candidates = []
    for l_name, l_code in all_leagues.items():
        fixs = fetch_fixtures(l_code, api_key)
        stds = fetch_standings(l_code, api_key)
        forms = fetch_recent_form(l_code, api_key)
        
        for m in fixs:
            h = m["homeTeam"]["name"]
            a = m["awayTeam"]["name"]
            m_id = m["id"]
            utc_dt = datetime.strptime(m["utcDate"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            local_dt = utc_dt.astimezone(LOCAL_TZ)
            
            hd = get_team_data(h, stds, forms)
            ad = get_team_data(a, stds, forms)
            pred = predict_match(hd, ad)
            
            # Confidence score = Max probability of outcome (Home win, Away win, or Exact score probability)
            max_1x2 = max(pred["probabilities"]["home_win"], pred["probabilities"]["away_win"])
            
            candidates.append({
                "league": l_name.split(" ")[0],
                "date": local_dt.strftime("%d-%b %H:%M"),
                "match": f"{h} vs {a}",
                "pred_score": f"{pred['best_score'][0]} - {pred['best_score'][1]}",
                "confidence": max_1x2 * 100,
                "score_prob": pred["best_probability"] * 100
            })
            
    # Sort by confidence/probability descending
    candidates = sorted(candidates, key=lambda x: x["confidence"], reverse=True)
    return candidates[:10]

if fb_key:
    with st.spinner("Calculating top probable matches across all leagues..."):
        top_10 = get_top_10_matches(LEAGUES, fb_key)
        if top_10:
            top_10_df = pd.DataFrame(top_10)
            top_10_df.columns = ["League", "Date (MMT)", "Match", "Predicted Score", "Win Confidence (%)", "Score Prob (%)"]
            st.dataframe(top_10_df, hide_index=True, use_container_width=True)
        else:
            st.info("လာမည့် 7 ရက်အတွင်း ပွဲစဉ်များ မရှိသေးပါ။")
else:
    st.info("Top 10 matches ကြည့်ရှုရန် Sidebar တွင် Football-data.org API Key ထည့်သွင်းပေးပါ။")

st.divider()
st.subheader(f"📅 {selected_league} | Next 7 Days")

if not fixtures:
    st.warning("ပွဲစဉ်မရှိပါ သို့မဟုတ် API Key မရရှိပါ။")
else:
    st.success(f"{len(fixtures)} matches found")

for index, match in enumerate(fixtures):
    home = match["homeTeam"]["name"]
    away = match["awayTeam"]["name"]
    match_id = match["id"]
    
    utc_dt = datetime.strptime(match["utcDate"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    local_dt = utc_dt.astimezone(LOCAL_TZ)
    match_date_str = local_dt.strftime("%Y-%m-%d %H:%M MMT")

    with st.expander(f"🏟️ {home} vs {away} | 🕒 {match_date_str}"):
        left, right = st.columns([1, 1.3])

        with left:
            st.markdown("### 🌐 Live World Odds")
            odds = extract_match_odds(home, away, odds_data)

            home_hdp = format_handicap(home, odds["home_hdp"], odds["home_price"])
            away_hdp = format_handicap(away, odds["away_hdp"], odds["away_price"])
            over = format_ou(odds["over_line"], odds["over_price"])
            under = format_ou(odds["under_line"], odds["under_price"])

            if not home_hdp and not away_hdp and not over and not under:
                st.info("ကမ္ဘာ့ကြေးများ မထွက်သေးပါ")
            else:
                if home_hdp: st.write("🏠", home_hdp)
                if away_hdp: st.write("✈️", away_hdp)
                if over: st.write("Over:", over)
                if under: st.write("Under:", under)

            st.divider()
            st.text_input("ဒေသတွင်း Handicap", key=f"hdp_{match_id}")
            st.text_input("ဒေသတွင်း O/U", key=f"ou_{match_id}")

        with right:
            st.markdown("### 🤖 AI Statistical & Value Analysis")
            hd = get_team_data(home, standings, recent_form)
            ad = get_team_data(away, standings, recent_form)
            prediction = predict_match(hd, ad)

            hl = prediction["home_lambda"]
            al = prediction["away_lambda"]
            score = prediction["best_score"]
            score_prob = prediction["best_probability"]
            probs = prediction["probabilities"]

            st.markdown(f"**Elo:** {home} {hd['elo']:.0f} | {away} {ad['elo']:.0f}")
            st.markdown(f"**Recent Form:** {home} `{hd['recent_form']}` | {away} `{ad['recent_form']}`")
            st.markdown(f"**Expected Goals:** {hl:.2f} - {al:.2f}")
            st.success(f"🎯 **Predicted Score (Most Probable): {home} {score[0]} - {score[1]} {away}** (Probability: {score_prob * 100:.1f}%)")

            # Value Bet Finder
            st.divider()
            st.markdown("### 💎 Value Bet Finder")
            h_price = odds.get("home_win_price")
            a_price = odds.get("away_win_price")
            value_found = False

            if h_price and h_price > 1.0:
                market_prob_home = 1.0 / h_price
                ai_prob_home = probs["home_win"]
                edge_home = (ai_prob_home - market_prob_home) * 100
                if (ai_prob_home * h_price) > 1.03:
                    value_found = True
                    st.success(f"🔥 **Value Bet ({home} အနိုင်):** Edge: **+{edge_home:.1f}%**")

            if a_price and a_price > 1.0:
                market_prob_away = 1.0 / a_price
                ai_prob_away = probs["away_win"]
                edge_away = (ai_prob_away - market_prob_away) * 100
                if (ai_prob_away * a_price) > 1.03:
                    value_found = True
                    st.success(f"🔥 **Value Bet ({away} အဝေးကွင်းအနိုင်):** Edge: **+{edge_away:.1f}%**")

            if not value_found:
                st.info("ℹ️ သိသာထင်ရှားသော Value Bet မတွေ့ရသေးပါ။")

            st.divider()
            st.markdown("### 📊 Over / Under Probability")
            ou_df = pd.DataFrame({
                "Market": ["Over 2.5", "Under 2.5"],
                "Probability (%)": [probs["over25"] * 100, probs["under25"] * 100],
            })
            st.dataframe(ou_df, hide_index=True, use_container_width=True)

            st.markdown("### 🏆 1X2 Probability")
            st.write(f"Home Win: {probs['home_win'] * 100:.2f}% | Draw: {probs['draw'] * 100:.2f}% | Away Win: {probs['away_win'] * 100:.2f}%")

            save_prediction(match_id, league_code, match_date_str, home, away, score, hl, al)

# =========================================================
# HISTORY & ACCURACY
# =========================================================

st.divider()
st.header("📚 Match History & Prediction Accuracy")

history = load_history()
if history.empty:
    st.info("Prediction History မရှိသေးပါ။")
else:
    completed = history.dropna(subset=["actual_home", "actual_away"])
    total_predictions = len(history)
    completed_count = len(completed)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", total_predictions)
    col2.metric("Completed Matches", completed_count)

    if completed_count > 0:
        exact_accuracy = (completed["prediction_correct"].sum() / completed_count) * 100
        col3.metric("Exact Score Accuracy", f"{exact_accuracy:.2f}%")
    else:
        col3.metric("Exact Score Accuracy", "N/A")

    st.dataframe(history, use_container_width=True, hide_index=True)

    csv_data = history.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Match History CSV",
        data=csv_data,
        file_name="football_prediction_history.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Professional Football Predictor Dashboard V2.4 (Italy Serie A & Top 10 Added)")
