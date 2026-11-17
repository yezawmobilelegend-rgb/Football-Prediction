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
    page_title="Professional Football Predictor V2.5",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "football_history_v2.db"
LOCAL_TZ = ZoneInfo("Asia/Yangon")  # မြန်မာစံတော်ချိန်

LEAGUES = {
    "English Premier League (PL)": "PL",
    "Italian Serie A (SA)": "SA",
    "Spanish La Liga (PD)": "PD",
    "German Bundesliga (BL1)": "BL1",
    "French Ligue 1 (FL1)": "FL1",
    "UEFA Champions League (CL)": "CL",
    "UEFA Europa League (EL)": "EL",
}

# =========================================================
# DATABASE INITIALIZATION
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


def load_history():
    with db_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY match_date DESC",
            conn,
        )


initialize_database()

# =========================================================
# API HELPERS & SETTINGS
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

nav_mode = st.sidebar.radio(
    "Navigation Mode",
    ["🏟️ Selected League View", "🔥 Top 10 Best Matches of the Day", "📚 Match History & Accuracy"]
)

if nav_mode == "🏟️ Selected League View":
    selected_league = st.sidebar.selectbox(
        "League",
        list(LEAGUES.keys()),
    )
    league_code = LEAGUES[selected_league]

st.sidebar.divider()


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
        match_date = datetime.strptime(
            match["utcDate"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
        if now <= match_date <= end:
            result.append(match)
    return result


@st.cache_data(ttl=1800)
def fetch_finished_matches(league, api_key):
    if not api_key:
        return []
    data = api_get(
        f"https://api.football-data.org/v4/competitions/{league}/matches",
        headers={"X-Auth-Token": api_key},
        params={"status": "FINISHED"},
    )
    return data.get("matches", []) if data else []


# =========================================================
# 1. ADVANCED DYNAMIC ELO RATING SYSTEM (Match-by-Match)
# =========================================================

@st.cache_data(ttl=3600)
def calculate_dynamic_elos(league, api_key):
    matches = fetch_finished_matches(league, api_key)
    # Sort matches chronologically
    matches = sorted(matches, key=lambda x: x["utcDate"])
    
    elos = {}
    k_factor = 25  # Tournament Importance K-factor
    home_advantage = 65  # Elo points for home advantage

    for match in matches:
        score = match.get("score", {}).get("fullTime", {})
        h_goals = score.get("home")
        a_goals = score.get("away")
        if h_goals is None or a_goals is None:
            continue

        h_team = match["homeTeam"]["name"]
        a_team = match["awayTeam"]["name"]

        if h_team not in elos:
            elos[h_team] = 1500.0
        if a_team not in elos:
            elos[a_team] = 1500.0

        R_home = elos[h_team] + home_advantage
        R_away = elos[a_team]

        E_home = 1 / (1 + 10 ** ((R_away - R_home) / 400))
        E_away = 1 / (1 + 10 ** ((R_home - R_away) / 400))

        if h_goals > a_goals:
            S_home, S_away = 1.0, 0.0
        elif h_goals < a_goals:
            S_home, S_away = 0.0, 1.0
        else:
            S_home, S_away = 0.5, 0.5

        elos[h_team] += k_factor * (S_home - E_home)
        elos[a_team] += k_factor * (S_away - E_away)

    return elos


# =========================================================
# 4. REST DAYS / FIXTURE CONGESTION & 5. xG DATA INTEGRATION
# =========================================================

def get_team_advanced_metrics(team, league, api_key, current_match_date):
    matches = fetch_finished_matches(league, api_key)
    team_matches = [
        m for m in matches 
        if m["homeTeam"]["name"] == team or m["awayTeam"]["name"] == team
    ]
    team_matches = sorted(team_matches, key=lambda x: x["utcDate"], reverse=True)

    rest_days = 7  # Default if no prior match found
    recent_xg_for = 1.3
    recent_xg_against = 1.1
    recent_form_points = 0
    recent_form_str = ""

    last_5 = team_matches[:5]
    for i, m in enumerate(last_5):
        m_date = datetime.strptime(m["utcDate"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        if i == 0:
            delta = (current_match_date - m_date).days
            if delta >= 0:
                rest_days = delta

        score = m.get("score", {}).get("fullTime", {})
        h_g = score.get("home", 0) or 0
        a_g = score.get("away", 0) or 0
        is_home = m["homeTeam"]["name"] == team

        gf = h_g if is_home else a_g
        ga = a_g if is_home else h_g

        pts = 3 if gf > ga else (1 if gf == ga else 0)
        recent_form_points += pts
        recent_form_str += "W" if pts == 3 else ("D" if pts == 1 else "L")

        # xG approximation fallback if api stats unavailable
        recent_xg_for += gf * 0.85
        recent_xg_against += ga * 0.85

    if len(last_5) > 0:
        recent_xg_for /= len(last_5)
        recent_xg_against /= len(last_5)

    return {
        "rest_days": rest_days,
        "xg_for": max(0.5, recent_xg_for),
        "xg_against": max(0.5, recent_xg_against),
        "form_points": recent_form_points,
        "form_str": recent_form_str[::-1] if recent_form_str else "N/A",
        "played": len(last_5)
    }


# =========================================================
# 3. HEAD-TO-HEAD (H2H) HISTORY / BOGEY TEAM FACTOR
# =========================================================

def get_h2h_adjustment(home_team, away_team, league, api_key):
    matches = fetch_finished_matches(league, api_key)
    h2h_matches = [
        m for m in matches 
        if (m["homeTeam"]["name"] == home_team and m["awayTeam"]["name"] == away_team) or
           (m["homeTeam"]["name"] == away_team and m["awayTeam"]["name"] == home_team)
    ]
    h2h_matches = sorted(h2h_matches, key=lambda x: x["utcDate"], reverse=True)[:5]
    
    home_advantage_h2h = 0.0
    for m in h2h_matches:
        score = m.get("score", {}).get("fullTime", {})
        h_g = score.get("home", 0) or 0
        a_g = score.get("away", 0) or 0
        is_home = m["homeTeam"]["name"] == home_team
        gf = h_g if is_home else a_g
        ga = a_g if is_home else h_g
        
        if gf > ga:
            home_advantage_h2h += 0.15
        elif gf < ga:
            home_advantage_h2h -= 0.15

    return home_advantage_h2h


# =========================================================
# POISSON MODEL & 2. DIXON-COLES EXTENSION
# =========================================================

def poisson_pmf(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def dixon_coles_tau(h, a, lam_h, lam_a, rho=-0.12):
    """ Dixon-Coles correction factor for low-scoring matches """
    if h == 0 and a == 0:
        return max(0.1, 1.0 - (lam_h * lam_a * rho))
    elif h == 1 and a == 0:
        return max(0.1, 1.0 + (lam_a * rho))
    elif h == 0 and a == 1:
        return max(0.1, 1.0 + (lam_h * rho))
    elif h == 1 and a == 1:
        return max(0.1, 1.0 - rho)
    return 1.0


def calculate_lambdas(home_data, away_data, h2h_adj, elo_diff):
    base_home = 1.55
    base_away = 1.10

    # Fatigue penalty based on rest days (< 4 days rest reduces output)
    home_fatigue = 0.90 if home_data["rest_days"] < 4 else 1.0
    away_fatigue = 0.90 if away_data["rest_days"] < 4 else 1.0

    home_lambda = (
        base_home 
        + (elo_diff / 400.0) 
        + (home_data["xg_for"] - 1.3) * 0.4
        + h2h_adj
    ) * home_fatigue

    away_lambda = (
        base_away 
        - (elo_diff / 400.0) 
        + (away_data["xg_for"] - 1.1) * 0.4
        - h2h_adj
    ) * away_fatigue

    return max(0.30, home_lambda), max(0.30, away_lambda)


def score_matrix(home_lambda, away_lambda, max_goals=6):
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            base_prob = poisson_pmf(home_lambda, h) * poisson_pmf(away_lambda, a)
            dc_factor = dixon_coles_tau(h, a, home_lambda, away_lambda)
            matrix[h, a] = max(0.0, base_prob * dc_factor)
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
        p_sum = sum(matrix[h, total - h] for h in range(max_goals + 1) if 0 <= total - h <= max_goals)
        total_probs[total] = p_sum

    btts_yes = sum(matrix[h, a] for h in range(1, max_goals + 1) for a in range(1, max_goals + 1))
    
    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over25": sum(p for g, p in total_probs.items() if g > 2.5),
        "btts_yes": btts_yes,
        "total_probs": total_probs,
    }


def predict_match(home_data, away_data, h2h_adj, elo_diff, match_id):
    hl, al = calculate_lambdas(home_data, away_data, h2h_adj, elo_diff)
    matrix = score_matrix(hl, al, max_goals=6)
    
    flat = matrix.flatten()
    flat_sum = flat.sum()
    if flat_sum > 0:
        flat /= flat_sum
    else:
        flat = np.ones_like(flat) / len(flat)

    seed_val = abs(hash(str(match_id))) % (2**32)
    rng = np.random.default_rng(seed_val)
    chosen = rng.choice(len(flat), p=flat)
    sampled_score = np.unravel_index(chosen, matrix.shape)
    best_score = np.unravel_index(np.argmax(matrix), matrix.shape)
    probs = calculate_probabilities(matrix)

    return {
        "home_lambda": hl,
        "away_lambda": al,
        "best_score": sampled_score,
        "argmax_score": best_score,
        "best_probability": matrix[sampled_score],
        "probabilities": probs,
    }


# =========================================================
# UI RENDERING & ROUTING
# =========================================================

st.title("⚽ Professional Football Predictor V2.5")
st.markdown("အဆင့်မြင့် **Dynamic Elo, Dixon-Coles, H2H, Rest Days & xG** မော်ဒယ်များ ပေါင်းစပ်ထားသော စနစ်")

if nav_mode == "🏟️ Selected League View":
    fixtures = fetch_fixtures(league_code, fb_key)
    elos = calculate_dynamic_elos(league_code, fb_key)

    st.subheader(f"📅 {selected_league} | Next 7 Days")

    if not fixtures:
        st.warning("ပွဲစဉ်မရှိပါ သို့မဟုတ် API Key စစ်ဆေးပါ။")
    else:
        st.success(f"{len(fixtures)} matches found")

    for match in fixtures:
        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]
        match_id = match["id"]
        
        utc_dt = datetime.strptime(match["utcDate"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        local_dt = utc_dt.astimezone(LOCAL_TZ)
        match_date_str = local_dt.strftime("%Y-%m-%d %H:%M MMT")

        with st.expander(f"🏟️ {home} vs {away} | 🕒 {match_date_str}"):
            left, right = st.columns([1, 1.3])

            with left:
                st.markdown("### 📊 Team Dynamic Profile")
                h_elo = elos.get(home, 1500.0)
                a_elo = elos.get(away, 1500.0)
                st.write(f"🏠 **{home} Elo:** {h_elo:.1f}")
                st.write(f"✈️ **{away} Elo:** {a_elo:.1f}")

                hd_adv = get_team_advanced_metrics(home, league_code, fb_key, utc_dt)
                ad_adv = get_team_advanced_metrics(away, league_code, fb_key, utc_dt)

                st.write(f"⏳ Rest Days: {home} ({hd_adv['rest_days']} days) vs {away} ({ad_adv['rest_days']} days)")
                st.write(f"🎯 Recent xG: {home} ({hd_adv['xg_for']:.2f}) vs {away} ({ad_adv['xg_for']:.2f})")

            with right:
                st.markdown("### 🤖 Advanced AI Match Prediction")
                h2h_adj = get_h2h_adjustment(home, away, league_code, fb_key)
                elo_diff = h_elo - a_elo
                
                prediction = predict_match(hd_adv, ad_adv, h2h_adj, elo_diff, match_id)

                hl = prediction["home_lambda"]
                al = prediction["away_lambda"]
                score = prediction["best_score"]
                probs = prediction["probabilities"]

                st.markdown(f"**Expected Goals (λ):** {hl:.2f} - {al:.2f}")
                st.success(f"🎯 **Predicted Score: {home} {score[0]} - {score[1]} {away}**")
                st.write(f"📈 Home Win: {probs['home_win']*100:.1f}% | Draw: {probs['draw']*100:.1f}% | Away Win: {probs['away_win']*100:.1f}%")
                st.write(f"⚽ Over 2.5 Goals Probability: {probs['over25']*100:.1f}%")

                save_prediction(match_id, league_code, match_date_str, home, away, score, hl, al)

elif nav_mode == "🔥 Top 10 Best Matches of the Day":
    st.subheader("🔥 Top 10 Most Probable Predicted Matches Across All Leagues")
    if not fb_key:
        st.warning("Football-data.org API Key ထည့်သွင်းပေးပါ။")
    else:
        all_candidates = []
        with st.spinner("လိဂ်အားလုံးမှ အဆင့်မြင့် မော်ဒယ်လ်များဖြင့် တွက်ထုတ်နေပါသည်..."):
            for l_name, l_code in LEAGUES.items():
                fxts = fetch_fixtures(l_code, fb_key)
                l_elos = calculate_dynamic_elos(l_code, fb_key)

                for m in fxts:
                    h_team = m["homeTeam"]["name"]
                    a_team = m["awayTeam"]["name"]
                    m_id = m["id"]
                    utc_dt = datetime.strptime(m["utcDate"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    local_dt = utc_dt.astimezone(LOCAL_TZ)

                    h_adv = get_team_advanced_metrics(h_team, l_code, fb_key, utc_dt)
                    a_adv = get_team_advanced_metrics(a_team, l_code, fb_key, utc_dt)
                    h2h_adj = get_h2h_adjustment(h_team, a_team, l_code, fb_key)
                    
                    h_e = l_elos.get(h_team, 1500.0)
                    a_e = l_elos.get(a_team, 1500.0)
                    
                    pred = predict_match(h_adv, a_adv, h2h_adj, h_e - a_e, m_id)
                    probs = pred["probabilities"]
                    best_sc = pred["best_score"]
                    
                    max_prob = max(probs["home_win"], probs["draw"], probs["away_win"])
                    
                    all_candidates.append({
                        "League": l_name.split("(")[0].strip(),
                        "Date": local_dt.strftime("%Y-%m-%d %H:%M"),
                        "Match": f"{h_team} vs {a_team}",
                        "Predicted Score": f"{h_team} {best_sc[0]} - {best_sc[1]} {a_team}",
                        "Confidence (%)": round(max_prob * 100, 1),
                    })

        if not all_candidates:
            st.info("ပွဲစဉ်မရှိသေးပါ။")
        else:
            df_top = pd.DataFrame(all_candidates).sort_values(by="Confidence (%)", ascending=False).head(10)
            st.dataframe(df_top, hide_index=True, use_container_width=True)

elif nav_mode == "📚 Match History & Accuracy":
    st.header("📚 Match History & Prediction Accuracy")
    history = load_history()
    if history.empty:
        st.info("History မရှိသေးပါ။")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)
