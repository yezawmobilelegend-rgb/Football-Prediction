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
    page_title="Professional Football Predictor V2.6",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "football_history_v26.db"
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
# DYNAMIC ELO RATING SYSTEM
# =========================================================

@st.cache_data(ttl=3600)
def calculate_dynamic_elos(league, api_key):
    matches = fetch_finished_matches(league, api_key)
    matches = sorted(matches, key=lambda x: x["utcDate"])
    
    elos = {}
    k_factor = 25
    home_advantage = 65

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
# ADVANCED METRICS (Rest Days & xG)
# =========================================================

def get_team_advanced_metrics(team, league, api_key, current_match_date):
    matches = fetch_finished_matches(league, api_key)
    team_matches = [
        m for m in matches 
        if m["homeTeam"]["name"] == team or m["awayTeam"]["name"] == team
    ]
    team_matches = sorted(team_matches, key=lambda x: x["utcDate"], reverse=True)

    rest_days = 7
    recent_xg_for = 1.3
    recent_xg_against = 1.1

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

        recent_xg_for += gf * 0.85
        recent_xg_against += ga * 0.85

    if len(last_5) > 0:
        recent_xg_for /= len(last_5)
        recent_xg_against /= len(last_5)

    return {
        "rest_days": rest_days,
        "xg_for": max(0.5, recent_xg_for),
        "xg_against": max(0.5, recent_xg_against),
    }


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
# POISSON MODEL & DIXON-COLES EXTENSION (ARGMAX SCORE)
# =========================================================

def poisson_pmf(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def dixon_coles_tau(h, a, lam_h, lam_a, rho=-0.12):
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
        "over15": sum(p for g, p in total_probs.items() if g > 1.5),
        "over25": sum(p for g, p in total_probs.items() if g > 2.5),
        "over35": sum(p for g, p in total_probs.items() if g > 3.5),
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

    # ARGMAX Score (Highest probability score directly)
    best_score = np.unravel_index(np.argmax(matrix), matrix.shape)
    probs = calculate_probabilities(matrix)

    return {
        "home_lambda": hl,
        "away_lambda": al,
        "best_score": best_score,
        "best_probability": matrix[best_score],
        "probabilities": probs,
    }


# =========================================================
# LOCAL ODDS PRICE PARSER & VALUE FINDER HELPER
# =========================================================

def parse_myanmar_price(price_str):
    """ Parses Myanmar local price string like '+20', '-80', 'ဆုံ', or decimal odds like '1.95' """
    price_str = str(price_str).strip().lower()
    if not price_str:
        return None
    try:
        if price_str in ["ဆုံ", "even", "1.0"]:
            return 2.0  # Decimal equivalent for even money
        
        if price_str.startswith("+"):
            val = float(price_str[1:])
            # e.g., +20 means betting 100 makes 120 return or profit ratio
            return 1.0 + (val / 100.0) + 1.0  # Decimal odds approx (e.g. 2.20)
        elif price_str.startswith("-"):
            val = float(price_str[1:])
            # e.g., -80 means risk 80 to win 100
            return 1.0 + (100.0 / val)
        else:
            return float(price_str)
    except ValueError:
        return None


# =========================================================
# UI RENDERING & ROUTING
# =========================================================

st.title("⚽ Professional Football Predictor V2.6")
st.markdown("အဆင့်မြင့် **Argmax Score, Dynamic Elo, Dixon-Coles & Local Myanmar Odds Value Finder** စနစ်")

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
                st.markdown("### 🇲🇲 ဒေသတွင်းပေါက်ကြေး (Local Odds Manual Input)")
                st.caption("မြန်မာပြည်ပေါက်ကြေးပုံစံ (ဥပမာ Handicap 1+20 / O/U 3+40) ဖြင့် ဖြည့်သွင်းပါ")
                
                local_hdp_team = st.selectbox("Handicap ပေးသည့်အသင်း", [home, away, "မရှိပါ"], key=f"hdp_team_{match_id}")
                local_hdp_line = st.text_input("Handicap နှုန်း (ဥပမာ 0.5, 1, 1.5)", key=f"hdp_line_{match_id}")
                local_hdp_price = st.text_input("Handicap ဈေး (ဥပမာ +20, -80, ဆုံ)", key=f"hdp_price_{match_id}")
                
                st.divider()
                local_ou_line = st.text_input("O/U လိုင်း (ဥပမာ 2.5, 3)", key=f"ou_line_{match_id}")
                local_ou_price = st.text_input("O/U ဈေး (ဥပမာ +40, -90, ဆုံ)", key=f"ou_price_{match_id}")

            with right:
                st.markdown("### 🤖 Advanced AI Match Prediction")
                h_elo = elos.get(home, 1500.0)
                a_elo = elos.get(away, 1500.0)
                hd_adv = get_team_advanced_metrics(home, league_code, fb_key, utc_dt)
                ad_adv = get_team_advanced_metrics(away, league_code, fb_key, utc_dt)
                h2h_adj = get_h2h_adjustment(home, away, league_code, fb_key)
                
                prediction = predict_match(hd_adv, ad_adv, h2h_adj, h_elo - a_elo, match_id)

                hl = prediction["home_lambda"]
                al = prediction["away_lambda"]
                score = prediction["best_score"]
                probs = prediction["probabilities"]

                st.markdown(f"**Expected Goals (λ):** {hl:.2f} - {al:.2f}")
                st.success(f"🎯 **Predicted Score (Argmax): {home} {score[0]} - {score[1]} {away}**")
                st.markdown(f"📈 **AI Probabilities:** Home Win: `{probs['home_win']*100:.1f}%` | Draw: `{probs['draw']*100:.1f}%` | Away Win: `{probs['away_win']*100:.1f}%`")
                st.markdown(f"⚽ **O/U Probabilities:** Over 2.5: `{probs['over25']*100:.1f}%` | Over 1.5: `{probs['over15']*100:.1f}%`")

                # =========================================================
                # LOCAL VALUE BET FINDER SYSTEM
                # =========================================================
                st.divider()
                st.markdown("### 💎 Local Value Bet Finder")
                
                value_found = False
                
                # Evaluate Handicap Value
                if local_hdp_team != "မရှိပါ" and local_hdp_line and local_hdp_price:
                    dec_price = parse_myanmar_price(local_hdp_price)
                    if dec_price and dec_price > 1.0:
                        implied_prob = 1.0 / dec_price
                        ai_prob = probs['home_win'] if local_hdp_team == home else probs['away_win']
                        edge = (ai_prob - implied_prob) * 100
                        if (ai_prob * dec_price) > 1.03:
                            value_found = True
                            st.success(f"🔥 **Value Bet (Handicap - {local_hdp_team} {local_hdp_line} @ {local_hdp_price}):** Edge +{edge:.1f}% (AI Prob: {ai_prob*100:.1f}% vs Implied: {implied_prob*100:.1f}%)")

                # Evaluate Over/Under Value
                if local_ou_line and local_ou_price:
                    dec_ou_price = parse_myanmar_price(local_ou_price)
                    if dec_ou_price and dec_ou_price > 1.0:
                        implied_ou_prob = 1.0 / dec_ou_price
                        try:
                            ou_val = float(local_ou_line)
                            if ou_val == 2.5:
                                ai_ou_prob = probs['over25']
                            elif ou_val == 1.5:
                                ai_ou_prob = probs['over15']
                            else:
                                ai_ou_prob = probs['over25'] # default fallback
                            
                            ou_edge = (ai_ou_prob - implied_ou_prob) * 100
                            if (ai_ou_prob * dec_ou_price) > 1.03:
                                value_found = True
                                st.success(f"🔥 **Value Bet (Over {ou_val} @ {local_ou_price}):** Edge +{ou_edge:.1f}% (AI Prob: {ai_ou_prob*100:.1f}% vs Implied: {implied_ou_prob*100:.1f}%)")
                        except ValueError:
                            pass

                if not value_found:
                    st.info("ℹ️ ထည့်သွင်းထားသော ဒေသတွင်းပေါက်ကြေးအပေါ် အခြေခံ၍ သိသာထင်ရှားသော Value Bet မတွေ့ရသေးပါ။")

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
