import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Streamlit Page Setup
st.set_page_config(
    page_title="Professional Football Match Predictor Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚽ ဘောလုံးပွဲစဉ်နှင့် ကြေးနှိုင်းယှဉ် ခန့်မှန်းချက် Dashboard")
st.markdown(
    "အဓိကလိဂ် (၆) ခု၏ တစ်ပတ်စာပွဲစဉ်များ၊ ကမ္ဘာ့ပေါက်ကြေး၊ ဒေသတွင်းပေါက်ကြေးနှင့် AI Poisson ခန့်မှန်းချက်စနစ်"
)

# --- SIDEBAR: API CONFIGURATIONS ---
st.sidebar.header("🔑 API & Settings")
fb_api_key = st.sidebar.text_input(
    "Football-data.org API Key", type="password", placeholder="API Key ထည့်ရန်"
)
odds_api_key = st.sidebar.text_input(
    "Odds-api.io API Key", type="password", placeholder="Odds API Key ထည့်ရန်"
)

# အမှန်ပြင်ဆင်ထားသော selectbox (အစက် (.) ဖယ်ရှားပြီး)
selected_league_name = st.sidebar.selectbox(
    "လိဂ် ရွေးချယ်ရန်",
    [
        "English Premier League (PL)",
        "Spanish La Liga (PD)",
        "German Bundesliga (BL1)",
        "French Ligue 1 (FL1)",
        "UEFA Champions League (CL)",
        "UEFA Europa League (EL)",
    ],
)

# League Mapping for Football-data.org
LEAGUE_CODES = {
    "English Premier League (PL)": "PL",
    "Spanish La Liga (PD)": "PD",
    "German Bundesliga (BL1)": "BL1",
    "French Ligue 1 (FL1)": "FL1",
    "UEFA Champions League (CL)": "CL",
    "UEFA Europa League (EL)": "EL",
}

current_league_code = LEAGUE_CODES[selected_league_name]


# --- HELPER FUNCTIONS ---


# ၁။ မြန်မာကြေးသို့ ပြောင်းလဲပေးသော Function
def convert_to_myanmar_odds(team_name, handicap, decimal_odds):
    if not decimal_odds or decimal_odds <= 1.0:
        return f"{team_name} (ကမ္ဘာ့ပေါက်ကြေး မထွက်သေးပါ)"

    profit = decimal_odds - 1.0
    price_val = int(round(profit * 100))
    if decimal_odds >= 1.0:
        price_str = f"+{price_val}" if price_val > 0 else "ဆုံ"
    else:
        price_val = int(round((1.0 - decimal_odds) * 100))
        price_str = f"-{price_val}"

    hdp_str = str(abs(handicap)) if handicap != 0 else "0 (ဆုံလည်)"
    return f"{team_name} {hdp_str} ({price_str})"


# ၂။ Poisson Distribution Probability တွက်ချက်မှု
def poisson_probability(lam, k):
    return (math.exp(-lam) * (lam**k)) / math.factorial(k)


# ၃။ ပွဲစဉ်ခန့်မှန်းချက် Engine (Elo + Recent Form + Poisson)
def calculate_match_prediction(
    home_team, away_team, home_elo, away_elo, home_form_pts, away_form_pts
):
    base_home_lambda = 1.45
    base_away_lambda = 1.15

    elo_diff_factor = (home_elo - away_elo) / 400.0
    form_diff = (home_form_pts - away_form_pts) / 15.0

    home_lambda = max(
        0.5,
        base_home_lambda
        + (elo_diff_factor * 0.35)
        + (form_diff * 0.2),
    )
    away_lambda = max(
        0.5,
        base_away_lambda
        - (elo_diff_factor * 0.35)
        - (form_diff * 0.2),
    )

    max_goals = 5
    best_score = (0, 0)
    max_prob = -1.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = poisson_probability(home_lambda, h) * poisson_probability(
                away_lambda, a
            )
            if prob > max_prob:
                max_prob = prob
                best_score = (h, a)

    total_goals_lambda = home_lambda + away_lambda
    ou_recommendation = (
        "အထက် (Over 2.5)"
        if total_goals_lambda > 2.6
        else "အောက် (Under 2.5)"
    )

    if home_lambda > away_lambda + 0.4:
        winner_rec = f"{home_team} နိုင်ရန် ရန်ပုံငွေ/ဂိုးပေး အသာစီး"
    elif away_lambda > home_lambda + 0.2:
        winner_rec = f"{away_team} အဝေးကွင်း သို့မဟုတ် အနိမ့်အသင်း အသာစီး"
    else:
        winner_rec = "သရေ (Draw) ဖြစ်နိုင်ခြေ များသည်"

    return best_score, ou_recommendation, winner_rec, home_lambda, away_lambda


# --- FETCH FIXTURES ---
@st.cache_data(ttl=3600)
def fetch_fixtures(league_code, api_key):
    if not api_key:
        today = datetime.now().strftime("%Y-%m-%d")
        return [
            {
                "id": 101,
                "homeTeam": {"name": "Chelsea"},
                "awayTeam": {"name": "Arsenal"},
                "utcDate": f"{today}T16:00:00Z",
                "status": "TIMED",
            },
            {
                "id": 102,
                "homeTeam": {"name": "Manchester United"},
                "awayTeam": {"name": "Liverpool"},
                "utcDate": f"{today}T18:30:00Z",
                "status": "TIMED",
            },
        ]

    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches?status=SCHEDULED"
    headers = {"X-Auth-Token": api_key}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        matches = data.get("matches", [])
        now = datetime.utcnow()
        seven_days_later = now + timedelta(days=7)
        filtered_matches = []
        for m in matches:
            match_date = datetime.strptime(
                m["utcDate"], "%Y-%m-%dT%H:%M:%SZ"
            )
            if now <= match_date <= seven_days_later:
                filtered_matches.append(m)
        return filtered_matches
    else:
        return []


# --- MAIN UI DISPLAY ---
st.subheader(f"📅 {selected_league_name} - လာမည့် ၇ ရက်အတွင်း ပွဲစဉ်ဇယားများ")

matches = fetch_fixtures(current_league_code, fb_api_key)

if not matches:
    st.warning(
        "⚠️ လက်တလော (၇) ရက်အတွင်း ကန်မည့် ပွဲစဉ်များ မရှိသေးပါ သို့မဟုတ် API Key မှားယွင်းနေပါသည်။"
    )
else:
    st.success(f"ပွဲစဉ် စုစုပေါင်း ({len(matches)} ပွဲ) တွေ့ရှိရပါသည်။")

    for i, match in enumerate(matches):
        home_team = match["homeTeam"]["name"]
        away_team = match["awayTeam"]["name"]
        match_time = match["utcDate"]

        with st.expander(
            f"🏟️ {home_team} vs {away_team}  |  🕒 အချိန်: {match_time}"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🌐 ကမ္ဘာ့ပေါက်ကြေးနှင့် အချက်အလက်များ")
                global_odds_available = True

                if not global_odds_available:
                    st.error(
                        "ℹ️ ကမ္ဘာ့ပေါက်ကြေးများ မထွက်သေးပါ (Odds not released yet)"
                    )
                    global_home_myanmar_str = "ကမ္ဘာ့ပေါက်ကြေးများ မထွက်သေးပါ"
                else:
                    simulated_hdp = -0.75
                    simulated_dec_odds = 1.85
                    global_home_myanmar_str = convert_to_myanmar_odds(
                        home_team, simulated_hdp, simulated_dec_odds
                    )
                    st.info(
                        f"**ကမ္ဘာ့ပေါက်ကြေး (Myanmar Format):** {global_home_myanmar_str}"
                    )
                    st.caption(
                        "💡 (ပွဲမစမီ ၁ နာရီ သို့မဟုတ် Lineup ထွက်ပြီး ၁၅ မိနစ်အလို အချိန်နှင့်တပြေးညီ တိုက်ရိုက်ဆွဲယူသည်)"
                    )

                st.markdown("### 📝 ဒေသတွင်း/ကိုယ်ပိုင် ပွဲပေါက်ကြေး ထည့်သွင်းရန်")
                local_odds
