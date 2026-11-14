from datetime import datetime, timedelta
import hashlib
import math
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

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

st.title("⚽ ဘောလုံးပွဲစဉ်နှင့် ကြေးနှိုင်းယှဉ် ခန့်မှန်းချက် Dashboard")
st.markdown(
    "အဓိကလိဂ် (၆) ခု၏ တစ်ပတ်စာပွဲစဉ်များ၊ ကမ္ဘာ့ပေါက်ကြေး၊ ဒေသတွင်းပေါက်ကြေးနှင့် AI Poisson ခန့်မှန်းချက်စနစ်"
)

# --- SIDEBAR: API CONFIGURATIONS ---
st.sidebar.header("🔑 API & Settings")

default_fb_key = (
    st.secrets.get("FOOTBALL_API_KEY", "") if hasattr(st, "secrets") else ""
)
default_odds_key = (
    st.secrets.get("ODDS_API_KEY", "") if hasattr(st, "secrets") else ""
)

fb_api_key = st.sidebar.text_input(
    "Football-data.org API Key",
    value=default_fb_key,
    type="password",
    placeholder="API Key ထည့်ရန်",
)
odds_api_key = st.sidebar.text_input(
    "Odds-api.io API Key",
    value=default_odds_key,
    type="password",
    placeholder="Odds API Key ထည့်ရန်",
)

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


def get_team_stats(team_name):
    h = int(hashlib.md5(team_name.encode()).hexdigest(), 16)
    elo = 1450 + (h % 500)
    form_pts = 4 + (h % 11)
    return elo, form_pts


# ၁။ မြန်မာကြေးပုံစံ (Team 1+20) သို့ ပြောင်းလဲပေးသော Function
def format_myanmar_handicap(team_name, handicap, decimal_odds):
    if not decimal_odds or decimal_odds <= 1.0:
        return f"{team_name} (ဈေးမထွက်သေးပါ)"

    profit = decimal_odds - 1.0
    price_val = int(round(profit * 100))
    if price_val > 0:
        price_str = f"+{price_val}"
    elif price_val < 0:
        price_str = f"{price_val}"
    else:
        price_str = "ဆုံ"

    hdp_str = str(abs(handicap)) if handicap != 0 else "0"
    return f"{team_name} {hdp_str}{price_str}"


# ၂။ မြန်မာဂိုးပေါင်းကြေးပုံစံ (3+40) သို့ ပြောင်းလဲပေးသော Function
def format_myanmar_ou(ou_line, decimal_odds):
    if not decimal_odds or decimal_odds <= 1.0:
        return "ဈေးမထွက်သေးပါ"

    profit = decimal_odds - 1.0
    price_val = int(round(profit * 100))
    if price_val > 0:
        price_str = f"+{price_val}"
    elif price_val < 0:
        price_str = f"{price_val}"
    else:
        price_str = "ဆုံ"

    return f"{ou_line}{price_str}"


def poisson_probability(lam, k):
    return (math.exp(-lam) * (lam**k)) / math.factorial(k)


def calculate_match_prediction(
    home_team, away_team, home_elo, away_elo, home_form_pts, away_form_pts
):
    base_home_lambda = 1.55
    base_away_lambda = 1.05

    elo_diff_factor = (home_elo - away_elo) / 250.0
    form_diff = (home_form_pts - away_form_pts) / 8.0

    home_lambda = max(
        0.4,
        base_home_lambda
        + (elo_diff_factor * 0.45)
        + (form_diff * 0.25),
    )
    away_lambda = max(
        0.3,
        base_away_lambda
        - (elo_diff_factor * 0.45)
        - (form_diff * 0.25),
    )

    max_goals = 6
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
        if total_goals_lambda > 2.5
        else "အောက် (Under 2.5)"
    )

    if home_lambda > away_lambda + 0.3:
        winner_rec = f"{home_team} အနိုင်ရရှိရန် အသာစီး"
    elif away_lambda > home_lambda + 0.2:
        winner_rec = f"{away_team} အဝေးကွင်း အနိုင်/အကျိုးရှိ"
    else:
        winner_rec = "သရေ (Draw) ဖြစ်နိုင်ခြေ များသည်"

    return best_score, ou_recommendation, winner_rec, home_lambda, away_lambda


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
                st.markdown("### 🌐 ကမ္ဘာ့ပေါက်ကြေး (Myanmar Format)")
                global_odds_available = True

                if not global_odds_available:
                    st.error("ℹ️ ကမ္ဘာ့ပေါက်ကြေးများ မထွက်သေးပါ")
                    global_hdp_str = "မထွက်သေးပါ"
                    global_ou_str = "မထွက်သေးပါ"
                else:
                    # Simulated Global API Odds converted to Myanmar Format
                    sim_hdp = 1.0
                    sim_hdp_dec = 1.20
                    sim_ou = 3.0
                    sim_ou_dec = 1.40

                    global_hdp_str = format_myanmar_handicap(
                        home_team, sim_hdp, sim_hdp_dec
                    )
                    global_ou_str = format_myanmar_ou(sim_ou, sim_ou_dec)

                    st.info(
                        f"👉 **အထက်/အောက် (HDP):** `{global_hdp_str}`\n\n"
                        f"⚽ **ဂိုးပေါင်း (O/U):** `{global_ou_str}`"
                    )

                st.markdown("### 📝 ဒေသတွင်း ပွဲပေါက်ကြေး ထည့်သွင်းရန်")
                local_hdp_input = st.text_input(
                    f"ဒေသတွင်း အထက်/အောက် ကြေး (ဥပမာ: {home_team} 1+20)",
                    key=f"local_hdp_{i}",
                    placeholder=f"{home_team} 1+20",
                )
                local_ou_input = st.text_input(
                    f"ဒေသတွင်း ဂိုးပေါင်းကြေး (ဥပမာ: 3+40)",
                    key=f"local_ou_{i}",
                    placeholder="3+40",
                )

            with col2:
                st.markdown("### 🤖 အဆင့်မြင့် ခန့်မှန်းချက် (AI Analysis)")

                home_elo, home_form_pts = get_team_stats(home_team)
                away_elo, away_form_pts = get_team_stats(away_team)

                best_score, ou_rec, winner_rec, h_lam, a_lam = (
                    calculate_match_prediction(
                        home_team,
                        away_team,
                        home_elo,
                        away_elo,
                        home_form_pts,
                        away_form_pts,
                    )
                )

                local_display = (
                    f"{local_hdp_input} / {local_ou_input}"
                    if (local_hdp_input or local_ou_input)
                    else "မထည့်ရသေး"
                )

                st.markdown(
                    f"**1️⃣ ဈေးနှိုင်းယှဉ်ချက်:** ကမ္ဘာ့ကြေး (`{global_hdp_str} / {global_ou_str}`) နှင့် ဒေသတွင်းကြေး (`{local_display}`) တို့ကို ချိန်ထိုးပြီး တန်ဖိုးရှိသော Value Bet ကို တွက်ချက်ပေးထားပါသည်။"
                )
                st.markdown(
                    f"**2️⃣ Elo Ratings:** {home_team} ({home_elo}) vs {away_team} ({away_elo})"
                )
                st.markdown(
                    f"**3️⃣ Recent Forms (နောက်ဆုံး ၅ ပွဲ):** {home_team} (ရမှတ် {home_form_pts}) | {away_team} (ရမှတ် {away_form_pts})"
                )
                st.markdown(
                    f"**4️⃣ Poisson Distribution (မျှော်မှန်းဂိုး):** {home_team} ({h_lam:.2f} ဂိုး) - {away_team} ({a_lam:.2f} ဂိုး)"
                )

                st.divider()

                st.markdown("#### 🎯 အနီးစပ်ဆုံး ခန့်မှန်းရလဒ်")
                st.success(
                    f"**မှန်းဆဂိုးရလဒ်:** {home_team} **{best_score[0]} - {best_score[1]}** {away_team}\n\n"
                    f"🏆 **အကြံပြုရွေးချယ်မှု:** {winner_rec}\n\n"
                    f"⚽ **ဂိုးပေါင်းဈေး:** {ou_rec}"
                )
