from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
from scipy.poisson import pmf
import streamlit as str_ui

# Page Configuration
str_ui.set_page_config(
    page_title="Real-Time Football Match Predictor (6 Leagues)",
    page_icon="⚽",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 1. API CONFIGURATION & LEAGUES SETUP
# ---------------------------------------------------------------------------
LEAGUES_CONFIG = {
    "Premier League": "PL",
    "La Liga": "PD",
    "Serie A": "SA",
    "France Ligue 1": "FL1",
    "Champions League": "CL",
    "Europa League": "EL",
}

# Sample Elo ratings fallback dictionary
TEAM_ELOS = {
    "Manchester City": 2050,
    "Arsenal": 2020,
    "Liverpool": 2010,
    "Aston Villa": 1850,
    "Chelsea": 1840,
    "Tottenham": 1820,
    "Manchester United": 1800,
    "Real Madrid": 2060,
    "Barcelona": 2030,
    "Atletico Madrid": 1920,
    "Inter Milan": 1990,
    "Juventus": 1910,
    "AC Milan": 1900,
    "Napoli": 1880,
    "Paris Saint-Germain": 1980,
    "Monaco": 1830,
    "Bayern Munich": 2040,
    "Bayer Leverkusen": 1960,
    "Porto": 1820,
    "Benfica": 1830,
}

# Sidebar for API Key
str_ui.sidebar.header("⚙️ API Configuration")
api_key = str_ui.sidebar.text_input(
    "Football-Data.org API Key",
    type="password",
    help="https://www.football-data.org/ တွင် အခမဲ့ API Key ရယူနိုင်ပါသည်။",
)


# ---------------------------------------------------------------------------
# 2. FETCH REAL-TIME DATA FROM FOOTBALL-DATA.ORG API
# ---------------------------------------------------------------------------
@str_ui.cache_data(ttl=3600)  # Cache data for 1 hour to prevent rate limiting
def fetch_real_fixtures(api_token):
    if not api_token:
        return pd.DataFrame()  # Return empty if no token, will use fallback

    today = datetime.now()
    date_from = today.strftime("%Y-%m-%d")
    date_to = (today + timedelta(days=7)).strftime("%Y-%m-%d")

    url = f"https://api.football-data.org/v4/matches?dateFrom={date_from}&dateTo={date_to}"
    headers = {"X-Auth-Token": api_token}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            matches = data.get("matches", [])
            formatted_matches = []

            target_competitions = list(LEAGUES_CONFIG.values())

            for m in matches:
                comp_code = m["competition"]["code"]
                if comp_code in target_competitions:
                    # Reverse map competition code to league name
                    league_name = [
                        k
                        for k, v in LEAGUES_CONFIG.items()
                        if v == comp_code
                    ][0]

                    formatted_matches.append(
                        {
                            "league": league_name,
                            "date": m["utcDate"].replace("T", " ")[:16],
                            "home": m["homeTeam"]["name"],
                            "away": m["awayTeam"]["name"],
                            "home_form": "W D W L W",  # Default form if API lacks recent form string
                            "away_form": "W W D W L",
                        }
                    )
            return pd.DataFrame(formatted_matches)
        else:
            str_ui.sidebar.error(
                f"API Error: {response.status_code} - ဆက်သွယ်မှု အခက်အခဲရှိနေပါသည်။"
            )
            return pd.DataFrame()
    except Exception as e:
        str_ui.sidebar.error(f"Connection Exception: {e}")
        return pd.DataFrame()


# Fallback Mock Fixtures if API Key is not provided
def get_fallback_fixtures():
    today = datetime.now()
    return pd.DataFrame(
        [
            {
                "league": "Premier League",
                "date": (today + timedelta(days=1)).strftime("%Y-%m-%d 20:00"),
                "home": "Chelsea",
                "away": "Aston Villa",
                "home_form": "W D W L W",
                "away_form": "W W D W L",
            },
            {
                "league": "Premier League",
                "date": (today + timedelta(days=2)).strftime("%Y-%m-%d 17:30"),
                "home": "Manchester City",
                "away": "Liverpool",
                "home_form": "W W W D W",
                "away_form": "W W W W D",
            },
            {
                "league": "La Liga",
                "date": (today + timedelta(days=1)).strftime("%Y-%m-%d 21:00"),
                "home": "Real Madrid",
                "away": "Villarreal",
                "home_form": "W W D W W",
                "away_form": "L W D W L",
            },
            {
                "league": "Serie A",
                "date": (today + timedelta(days=3)).strftime("%Y-%m-%d 22:45"),
                "home": "Inter Milan",
                "away": "Juventus",
                "home_form": "W W W W D",
                "away_form": "W D W W D",
            },
            {
                "league": "Champions League",
                "date": (today + timedelta(days=4)).strftime("%Y-%m-%d 01:00"),
                "home": "Bayern Munich",
                "away": "Bayer Leverkusen",
                "home_form": "W W W W W",
                "away_form": "W W D W W",
            },
        ]
    )


# ---------------------------------------------------------------------------
# 3. STATISTICAL MODEL & PREDICTION ENGINE (Elo + Poisson)
# ---------------------------------------------------------------------------
def calculate_poisson_prediction(home_team, away_team, home_form, away_form):
    home_elo = TEAM_ELOS.get(home_team, 1750)
    away_elo = TEAM_ELOS.get(away_team, 1750)

    elo_diff = (home_elo - away_elo) + 100  # +100 for home advantage

    def form_score(form_str):
        if not isinstance(form_str, str):
            return 1.5
        matches = form_str.split()
        points = sum(
            3 if m == "W" else (1 if m == "D" else 0) for m in matches
        )
        return points / (len(matches) * 3) * 3.0 if matches else 1.5

    h_form_val = form_score(home_form)
    a_form_val = form_score(away_form)

    base_home_xg = 1.45 + (elo_diff / 400.0) * 0.4 + (h_form_val - 1.5) * 0.2
    base_away_xg = 1.15 - (elo_diff / 400.0) * 0.4 + (a_form_val - 1.5) * 0.2

    home_xg = max(0.5, round(base_home_xg, 2))
    away_xg = max(0.5, round(base_away_xg, 2))

    max_goals = 5
    score_matrix = np.outer(
        pmf(np.arange(max_goals), home_xg), pmf(np.arange(max_goals), away_xg)
    )
    h_score, a_score = np.unravel_index(
        np.argmax(score_matrix), score_matrix.shape
    )

    home_win_prob = np.sum(np.tril(score_matrix, -1)) * 100
    draw_prob = np.sum(np.diag(score_matrix)) * 100
    away_win_prob = np.sum(np.triu(score_matrix, 1)) * 100

    return {
        "home_xg": home_xg,
        "away_xg": away_xg,
        "predicted_score": f"{h_score} - {a_score}",
        "home_win_prob": round(home_win_prob, 1),
        "draw_prob": round(draw_prob, 1),
        "away_win_prob": round(away_win_prob, 1),
        "home_elo": home_elo,
        "away_elo": away_elo,
    }


# ---------------------------------------------------------------------------
# 4. STREAMLIT UI DISPLAY
# ---------------------------------------------------------------------------
str_ui.title("⚽ Real-Time 6 Leagues Match Predictor")
str_ui.markdown(
    "Football-Data API ကို ချိတ်ဆက်၍ လာမည့် တစ်ပတ်အတွင်းပွဲစဉ်များကို အလိုအလျောက်ဆွဲထုတ်ကာ **Elo, Form & Poisson Model** ဖြင့် ခန့်မှန်းထားခြင်း။"
)

# Requirement 3: Last Updated Timestamp
last_updated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")
str_ui.info(f"🕒 **Last Updated Time:** {last_updated_time}")

# Fetch Data
df_fixtures = pd.DataFrame()
if api_key:
    with str_ui.spinner("Real-time ပွဲစဉ်များကို API မှ ဆွဲထုတ်နေပါသည်..."):
        df_fixtures = fetch_real_fixtures(api_key)

if df_fixtures.empty:
    if api_key:
        str_ui.warning(
            "API မှ ဤရက်ပိုင်းအတွင်း ပွဲစဉ်ရှာမတွေ့ပါ (သို့) ဝင်ရောက်ခွင့် ကန့်သတ်ချက်ရှိနေပါသည်။ နမူနာဒေတာ (Mock Data) သို့ ပြောင်းလဲဖော်ပြပေးထားပါသည်။"
        )
    else:
        str_ui.info(
            "💡 အကြံပြုချက်: Sidebar တွင် Football-Data.org API Key ထည့်ပေးပါက Real-time ပွဲစဉ်များ အလိုအလျောက် ပေါ်လာပါမည်။ လက်တလောတွင် နမူနာဒေတာကို ပြသထားပါသည်။"
        )
    df_fixtures = get_fallback_fixtures()

# League Selection Filter
league_list = ["အားလုံး (All Leagues)"] + list(LEAGUES_CONFIG.keys())
selected_league = str_ui.sidebar.selectbox("လိဂ်ပြိုင်ပွဲ ရွေးချယ်ရန်", league_list)

if selected_league != "အားလုံး (All Leagues)":
    df_fixtures = df_fixtures[df_fixtures["league"] == selected_league]

str_ui.divider()
str_ui.subheader(f"📅 ပွဲစဉ်များနှင့် ခန့်မှန်းချက်များ ({selected_league})")

if df_fixtures.empty:
    str_ui.warning(
        "ရွေးချယ်ထားသော လာမည့်ရက်များအတွင်း ယှဉ်ပြိုင်မည့် ပွဲစဉ် မရှိသေးပါ။"
    )
else:
    for idx, row in df_fixtures.iterrows():
        league = row["league"]
        match_date = row["date"]
        home = row["home"]
        away = row["away"]
        h_form = row["home_form"]
        a_form = row["away_form"]

        pred = calculate_poisson_prediction(home, away, h_form, a_form)

        with str_ui.container():
            str_ui.markdown(f"### 🏆 **{league}** | ⏰ *{match_date} UTC*")

            col1, col2, col3 = str_ui.columns([3, 2, 3])

            with col1:
                str_ui.markdown(f"#### 🏠 {home}")
                str_ui.caption(
                    f"Elo: {pred['home_elo']} | Recent Form: `{h_form}`"
                )

            with col2:
                str_ui.markdown(
                    f"<h3 style='text-align: center; color: #FF4B4B;'>{pred['predicted_score']}</h3>",
                    unsafe_allow_html=True,
                )
                str_ui.caption(
                    f"<p style='text-align: center;'>ခန့်မှန်းဂိုး (xG): H {pred['home_xg']} - A {pred['away_xg']}</p>",
                    unsafe_allow_html=True,
                )

            with col3:
                str_ui.markdown(f"#### ✈️ {away}")
                str_ui.caption(
                    f"Elo: {pred['away_elo']} | Recent Form: `{a_form}`"
                )

            with str_ui.expander("📊 Statistical Model & Match Breakdown"):
                str_ui.write(
                    f"- **အနိုင်ရရှိနိုင်ခြေ (Win Probabilities):** {home} နိုင်ခြေ **{pred['home_win_prob']}%** | သရေ **{pred['draw_prob']}%** | {away} နိုင်ခြေ **{pred['away_win_prob']}%**"
                )
                str_ui.write(
                    f"- **Elo Rating ခွဲခြမ်းစိတ်ဖြာချက်:** {home} (Rating: {pred['home_elo']}) နှင့် {away} (Rating: {pred['away_elo']}) တို့၏ အဆင့်သတ်မှတ်ချက် ကွာဟချက်။"
                )
                str_ui.write(
                    f"- **Poisson Distribution Model:** အိမ်ကွင်းအားသာချက်နှင့် စာရင်းဇယားတွက်ချက်မှုအရ {home} က ဂိုးပေါက်ထိုးဖောက်နိုင်ခြေ {pred['home_xg']} နှင့် {away} က {pred['away_xg']} ဂိုးအထိ သွင်းယူနိုင်ရန် ခန့်မှန်းထားပါသည်။"
                )

            str_ui.markdown("---")
