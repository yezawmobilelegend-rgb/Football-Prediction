from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson  # <-- ပြင်ဆင်ထားသည့်နေရာ (Correct Import)
import streamlit as str_ui

# Page Configuration
str_ui.set_page_config(
    page_title="Real-Time Football Predictor - 6 Major Leagues",
    page_icon="⚽",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 1. SIDEBAR CONFIGURATION & API KEY INPUT
# ---------------------------------------------------------------------------
str_ui.sidebar.title("⚙️ API Settings")
api_key_input = str_ui.sidebar.text_input(
    "football-data.org API Key ထည့်ရန်", type="password"
)
str_ui.sidebar.caption(
    "သင်၏ registered လုပ်ထားသော API Key ကို ဤနေရာတွင် ထည့်သွင်းပါ။"
)

LEAGUES = [
    "Premier League",
    "La Liga",
    "Serie A",
    "France Ligue 1",
    "Champions League",
    "Europa League",
]

# Sample Elo ratings database for prediction model
TEAM_ELOS = {
    # Premier League
    "Manchester City": 2050,
    "Arsenal": 2020,
    "Liverpool": 2010,
    "Aston Villa": 1850,
    "Chelsea": 1840,
    "Tottenham Hotspur FC": 1820,
    "Manchester United FC": 1800,
    "Newcastle United FC": 1790,
    # La Liga
    "Real Madrid CF": 2060,
    "FC Barcelona": 2030,
    "Atlético de Madrid": 1920,
    "Real Sociedad de Fútbol": 1840,
    "Villarreal CF": 1810,
    "Athletic Club": 1820,
    # Serie A
    "Inter Milan": 1990,
    "Juventus FC": 1910,
    "AC Milan": 1900,
    "SSC Napoli": 1880,
    "Atalanta BC": 1870,
    "AS Roma": 1840,
    # Ligue 1
    "Paris Saint-Germain FC": 1980,
    "AS Monaco FC": 1830,
    "Olympique de Marseille": 1810,
    "LOSC Lille": 1800,
    # UCL / UEL
    "FC Bayern Munich": 2040,
    "Borussia Dortmund": 1910,
    "Bayer 04 Leverkusen": 1960,
    "SL Benfica": 1830,
}


# ---------------------------------------------------------------------------
# 2. REAL-TIME API FETCHING FUNCTION
# ---------------------------------------------------------------------------
def fetch_live_fixtures(api_key):
  if not api_key:
    return None, "⚠️ ကျေးဇူးပြု၍ Sidebar တွင် API Key ထည့်သွင်းပေးပါ။"

  today = datetime.now()
  date_from = today.strftime("%Y-%m-%d")
  date_to = (today + timedelta(days=7)).strftime("%Y-%m-%d")

  url = f"https://api.football-data.org/v4/matches?dateFrom={date_from}&dateTo={date_to}"
  headers = {"X-Auth-Token": api_key}

  try:
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code == 200:
      data = response.json()
      matches = data.get("matches", [])
      parsed_fixtures = []

      comp_mapping = {
          "Premier League": "Premier League",
          "Primera Division": "La Liga",
          "Serie A": "Serie A",
          "Ligue 1": "France Ligue 1",
          "UEFA Champions League": "Champions League",
          "UEFA Europa League": "Europa League",
      }

      for m in matches:
        comp_name = m.get("competition", {}).get("name", "")
        if comp_name in comp_mapping:
          league_title = comp_mapping[comp_name]
          match_date = m.get("utcDate", "").replace("T", " ")[:16]
          home_team = m.get("homeTeam", {}).get("name", "Home Team")
          away_team = m.get("awayTeam", {}).get("name", "Away Team")

          parsed_fixtures.append({
              "league": league_title,
              "date": match_date,
              "home": home_team,
              "away": away_team,
              "home_form": "W D W L W",
              "away_form": "W W D W L",
          })

      if not parsed_fixtures:
        return (
            pd.DataFrame(),
            "ℹ️ လာမည့် ၇ ရက်အတွင်း ရွေးချယ်ထားသော လိဂ်ပွဲစဉ်များ မရှိသေးပါ။",
        )
      return pd.DataFrame(parsed_fixtures), None

    elif response.status_code == 429:
      return (
          None,
          "⚠️ API Request Limit ကျော်လွန်သွားပါပြီ (Free Plan: 10 req/min)။"
          " ခေတ္တစောင့်ဆိုင်းပါ။",
      )
    else:
      return (
          None,
          f"⚠️ API Error (Code {response.status_code}): {response.text}",
      )
  except Exception as e:
    return None, f"⚠️ Connection Error: {str(e)}"


# Fallback Mock Data in case API key is missing
def get_fallback_fixtures():
  today = datetime.now()
  return pd.DataFrame([
      {
          "league": "Premier League",
          "date": (today + timedelta(days=1)).strftime("%Y-%m-%d %H:%M"),
          "home": "Chelsea",
          "away": "Aston Villa",
          "home_form": "W D W L W",
          "away_form": "W W D W L",
      },
      {
          "league": "Premier League",
          "date": (today + timedelta(days=2)).strftime("%Y-%m-%d %H:%M"),
          "home": "Manchester City",
          "away": "Liverpool",
          "home_form": "W W W D W",
          "away_form": "W W W W D",
      },
      {
          "league": "La Liga",
          "date": (today + timedelta(days=1)).strftime("%Y-%m-%d %H:%M"),
          "home": "Real Madrid CF",
          "away": "Villarreal CF",
          "home_form": "W W D W W",
          "away_form": "L W D W L",
      },
      {
          "league": "Serie A",
          "date": (today + timedelta(days=3)).strftime("%Y-%m-%d %H:%M"),
          "home": "Inter Milan",
          "away": "Juventus FC",
          "home_form": "W W W W D",
          "away_form": "W D W W D",
      },
  ])


# ---------------------------------------------------------------------------
# 3. STATISTICAL PREDICTION MODEL (Elo + Poisson + Form)
# ---------------------------------------------------------------------------
def calculate_poisson_prediction(home_team, away_team, home_form, away_form):
  home_elo = TEAM_ELOS.get(home_team, 1750)
  away_elo = TEAM_ELOS.get(away_team, 1750)

  elo_diff = (home_elo - away_elo) + 100

  def form_score(form_str):
    if not isinstance(form_str, str):
      return 1.5
    matches = form_str.split()
    points = sum(3 if m == "W" else (1 if m == "D" else 0) for m in matches)
    return points / (len(matches) * 3) * 3.0 if matches else 1.5

  h_form_val = form_score(home_form)
  a_form_val = form_score(away_form)

  base_home_xg = 1.45 + (elo_diff / 400.0) * 0.4 + (h_form_val - 1.5) * 0.2
  base_away_xg = 1.15 - (elo_diff / 400.0) * 0.4 + (a_form_val - 1.5) * 0.2

  home_xg = max(0.5, round(base_home_xg, 2))
  away_xg = max(0.5, round(base_away_xg, 2))

  max_goals = 5
  # Updated usage: poisson.pmf(...) ကို သုံးပေးထားသည်
  score_matrix = np.outer(
      poisson.pmf(np.arange(max_goals), home_xg),
      poisson.pmf(np.arange(max_goals), away_xg),
  )
  h_score, a_score = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)

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
    "football-data.org API ဖြင့် ချိတ်ဆက်ထားပြီး **Elo Ratings**, **Recent"
    " Forms** နှင့် **Poisson Distribution** တို့ကို အခြေခံ၍ အနီးစပ်ဆုံး"
    " ခန့်မှန်းပေးသော Dashboard ဖြစ်ပါသည်။"
)

# Requirement 3: Last Updated Timestamp
last_updated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")
str_ui.info(f"🕒 **Last Updated:** {last_updated_time}")

# Fetch Data from API or Fallback
df_fixtures, api_error = fetch_live_fixtures(api_key_input)

if api_error:
  str_ui.warning(api_error)
  str_ui.info(
      "ℹ️ လက်ရှိတွင် API Key မရှိသေးပါက (သို့မဟုတ် Error ဖြစ်နေလျှင်) စမ်းသပ်ရန်"
      " Sample Data ဖြင့် ပြသပေးနေပါသည်။"
  )
  df_fixtures = get_fallback_fixtures()

# League Filter Selection
selected_league = str_ui.selectbox(
    "လိဂ်ပြိုင်ပွဲ ရွေးချယ်ရန်", ["အားလုံး (All Leagues)"] + LEAGUES
)

if (
    not df_fixtures.empty
    and selected_league != "အားလုံး (All Leagues)"
    and "league" in df_fixtures.columns
):
  df_fixtures = df_fixtures[df_fixtures["league"] == selected_league]

str_ui.divider()
str_ui.subheader(
    f"📅 လာမည့် တစ်ပတ်အတွင်း ပွဲစဉ်များနှင့် ခန့်မှန်းချက်များ"
    f" ({selected_league})"
)

if df_fixtures is None or df_fixtures.empty:
  str_ui.warning("ရွေးချယ်ထားသော အချက်အလက်များ မရှိပါ။")
else:
  for idx, row in df_fixtures.iterrows():
    league = row["league"]
    match_date = row["date"]
    home = row["home"]
    away = row["away"]
    h_form = row["home_form"]
    a_form = row["away_form"]

    # Run Prediction Model
    pred = calculate_poisson_prediction(home, away, h_form, a_form)

    with str_ui.container():
      str_ui.markdown(f"### 🏆 **{league}** | ⏰ *{match_date} UTC*")

      col1, col2, col3 = str_ui.columns([3, 2, 3])

      with col1:
        str_ui.markdown(f"#### 🏠 {home}")
        str_ui.caption(f"Elo: {pred['home_elo']} | Form: `{h_form}`")

      with col2:
        str_ui.markdown(
            f"<h3 style='text-align: center; color:"
            f" #FF4B4B;'>{pred['predicted_score']}</h3>",
            unsafe_allow_html=True,
        )
        str_ui.caption(
            f"<p style='text-align: center;'>ခန့်မှန်းဂိုး (xG): H"
            f" {pred['home_xg']} - A {pred['away_xg']}</p>",
            unsafe_allow_html=True,
        )

      with col3:
        str_ui.markdown(f"#### ✈️ {away}")
        str_ui.caption(f"Elo: {pred['away_elo']} | Form: `{a_form}`")

      # Detailed Rationale Expandable Box
      with str_ui.expander("📊 Statistical Model & Match Breakdown"):
        str_ui.write(
            f"- **အနိုင်ရရှိနိုင်ခြေ:** {home} နိုင်ခြေ"
            f" **{pred['home_win_prob']}%** | သရေ **{pred['draw_prob']}%** |"
            f" {away} နိုင်ခြေ **{pred['away_win_prob']}%**"
        )
        str_ui.write(
            f"- **Elo Rating နှိုင်းယှဉ်ချက်:** {home} (Rating:"
            f" {pred['home_elo']}) နှင့် {away} (Rating: {pred['away_elo']})"
            f" တို့၏ အဆင့်ကွာဟချက်နှင့် အိမ်ကွင်းအားသာချက်ကို"
            f" တွက်ချက်ထားသည်။"
        )
        str_ui.write(
            f"- **Poisson Distribution Model:** မျှော်လင့်ထားသည့် သွင်းဂိုးနှုန်း"
            f" (xG) အရ {home} တွင် ဂိုးသွင်းရန် {pred['home_xg']} နှင့် {away}"
            f" တွင် {pred['away_xg']} အသီးသီး ရှိနေပါသည်။"
        )

      str_ui.markdown("---")
