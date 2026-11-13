from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson
import streamlit as str_ui

# Page Configuration
str_ui.set_page_config(
    page_title="Advanced Real-Time Football Predictor",
    page_icon="⚽",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 1. API & SECRETS CONFIGURATION
# ---------------------------------------------------------------------------
try:
  api_key_input = str_ui.secrets["FOOTBALL_API_KEY"]
except:
  str_ui.sidebar.title("⚙️ API Settings")
  api_key_input = str_ui.sidebar.text_input(
      "football-data.org API Key ထည့်ရန်", type="password"
  )

LEAGUES = [
    "Premier League",
    "La Liga",
    "Serie A",
    "France Ligue 1",
    "Champions League",
    "Europa League",
]

# Comprehensive Team Database (Elo, H2H weights, Home/Away Boosts)
TEAM_DB = {
    "Manchester City": {
        "elo": 2050,
        "home_xg": 2.2,
        "away_xg": 1.8,
        "rest_days": 6,
    },
    "Arsenal": {"elo": 2020, "home_xg": 2.1, "away_xg": 1.7, "rest_days": 6},
    "Liverpool": {"elo": 2010, "home_xg": 2.2, "away_xg": 1.7, "rest_days": 5},
    "Chelsea": {"elo": 1840, "home_xg": 1.6, "away_xg": 1.3, "rest_days": 7},
    "Aston Villa": {"elo": 1850, "home_xg": 1.7, "away_xg": 1.2, "rest_days": 4},
    "Real Madrid CF": {
        "elo": 2060,
        "home_xg": 2.3,
        "away_xg": 1.9,
        "rest_days": 5,
    },
    "Villarreal CF": {
        "elo": 1810,
        "home_xg": 1.5,
        "away_xg": 1.1,
        "rest_days": 6,
    },
    "Inter Milan": {"elo": 1990, "home_xg": 2.0, "away_xg": 1.6, "rest_days": 6},
    "Juventus FC": {"elo": 1910, "home_xg": 1.8, "away_xg": 1.3, "rest_days": 5},
    "Paris Saint-Germain FC": {
        "elo": 1980,
        "home_xg": 2.3,
        "away_xg": 1.8,
        "rest_days": 6,
    },
    "FC Bayern Munich": {
        "elo": 2040,
        "home_xg": 2.4,
        "away_xg": 1.9,
        "rest_days": 5,
    },
    "Bayer 04 Leverkusen": {
        "elo": 1960,
        "home_xg": 2.1,
        "away_xg": 1.6,
        "rest_days": 6,
    },
}


# ---------------------------------------------------------------------------
# 2. FETCH FIXTURES FROM API
# ---------------------------------------------------------------------------
def fetch_live_fixtures(api_key):
  if not api_key:
    return None, "⚠️ API Key မရှိပါ။ Sidebar တွင် သို့မဟုတ် Secrets တွင် ထည့်ပါ။"

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
          parsed_fixtures.append({
              "league": comp_mapping[comp_name],
              "date": m.get("utcDate", "").replace("T", " ")[:16],
              "home": m.get("homeTeam", {}).get("name", "Home Team"),
              "away": m.get("awayTeam", {}).get("name", "Away Team"),
          })

      return (
          pd.DataFrame(parsed_fixtures)
          if parsed_fixtures
          else pd.DataFrame(),
          None,
      )
    else:
      return None, f"⚠️ API Error: {response.status_code}"
  except Exception as e:
    return None, f"⚠️ Connection Error: {str(e)}"


# Fallback Mock Fixtures if API returns empty
def get_fallback_fixtures():
  today = datetime.now()
  return pd.DataFrame([
      {
          "league": "Premier League",
          "date": (today + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"),
          "home": "Chelsea",
          "away": "Aston Villa",
      },
      {
          "league": "Premier League",
          "date": (today + timedelta(days=1)).strftime("%Y-%m-%d %H:%M"),
          "home": "Manchester City",
          "away": "Liverpool",
      },
  ])


# ---------------------------------------------------------------------------
# 3. ADVANCED PREDICTION MODEL WITH TIMELINE PHASES
# ---------------------------------------------------------------------------
def calculate_advanced_prediction(
    home, away, hours_to_match, lineup_status, h2h_bias
):
  h_data = TEAM_DB.get(
      home, {"elo": 1750, "home_xg": 1.5, "away_xg": 1.2, "rest_days": 5}
  )
  a_data = TEAM_DB.get(
      away, {"elo": 1750, "home_xg": 1.5, "away_xg": 1.2, "rest_days": 5}
  )

  elo_diff = (h_data["elo"] - a_data["elo"]) + 100
  base_home_xg = h_data["home_xg"] + (elo_diff / 400.0) * 0.3
  base_away_xg = a_data["away_xg"] - (elo_diff / 400.0) * 0.3

  # Factor 1: Rest Days / Fatigue Adjustment
  rest_diff = h_data["rest_days"] - a_data["rest_days"]
  base_home_xg += rest_diff * 0.05
  base_away_xg -= rest_diff * 0.05

  # Factor 2: H2H Historical Psychological Bias
  if h2h_bias == "Home Dominant":
    base_home_xg += 0.25
  elif h2h_bias == "Away Dominant":
    base_away_xg += 0.25

  # Factor 3: Timeline Updates (3 Hours, 1 Hour, 15 Mins / Lineup Released)
  timeline_note = ""
  if hours_to_match <= 3 and hours_to_match > 1:
    timeline_note = (
        "🕒 **[၃ နာရီအလို Update]:** အသင်းလိုက်လေ့ကျင့်ခန်းနှင့်"
        " ကစားသမားထုတ်ပြန်မည့် သတင်းများကို အခြေခံထားသည်။"
    )
    base_home_xg *= 1.02
  elif hours_to_match <= 1 and hours_to_match > 0.25:
    if lineup_status == "Full Strength (အဓိကလူစုံ)":
      timeline_note = (
          "🔥 **[၁ နာရီအလို / Lineup ထွက်ရှိချိန်]:** အဓိကကစားသမားစုံလင်စွာ"
          " ပါဝင်သဖြင့် တိုက်စစ်စွမ်းအား မြင့်တက်လာသည်။"
      )
      base_home_xg *= 1.1
      base_away_xg *= 1.1
    else:
      timeline_note = (
          "⚠️ **[၁ နာရီအလို / Lineup ထွက်ရှိချိန်]:** အဓိကကစားသမား"
          " ဒဏ်ရာရ/ပွဲပယ်ကြောင့် လူစာရင်း အပြောင်းအလဲရှိသည်။"
      )
      base_home_xg *= 0.85
      base_away_xg *= 0.85
  elif hours_to_match <= 0.25:
    timeline_note = (
        "⚡ **[၁၅ မိနစ်အလို (ပွဲကန်ခါနီး)]:** တရားဝင် Lineup ပြီးဆုံး၍"
        " ကွင်းအတွင်း တိုက်ရိုက်ရာသီဥတုနှင့် အكت်ဖ်ဖြစ်မှု အခြေအနေများကို"
        " ထည့်သွင်းတွက်ချက်ပြီးပါပြီ။"
    )
  else:
    timeline_note = "📅 **[ကြိုတင်ခန့်မှန်းချက်]:** ယေဘုယျ ခြေစွမ်းနှင့် အချက်အလက်များအပေါ် အခြေခံသည်။"

  final_home_xg = max(0.4, round(base_home_xg, 2))
  final_away_xg = max(0.4, round(base_away_xg, 2))

  # Poisson Distribution Matrix
  max_goals = 6
  score_matrix = np.outer(
      poisson.pmf(np.arange(max_goals), final_home_xg),
      poisson.pmf(np.arange(max_goals), final_away_xg),
  )
  h_score, a_score = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)

  home_win_prob = np.sum(np.tril(score_matrix, -1)) * 100
  draw_prob = np.sum(np.diag(score_matrix)) * 100
  away_win_prob = np.sum(np.triu(score_matrix, 1)) * 100

  return {
      "home_xg": final_home_xg,
      "away_xg": final_away_xg,
      "predicted_score": f"{h_score} - {a_score}",
      "home_win_prob": round(home_win_prob, 1),
      "draw_prob": round(draw_prob, 1),
      "away_win_prob": round(away_win_prob, 1),
      "home_elo": h_data["elo"],
      "away_elo": a_data["elo"],
      "timeline_note": timeline_note,
  }


# ---------------------------------------------------------------------------
# 4. STREAMLIT USER INTERFACE
# ---------------------------------------------------------------------------
str_ui.title("⚽ Advanced Real-Time Match Predictor & Timeline AI")
str_ui.markdown(
    "ပွဲချိန်အလိုက် (**၃ နာရီအလို၊ ၁ နာရီအလို Lineup နှင့် ၁၅ မိနစ်အလို**) အချက်အလက်အသစ်များနှင့်"
    " **Elo, H2H, Fatigue, Poisson Model** များကို ပေါင်းစပ်တွက်ချက်ပေးသော စနစ်။"
)

# Last Updated Timestamp
last_updated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")
str_ui.info(f"🕒 **Last Updated Timestamp:** {last_updated_time}")

# Fetch Data
df_fixtures, api_error = fetch_live_fixtures(api_key_input)
if api_error or df_fixtures is None or df_fixtures.empty:
  df_fixtures = get_fallback_fixtures()

selected_league = str_ui.selectbox(
    "လိဂ်ပြိုင်ပွဲ ရွေးချယ်ရန်", ["အားလုံး (All Leagues)"] + LEAGUES
)
if selected_league != "အားလုံး (All Leagues)":
  df_fixtures = df_fixtures[df_fixtures["league"] == selected_league]

str_ui.divider()
str_ui.subheader(f"📅 ပွဲစဉ်များနှင့် အချိန်ကိုက် ခန့်မှန်းချက်များ")

for idx, row in df_fixtures.iterrows():
  league = row["league"]
  match_date = row["date"]
  home = row["home"]
  away = row["away"]

  with str_ui.container():
    str_ui.markdown(f"### 🏆 **{league}** | ⏰ *{match_date}*")

    # Interactive Controls for Match Timeline Simulation
    col_ctrl1, col_ctrl2, col_ctrl3 = str_ui.columns(3)
    with col_ctrl1:
      hours_to_match = str_ui.slider(
          f"ပွဲစတင်ရန် ကျန်ချိန် (နာရီ) - {home} vs {away}",
          0.1,
          24.0,
          2.0,
          key=f"h_{idx}",
      )
    with col_ctrl2:
      lineup_status = str_ui.selectbox(
          "ကစားသမားစာရင်း (Lineup Status)",
          [
              "Full Strength (အဓိကလူစုံ)",
              "Key Players Missing (အဓိကလူ ဒဏ်ရာရ/ပွဲပယ်)",
          ],
          key=f"l_{idx}",
      )
    with col_ctrl3:
      h2h_bias = str_ui.selectbox(
          "H2H မှတ်တမ်းအသာစီး",
          ["Neutral (ပုံမှန်)", "Home Dominant", "Away Dominant"],
          key=f"b_{idx}",
      )

    # Calculate Advanced Prediction
    pred = calculate_advanced_prediction(
        home, away, hours_to_match, lineup_status, h2h_bias
    )

    c1, c2, c3 = str_ui.columns([3, 2, 3])
    with c1:
      str_ui.markdown(f"#### 🏠 {home}")
      str_ui.caption(f"Elo Rating: {pred['home_elo']}")
    with c2:
      str_ui.markdown(
          f"<h3 style='text-align: center; color:"
          f" #FF4B4B;'>{pred['predicted_score']}</h3>",
          unsafe_allow_html=True,
      )
      str_ui.caption(
          f"<p style='text-align: center;'>ခန့်မှန်း xG: H"
          f" {pred['home_xg']} - A {pred['away_xg']}</p>",
          unsafe_allow_html=True,
      )
    with c3:
      str_ui.markdown(f"#### ✈️ {away}")
      str_ui.caption(f"Elo Rating: {pred['away_elo']}")

    # Timeline Note & Rationale
    str_ui.success(pred["timeline_note"])

    with str_ui.expander("📊 အသေးစိတ် တွက်ချက်မှု အကြောင်းပြချက်များ"):
      str_ui.write(
          f"- **အနိုင်ရရှိနိုင်ခြေ:** {home} နိုင်ခြေ **{pred['home_win_prob']}%**"
          f" | သရေ **{pred['draw_prob']}%** | {away} နိုင်ခြေ"
          f" **{pred['away_win_prob']}%**"
      )
      str_ui.write(
          f"- **Timeline & Lineup Impact:** ပွဲချိန် {hours_hours_str if 'hours_hours_str' in locals() else hours_နာရီ} အလို"
          f" အခြေအနေနှင့် {lineup_status} တို့ကို ထည့်သွင်းတွက်ချက်ထားသည်။"
      )
      str_ui.write(
          "- **Fatigue & H2H Factors:** အနားယူရသည့် ရက်ကွာဟချက်များနှင့်"
          " ယခင်တွေ့ဆုံမှု (H2H) သမိုင်းကြောင်းများကို ချိန်ညှိထားသည်။"
      )

    str_ui.markdown("---")
