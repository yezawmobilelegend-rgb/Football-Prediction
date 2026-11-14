from datetime import datetime, timedelta
import math
import re
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
    "အဓိကလိဂ် (၆) ခု၏ တစ်ပတ်စာပွဲစဉ်များ၊ ကမ္ဘာ့ပေါက်ကြေး (တိုက်ရိုက် API)၊"
    " တကယ့် Form ရမှတ်များနှင့် AI Poisson ခန့်မှန်းချက်စနစ်"
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


# ၁။ Football-data.org မှ အသင်းများ၏ တကယ့် Standings နှင့် Form ရမှတ်များကို ဆွဲထုတ်ခြင်း
@st.cache_data(ttl=3600)
def fetch_team_standings(league_code, api_key):
  if not api_key:
    return {}
  url = f"https://api.football-data.org/v4/competitions/{league_code}/standings"
  headers = {"X-Auth-Token": api_key}
  team_stats_map = {}
  try:
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code == 200:
      data = response.json()
      standings = data.get("standings", [])
      for table in standings:
        if table.get("type") == "TOTAL":
          for row in table.get("table", []):
            team_name = row["team"]["name"]
            points = row.get("points", 10)
            # အသင်း၏ နိုင်/ရှုံး အပေါ်မူတည်၍ Elo တန်ဖိုး တွက်ချက်ခြင်း
            goal_diff = row.get("goalDifference", 0)
            elo = 1500 + (points * 15) + (goal_diff * 5)
            team_stats_map[team_name] = {
                "elo": elo,
                "form_pts": points,  # လိဂ်ဇယားပါ တကယ့်ရမှတ်
            }
  except Exception as e:
    print(f"Standings API Error: {e}")
  return team_stats_map


# အကယ်၍ API ထဲတွင် မပါသေးပါက Default တန်ဖိုး သတ်မှတ်ပေးရန်
def get_team_stats(team_name, standings_map):
  if team_name in standings_map:
    return standings_map[team_name]["elo"], standings_map[team_name]["form_pts"]
  # အကယ်၍ ရှာမတွေ့ပါက ပုံမှန်တန်ဖိုးပေးမည်
  return 1650, 12


# ၂။ မြန်မာကြေးပုံစံ (Team 1+20) သို့ ပြောင်းလဲပေးသော Function
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


# ၃။ မြန်မာဂိုးပေါင်းကြေးပုံစံ (3+40) သို့ ပြောင်းလဲပေးသော Function
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


# ၄။ odds-api.io မှ Live Odds ဆွဲထုတ်သည့် Function
@st.cache_data(ttl=600)
def fetch_live_odds_from_api(api_key):
  if not api_key:
    return []
  url = "https://api.odds-api.io/v1/odds"
  params = {"apiKey": api_key, "sport": "soccer", "region": "eu"}
  try:
    response = requests.get(url, params=params, timeout=10)
    if response.status_code == 200:
      data = response.json()
      if isinstance(data, list):
        return data
      elif isinstance(data, dict) and "data" in data:
        return data["data"]
  except Exception as e:
    print(f"Odds API Error: {e}")
  return []


def get_match_odds(home_team, away_team, odds_list):
  hdp, hdp_dec, ou, ou_dec = 1.0, 1.85, 2.5, 1.85
  found = False

  for item in odds_list:
    h_name = item.get("home_team", "")
    a_name = item.get("away_team", "")
    if (
        home_team.lower() in h_name.lower()
        or h_name.lower() in home_team.lower()
    ) and (
        away_team.lower() in a_name.lower()
        or a_name.lower() in away_team.lower()
    ):
      markets = item.get("bookmakers", [])
      for bm in markets:
        for market in bm.get("markets", []):
          if market.get("key") == "spreads":
            outcomes = market.get("outcomes", [])
            for out in outcomes:
              if (
                  home_team.lower() in out.get("name", "").lower()
                  or out.get("name", "").lower() in home_team.lower()
              ):
                hdp = out.get("point", 1.0)
                hdp_dec = out.get("price", 1.85)
                found = True
          if market.get("key") == "totals":
            outcomes = market.get("outcomes", [])
            for out in outcomes:
              if "over" in out.get("name", "").lower():
                ou = out.get("point", 2.5)
                ou_dec = out.get("price", 1.85)
                found = True
      if found:
        break

  return hdp, hdp_dec, ou, ou_dec


def poisson_probability(lam, k):
  return (math.exp(-lam) * (lam**k)) / math.factorial(k)


# ၅။ AI ခန့်မှန်းချက် Engine
def calculate_match_prediction(
    home_team,
    away_team,
    home_elo,
    away_elo,
    home_form_pts,
    away_form_pts,
    local_hdp_input,
    local_ou_input,
):
  base_home_lambda = 1.55
  base_away_lambda = 1.05

  elo_diff_factor = (home_elo - away_elo) / 250.0
  form_diff = (home_form_pts - away_form_pts) / 8.0

  home_lambda = max(
      0.4,
      base_home_lambda + (elo_diff_factor * 0.45) + (form_diff * 0.25),
  )
  away_lambda = max(
      0.3,
      base_away_lambda - (elo_diff_factor * 0.45) - (form_diff * 0.25),
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
      f"အထက် (Over 2.5) [AI မျှော်မှန်းဂိုး: {total_goals_lambda:.2f}]"
      if total_goals_lambda > 2.5
      else f"အောက် (Under 2.5) [AI မျှော်မှန်းဂိုး: {total_goals_lambda:.2f}]"
  )

  if local_ou_input:
    nums = re.findall(r"\d+\.?\d*", local_ou_input)
    if nums:
      user_ou_line = float(nums[0])
      if total_goals_lambda > user_ou_line:
        ou_recommendation = (
            f"ဒေသတွင်းဂိုးပေါင်းကြေး (`{local_ou_input}`) လိုင်းနှင့် AI"
            f" မျှော်မှန်းဂိုး ({total_goals_lambda:.2f}) ချိန်ထိုးချက်အရ 👉"
            f" **အထက် (Over {user_ou_line})** ကို အဓိကရွေးချယ်သင့်သည်"
        )
      else:
        ou_recommendation = (
            f"ဒေသတွင်းဂိုးပေါင်းကြေး (`{local_ou_input}`) လိုင်းနှင့် AI"
            f" မျှော်မှန်းဂိုး ({total_goals_lambda:.2f}) ချိန်ထိုးချက်အရ 👉"
            f" **အောက် (Under {user_ou_line})** ကို အဓိကရွေးချယ်သင့်သည်"
        )

  if local_hdp_input:
    if home_team.lower() in local_hdp_input.lower() and (
        "+" in local_hdp_input or "ဆုံ" in local_hdp_input
    ):
      winner_rec = (
          f"ဒေသတွင်းကြေး (`{local_hdp_input}`) အရ ဈေးကွက်မှ {home_team}"
          f" ဘက်ကို ပိုမိုအားသန်နေသဖြင့် 👉 **{home_team} အနိုင်/အသာစီး**"
          " ရယူရန် အဓိက အကြံပြုသည်"
      )
    else:
      winner_rec = (
          f"ဒေသတွင်းကြေး (`{local_hdp_input}`) နှင့် AI ချိန်ထိုးချက်အရ 👉"
          f" **{away_team} အဝေးကွင်း သို့မဟုတ် အနိမ့်အသင်းဘက်မှ** အသာစီးရှိသည်"
      )
  else:
    if home_lambda > away_lambda + 0.3:
      winner_rec = (
          f"AI တွက်ချက်မှုအရ 👉 **{home_team} အနိုင်ရရှိရန် အသာစီး**"
          " (ဒေသတွင်းကြေး ထည့်သွင်းရန် လိုအပ်သည်)"
      )
    elif away_lambda > home_lambda + 0.2:
      winner_rec = (
          f"AI တွက်ချက်မှုအရ 👉 **{away_team} အဝေးကွင်း အနိုင်/အကျိုးရှိ**"
          " (ဒေသတွင်းကြေး ထည့်သွင်းရန် လိုအပ်သည်)"
      )
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
            "homeTeam": {"name": "FC Bayern München"},
            "awayTeam": {"name": "1. FC Köln"},
            "utcDate": f"{today}T19:30:00Z",
            "status": "TIMED",
        }
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
      match_date = datetime.strptime(m["utcDate"], "%Y-%m-%dT%H:%M:%SZ")
      if now <= match_date <= seven_days_later:
        filtered_matches.append(m)
    return filtered_matches
  else:
    return []


# --- MAIN UI DISPLAY ---
st.subheader(f"📅 {selected_league_name} - လာမည့် ၇ ရက်အတွင်း ပွဲစဉ်ဇယားများ")

matches = fetch_fixtures(current_league_code, fb_api_key)
live_odds_data = fetch_live_odds_from_api(odds_api_key)
team_standings_map = fetch_team_standings(current_league_code, fb_api_key)

if not matches:
  st.warning(
      "⚠️ လက်တလော (၇) ရက်အတွင်း ကန်မည့် ပွဲစဉ်များ မရှိသေးပါ သို့မဟုတ် API Key"
      " မှားယွင်းနေပါသည်။"
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
        st.markdown("### 🌐 ကမ္ဘာ့ပေါက်ကြေး (Live API Format)")

        sim_hdp, sim_hdp_dec, sim_ou, sim_ou_dec = get_match_odds(
            home_team, away_team, live_odds_data
        )

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

        # တကယ့် API Standings မှ ရလာသော Elo နှင့် Form (ရမှတ်) များကို သုံးမည်
        home_elo, home_form_pts = get_team_stats(home_team, team_standings_map)
        away_elo, away_form_pts = get_team_stats(away_team, team_standings_map)

        best_score, ou_rec, winner_rec, h_lam, a_lam = (
            calculate_match_prediction(
                home_team,
                away_team,
                home_elo,
                away_elo,
                home_form_pts,
                away_form_pts,
                local_hdp_input,
                local_ou_input,
            )
        )

        local_display = (
            f"{local_hdp_input} / {local_ou_input}"
            if (local_hdp_input or local_ou_input)
            else "မထည့်ရသေး"
        )

        st.markdown(
            f"**1️⃣ ဈေးနှိုင်းယှဉ်ချက်:** ကမ္ဘာ့ကြေး (`{global_hdp_str} /"
            f" {global_ou_str}`) နှင့် ဒေသတွင်းကြေး (`{local_display}`)"
            " တို့ကို ချိန်ထိုးပြီး တန်ဖိုးရှိသော Value Bet ကို"
            " တွက်ချက်ပေးထားပါသည်။"
        )
        st.markdown(
            f"**2️⃣ Elo Ratings:** {home_team} ({home_elo}) vs {away_team}"
            f" ({away_elo})"
        )
        st.markdown(
            f"**3️⃣ လိဂ်ဇယား ရမှတ် (Points):** {home_team} (ရမှတ်"
            f" {home_form_pts}) | {away_team} (ရမှတ် {away_form_pts})"
        )
        st.markdown(
            f"**4️⃣ Poisson Distribution (မျှော်မှန်းဂိုး):** {home_team}"
            f" ({h_lam:.2f} ဂိုး) - {away_team} ({a_lam:.2f} ဂိုး)"
        )

        st.divider()

        st.markdown("#### 🎯 အနီးစပ်ဆုံး ခန့်မှန်းရလဒ်")
        st.success(
            f"**မှန်းဆဂိုးရလဒ်:** {home_team} **{best_score[0]} - {best_score[1]}**"
            f" {away_team}\n\n"
            f"🏆 **အကြံပြုရွေးချယ်မှု:** {winner_rec}\n\n"
            f"⚽ **ဂိုးပေါင်းဈေး:** {ou_rec}"
        )
