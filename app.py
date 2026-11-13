from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson  # Corrected Import
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

TEAM_ELOS = {
    "Manchester City": 2050,
    "Arsenal": 2020,
    "Liverpool": 2010,
    "Aston Villa": 1850,
    "Chelsea": 1840,
    "Tottenham Hotspur FC": 1820,
    "Manchester United FC": 1800,
    "Newcastle United FC": 1790,
    "Real Madrid CF": 2060,
    "FC Barcelona": 2030,
    "Atlético de Madrid": 1920,
    "Real Sociedad de Fútbol": 1840,
    "Villarreal CF": 1810,
    "Athletic Club": 1820,
    "Inter Milan": 1990,
    "Juventus FC": 1910,
    "AC Milan": 1900,
    "SSC Napoli": 1880,
    "Atalanta BC": 1870,
    "AS Roma": 1840,
    "Paris Saint-Germain FC": 1980,
    "AS Monaco FC": 1830,
    "Olympique de Marseille": 1810,
    "LOSC Lille": 1800,
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
          "⚠️ API Request Limit ကျော်လွန်သွားပါပြီ (Free Plan: 10 req/min)။",
      )
    else:
      return (
          None,
          f"⚠️ API Error (Code {response.status_code}): {response.text}",
      )
  except Exception as e:
    return None, f"⚠️ Connection Error: {str(e)}"


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
  # Corrected usage: poisson.pmf instead of pmf directly
  score_matrix = np.outer(
      poisson.pmf(np.arange(max_goals), home_xg),
      poisson.pmf(np.arange(max_goals), away_xg),
  )
  h_score, a_score = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)

  home_win_prob = np.sum(np.tril(score_matrix, -1)) * 100
  draw_prob = np.sum(np.diag(score_matrix)) * 100
  away_win_prob = np.sum(np.triu(score_matrix, 1)) * 
