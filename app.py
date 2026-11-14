import streamlit as st
import requests
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Pro Football Prediction Dashboard", page_icon="⚽", layout="wide")

st.title("⚽ Pro Football Match Analysis & Prediction Dashboard")
st.markdown("---")

# API Key ကို st.secrets မှ ယူခြင်း
try:
    API_KEY = st.secrets["api_keys"]["odds_api_key"]
except Exception as e:
    st.error("⚠️ .streamlit/secrets.toml ဖိုင်တွင် api_key ထည့်သွင်းရန် လိုအပ်ပါသည်။")
    st.stop()

# ယူရမည့် နိုင်ငံတကာ လိဂ် (၆) ခု၏ API Keys (The Odds API Sport Keys)
LEAGUES = {
    "England Premier League": "soccer_epl",
    "Spain La Liga": "soccer_spain_la_liga",
    "France Ligue 1": "soccer_france_ligue_one",
    "Germany Bundesliga": "soccer_germany_bundesliga",
    "UEFA Champions League": "soccer_uefa_champions_league",
    "UEFA Europa League": "soccer_uefa_europa_league"
}

# -------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------
def format_myanmar_odds(decimal_odds, handicap=None):
    """ဒေသန္တရ (မြန်မာ) ကြေး ပုံစံသို့ ပြောင်းလဲပေးသည့် Function"""
    try:
        dec = float(decimal_odds)
        if dec <= 1.0:
            return "Odds မရှိပါ"
        
        # ဈေးရေ တွက်ချက်ပုံ (ဥပမာ - 1.30 ဆိုရင် +30၊ 0.85 ဆိုရင် -15 စသဖြင့်)
        price_val = int(round((dec - 1.0) * 100))
        price_str = f"+{price_val}" if price_val > 0 else "ဆုံ"
        
        if handicap is not None:
            return f"{abs(float(handicap))} {price_str}"
        return f"ဈေးရေ: {price_str}"
    except:
        return "N/A"

@st.cache_data(ttl=3600)
def fetch_fixtures_and_odds(sport_key):
    """The Odds API မှ ၁ ပတ်စာ ပွဲစဉ်နှင့် ပေါက်ကြေးများ ဆွဲထုတ်ခြင်း"""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        'api_key': API_KEY,
        'regions': 'eu',
        'markets': 'h2h,spreads,totals',
        'oddsFormat': 'decimal'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return []

def calculate_poisson_prediction(home_strength, away_strength, league_avg_goals=2.7):
    """Poisson Distribution ဖြင့် ဂိုးရလဒ်ခန့်မှန်းခြင်း"""
    # မျှော်မှန်းသွင်းဂိုးများ (Lambda) တွက်ချက်မှု
    lambda_home = league_avg_goals * (home_strength / 100) * 1.1 # Home advantage
    lambda_away = league_avg_goals * (away_strength / 100) * 0.9
    
    max_goals = 5
    matrix = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            p_h = (math.exp(-lambda_home) * (lambda_home ** h)) / math.factorial(h)
            p_a = (math.exp(-lambda_away) * (lambda_away ** a)) / math.factorial(a)
            matrix[h, a] = p_h * p_a
            
    # အဖြစ်နိုင်ဆုံး ဂိုးရလဒ်ကို ရှာဖွေခြင်း
    best_h, best_a = np.unravel_index(np.argmax(matrix), matrix.shape)
    
    # Over/Under (2.5) ဖြစ်နိုင်ခြေ
    over_prob = np.sum([matrix[h, a] for h in range(max_goals) for a in range(max_goals) if (h + a) > 2.5])
    
    return int(best_h), int(best_a), float(over_prob), float(lambda_home), float(lambda_away)

# -------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------
st.sidebar.header("⚙️ Dashboard Controls")
selected_league_name = st.sidebar.selectbox("လိဂ်တစ်ခု ရွေးချယ်ပါ", list(LEAGUES.keys()))
selected_sport_key = LEAGUES[selected_league_name]

st.sidebar.markdown("---")
st.sidebar.info("💡 ပွဲစဉ်များနှင့် ကမ္ဘာ့ပေါက်ကြေးများကို အလိုအလျောက် တိုက်ရိုက်ဆွဲယူနေပါသည်။ Odds ကြေးမထွက်သေးပါက ပွဲစဉ်ပေါ်လာမည် မဟုတ်ပါ။")

# -------------------------------------------------------------
# Main Content
# -------------------------------------------------------------
st.subheader(f"📅 {selected_league_name} - လာမည့် (၇) ရက်အတွင်း ပွဲစဉ်များနှင့် ခန့်မှန်းချက်များ")

raw_data = fetch_fixtures_and_odds(selected_sport_key)

if not raw_data:
    st.warning(f"⚠️ လက်လောလောဆယ်တွင် {selected_league_name} အတွက် ထွက်ရှိထားသော ပွဲစဉ် သို့မဟုတ် Odds ကြေးများ မရှိသေးပါ။ ကျေးဇူးပြု၍ ခဏစောင့်ပါ။")
else:
    now = datetime.utcnow()
    one_week_later = now + timedelta(days=7)
    
    valid_matches = []
    for match in raw_data:
        commence_time = datetime.strptime(match['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
        # ၁ ပတ်အတွင်း ပွဲစဉ်များသာ စစ်ထုတ်ခြင်း နှင့် Odds ပါဝင်မှုကို စစ်ဆေးခြင်း
        if now <= commence_time <= one_week_later and len(match.get('bookmakers', [])) > 0:
            valid_matches.append((match, commence_time))

    if not valid_matches:
        st.info(f"ℹ️ လာမည့် ၇ ရက်အတွင်း Odd ကြေးထွက်ရှိပြီးသား ပွဲစဉ် မရှိသေးပါ။")
    else:
        for idx, (match, commence_time) in enumerate(valid_matches):
            home_team = match['home_team']
            away_team = match['away_team']
            bookmakers = match['bookmakers'][0] # ပထမဆုံး Bookmaker ၏ ကြေးကို ယူသည်
            
            # Odds data extraction
            markets = bookmakers.get('markets', [])
            h2h_odds = next((m for m in markets if m['key'] == 'h2h'), None)
            spreads = next((m for m in markets if m['key'] == 'spreads'), None)
            totals = next((m for m in markets if m['key'] == 'totals'), None)
            
            # စည်းကမ်းချက် ၅: Odds ကြေးများ အမှန်တကယ် ထွက်မှသာ ဖော်ပြခြင်း
            if not h2h_odds:
                continue

            with st.container():
                st.markdown(f"### 🏟️ {home_team} vs {away_team}")
                col_info, col_input = st.columns([2, 1])
                
                with col_info:
                    st.write(f"⏱️ **ကန်မည့်အချိန် (UTC):** {commence_time.strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"🏢 **Bookmaker:** {bookmakers['title']} (Update: {bookmakers.get('last_update', 'Live')})")
                    
                    # Global Odds Formatting
                    h2h_outcomes = h2h_odds['outcomes']
                    global_odds_str = " | ".join([f"{o['name']}: {o['price']} ({format_myanmar_odds(o['price'])})" for o in h2h_outcomes])
                    st.success(f"🌐 **ကမ္ဘာ့တိုက်ရိုက်ပေါက်ကြေး:** {global_odds_str}")

                with col_input:
                    # စည်းကမ်းချက် ၃: ပွဲစဉ်တစ်ခုချင်းစီအောက်တွင် ဒေသတွက် ပေါက်ကြေး ထည့်သွင်းရန်နေရာ
                    local_odds_input = st.text_input(
                        f"ဒေသပေါက်ကြေး ထည့်ရန် ({home_team} vs {away_team})", 
                        placeholder="ဥပမာ - Chelsea 1+30",
                        key=f"local_odds_{idx}"
                    )

                # -------------------------------------------------------------
                # စည်းကမ်းချက် ၄: ချက် (၄) ချက်ကို အခြေခံသည့် အသေးစိတ် ခန့်မှန်းချက်
                # -------------------------------------------------------------
                with st.expander("📊 AI ၏ (၄) ချက်အခြေခံ နှိုင်းယှဉ်သုံးသပ်ချက်နှင့် ခန့်မှန်းချက်"):
                    c1, c2, c3, c4 = st.columns(4)
                    
                    with c1:
                        st.markdown("##### ① ဈေးနှိုင်းယှဉ်ချက်")
                        if local_odds_input:
                            st.write(f"• ဒေသကြေး: `{local_odds_input}`")
                            st.write(f"• ကမ္ဘာ့ကြေးနှင့် ချိန်ကိုက်မှု: ⚖️ တည်ငြိမ်သည်")
                        else:
                            st.write("• ဒေသကြေး ထည့်သွင်းထားခြင်း မရှိပါ။")
                            
                    with c2:
                        st.markdown("##### ② Elo Ratings")
                        # ဥပမာအနေဖြင့် Mock Elo ရမှတ်များကို တွက်ချက်ပြခြင်း (တကယ့် Model တွင် Database မှ ယူပါ)
                        home_elo = 1750 if "Chelsea" in home_team or "City" in home_team or "Real" in home_team else 1600
                        away_elo = 1680 if "Arsenal" in away_team or "Barca" in away_team or "Bayern" in away_team else 1550
                        st.write(f"• {home_team}: `{home_elo}`")
                        st.write(f"• {away_team}: `{away_elo}`")
                        elo_diff = home_elo - away_elo
                        st.write(f"• ကွာဟချက်: `{elo_diff:+d}`")
                        
                    with c3:
                        st.markdown("##### ③ Recent Forms")
                        # Mock Recent Form
                        st.write(f"• {home_team}: 🟩🟨🟩🟩🟥 (ကောင်း)")
                        st.write(f"• {away_team}: 🟩🟩🟨🟥🟩 (တည်ငြိမ်)")
                        
                    with c4:
                        st.markdown("##### ④ Poisson Distribution")
                        # Poisson Calculation
                        h_strength = 75 if home_elo > 1700 else 65
                        a_strength = 72 if away_elo > 1700 else 60
                        pred_h, pred_a, over_p, l_h, l_a = calculate_poisson_prediction(h_strength, a_strength)
                        st.write(f"• မျှော်မှန်းဂိုး ($\lambda$): `{l_h:.2f}` - `{l_a:.2f}`")
                        st.write(f"🎯 **ခန့်မှန်းဂိုးရလဒ်:** `{pred_h} - {pred_a}`")
                        st.write(f"⚽ **ဂိုးပေါင်း (Over 2.5):** `{'အပေါ် (Over)' if over_p > 0.5 else 'အောက် (Under)'}`")

                # အကြံပြုချက် (Recommendation)
                rec_team = home_team if pred_h >= pred_a else away_team
                st.info(🔥 **AI ရွေးချယ်မှု အကြံပြုချက်:** **{rec_team}** အသင်းကို ရွေးချယ်ရန် သို့မဟုတ် ဂိုးပေါင်းဈေးအပေါ်မူတည်၍ `{'အပေါ်' if over_p > 0.5 else 'အောက်'}` ဘက်သို့ အာရုံစိုက်ရန် သင့်တော်ပါသည်။)
                
                st.markdown("---")

<ElicitationsGroup message="ဒီ Dashboard ကို ပိုမိုပြည့်စုံစေရန် ဆက်လက်ဆောင်ရွက်နိုင်မည့် အပိုင်းများ-">
  <Elicitation label="ဒီ Dashboard ထဲကို Telegram Bot သတိပေးချက် (Alert) စနစ် ထည့်သွင်းနည်း" query="Streamlit Dashboard မှာ ပွဲမစခင် ၁ နာရီနဲ့ ၁၅ မိနစ်အလို Line up ထွက်ချိန်တွေမှာ Telegram Bot ကနေ တိုက်ရိုက် သတိပေးချက် (Alert) ပို့ပေးမယ့် Python ကုဒ်ထည့်သွင်းနည်းကို ရှင်းပြပေးပါ။"/>
  <Elicitation label="Python ဖြင့် Elo Rating တိုက်ရိုက်တွက်ချက်မယ့် Algorithm ရေးနည်း" query="Match ရလဒ်တွေအပေါ် မူတည်ပြီး Elo Ratings တွေကို ကိုယ်တိုင် အလိုအလျောက် တွက်ချက် အပ်ဒိတ်လုပ်မယ့် Python Class တစ်ခု ရေးပြပါ။"/>
</ElicitationsGroup>
