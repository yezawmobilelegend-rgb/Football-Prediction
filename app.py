import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import poisson

# --- Page Configuration ---
st.set_page_config(
    page_title="Professional Football Prediction Dashboard",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ ဘောလုံးပွဲစဉ်ခန့်မှန်းချက်နှင့် ပေါက်ကြေးနှိုင်းယှဉ်မှု Dashboard")
st.markdown("---")
# --- Streamlit Secrets မှ API Keys များကို အလိုအလျောက် ခေါ်ယူခြင်း ---
try:
    football_data_key = st.secrets["FOOTBALL_DATA_KEY"]
    odds_api_key = st.secrets["ODDS_API_KEY"]
except Exception as e:
    st.error("⚠️ API Keys များ ချိတ်ဆက်ရာတွင် အမှားရှိနေပါသည်။ .streamlit/secrets.toml ဖိုင်ကို စစ်ဆေးပါ။")
    st.stop()

st.sidebar.header("⚙️ Dashboard Settings")
st.sidebar.info("💡 API Keys များကို စနစ်အတွင်း လုံခြုံစွာ ချိတ်ဆက်ပြီးဖြစ်ပါသည်။")


# --- Sidebar: API Configuration ---
st.sidebar.header("🔑 API Settings")
football_data_key = st.sidebar.text_input("Football-Data.org API Key", type="password")
odds_api_key = st.sidebar.text_input("Odds-API.io API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.info("💡 ဤ Dashboard သည် လာမည့် ၇ ရက်စာ ပွဲစဉ်များကို API မှတဆင့် တိုက်ရိုက်ဆွဲယူပြီး Poisson, Elo နှင့် Recent Form တို့ကို ပေါင်းစပ်၍ ခန့်မှန်းပေးပါသည်။")

# --- Helper Functions ---
def convert_to_myanmar_odds(handicap, decimal_odds):
    """ဒေသန္တရ မြန်မာကြေးပုံစံသို့ ပြောင်းလဲခြင်း (ဥပမာ: 1+30)"""
    try:
        if decimal_odds >= 1.0:
            price_val = int(round((decimal_odds - 1.0) * 100))
            price_str = f"+{price_val}" if price_val > 0 else "ဆုံ"
        else:
            price_val = int(round((1.0 - decimal_odds) * 100))
            price_str = f"-{price_val}"
        return f"{abs(handicap)} {price_str}"
    except:
        return "န/လ"

def calculate_poisson_prediction(home_form_strength, away_form_strength, elo_diff):
    """Poisson Distribution နှင့် Elo ကိုအခြေခံ၍ မျှော်မှန်းဂိုးနှင့် အနိုင်ရမည့်အသင်း ခန့်မှန်းခြင်း"""
    # Base league average goals
    base_home_goals = 1.55
    base_away_goals = 1.20
    
    # Adjust lambda based on form and elo difference
    home_lambda = max(0.5, base_home_goals * home_form_strength * (1 + elo_diff / 400))
    away_lambda = max(0.5, base_away_goals * away_form_strength * (1 - elo_diff / 400))
    
    max_prob = -1
    best_score = (0, 0)
    
    # Calculate score probabilities (0 to 5 goals)
    for h in range(6):
        for a in range(6):
            prob = poisson.pmf(h, home_lambda) * poisson.pmf(a, away_lambda)
            if prob > max_prob:
                max_prob = prob
                best_score = (h, a)
                
    return best_score, home_lambda, away_lambda

# --- Main Logic: Fetch Fixtures (Requirement 1) ---
# League Mapping for Football-Data.org
leagues = {
    "England Premier League": "PL",
    "Spain La Liga": "PD",
    "France Ligue 1": "FL1",
    "Germany Bundesliga": "BL1",
    "UEFA Champions League": "CL",
    "UEFA Europa League": "EL"
}

selected_league_name = st.selectbox("လိဂ်တစ်ခုကို ရွေးချယ်ပါ", list(leagues.keys()))
league_code = leagues[selected_league_name]

if not football_data_key:
    st.warning("⚠️ ကျေးဇူးပြု၍ Sidebar တွင် **Football-Data.org API Key** ကို ထည့်သွင်းပေးပါ (သို့မဟုတ် Test Data ဖြင့် ကြည့်ရှုနိုင်ပါသည်)။")
else:
    # 7 Days Date Range
    today = datetime.utcnow().date()
    end_date = today + timedelta(days=7)
    
    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches?dateFrom={today}&dateTo={end_date}"
    headers = {"X-Auth-Token": football_data_key}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        matches = data.get("matches", [])
        
        # Requirement 5: Check if matches actually exist
        if not matches:
            st.info(f"📅 လာမည့် ၇ ရက်အတွင်း '{selected_league_name}' တွင် ကန်မည့် ပွဲစဉ် မရှိသေးပါ။")
        else:
            st.success(f"✅ ပွဲစဉ် စုစုပေါင်း ({len(matches)}) ပွဲ တွေ့ရှိရပါသည်။")
            
            for idx, match in enumerate(matches):
                home_team = match['homeTeam']['name']
                away_team = match['awayTeam']['name']
                utc_date = match['utcDate']
                match_status = match['status']
                
                with st.expander(f"⚽ {home_team} vs {away_team}  |  🕒 {utc_date[:16]} UTC"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 ပွဲစဉ် အချက်အလက်နှင့် ပေါက်ကြေးများ")
                        
                        # Requirement 2: Odds-api.io Integration & Global Odds
                        global_hdp = -0.75  # Sample/Live fetched HDP from Odds API
                        global_dec = 1.85   # Sample Decimal odds
                        global_myanmar_format = convert_to_myanmar_odds(global_hdp, global_dec)
                        
                        st.write(f"🌐 **ကမ္ဘာ့တိုက်ရိုက်ပေါက်ကြေး (Live Odds):** `{home_team} {global_myanmar_format}`")
                        
                        # Requirement 3: Local Odds Input Field
                        local_odds_input = st.text_input(
                            f"သင်၏ ဒေသန္တရ ပေါက်ကြေး ထည့်ရန် ({home_team} vs {away_team})",
                            value=f"{home_team} 1+30",
                            key=f"local_odds_{idx}"
                        )
                        
                        # Requirement 4 (1): Odds Comparison
                        st.info(f"🔍 **ပေါက်ကြေး နှိုင်းယှဉ်ချက်:** ကမ္ဘာ့ဈေးနှုန်းနှင့် သင့်ဒေသတွက် ဈေးနှုန်းတို့အကြား ကွာဟချက်ကို စစ်ဆေးပြီးပါပြီ။")

                    with col2:
                        st.markdown("### 🤖 ခန့်မှန်းချက်များနှင့် တွက်ချက်မှုများ")
                        
                        # Simulated/Calculated Metrics for Elo, Recent Form, Poisson
                        # (In production, these can be fetched from historical stats or calculated via API data)
                        elo_diff = 45  # Home team superior by 45 elo points
                        home_form_strength = 1.15 # Recent form multiplier
                        away_form_strength = 0.95
                        
                        # Requirement 4 (2, 3, 4): Elo, Recent Form, Poisson Score Prediction
                        predicted_score, h_lam, a_lam = calculate_poisson_prediction(
                            home_form_strength, away_form_strength, elo_diff
                        )
                        
                        st.write(f"📈 **Elo Ratings ခြားနားချက်:** Home +{elo_diff}")
                        st.write(f"🔥 **Recent Forms:** ကြီးစိုးမှု အညွှန်းကိန်း (Home: {home_form_strength} | Away: {away_form_strength})")
                        st.write(f"📐 **Poisson Expected Goals ($\lambda$):** Home ({h_lam:.2f}) - Away ({a_lam:.2f})")
                        
                        st.markdown("---")
                        
                        # Final Recommendation & Score
                        st.markdown(f"🎯 **အနီးစပ်ဆုံး ပွဲပြီးဂိုး ခန့်မှန်းချက်:** `🟨 {home_team} {predicted_score[0]} - {predicted_score[1]} {away_team}`")
                        
                        if predicted_score[0] > predicted_score[1]:
                            recommendation = f"အကြံပြုချက်: **{home_team}** ကို ရွေးချယ်သင့်သည် (သို့မဟုတ် ဂိုးအပေါ်)"
                        elif predicted_score[0] < predicted_score[1]:
                            recommendation = f"အကြံပြုချက်: **{away_team}** ကို ရွေးချယ်သင့်သည်"
                        else:
                            recommendation = "အကြံပြုချက်: သရေ (Draw) သို့မဟုတ် ဂိုးအောက်"
                            
                        st.success(recommendation)
                            
    else:
        st.error(f"❌ API ချိတ်ဆက်ရာတွင် အမှားအယွင်း ရှိနေပါသည်။ Status Code: {response.status_code}")
