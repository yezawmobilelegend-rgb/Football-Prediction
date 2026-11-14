import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import poisson

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Professional Football Odds & Prediction Dashboard",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ ဘောလုံးပွဲစဉ်များနှင့် အဆင့်မြင့် ခန့်မှန်းတွက်ချက်မှု ဒိုင်ခွက် (Dashboard)")
st.markdown("---")

# --- SECRETS & API CONFIGURATION ---
try:
    API_KEY = st.secrets["api"]["odds_api_key"]
except Exception:
    st.error("⚠️ .streamlit/secrets.toml ထဲတွင် API Key ထည့်သွင်းရန် လိုအပ်ပါသည်။")
    st.stop()

# နိုင်ငံတကာ လိဂ်များနှင့် The Odds API ၏ Sport Keys များ
LEAGUES = {
    "English Premier League": "soccer_epl",
    "Spanish La Liga": "soccer_spain_la_liga",
    "French Ligue 1": "soccer_france_ligue_one",
    "German Bundesliga": "soccer_germany_bundesliga",
    "UEFA Champions League": "soccer_uefa_champions_league",
    "UEFA Europa League": "soccer_uefa_europa_league"
}

# --- FUNCTIONS ---

@st.cache_data(ttl=3600)
def fetch_odds_and_fixtures(sport_key):
    """The Odds API မှ ၇ ရက်စာ ပွဲစဉ်များနှင့် Live Odds များကို ဆွဲထုတ်ခြင်း"""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        'api_key': API_KEY,
        'regions': 'eu',
        'markets': 'h2h,spreads,totals',
        'oddsFormat': 'decimal'
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        return []

def decimal_to_myanmar_odds(hdp_val, dec_odds):
    """ကမ္ဘာ့ဆိုက် Decimal Odds များကို မြန်မာကြေး ပုံစံသို့ ပြောင်းလဲခြင်း"""
    if dec_odds is None or dec_odds <= 1.0:
        dec_odds = 1.01
    
    price_val = int(round((dec_odds - 1.0) * 100))
    if price_val > 0:
        price_str = f"+{price_val}"
    elif price_val == 0:
        price_str = "ဆုံ"
    else:
        price_str = f"-{abs(price_val)}"
        
    if hdp_val is not None:
        return f"{hdp_val} {price_str}"
    else:
        return f"ဆုံ {price_str}"

def calculate_poisson_prediction(home_form_score, away_form_score, elo_diff):
    """Elo, Recent Form တို့ကို အခြေခံ၍ Poisson Distribution ဖြင့် ဂိုးခန့်မှန်းခြင်း"""
    # ခြေစွမ်းနှင့် Elo ကွာဟချက်အပေါ်မူတည်၍ မျှော်မှန်းသွင်းဂိုး (Lambda) တွက်ချက်ခြင်း
    base_home_xg = 1.5
    base_away_xg = 1.1
    
    # ချိန်ညှိချက်
    home_lambda = max(0.5, base_home_xg + (elo_diff / 400.0) + (home_form_score * 0.1))
    away_lambda = max(0.5, base_away_xg - (elo_diff / 400.0) + (away_form_score * 0.1))
    
    # Poisson ဖြင့် 0 မှ 5 ဂိုးအထိ ဖြစ်နိုင်ခြေ Matrix တည်ဆောက်ခြင်း
    max_goals = 5
    matrix = np.outer(
        [poisson.pmf(i, home_lambda) for i in range(max_goals + 1)],
        [poisson.pmf(j, away_lambda) for j in range(max_goals + 1)]
    )
    
    # အဖြစ်နိုင်ဆုံး ရလဒ် (Score) ကို ရှာဖွေခြင်း
    home_idx, away_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    
    # 1X2 ရာခိုင်နှုန်းများ
    home_win_prob = np.sum(np.tril(matrix, -1)) * 100
    draw_prob = np.sum(np.diag(matrix)) * 100
    away_win_prob = np.sum(np.triu(matrix, 1)) * 100
    
    return {
        "home_goals": home_idx,
        "away_goals": away_idx,
        "home_prob": round(home_win_prob, 1),
        "draw_prob": round(draw_prob, 1),
        "away_prob": round(away_win_prob, 1),
        "expected_home_lambda": round(home_lambda, 2),
        "expected_away_lambda": round(away_lambda, 2)
    }

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ ဆက်တင်များနှင့် စစ်ထုတ်ရန်များ")
selected_league_name = st.sidebar.selectbox("လိဂ်တစ်ခုကို ရွေးချယ်ပါ", list(LEAGUES.keys()))
sport_key = LEAGUES[selected_league_name]

st.sidebar.markdown("---")
st.sidebar.info("💡 **အကြံပြုချက်:** ပွဲစဉ်များ အမှန်တကယ် ရှိပြီး API မှ Odds (ပေါက်ကြေး) များ ထွက်ရှိလာမှသာ အောက်ပါ Card များတွင် အလိုအလျောက် ပေါ်လာမည် ဖြစ်ပါသည်။")

# --- MAIN CONTENT ---
with st.spinner(f"⏳ {selected_league_name} ၏ ၇ ရက်စာ ပွဲစဉ်များနှင့် ပေါက်ကြေးများကို ရယူနေပါသည်..."):
    matches_data = fetch_odds_and_fixtures(sport_key)

if not matches_data:
    st.warning(f"⚠️ လက်တလောတွင် {selected_league_name} အတွက် ကစားမည့်ပွဲစဉ်များ (သို့) အချိန်ကိုက် Odds ဒေတာများ မရှိသေးပါ။ ကျေးဇူးပြု၍ ခဏစောင့်ဆိုင်းပါ။")
else:
    # ၇ ရက်စာ (Current Time to +7 Days) ဖिल्တာ လုပ်ခြင်း
    now = datetime.utcnow()
    seven_days_later = now + timedelta(days=7)
    
    filtered_matches = []
    for match in matches_data:
        match_time = datetime.strptime(match['commence_time'].replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
        if now <= match_time <= seven_days_later:
            filtered_matches.append((match, match_time))
            
    if not filtered_matches:
        st.info(jsons := f"လာမည့် ၇ ရက်အတွင်း {selected_league_name} တွင် အတည်ပြုထားသော ပွဲစဉ်များ မရှိသေးပါ။")
    else:
        st.success(f"🎯 တွေ့ရှိရသော ပွဲစဉ်အရေအတွက်: **{len(filtered_matches)} ပွဲ** (လာမည့် ၇ ရက်အတွင်း)")
        
        for idx, (match, match_time) in enumerate(filtered_matches):
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Bookmakers data ထုတ်ယူခြင်း
            bookmakers = match.get('bookmakers', [])
            world_hdp = "ဒေတာမရှိ"
            world_dec = 1.85
            over_under_line = "2.5"
            
            if bookmakers:
                # ပထမဆုံးရရှိနိုင်သော Bookmaker ၏ ကြေးကို ယူသုံးခြင်း
                bm = bookmakers[0]
                for market in bm.get('markets', []):
                    if market['key'] == 'spreads':
                        outcomes = market['outcomes']
                        for out in outcomes:
                            if out['name'] == home_team:
                                world_hdp = out.get('point', 0)
                                world_dec = out.get('price', 1.85)
                    elif market['key'] == 'totals':
                        for out in market['outcomes']:
                            if out.get('name') == 'Over':
                                over_under_line = out.get('point', 2.5)

            world_myanmar_odds_str = decimal_to_myanmar_odds(world_hdp, world_dec)

            # UI Card for each match
            with st.container():
                st.markdown(f"### 🏟️ ပွဲစဉ် ({idx+1}): {home_team} vs {away_team}")
                col_info, col_odds, col_analysis = st.columns([1.2, 1.2, 1.8])
                
                with col_info:
                    st.markdown(f"**⏰ ပွဲချိန်:** {match_time.strftime('%Y-%m-%d %H:%M')} (UTC)")
                    st.markdown(f"**လိဂ်:** {selected_league_name}")
                    st.markdown(f"**အိမ်ရှင်:** {home_team}")
                    st.markdown(f"**ဧည့်သည်:** {away_team}")
                    
                with col_odds:
                    st.markdown("#### 💰 ပေါက်ကြေးများ နှိုင်းယှဉ်ချက်")
                    st.info(f"🌐 **ကမ္ဘာ့တိုက်ရိုက်ကြေး:**\n{home_team} {world_myanmar_odds_str}\n(ဂိုးပေါင်းလိုင်း: {over_under_line})")
                    
                    # 3) Local Odds Input Field (ဒေသတွက် ပေါက်ကြေးထည့်ရန်နေရာ)
                    local_input_key = f"local_odds_{match.get('id', idx)}"
                    local_odds_val = st.text_input(
                        f"ယခုဒေသတွက် ပေါက်နေသော ကြေးထည့်ရန် ({home_team})", 
                        value=world_myanmar_odds_str,
                        key=local_input_key
                    )
                    
                with col_analysis:
                    st.markdown("#### 📊 အဆင့်မြင့် ခန့်မှန်းသုံးသပ်ချက် (Analysis)")
                    
                    # Mock/Simulated Values for demonstration based on user request (Elo, Recent Form, Poisson)
                    # တကယ့် production တွင် ဤနေရာ၌ သီးသန့် Elo/Form API သို့မဟုတ် DB မှ ဒေတာကို ထည့်သွင်းရပါမည်။
                    sim_elo_diff = 45 # ဥပမာ - အိမ်ရှင် အသာစီးရမှု
                    sim_home_form = 2 # ဥပမာ - ခြေစွမ်းကောင်း
                    sim_away_form = 0
                    
                    pred = calculate_poisson_prediction(sim_home_form, sim_away_form, sim_elo_diff)
                    
                    # (4) ချက်များကို အခြေခံပြီး ခန့်မှန်းချက်ထုတ်ပေးခြင်း
                    st.write(f"• **Elo Ratings ကွာဟချက်:** အိမ်ရှင်က +{sim_elo_diff} အသာစီးရှိ")
                    st.write(f"• **လတ်တလော ခြေစွမ်း (Recent Forms):** အိမ်ရှင် (ကောင်း) vs ဧည့်သည် (သာမန်)")
                    st.write(f"• **Poisson ခန့်မှန်းဂိုးရလဒ်:** **{home_team} {pred['home_goals']} - {pred['away_goals']} {away_team}**")
                    st.write(f"• **အနိုင်ရနိုင်ခြေ (Win %):** 🏠 {home_team} ({pred['home_prob']}%) | 🤝 သရေ ({pred['draw_prob']}%) | ✈️ {away_team} ({pred['away_prob']}%)")
                    
                    # Recommendation summary
                    if pred['home_prob'] > 55:
                        recommendation = f"🔥 **အကြံပြုချက်:** {home_team} ကို ရွေးချယ်သင့်သည် (သို့) ဂိုးပေါ် (Over {over_under_line})"
                    elif pred['away_prob'] > 45:
                        recommendation = f"🔥 **အကြံပြုချက်:** ဧည့်သည် {away_team} ကို ရွေးချယ်သင့်သည်"
                    else:
                        recommendation = f"⚖️ **အကြံပြုချက်:** သရေ (သို့) ဂိုးအောက် ရွေးချယ်ရန် သင့်တော်သည်"
                        
                    st.success(recommendation)
                    
                st.markdown("---")
