import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.express as px
import plotly.graph_objects as go
import requests

# ---------------------------------------------------------
# PAGE CONFIGURATION & CUSTOM STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pro Football Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1E212B;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2E3344;
        text-align: center;
    }
    .big-score {
        font-size: 42px;
        font-weight: 800;
        color: #00E676;
    }
    .team-title {
        font-size: 22px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# TEAM DATABASE (ELO RATINGS & ATTACK/DEFENSE MULTIPLIERS)
# ---------------------------------------------------------
# Elo ~1500 is average top-flight; ~1950+ is elite European contender
TEAM_DATA = {
    "England Premier League": {
        "Arsenal": {"elo": 1955, "att": 1.38, "def": 0.72},
        "Manchester City": {"elo": 1965, "att": 1.45, "def": 0.78},
        "Liverpool": {"elo": 1950, "att": 1.42, "def": 0.76},
        "Chelsea": {"elo": 1860, "att": 1.25, "def": 0.92},
        "Aston Villa": {"elo": 1845, "att": 1.20, "def": 0.95},
        "Newcastle United": {"elo": 1835, "att": 1.18, "def": 0.94},
        "Tottenham Hotspur": {"elo": 1825, "att": 1.24, "def": 1.05},
        "Manchester United": {"elo": 1810, "att": 1.12, "def": 1.02},
        "Brighton": {"elo": 1790, "att": 1.15, "def": 1.04},
        "West Ham": {"elo": 1750, "att": 1.02, "def": 1.12},
    },
    "Spain La Liga": {
        "Real Madrid": {"elo": 1985, "att": 1.48, "def": 0.74},
        "FC Barcelona": {"elo": 1960, "att": 1.50, "def": 0.80},
        "Atletico Madrid": {"elo": 1890, "att": 1.24, "def": 0.78},
        "Athletic Bilbao": {"elo": 1830, "att": 1.14, "def": 0.88},
        "Real Sociedad": {"elo": 1815, "att": 1.08, "def": 0.90},
        "Villarreal": {"elo": 1805, "att": 1.22, "def": 1.08},
        "Real Betis": {"elo": 1785, "att": 1.06, "def": 1.00},
        "Sevilla": {"elo": 1745, "att": 0.98, "def": 1.08},
    },
    "Italy Serie A": {
        "Inter Milan": {"elo": 1945, "att": 1.40, "def": 0.73},
        "Atalanta": {"elo": 1895, "att": 1.36, "def": 0.88},
        "Juventus": {"elo": 1875, "att": 1.16, "def": 0.76},
        "AC Milan": {"elo": 1865, "att": 1.25, "def": 0.92},
        "Napoli": {"elo": 1870, "att": 1.22, "def": 0.85},
        "Lazio": {"elo": 1820, "att": 1.15, "def": 0.96},
        "AS Roma": {"elo": 1810, "att": 1.10, "def": 0.95},
        "Fiorentina": {"elo": 1800, "att": 1.12, "def": 0.96},
    },
    "France Ligue 1": {
        "Paris Saint-Germain": {"elo": 1935, "att": 1.46, "def": 0.80},
        "AS Monaco": {"elo": 1835, "att": 1.24, "def": 0.94},
        "Lille": {"elo": 1820, "att": 1.14, "def": 0.89},
        "Olympique Marseille": {"elo": 1815, "att": 1.20, "def": 0.98},
        "Olympique Lyon": {"elo": 1795, "att": 1.18, "def": 1.04},
        "OGC Nice": {"elo": 1780, "att": 1.04, "def": 0.92},
        "RC Lens": {"elo": 1775, "att": 1.06, "def": 0.94},
    }
}

# Build combined pools for UEFA competitions
ALL_CLUBS = {}
for league, teams in TEAM_DATA.items():
    for team, stats in teams.items():
        ALL_CLUBS[f"{team} ({league.split()[0]})"] = stats

LEAGUES = {
    "England Premier League": TEAM_DATA["England Premier League"],
    "Spain La Liga": TEAM_DATA["Spain La Liga"],
    "Italy Serie A": TEAM_DATA["Italy Serie A"],
    "France Ligue 1": TEAM_DATA["France Ligue 1"],
    "UEFA Champions League": ALL_CLUBS,
    "UEFA Europa League": ALL_CLUBS,
}

LEAGUE_AVG_GOALS = {
    "England Premier League": 2.85,
    "Spain La Liga": 2.60,
    "Italy Serie A": 2.65,
    "France Ligue 1": 2.70,
    "UEFA Champions League": 2.95,
    "UEFA Europa League": 2.80,
}

# ---------------------------------------------------------
# OPTIONAL: LIVE STANDINGS FETCH (FOOTBALL-DATA.ORG V4)
# ---------------------------------------------------------
COMPETITION_CODES = {
    "England Premier League": "PL",
    "Spain La Liga": "PD",
    "Italy Serie A": "SA",
    "France Ligue 1": "FL1",
    "UEFA Champions League": "CL",
}

@st.cache_data(ttl=3600)
def fetch_live_standings(comp_code, api_key):
    url = f"https://api.football-data.org/v4/competitions/{comp_code}/standings"
    headers = {"X-Auth-Token": api_key}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            table = data["standings"][0]["table"]
            rows = []
            for row in table:
                rows.append({
                    "Pos": row["position"],
                    "Team": row["team"]["name"],
                    "Played": row["playedGames"],
                    "W": row["won"],
                    "D": row["draw"],
                    "L": row["lost"],
                    "GF": row["goalsFor"],
                    "GA": row["goalsAgainst"],
                    "GD": row["goalDifference"],
                    "Pts": row["points"]
                })
            return pd.DataFrame(rows)
    except Exception:
        return None
    return None

# ---------------------------------------------------------
# PREDICTION ENGINE (HYBRID ELO + BIVARIATE POISSON)
# ---------------------------------------------------------
def calculate_prediction(home_stats, away_stats, league_avg, home_adv, home_form, away_form, is_neutral=False):
    # 1. Elo Win Expectancy Adjustment
    effective_home_adv = 0 if is_neutral else home_adv
    elo_diff = (home_stats["elo"] + effective_home_adv) - away_stats["elo"]
    elo_factor = 10 ** (elo_diff / 800)  # Damped Elo multiplier for goal scaling

    # 2. Expected Goals (xG) Calculation
    base_goals = league_avg / 2.0
    home_xg = base_goals * home_stats["att"] * away_stats["def"] * (elo_factor ** 0.35) * home_form
    away_xg = base_goals * away_stats["att"] * home_stats["def"] * ((1 / elo_factor) ** 0.35) * away_form

    # Clamp realistic boundaries
    home_xg = float(np.clip(home_xg, 0.25, 4.50))
    away_xg = float(np.clip(away_xg, 0.20, 4.20))

    # 3. Scoreline Probability Matrix (0 to 6 goals)
    max_goals = 7
    h_probs = [poisson.pmf(i, home_xg) for i in range(max_goals)]
    a_probs = [poisson.pmf(i, away_xg) for i in range(max_goals)]
    matrix = np.outer(h_probs, a_probs)

    # Normalize matrix
    matrix = matrix / np.sum(matrix)

    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))

    # Over/Under 2.5 Goals & BTTS (Both Teams To Score)
    over_25 = 0.0
    btts_yes = 0.0
    for h in range(max_goals):
        for a in range(max_goals):
            if (h + a) > 2.5:
                over_25 += matrix[h, a]
            if h > 0 and a > 0:
                btts_yes += matrix[h, a]

    # Top 5 Most Likely Exact Scores
    scores = []
    for h in range(max_goals):
        for a in range(max_goals):
            scores.append((f"{h} - {a}", matrix[h, a] * 100))
    scores.sort(key=lambda x: x[1], reverse=True)

    return {
        "home_xg": round(home_xg, 2),
        "away_xg": round(away_xg, 2),
        "home_win": round(home_win * 100, 1),
        "draw": round(draw * 100, 1),
        "away_win": round(away_win * 100, 1),
        "over_25": round(over_25 * 100, 1),
        "under_25": round((1 - over_25) * 100, 1),
        "btts_yes": round(btts_yes * 100, 1),
        "top_scores": scores[:5],
        "matrix": matrix[:6, :6]
    }

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Model Settings")
    selected_league = st.selectbox("Select Competition", list(LEAGUES.keys()))
    
    st.markdown("---")
    st.subheader("Match Context Adjustments")
    is_neutral = st.checkbox("Neutral Venue (e.g., Cup Final)", value=False)
    home_adv_elo = st.slider("Home Advantage (Elo Points)", 0, 100, 55, disabled=is_neutral)
    
    home_form = st.slider("Home Recent Form Multiplier", 0.80, 1.20, 1.00, 0.02)
    away_form = st.slider("Away Recent Form Multiplier", 0.80, 1.20, 1.00, 0.02)

    st.markdown("---")
    st.subheader("📡 Live API (Optional)")
    api_key = st.text_input("Football-Data.org API Key", type="password", help="Free key from football-data.org for live league tables.")

# ---------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------
st.title(f"⚽ {selected_league} — Match Predictor")
st.caption("Hybrid Elo Rating & Bivariate Poisson Statistical Engine")

teams_dict = LEAGUES[selected_league]
team_names = list(teams_dict.keys())

col1, col_vs, col2 = st.columns([4, 1, 4])

with col1:
    home_team = st.selectbox("🏠 Home Team", team_names, index=0)
    st.caption(f"Elo Rating: **{teams_dict[home_team]['elo']}** | Attack: **{teams_dict[home_team]['att']}** | Defense: **{teams_dict[home_team]['def']}**")

with col_vs:
    st.markdown("<h2 style='text-align: center; padding-top: 25px;'>VS</h2>", unsafe_allow_html=True)

with col2:
    away_team = st.selectbox("✈️ Away Team", team_names, index=1 if len(team_names) > 1 else 0)
    st.caption(f"Elo Rating: **{teams_dict[away_team]['elo']}** | Attack: **{teams_dict[away_team]['att']}** | Defense: **{teams_dict[away_team]['def']}**")

if home_team == away_team:
    st.warning("⚠️ Please select two different teams to run the prediction.")
    st.stop()

# Run Prediction
res = calculate_prediction(
    home_stats=teams_dict[home_team],
    away_stats=teams_dict[away_team],
    league_avg=LEAGUE_AVG_GOALS[selected_league],
    home_adv=home_adv_elo,
    home_form=home_form,
    away_form=away_form,
    is_neutral=is_neutral
)

st.markdown("---")

# ---------------------------------------------------------
# KEY METRICS ROW (1X2 & EXPECTED GOALS)
# ---------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color:#A0AAB8;">Expected Goals (xG)</div>
        <div class="big-score">{res['home_xg']} - {res['away_xg']}</div>
        <div style="font-size:13px; color:#888;">Projected xG Output</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color:#A0AAB8;">{home_team} Win (1)</div>
        <div class="big-score" style="color:#4FC3F7;">{res['home_win']}%</div>
        <div style="font-size:13px; color:#888;">Fair Odds: {round(100/max(res['home_win'],1), 2)}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color:#A0AAB8;">Draw (X)</div>
        <div class="big-score" style="color:#FFD54F;">{res['draw']}%</div>
        <div style="font-size:13px; color:#888;">Fair Odds: {round(100/max(res['draw'],1), 2)}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color:#A0AAB8;">{away_team} Win (2)</div>
        <div class="big-score" style="color:#FF8A65;">{res['away_win']}%</div>
        <div style="font-size:13px; color:#888;">Fair Odds: {round(100/max(res['away_win'],1), 2)}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("###")

# ---------------------------------------------------------
# VISUALIZATIONS & GOAL MARKETS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Scoreline Matrix & Markets", "🎯 Most Likely Scores", "🏆 Live League Table"])

with tab1:
    left_chart, right_stats = st.columns([3, 2])
    
    with left_chart:
        st.subheader("Exact Score Probability Heatmap (%)")
        heatmap_df = pd.DataFrame(
            np.round(res["matrix"] * 100, 1),
            index=[f"{home_team} {i}" for i in range(6)],
            columns=[f"{away_team} {j}" for j in range(6)]
        )
        fig = px.imshow(
            heatmap_df,
            text_auto=True,
            color_continuous_scale="Viridis",
            aspect="auto"
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right_stats:
        st.subheader("Goals & BTTS Markets")
        markets_df = pd.DataFrame({
            "Market": ["Over 2.5 Goals", "Under 2.5 Goals", "Both Teams to Score (Yes)", "Both Teams to Score (No)"],
            "Probability (%)": [res["over_25"], res["under_25"], res["btts_yes"], round(100 - res["btts_yes"], 1)],
            "Fair Decimal Odds": [
                round(100 / max(res["over_25"], 1), 2),
                round(100 / max(res["under_25"], 1), 2),
                round(100 / max(res["btts_yes"], 1), 2),
                round(100 / max(100 - res["btts_yes"], 1), 2),
            ]
        })
        st.dataframe(markets_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Top 5 Projected Scorelines")
    top_df = pd.DataFrame(res["top_scores"], columns=["Scoreline (Home - Away)", "Probability (%)"])
    top_df["Probability (%)"] = top_df["Probability (%)"].round(2)
    
    fig_bar = px.bar(
        top_df,
        x="Scoreline (Home - Away)",
        y="Probability (%)",
        text="Probability (%)",
        color="Probability (%)",
        color_continuous_scale="Tealgrn"
    )
    fig_bar.update_layout(height=350)
    st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    if api_key and selected_league in COMPETITION_CODES:
        standings_df = fetch_live_standings(COMPETITION_CODES[selected_league], api_key)
        if standings_df is not None:
            st.dataframe(standings_df, use_container_width=True, hide_index=True)
        else:
            st.error("Could not fetch standings. Check your API key or rate limit.")
    else:
        st.info("Enter a free API key from **football-data.org** in the sidebar to view live standings for Premier League, La Liga, Serie A, Ligue 1, and Champions League.")
