# Football Analytics V3 (Streamlit)
# Educational statistics dashboard. Does not provide betting picks or stake advice.

import math
import re
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Football Analytics V3", page_icon="⚽", layout="wide")

DB_PATH = "football_v3_history.db"
FD_BASE = "https://api.football-data.org/v4"
LEAGUES = {
    "Premier League": "PL",
    "La Liga": "PD",
    "Bundesliga": "BL1",
    "Serie A": "SA",
    "Ligue 1": "FL1",
    "Champions League": "CL",
}

# -------------------- Database --------------------
def db_init():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS match_history (
            match_id INTEGER PRIMARY KEY,
            competition TEXT,
            kickoff TEXT,
            home_team TEXT,
            away_team TEXT,
            predicted_home INTEGER,
            predicted_away INTEGER,
            actual_home INTEGER,
            actual_away INTEGER,
            saved_at TEXT
        )
        """)
db_init()

def save_prediction(match, score):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        INSERT INTO match_history
        (match_id, competition, kickoff, home_team, away_team,
         predicted_home, predicted_away, saved_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(match_id) DO UPDATE SET
          predicted_home=excluded.predicted_home,
          predicted_away=excluded.predicted_away
        """, (
            match["id"], match.get("competition", {}).get("name", ""),
            match.get("utcDate", ""), match["homeTeam"]["name"],
            match["awayTeam"]["name"], score[0], score[1],
            datetime.now(timezone.utc).isoformat()
        ))

def history_df():
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query("SELECT * FROM match_history ORDER BY kickoff DESC", con)

# -------------------- API --------------------
def fd_headers():
    token = st.session_state.get("fd_token", "").strip()
    return {"X-Auth-Token": token} if token else {}

@st.cache_data(ttl=900, show_spinner=False)
def get_matches(code, token, days=7):
    if not token:
        return []
    start = datetime.now(timezone.utc).date()
    end = start + timedelta(days=days)
    url = f"{FD_BASE}/competitions/{code}/matches"
    try:
        r = requests.get(url, headers={"X-Auth-Token": token},
                         params={"dateFrom": start.isoformat(), "dateTo": end.isoformat()},
                         timeout=20)
        if r.status_code == 200:
            return r.json().get("matches", [])
        return []
    except requests.RequestException:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_standings(code, token):
    if not token:
        return []
    try:
        r = requests.get(f"{FD_BASE}/competitions/{code}/standings",
                         headers={"X-Auth-Token": token}, timeout=20)
        if r.status_code == 200:
            tables = r.json().get("standings", [])
            return tables[0].get("table", []) if tables else []
    except requests.RequestException:
        pass
    return []

@st.cache_data(ttl=1800, show_spinner=False)
def recent_matches(team_id, code, token):
    if not token:
        return []
    today = datetime.now(timezone.utc).date()
    since = today - timedelta(days=120)
    try:
        r = requests.get(f"{FD_BASE}/teams/{team_id}/matches",
            headers={"X-Auth-Token": token},
            params={"competitions": code, "status": "FINISHED",
                    "dateFrom": since.isoformat(), "dateTo": today.isoformat(),
                    "limit": 20}, timeout=20)
        if r.status_code == 200:
            return r.json().get("matches", [])
    except requests.RequestException:
        pass
    return []

# -------------------- Analytics --------------------
def poisson(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)

def score_matrix(lam_home, lam_away, max_goals=8):
    m = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            m[h, a] = poisson(h, lam_home) * poisson(a, lam_away)
    # Normalize truncated matrix so displayed probabilities sum to 100%.
    total = m.sum()
    if total:
        m /= total
    return m

def outcomes(m):
    home = sum(m[h, a] for h in range(m.shape[0]) for a in range(m.shape[1]) if h > a)
    draw = sum(m[h, a] for h in range(m.shape[0]) for a in range(m.shape[1]) if h == a)
    away = sum(m[h, a] for h in range(m.shape[0]) for a in range(m.shape[1]) if h < a)
    total_goals = np.fromfunction(lambda h, a: h + a, m.shape)
    over = {}
    for line in (1.5, 2.5, 3.5):
        over[line] = float(m[total_goals > line].sum())
    btts = float(sum(m[h, a] for h in range(1, m.shape[0]) for a in range(1, m.shape[1])))
    return home, draw, away, over, btts

def recent_form(matches, team_id):
    matches = sorted(matches, key=lambda x: x.get("utcDate", ""), reverse=True)[:5]
    letters, pts, gf, ga = [], 0, 0, 0
    for x in matches:
        h_id = x["homeTeam"]["id"]
        is_home = h_id == team_id
        hs = x["score"]["fullTime"]["home"]
        aws = x["score"]["fullTime"]["away"]
        own, opp = (hs, aws) if is_home else (aws, hs)
        gf += own or 0
        ga += opp or 0
        if own > opp:
            letters.append("W"); pts += 3
        elif own == opp:
            letters.append("D"); pts += 1
        else:
            letters.append("L")
    return "".join(letters) or "N/A", pts, gf, ga, len(matches)

def team_metrics(team, table, token, league_code):
    row = next((r for r in table if r.get("team", {}).get("id") == team["id"]), None)
    base_pts = row.get("points", 0) if row else 0
    gd = row.get("goalDifference", 0) if row else 0
    elo = 1500 + base_pts * 8 + gd * 3
    rec = recent_form(recent_matches(team["id"], league_code, token), team["id"])
    form, pts5, gf5, ga5, n = rec
    avg_for = gf5 / n if n else 1.25
    avg_against = ga5 / n if n else 1.25
    return {"elo": elo, "form": form, "pts5": pts5,
            "gf5": gf5, "ga5": ga5, "avg_for": avg_for,
            "avg_against": avg_against, "played5": n, "table_pts": base_pts}

def estimate_lambdas(home, away):
    # Transparent lightweight estimate; not a guaranteed prediction.
    h_attack = max(0.45, home["avg_for"])
    a_attack = max(0.45, away["avg_for"])
    h_def = max(0.45, home["avg_against"])
    a_def = max(0.45, away["avg_against"])
    elo_delta = (home["elo"] - away["elo"]) / 400
    lh = 0.58 * h_attack + 0.32 * a_def + 0.10 * (1 + elo_delta) + 0.15
    la = 0.58 * a_attack + 0.32 * h_def + 0.10 * (1 - elo_delta)
    return max(0.2, lh), max(0.2, la)

def local_time(utc_text):
    try:
        dt = datetime.fromisoformat(utc_text.replace("Z", "+00:00"))
        return dt.astimezone(timezone(timedelta(hours=6, minutes=30))).strftime("%Y-%m-%d %I:%M %p")
    except Exception:
        return utc_text

def clean_name(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())

# -------------------- UI --------------------
st.title("⚽ Football Analytics V3")
st.caption("ပွဲစဉ်စာရင်း၊ အသင်း Form၊ Statistics နှင့် Score Probability ကို လေ့လာရန်")

with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state["fd_token"] = st.text_input("Football-data.org API Token",
        value=st.session_state.get("fd_token", ""), type="password")
    league_name = st.selectbox("League", list(LEAGUES.keys()))
    league_code = LEAGUES[league_name]
    days = st.slider("လာမည့်ရက်", 1, 14, 7)
    if st.button("🔄 Refresh Data", use_container_width=True):
        get_matches.clear(); get_standings.clear(); recent_matches.clear()
        st.rerun()
    st.info("API Token မရှိပါက Fixtures/Standings ကို မရယူနိုင်ပါ။")

token = st.session_state.get("fd_token", "").strip()
matches = get_matches(league_code, token, days)
table = get_standings(league_code, token)

tabs = st.tabs(["📅 Fixtures & Analysis", "🏆 Standings", "🗂 History"])

with tabs[0]:
    if not token:
        st.warning("Sidebar တွင် Football-data.org API Token ထည့်ပါ။")
    elif not matches:
        st.info("ဒီအချိန်အတွင်း ပွဲစဉ်မတွေ့ပါ။ API Token/League ကို စစ်ဆေးပါ။")
    else:
        st.success(f"{len(matches)} ပွဲစဉ် တွေ့ရှိသည်")
    for match in matches:
        home_team = match.get("homeTeam", {})
        away_team = match.get("awayTeam", {})
        title = f'{home_team.get("name","Home")} vs {away_team.get("name","Away")} | {local_time(match.get("utcDate",""))}'
        with st.expander(title):
            hid, aid = home_team.get("id"), away_team.get("id")
            if not hid or not aid:
                st.warning("အသင်း ID မရရှိပါ။")
                continue
            hm = team_metrics(home_team, table, token, league_code)
            am = team_metrics(away_team, table, token, league_code)
            c1, c2 = st.columns(2)
            with c1:
                st.subheader(home_team.get("name", "Home"))
                st.metric("Elo (အနီးစပ်ဆုံး)", f'{hm["elo"]:.0f}')
                st.write("Last 5:", hm["form"] or "N/A")
                st.write(f'Points (Last 5): {hm["pts5"]}')
                st.write(f'Goals: {hm["gf5"]} scored / {hm["ga5"]} conceded')
            with c2:
                st.subheader(away_team.get("name", "Away"))
                st.metric("Elo (အနီးစပ်ဆုံး)", f'{am["elo"]:.0f}')
                st.write("Last 5:", am["form"] or "N/A")
                st.write(f'Points (Last 5): {am["pts5"]}')
                st.write(f'Goals: {am["gf5"]} scored / {am["ga5"]} conceded')

            lh, la = estimate_lambdas(hm, am)
            matrix = score_matrix(lh, la, 8)
            best = np.unravel_index(np.argmax(matrix), matrix.shape)
            homep, drawp, awayp, overp, bttsp = outcomes(matrix)
            st.markdown("### 📊 Statistical Analysis")
            st.write(f"Expected Goals (model estimate): **{lh:.2f} – {la:.2f}**")
            st.write(f"Most likely score in displayed matrix: **{best[0]} – {best[1]}**")
            st.caption("Most likely exact score is not the same as the most likely match outcome.")
            p1, p2, p3 = st.columns(3)
            p1.metric("Home win", f"{homep*100:.1f}%")
            p2.metric("Draw", f"{drawp*100:.1f}%")
            p3.metric("Away win", f"{awayp*100:.1f}%")
            p4, p5, p6 = st.columns(3)
            p4.metric("Over 1.5 goals", f'{overp[1.5]*100:.1f}%')
            p5.metric("Over 2.5 goals", f'{overp[2.5]*100:.1f}%')
            p6.metric("Over 3.5 goals", f'{overp[3.5]*100:.1f}%')
            st.metric("Both teams score (BTTS)", f"{bttsp*100:.1f}%")
            st.markdown("### Score Probability Matrix")
            matrix_df = pd.DataFrame(matrix * 100,
                index=[f"{i} Home" for i in range(9)],
                columns=[f"{i} Away" for i in range(9)])
            st.dataframe(matrix_df.style.format("{:.1f}%").background_gradient(),
                         use_container_width=True)
            st.caption("Matrix ကို 0–8 ဂိုးအတွင်း normalize လုပ်ထားသည်။ အပြင်ဘက် score များ မပါဝင်ပါ။")
            if match.get("status") == "FINISHED":
                st.info(f'Actual score: {match["score"]["fullTime"]["home"]} – {match["score"]["fullTime"]["away"]}')
            if st.button("💾 Save Statistical Record", key=f'save_{match["id"]}'):
                save_prediction(match, best)
                st.success("History ထဲ သိမ်းဆည်းပြီးပါပြီ။")

with tabs[1]:
    st.subheader(f"🏆 {league_name} Standings")
    if table:
        rows = []
        for r in table:
            t = r.get("team", {})
            rows.append({
                "Pos": r.get("position"), "Team": t.get("name"),
                "Played": r.get("playedGames"), "W": r.get("won"),
                "D": r.get("draw"), "L": r.get("lost"),
                "GF": r.get("goalsFor"), "GA": r.get("goalsAgainst"),
                "GD": r.get("goalDifference"), "Pts": r.get("points")
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Standings မရရှိပါ။ API Token ကို စစ်ဆေးပါ။")

with tabs[2]:
    st.subheader("🗂 Match History")
    hist = history_df()
    if hist.empty:
        st.info("မှတ်တမ်းမရှိသေးပါ။ Fixture မှ Save Statistical Record ကိုနှိပ်ပါ။")
    else:
        st.dataframe(hist, use_container_width=True, hide_index=True)
        done = hist.dropna(subset=["actual_home", "actual_away"])
        if len(done):
            exact = ((done["predicted_home"] == done["actual_home"]) &
                     (done["predicted_away"] == done["actual_away"])).mean()
            st.metric("Exact Score Match Rate (recorded finished games)", f"{exact*100:.1f}%")
        st.download_button("⬇️ Export CSV", hist.to_csv(index=False).encode("utf-8"),
                           "football_v3_history.csv", "text/csv")

st.divider()
st.caption("ဤ Dashboard သည် အားကစားစာရင်းအင်း/ပညာရေးဆိုင်ရာ ခန့်မှန်းချက်များသာဖြစ်ပြီး အာမခံရလဒ် မဟုတ်ပါ။")
