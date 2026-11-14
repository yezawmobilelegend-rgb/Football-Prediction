
from datetime import datetime, timedelta, timezone
import math
import re
import requests
import streamlit as st

# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Professional Football Match Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚽ ဘောလုံးပွဲစဉ်နှင့် ကြေးနှိုင်းယှဉ် ခန့်မှန်းချက် Dashboard")

st.markdown(
    """
    အဓိကလိဂ် (၆) ခု၏ ပွဲစဉ်များ၊ Live Odds API၊
    အသင်း Form၊ Elo Rating နှင့် Poisson Score Prediction
    """
)

# ============================================================
# API CONFIGURATION
# ============================================================

st.sidebar.header("🔑 API & Settings")

default_fb_key = st.secrets.get("FOOTBALL_API_KEY", "")
default_odds_key = st.secrets.get("ODDS_API_KEY", "")

fb_api_key = st.sidebar.text_input(
    "Football-data.org API Key",
    value=default_fb_key,
    type="password",
)

odds_api_key = st.sidebar.text_input(
    "Odds-api.io API Key",
    value=default_odds_key,
    type="password",
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


# ============================================================
# TIME HELPERS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def format_match_time(utc_date):
    try:
        dt = datetime.strptime(
            utc_date,
            "%Y-%m-%dT%H:%M:%SZ",
        ).replace(tzinfo=timezone.utc)

        # Myanmar Time UTC+6:30
        myanmar_time = dt + timedelta(hours=6, minutes=30)

        return myanmar_time.strftime("%Y-%m-%d %H:%M")

    except Exception:
        return utc_date


# ============================================================
# FOOTBALL DATA API
# ============================================================

@st.cache_data(ttl=3600)
def fetch_team_standings(league_code, api_key):

    if not api_key:
        return {}

    url = (
        f"https://api.football-data.org/v4/"
        f"competitions/{league_code}/standings"
    )

    headers = {"X-Auth-Token": api_key}

    team_stats_map = {}

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=15,
        )

        if response.status_code == 200:

            data = response.json()

            for table in data.get("standings", []):

                if table.get("type") == "TOTAL":

                    for row in table.get("table", []):

                        team_name = row["team"]["name"]

                        points = row.get("points", 0)
                        goal_diff = row.get("goalDifference", 0)

                        elo = (
                            1500
                            + (points * 15)
                            + (goal_diff * 5)
                        )

                        team_stats_map[team_name] = {
                            "elo": elo,
                            "form_pts": points,
                        }

        else:
            print("Standings API:", response.status_code)

    except requests.RequestException as e:
        print("Standings Error:", e)

    return team_stats_map


def get_team_stats(team_name, standings_map):

    if team_name in standings_map:

        return (
            standings_map[team_name]["elo"],
            standings_map[team_name]["form_pts"],
        )

    # Standings မရရှိသောအသင်း
    return 1500, 0


# ============================================================
# FETCH FIXTURES
# ============================================================

@st.cache_data(ttl=900)
def fetch_fixtures(league_code, api_key):

    if not api_key:
        return []

    url = (
        f"https://api.football-data.org/v4/"
        f"competitions/{league_code}/matches"
    )

    headers = {"X-Auth-Token": api_key}

    params = {
        "status": "SCHEDULED",
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=15,
        )

        if response.status_code != 200:
            return []

        data = response.json()

        matches = data.get("matches", [])

        now = utc_now()
        seven_days_later = now + timedelta(days=7)

        filtered_matches = []

        for match in matches:

            match_date = datetime.strptime(
                match["utcDate"],
                "%Y-%m-%dT%H:%M:%SZ",
            ).replace(tzinfo=timezone.utc)

            if now <= match_date <= seven_days_later:
                filtered_matches.append(match)

        return filtered_matches

    except Exception as e:

        print("Fixtures Error:", e)
        return []


# ============================================================
# LIVE ODDS API
# ============================================================

@st.cache_data(ttl=300)
def fetch_live_odds_from_api(api_key):

    if not api_key:
        return []

    url = "https://api.odds-api.io/v1/odds"

    params = {
        "apiKey": api_key,
        "sport": "soccer",
        "region": "eu",
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=20,
        )

        if response.status_code == 200:

            data = response.json()

            if isinstance(data, list):
                return data

            if isinstance(data, dict):

                return data.get("data", [])

        else:

            print("Odds API Status:", response.status_code)

    except requests.RequestException as e:

        print("Odds API Error:", e)

    return []


# ============================================================
# TEAM NAME MATCHING
# ============================================================

def normalize_team_name(name):

    name = str(name).lower()

    name = re.sub(
        r"[^a-z0-9]",
        "",
        name,
    )

    return name


def team_names_match(team1, team2):

    if not team1 or not team2:
        return False

    a = normalize_team_name(team1)
    b = normalize_team_name(team2)

    return (
        a == b
        or a in b
        or b in a
    )


# ============================================================
# LIVE ODDS EXTRACTION
# DEFAULT ODDS မရှိပါ
# ============================================================

def get_match_odds(home_team, away_team, odds_list):

    # Default Odds မထည့်ပါ
    result = {
        "home_handicap": None,
        "home_price": None,
        "away_handicap": None,
        "away_price": None,
        "over_line": None,
        "over_price": None,
        "under_line": None,
        "under_price": None,
        "found": False,
    }

    if not odds_list:
        return result

    for item in odds_list:

        h_name = item.get("home_team", "")
        a_name = item.get("away_team", "")

        if not (
            team_names_match(home_team, h_name)
            and team_names_match(away_team, a_name)
        ):
            continue

        bookmakers = item.get("bookmakers", [])

        for bookmaker in bookmakers:

            markets = bookmaker.get("markets", [])

            for market in markets:

                market_key = market.get("key", "")

                outcomes = market.get("outcomes", [])

                # --------------------------------------------
                # HANDICAP / SPREADS
                # --------------------------------------------

                if market_key == "spreads":

                    for outcome in outcomes:

                        outcome_name = outcome.get("name", "")

                        point = outcome.get("point")
                        price = outcome.get("price")

                        if (
                            team_names_match(
                                home_team,
                                outcome_name,
                            )
                        ):

                            result["home_handicap"] = point
                            result["home_price"] = price

                        elif (
                            team_names_match(
                                away_team,
                                outcome_name,
                            )
                        ):

                            result["away_handicap"] = point
                            result["away_price"] = price

                # --------------------------------------------
                # TOTALS / OVER UNDER
                # --------------------------------------------

                elif market_key == "totals":

                    for outcome in outcomes:

                        outcome_name = str(
                            outcome.get("name", "")
                        ).lower()

                        point = outcome.get("point")
                        price = outcome.get("price")

                        if "over" in outcome_name:

                            result["over_line"] = point
                            result["over_price"] = price

                        elif "under" in outcome_name:

                            result["under_line"] = point
                            result["under_price"] = price

        if (
            result["home_price"] is not None
            or result["over_price"] is not None
        ):

            result["found"] = True

            return result

    return result


# ============================================================
# MYANMAR ODDS FORMATTING
# ============================================================

def format_myanmar_price(decimal_odds):

    if decimal_odds is None:
        return None

    try:

        decimal_odds = float(decimal_odds)

        if decimal_odds <= 1:
            return None

        profit = decimal_odds - 1

        price_val = int(round(profit * 100))

        if price_val > 0:
            return f"+{price_val}"

        elif price_val < 0:
            return str(price_val)

        return "ဆုံ"

    except (TypeError, ValueError):

        return None


def format_myanmar_handicap(
    team_name,
    handicap,
    decimal_odds,
):

    if handicap is None or decimal_odds is None:

        return None

    price_str = format_myanmar_price(decimal_odds)

    if price_str is None:
        return None

    try:

        handicap = float(handicap)

        if handicap == 0:
            hdp_str = "0"

        else:
            hdp_str = str(abs(handicap))

        return (
            f"{team_name} "
            f"{hdp_str}{price_str}"
        )

    except (TypeError, ValueError):

        return None


def format_myanmar_ou(line, decimal_odds):

    if line is None or decimal_odds is None:
        return None

    price_str = format_myanmar_price(decimal_odds)

    if price_str is None:
        return None

    return f"{line}{price_str}"


# ============================================================
# POISSON DISTRIBUTION
# ============================================================

def poisson_probability(lam, k):

    return (
        math.exp(-lam)
        * (lam ** k)
        / math.factorial(k)
    )


# ============================================================
# AI MATCH PREDICTION
# ============================================================

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

    # --------------------------------------------
    # EXPECTED GOALS
    # --------------------------------------------

    base_home_lambda = 1.55
    base_away_lambda = 1.05

    elo_diff_factor = (
        home_elo - away_elo
    ) / 250.0

    form_diff = (
        home_form_pts - away_form_pts
    ) / 8.0

    home_lambda = max(
        0.4,
        base_home_lambda
        + (elo_diff_factor * 0.45)
        + (form_diff * 0.25),
    )

    away_lambda = max(
        0.3,
        base_away_lambda
        - (elo_diff_factor * 0.45)
        - (form_diff * 0.25),
    )

    # --------------------------------------------
    # BEST EXACT SCORE
    # --------------------------------------------

    max_goals = 6

    best_score = (0, 0)
    max_prob = -1.0

    for h in range(max_goals + 1):

        for a in range(max_goals + 1):

            prob = (
                poisson_probability(home_lambda, h)
                * poisson_probability(away_lambda, a)
            )

            if prob > max_prob:

                max_prob = prob
                best_score = (h, a)

    # --------------------------------------------
    # BEST SCORE TOTAL GOALS
    # --------------------------------------------

    predicted_total_goals = (
        best_score[0] + best_score[1]
    )

    # --------------------------------------------
    # OVER / UNDER BASED ON BEST SCORE
    # --------------------------------------------

    default_ou_line = 2.5

    if predicted_total_goals > default_ou_line:

        ou_recommendation = (
            f"အထက် (Over {default_ou_line}) "
            f"[မှန်းဆဂိုး: "
            f"{best_score[0]}-{best_score[1]} | "
            f"ဂိုးပေါင်း: {predicted_total_goals}]"
        )

    else:

        ou_recommendation = (
            f"အောက် (Under {default_ou_line}) "
            f"[မှန်းဆဂိုး: "
            f"{best_score[0]}-{best_score[1]} | "
            f"ဂိုးပေါင်း: {predicted_total_goals}]"
        )

    # --------------------------------------------
    # LOCAL O/U INPUT
    # --------------------------------------------

    if local_ou_input:

        nums = re.findall(
            r"\d+\.?\d*",
            local_ou_input,
        )

        if nums:

            user_ou_line = float(nums[0])

            if predicted_total_goals > user_ou_line:

                ou_recommendation = (
                    f"ဒေသတွင်း O/U Line "
                    f"({local_ou_input}) နှင့် "
                    f"မှန်းဆဂိုး ({predicted_total_goals}) "
                    f"အရ 👉 Over {user_ou_line}"
                )

            elif predicted_total_goals < user_ou_line:

                ou_recommendation = (
                    f"ဒေသတွင်း O/U Line "
                    f"({local_ou_input}) နှင့် "
                    f"မှန်းဆဂိုး ({predicted_total_goals}) "
                    f"အရ 👉 Under {user_ou_line}"
                )

            else:

                ou_recommendation = (
                    f"မှန်းဆဂိုးပေါင်း ({predicted_total_goals}) "
                    f"နှင့် Line ({user_ou_line}) "
                    f"တူညီနေသည်"
                )

    # --------------------------------------------
    # HOME / AWAY PREDICTION
    # --------------------------------------------

    if home_lambda > away_lambda + 0.3:

        winner_rec = (
            f"AI တွက်ချက်မှုအရ {home_team} "
            f"အသာစီးရှိသည်"
        )

    elif away_lambda > home_lambda + 0.2:

        winner_rec = (
            f"AI တွက်ချက်မှုအရ {away_team} "
            f"အသာစီးရှိသည်"
        )

    else:

        winner_rec = (
            "နှစ်သင်း အားသာချက်နီးစပ်၍ "
            "သရေဖြစ်နိုင်ခြေရှိသည်"
        )

    return (
        best_score,
        ou_recommendation,
        winner_rec,
        home_lambda,
        away_lambda,
        max_prob,
        predicted_total_goals,
    )


# ============================================================
# MAIN DASHBOARD
# ============================================================

st.subheader(
    f"📅 {selected_league_name} "
    f"- လာမည့် ၇ ရက်အတွင်း ပွဲစဉ်များ"
)

matches = fetch_fixtures(
    current_league_code,
    fb_api_key,
)

live_odds_data = fetch_live_odds_from_api(
    odds_api_key,
)

team_standings_map = fetch_team_standings(
    current_league_code,
    fb_api_key,
)


# ============================================================
# STATUS DISPLAY
# ============================================================

if not fb_api_key:

    st.warning(
        "Football-data.org API Key ထည့်သွင်းရန် လိုအပ်ပါသည်။"
    )

elif not matches:

    st.warning(
        "လာမည့် (၇) ရက်အတွင်း ပွဲစဉ်မရှိပါ "
        "သို့မဟုတ် API မှ ပွဲစဉ်မရရှိပါ။"
    )

else:

    st.success(
        f"ပွဲစဉ်စုစုပေါင်း "
        f"({len(matches)} ပွဲ) တွေ့ရှိရပါသည်။"
    )


# ============================================================
# MATCH DISPLAY
# ============================================================

for i, match in enumerate(matches):

    home_team = match["homeTeam"]["name"]
    away_team = match["awayTeam"]["name"]

    match_time = match["utcDate"]

    myanmar_match_time = format_match_time(
        match_time
    )

    with st.expander(
        f"🏟️ {home_team} vs {away_team} "
        f"| 🕒 {myanmar_match_time} (Myanmar Time)"
    ):

        col1, col2 = st.columns(2)

        # ====================================================
        # GLOBAL ODDS
        # ====================================================

        with col1:

            st.markdown(
                "### 🌐 ကမ္ဘာ့ပေါက်ကြေး (Live API)"
            )

            odds = get_match_odds(
                home_team,
                away_team,
                live_odds_data,
            )

            # --------------------------------------------
            # NO LIVE ODDS
            # --------------------------------------------

            if not odds["found"]:

                st.info(
                    "🌐 ကမ္ဘာ့ကြေးများ မထွက်သေးပါ"
                )

            else:

                # --------------------------------------------
                # HANDICAP
                # --------------------------------------------

                global_home_hdp = format_myanmar_handicap(
                    home_team,
                    odds["home_handicap"],
                    odds["home_price"],
                )

                global_away_hdp = format_myanmar_handicap(
                    away_team,
                    odds["away_handicap"],
                    odds["away_price"],
                )

                st.markdown(
                    "#### ⚽ Handicap"
                )

                if global_home_hdp:

                    st.success(
                        f"🏠 {global_home_hdp}"
                    )

                if global_away_hdp:

                    st.success(
                        f"✈️ {global_away_hdp}"
                    )

                if (
                    global_home_hdp is None
                    and global_away_hdp is None
                ):

                    st.info(
                        "Handicap ကြေး မရရှိသေးပါ"
                    )

                # --------------------------------------------
                # OVER / UNDER
                # --------------------------------------------

                global_over = format_myanmar_ou(
                    odds["over_line"],
                    odds["over_price"],
                )

                global_under = format_myanmar_ou(
                    odds["under_line"],
                    odds["under_price"],
                )

                st.markdown(
                    "#### ⚽ ဂိုးပေါင်း (O/U)"
                )

                if global_over:

                    st.success(
                        f"Over: {global_over}"
                    )

                if global_under:

                    st.success(
                        f"Under: {global_under}"
                    )

                if (
                    global_over is None
                    and global_under is None
                ):

                    st.info(
                        "ဂိုးပေါင်းကြေး မရရှိသေးပါ"
                    )

            # --------------------------------------------
            # LOCAL ODDS INPUT
            # --------------------------------------------

            st.markdown(
                "### 📝 ဒေသတွင်းကြေး ထည့်သွင်းရန်"
            )

            local_hdp_input = st.text_input(
                "ဒေသတွင်း Handicap",
                key=f"local_hdp_{i}",
                placeholder=f"{home_team} 1+20",
            )

            local_ou_input = st.text_input(
                "ဒေသတွင်း O/U",
                key=f"local_ou_{i}",
                placeholder="2.5",
            )

        # ====================================================
        # AI ANALYSIS
        # ====================================================

        with col2:

            st.markdown(
                "### 🤖 AI Match Analysis"
            )

            home_elo, home_form_pts = get_team_stats(
                home_team,
                team_standings_map,
            )

            away_elo, away_form_pts = get_team_stats(
                away_team,
                team_standings_map,
            )

            (
                best_score,
                ou_rec,
                winner_rec,
                h_lam,
                a_lam,
                score_probability,
                predicted_total_goals,
            ) = calculate_match_prediction(
                home_team,
                away_team,
                home_elo,
                away_elo,
                home_form_pts,
                away_form_pts,
                local_hdp_input,
                local_ou_input,
            )

            # --------------------------------------------
            # ELO
            # --------------------------------------------

            st.markdown(
                f"**Elo Rating:** "
                f"{home_team} ({home_elo:.0f}) "
                f"vs {away_team} ({away_elo:.0f})"
            )

            # --------------------------------------------
            # FORM POINTS
            # --------------------------------------------

            st.markdown(
                f"**League Points:** "
                f"{home_team} ({home_form_pts}) "
                f"| {away_team} ({away_form_pts})"
            )

            # --------------------------------------------
            # EXPECTED GOALS
            # --------------------------------------------

            st.markdown(
                f"**Expected Goals:** "
                f"{home_team} ({h_lam:.2f}) "
                f"- {away_team} ({a_lam:.2f})"
            )

            st.markdown(
                f"**Total Expected Goals:** "
                f"{h_lam + a_lam:.2f}"
            )

            # --------------------------------------------
            # BEST SCORE
            # --------------------------------------------

            st.divider()

            st.markdown(
                "#### 🎯 အနီးစပ်ဆုံး ခန့်မှန်းရလဒ်"
            )

            st.success(
                f"⚽ **{home_team} "
                f"{best_score[0]} - {best_score[1]} "
                f"{away_team}**"
            )

            st.markdown(
                f"**မှန်းဆဂိုးပေါင်း:** "
                f"{predicted_total_goals} ဂိုး"
            )

            st.markdown(
                f"**Best Score Probability:** "
                f"{score_probability * 100:.2f}%"
            )

            st.markdown(
                f"**ဂိုးပေါင်းအခြေအနေ:** {ou_rec}"
            )

            st.markdown(
                f"**အသင်းအားသာချက်:** {winner_rec}"
            )

            # --------------------------------------------
            # SCORE DETAILS
            # --------------------------------------------

            st.caption(
                "မှတ်ချက် - Exact Score သည် Poisson "
                "Distribution အရ အဖြစ်နိုင်ဆုံး "
                "တစ်ကွက်တည်းကို ဖော်ပြထားခြင်းဖြစ်သည်။"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Professional Football Match Predictor | "
    "Football-data.org + Odds API + Poisson Model"
)

st.caption(
    "API မှ မရရှိသော Odds များကို Default ကြေးဖြင့် "
    "အစားထိုးမပြသပါ။"
)
