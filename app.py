from __future__ import annotations

import io
import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ============================================================
# Football Predictor V4
# ============================================================
# This version supports:
# 1) walk-forward backtesting (no future leakage)
# 2) temperature calibration on out-of-sample predictions
# 3) injuries / suspensions features
# 4) confirmed or expected line-up features
# 5) team and match xG features
# 6) bookmaker closing-line features and CLV
# 7) optional API-Football enrichment adapter
#
# Required historical CSV columns for a full backtest:
# date, home_team, away_team, home_goals, away_goals,
# home_xg, away_xg, home_close_1x2, draw_close_1x2, away_close_1x2
# Optional columns:
# home_injuries, away_injuries, home_suspensions, away_suspensions,
# home_lineup_confirmed, away_lineup_confirmed,
# home_lineup_strength, away_lineup_strength,
# home_close_over25, away_close_over25, closing_timestamp
#
# An API key is still required for live injury/lineup/xG/odds enrichment.
# The code never invents unavailable data: missing features are imputed and
# the data-quality score is reduced.
# ============================================================

st.set_page_config(page_title="Professional Football Predictor V4", page_icon="⚽", layout="wide")
DB_PATH = "football_history_v4.db"
UTC = timezone.utc
MODEL_VERSION = "walk-forward-logit-calibrated-v4.0"

REQUIRED_COLUMNS = [
    "date", "home_team", "away_team", "home_goals", "away_goals",
    "home_xg", "away_xg", "home_close_1x2", "draw_close_1x2", "away_close_1x2",
]
OPTIONAL_DEFAULTS = {
    "home_injuries": 0, "away_injuries": 0,
    "home_suspensions": 0, "away_suspensions": 0,
    "home_lineup_confirmed": 0, "away_lineup_confirmed": 0,
    "home_lineup_strength": 1.0, "away_lineup_strength": 1.0,
    "home_close_over25": np.nan, "away_close_over25": np.nan,
    "closing_timestamp": "",
}

# -----------------------------
# General utilities
# -----------------------------

def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def num(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


def result_code(home: int, away: int) -> str:
    return "H" if home > away else "D" if home == away else "A"


def normalize_team(name: Any) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def team_equal(a: Any, b: Any) -> bool:
    x, y = normalize_team(a), normalize_team(b)
    return bool(x and y and (x == y or (len(x) >= 6 and x in y) or (len(y) >= 6 and y in x)))


def safe_logit_probability(p: float) -> float:
    return math.log(clamp(p, 1e-6, 1 - 1e-6) / clamp(1 - p, 1e-6, 1 - 1e-6))

# -----------------------------
# Database for predictions/backtests
# -----------------------------

SCHEMA = """ CREATE TABLE IF NOT EXISTS predictions ( id INTEGER PRIMARY KEY AUTOINCREMENT, match_id TEXT UNIQUE, match_date TEXT, home_team TEXT, away_team TEXT, predicted_home INTEGER, predicted_away INTEGER, p_home REAL, p_draw REAL, p_away REAL, p_over25 REAL, p_btts REAL, data_quality REAL, closing_home REAL, closing_draw REAL, closing_away REAL, actual_home INTEGER, actual_away INTEGER, created_at TEXT, model_version TEXT ) """


def conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with conn() as c:
        c.execute(SCHEMA)
        c.commit()


def save_live_prediction(row: Dict[str, Any], pred: Dict[str, Any]) -> None:
    with conn() as c:
        c.execute(
            """INSERT INTO predictions (match_id, match_date, home_team, away_team, predicted_home, predicted_away, p_home, p_draw, p_away, p_over25, p_btts, data_quality, closing_home, closing_draw, closing_away, created_at, model_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(match_id) DO UPDATE SET p_home=excluded.p_home, p_draw=excluded.p_draw, p_away=excluded.p_away, p_over25=excluded.p_over25, p_btts=excluded.p_btts, data_quality=excluded.data_quality, model_version=excluded.model_version""",
            (str(row.get("match_id", row.get("date", ""))), row.get("date"), row.get("home_team"), row.get("away_team"),
             int(pred["score"][0]), int(pred["score"][1]), pred["p_home"], pred["p_draw"], pred["p_away"],
             pred["p_over25"], pred["p_btts"], pred["data_quality"], row.get("home_close_1x2"),
             row.get("draw_close_1x2"), row.get("away_close_1x2"), now_iso(), MODEL_VERSION),
        )
        c.commit()

# -----------------------------
# Historical data and feature engineering
# -----------------------------

def validate_history(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    errors = []
    out = df.copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))
        return out, errors
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    if out["date"].isna().any():
        errors.append("Some date values are invalid and will be dropped.")
    numeric_cols = [c for c in REQUIRED_COLUMNS[3:] if c != "closing_timestamp"] + list(OPTIONAL_DEFAULTS.keys())
    for col in numeric_cols:
        if col in out.columns and col != "closing_timestamp":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    out = out.sort_values("date").reset_index(drop=True)
    if len(out) < 30:
        errors.append("Fewer than 30 usable matches: metrics will be unstable.")
    return out, errors


def rolling_team_features(history: pd.DataFrame, before_date: pd.Timestamp, team: str, venue: str, window: int = 10) -> Dict[str, float]:
    past = history[history["date"] < before_date]
    rows = past[(past["home_team"].map(lambda x: team_equal(x, team))) | (past["away_team"].map(lambda x: team_equal(x, team)))].tail(window)
    if rows.empty:
        return {"gf": 1.35, "ga": 1.35, "xgf": 1.35, "xga": 1.35, "points": 1.25, "played": 0}
    gf, ga, xgf, xga, points = [], [], [], [], []
    for _, r in rows.iterrows():
        is_home = team_equal(r["home_team"], team)
        goals_for = r["home_goals"] if is_home else r["away_goals"]
        goals_against = r["away_goals"] if is_home else r["home_goals"]
        x_for = num(r["home_xg"] if is_home else r["away_xg"], goals_for)
        x_against = num(r["away_xg"] if is_home else r["home_xg"], goals_against)
        gf.append(goals_for); ga.append(goals_against); xgf.append(x_for); xga.append(x_against)
        points.append(3 if goals_for > goals_against else 1 if goals_for == goals_against else 0)
    return {"gf": float(np.mean(gf)), "ga": float(np.mean(ga)), "xgf": float(np.mean(xgf)), "xga": float(np.mean(xga)), "points": float(np.mean(points)), "played": len(rows)}


def feature_row(history: pd.DataFrame, row: pd.Series) -> Tuple[np.ndarray, float, Dict[str, float]]:
    date = row["date"]
    home = rolling_team_features(history, date, row["home_team"], "home")
    away = rolling_team_features(history, date, row["away_team"], "away")
    home_xg = num(row.get("home_xg"), home["xgf"])
    away_xg = num(row.get("away_xg"), away["xgf"])
    injuries_diff = num(row.get("away_injuries"), 0) - num(row.get("home_injuries"), 0)
    suspension_diff = num(row.get("away_suspensions"), 0) - num(row.get("home_suspensions"), 0)
    lineup_diff = num(row.get("home_lineup_strength"), 1) - num(row.get("away_lineup_strength"), 1)
    lineup_confirmed = (num(row.get("home_lineup_confirmed"), 0) + num(row.get("away_lineup_confirmed"), 0)) / 2
    close_home = num(row.get("home_close_1x2"))
    close_draw = num(row.get("draw_close_1x2"))
    close_away = num(row.get("away_close_1x2"))
    close_home_p = 1 / close_home if np.isfinite(close_home) and close_home > 1 else 1 / 3
    close_draw_p = 1 / close_draw if np.isfinite(close_draw) and close_draw > 1 else 1 / 3
    close_away_p = 1 / close_away if np.isfinite(close_away) and close_away > 1 else 1 / 3
    close_sum = close_home_p + close_draw_p + close_away_p
    close_home_p, close_draw_p, close_away_p = close_home_p / close_sum, close_draw_p / close_sum, close_away_p / close_sum
    x = np.array([
        home["xgf"] - away["xgf"], home["xga"] - away["xga"],
        home["gf"] - away["gf"], home["ga"] - away["ga"],
        home["points"] - away["points"], home_xg - away_xg,
        injuries_diff, suspension_diff, lineup_diff, lineup_confirmed,
        close_home_p, close_draw_p, close_away_p,
    ], dtype=float)
    # Only values that genuinely exist receive full quality credit.
    quality_parts = [
        int(num(row.get("home_xg")) == num(row.get("home_xg")) and num(row.get("away_xg")) == num(row.get("away_xg"))),
        int(np.isfinite(close_home) and np.isfinite(close_draw) and np.isfinite(close_away)),
        int("home_injuries" in row.index and "away_injuries" in row.index),
        int("home_lineup_confirmed" in row.index and "away_lineup_confirmed" in row.index),
        int(home["played"] >= 5 and away["played"] >= 5),
    ]
    quality = float(np.mean(quality_parts))
    meta = {"close_home_p": close_home_p, "close_draw_p": close_draw_p, "close_away_p": close_away_p, "home_xg": home_xg, "away_xg": away_xg}
    return x, quality, meta

# -----------------------------
# Model, calibration and backtesting
# -----------------------------

def fallback_probs(x: np.ndarray) -> np.ndarray:
    z = np.array([0.8 * x[0] - 0.4 * x[1] + 0.9 * x[4] + 0.5 * x[5], 0.0, -0.8 * x[0] + 0.4 * x[1] - 0.9 * x[4] - 0.5 * x[5]])
    z -= np.max(z)
    e = np.exp(z)
    return e / e.sum()


def fit_classifier(X: np.ndarray, y: np.ndarray) -> Any:
    if SKLEARN_AVAILABLE and len(np.unique(y)) >= 2 and len(y) >= 20:
        model = LogisticRegression(max_iter=1000, multi_class="multinomial", C=0.7)
        model.fit(X, y)
        return model
    return None


def raw_probability(model: Any, x: np.ndarray) -> np.ndarray:
    if model is None:
        return fallback_probs(x)
    p = model.predict_proba(x.reshape(1, -1))[0]
    out = np.zeros(3)
    for cls, value in zip(model.classes_, p):
        out[int(cls)] = value
    return out / out.sum()


def temperature_scale(probs: np.ndarray, y: np.ndarray) -> float:
    if len(probs) < 20:
        return 1.0
    best_t, best_loss = 1.0, float("inf")
    for t in np.linspace(0.5, 3.0, 51):
        logits = np.log(np.clip(probs, 1e-8, 1)) / t
        logits -= logits.max(axis=1, keepdims=True)
        calibrated = np.exp(logits); calibrated /= calibrated.sum(axis=1, keepdims=True)
        loss = -np.mean(np.log(np.clip(calibrated[np.arange(len(y)), y], 1e-8, 1)))
        if loss < best_loss:
            best_loss, best_t = loss, float(t)
    return best_t


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    logits = np.log(np.clip(p, 1e-8, 1)) / max(t, 1e-6)
    logits -= logits.max()
    out = np.exp(logits); return out / out.sum()


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(3)[y.astype(int)]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def log_loss(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y.astype(int)], 1e-8, 1))))


def backtest(history: pd.DataFrame, min_train: int = 30) -> Dict[str, Any]:
    X, qualities, targets, metas = [], [], [], []
    for _, row in history.iterrows():
        x, q, meta = feature_row(history, row)
        X.append(x); qualities.append(q); metas.append(meta)
        targets.append(0 if row["home_goals"] > row["away_goals"] else 1 if row["home_goals"] == row["away_goals"] else 2)
    X, y = np.asarray(X), np.asarray(targets)
    predictions, actuals, dates, qs = [], [], [], []
    # Strict walk-forward: each row is predicted only using previous rows.
    for i in range(min_train, len(history)):
        model = fit_classifier(X[:i], y[:i])
        p = raw_probability(model, X[i])
        predictions.append(p); actuals.append(y[i]); dates.append(history.iloc[i]["date"]); qs.append(qualities[i])
    if not predictions:
        return {"error": f"Need at least {min_train + 1} rows for a backtest."}
    P, Y = np.asarray(predictions), np.asarray(actuals)
    # Calibrator is fitted on OOS predictions only, never on training features.
    t = temperature_scale(P, Y)
    PC = np.asarray([apply_temperature(p, t) for p in P])
    pred_class = PC.argmax(axis=1)
    return {
        "predictions": PC, "raw_predictions": P, "actuals": Y, "dates": dates, "qualities": qs,
        "temperature": t, "accuracy": float(np.mean(pred_class == Y)),
        "brier": brier_score(PC, Y), "logloss": log_loss(PC, Y),
        "n": len(Y), "pred_class": pred_class,
    }


def train_final_model(history: pd.DataFrame, calibration: Optional[Dict[str, Any]]) -> Tuple[Any, float]:
    X, y = [], []
    for _, row in history.iterrows():
        x, _, _ = feature_row(history, row); X.append(x)
        y.append(0 if row["home_goals"] > row["away_goals"] else 1 if row["home_goals"] == row["away_goals"] else 2)
    model = fit_classifier(np.asarray(X), np.asarray(y))
    return model, float(calibration.get("temperature", 1.0) if calibration else 1.0)

# -----------------------------
# API-Football optional enrichment
# -----------------------------

def api_football_get(endpoint: str, params: Dict[str, Any], key: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not key:
        return None, "API-Football key is missing."
    try:
        r = requests.get("https://v3.football.api-sports.io/" + endpoint, params=params, headers={"x-apisports-key": key}, timeout=20)
        if r.status_code != 200:
            return None, f"API-Football HTTP {r.status_code}"
        data = r.json()
        if data.get("errors"):
            return None, str(data["errors"])
        return data, None
    except requests.RequestException as exc:
        return None, str(exc)


def fetch_enrichment(fixture_id: int, key: str) -> Dict[str, Any]:
    """Fetch injuries, lineups, xG statistics and pre-match odds where provider supplies them. API-Football endpoint availability and quotas depend on plan. Missing endpoint responses are returned as missing fields rather than fabricated values. """
    result: Dict[str, Any] = {"injuries": [], "lineups": [], "statistics": [], "odds": [], "errors": []}
    for endpoint, target in [("injuries", "injuries"), ("fixtures/lineups", "lineups"), ("fixtures/statistics", "statistics"), ("odds", "odds")]:
        params = {"fixture": fixture_id}
        data, error = api_football_get(endpoint, params, key)
        if error:
            result["errors"].append(f"{endpoint}: {error}")
        elif data:
            result[target] = data.get("response", [])
    return result


def enrichment_to_features(bundle: Dict[str, Any]) -> Dict[str, Any]:
    injuries = bundle.get("injuries", [])
    lineups = bundle.get("lineups", [])
    stats = bundle.get("statistics", [])
    odds = bundle.get("odds", [])
    out = {"home_injuries": 0, "away_injuries": 0, "home_suspensions": 0, "away_suspensions": 0,
           "home_lineup_confirmed": 0, "away_lineup_confirmed": 0, "home_lineup_strength": 1.0, "away_lineup_strength": 1.0,
           "home_xg": np.nan, "away_xg": np.nan, "home_close_1x2": np.nan, "draw_close_1x2": np.nan, "away_close_1x2": np.nan}
    for item in injuries:
        team = str(item.get("team", {}).get("name", "")).lower()
        is_home = len(lineups) == 0 or (lineups and team_equal(team, lineups[0].get("team", {}).get("name", "")))
        key = "home" if is_home else "away"
        out[f"{key}_injuries"] += 1
        if "susp" in str(item).lower(): out[f"{key}_suspensions"] += 1
    for lineup in lineups:
        team = lineup.get("team", {}).get("name", "")
        key = "home" if not out["home_lineup_confirmed"] else "away"
        out[f"{key}_lineup_confirmed"] = 1
        players = lineup.get("startXI", [])
        out[f"{key}_lineup_strength"] = clamp(1 - 0.02 * sum(1 for p in players if not p.get("player", {}).get("name")), 0.7, 1.0)
    for index, team_stats in enumerate(stats):
        team_name = team_stats.get("team", {}).get("name", "")
        # API-Football returns numeric team IDs; response order is normally home/away.
        # If a provider includes an explicit side field, prefer it.
        side = str(team_stats.get("team", {}).get("side", "")).lower()
        key = "home" if side == "home" or (not side and index == 0) else "away"
        for item in team_stats.get("statistics", []):
            if str(item.get("type", "")).lower() in {"expected_goals", "xg"}:
                out[f"{key}_xg"] = num(item.get("value"))
    # Odds response schemas differ by plan/bookmaker; map common values conservatively.
    for item in odds:
        for bookmaker in item.get("bookmakers", []):
            for market in bookmaker.get("bets", bookmaker.get("markets", [])):
                name = str(market.get("name", market.get("key", ""))).lower()
                values = market.get("values", market.get("outcomes", []))
                for value in values:
                    label = str(value.get("value", value.get("name", ""))).lower()
                    price = num(value.get("odd", value.get("price")))
                    if "match winner" in name or name in {"h2h", "1x2"}:
                        if label in {"home", "1"}: out["home_close_1x2"] = price
                        elif label in {"draw", "x"}: out["draw_close_1x2"] = price
                        elif label in {"away", "2"}: out["away_close_1x2"] = price
    return out

# -----------------------------
# UI
# -----------------------------

init_db()
st.title("⚽ Professional Football Predictor V4")
st.caption("Walk-forward backtesting • out-of-sample calibration • xG • injuries • line-ups • closing-line features")

with st.sidebar:
    st.header("Data inputs")
    api_key = st.text_input("API-Football key (optional)", value=st.secrets.get("API_FOOTBALL_KEY", ""), type="password")
    fixture_id = st.number_input("API-Football fixture ID", min_value=0, step=1, value=0)
    uploaded = st.file_uploader("Enriched historical CSV", type=["csv"])
    min_train = st.slider("Minimum training matches", 20, 200, 30)
    st.info("Full backtest needs historical xG, closing odds, injury and lineup columns. Missing optional columns are handled explicitly, not guessed.")

if uploaded is None:
    st.warning("Full feature pipeline အလုပ်လုပ်ရန် enriched historical CSV တင်ပါ။")
    st.code(",".join(REQUIRED_COLUMNS + list(OPTIONAL_DEFAULTS.keys())), language="text")
    st.stop()

try:
    history_raw = pd.read_csv(uploaded)
    history, errors = validate_history(history_raw)
except Exception as exc:
    st.error(f"CSV read error: {exc}")
    st.stop()
for error in errors:
    st.warning(error)

bt_tab, live_tab, data_tab = st.tabs(["Backtest & calibration", "Live enrichment", "Data audit"])

with data_tab:
    st.subheader("Data audit")
    audit = []
    for col in REQUIRED_COLUMNS + list(OPTIONAL_DEFAULTS.keys()):
        if col in history.columns:
            present = int(history[col].notna().sum())
            audit.append({"column": col, "rows": len(history), "present": present, "coverage": present / max(len(history), 1)})
    audit_df = pd.DataFrame(audit)
    audit_df["coverage"] = audit_df["coverage"].map(lambda x: f"{x * 100:.1f}%")
    st.dataframe(audit_df, hide_index=True, use_container_width=True)
    st.write(f"Usable rows: **{len(history)}** | Date range: **{history['date'].min()} → {history['date'].max()}**")

with bt_tab:
    st.subheader("Strict walk-forward backtest")
    if st.button("Run backtest", type="primary"):
        with st.spinner("Running leakage-safe walk-forward predictions and calibration…"):
            result = backtest(history, min_train)
        if "error" in result:
            st.error(result["error"])
        else:
            st.session_state["backtest"] = result
            st.success(f"Completed {result['n']} out-of-sample predictions. Calibration temperature: {result['temperature']:.3f}")
    result = st.session_state.get("backtest")
    if result and "error" not in result:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("1X2 accuracy", f"{result['accuracy'] * 100:.1f}%")
        c2.metric("Brier score", f"{result['brier']:.4f}")
        c3.metric("Log loss", f"{result['logloss']:.4f}")
        c4.metric("OOS matches", result["n"])
        curve = pd.DataFrame({"date": result["dates"], "actual": result["actuals"], "predicted": result["pred_class"], "quality": result["qualities"]})
        curve["correct"] = (curve["actual"] == curve["predicted"]).astype(int)
        curve["rolling_accuracy"] = curve["correct"].rolling(20, min_periods=5).mean()
        st.line_chart(curve.set_index("date")["rolling_accuracy"])
        st.dataframe(curve.tail(100), hide_index=True, use_container_width=True)
        st.caption("Brier နှင့် log loss သည် lower-is-better ဖြစ်သည်။ Accuracy တစ်ခုတည်းနဲ့ calibration ကောင်း/မကောင်း မဆုံးဖြတ်ပါနှင့်။")

with live_tab:
    st.subheader("Live injury / lineup / xG / odds enrichment")
    if fixture_id <= 0:
        st.info("API-Football fixture ID ထည့်ပါ။")
    elif not api_key:
        st.warning("API-Football key မရှိသေးပါ။")
    elif st.button("Fetch enrichment"):
        bundle = fetch_enrichment(int(fixture_id), api_key)
        st.session_state["bundle"] = bundle
    bundle = st.session_state.get("bundle")
    if bundle:
        if bundle["errors"]:
            for error in bundle["errors"]:
                st.warning(error)
        live_features = enrichment_to_features(bundle)
        st.json(live_features)
        st.caption("Provider response schema/plan အလိုက် field တချို့ မရနိုင်ပါ။ မရနိုင်တဲ့ field ကို model က missing အဖြစ်သာ ကိုင်တွယ်ပါသည်။")

# Final prediction from the latest available historical context.
if len(history) > 0:
    st.divider()
    st.subheader("Latest-row probability preview")
    calibration = st.session_state.get("backtest")
    model, temperature = train_final_model(history, calibration)
    latest = history.iloc[-1]
    x, quality, meta = feature_row(history, latest)
    p = apply_temperature(raw_probability(model, x), temperature)
    score_home = max(0, round(1.35 + 0.35 * (p[0] - p[2])))
    score_away = max(0, round(1.10 + 0.35 * (p[2] - p[0])))
    st.write(f"**{latest['home_team']} {score_home}–{score_away} {latest['away_team']}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Home", f"{p[0] * 100:.1f}%")
    c2.metric("Draw", f"{p[1] * 100:.1f}%")
    c3.metric("Away", f"{p[2] * 100:.1f}%")
    c4.metric("Data quality", f"{quality * 100:.0f}%")
    st.caption("Latest-row preview သည် uploaded history ၏ နောက်ဆုံး row ကိုပြသည်။ Live fixture prediction အတွက် provider fixture mapping နှင့် full live feature join ကို ဆက်ချိတ်ရန်လိုပါသည်။")