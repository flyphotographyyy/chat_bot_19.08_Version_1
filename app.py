# app.py  — Streamlit Watchlist Swing-Assistant + Supabase auth (username+password)
# -------------------------------------------------------------------------------
# Запазен UI/логика; поправки:
# - st.experimental_rerun -> st.rerun (Streamlit 1.30+)
# - Login/Register през st.form + form_submit_button (фиксира „натисни 2 пъти“)

from __future__ import annotations
import json, os, time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional deps
try:
    import plotly.graph_objects as go
except Exception:
    go = None

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOR = True
except Exception:
    HAS_AUTOR = False

# --- Supabase (free, no disk needed) ---
from slugify import slugify
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "").strip()
sb: Client | None = create_client(SUPABASE_URL, SUPABASE_ANON_KEY) if (SUPABASE_URL and SUPABASE_ANON_KEY) else None

# ---------------- Config ----------------
APP_TITLE = "📈 Watchlist Swing-Assistant"
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]
DEFAULT_PROFILE = "Balanced"
CACHE_TTL_SECONDS = 60 * 15
MAX_LOOKBACK_DAYS = 240
CHART_WINDOW_DAYS = 180

PROFILE_CONFIG = {
    "Aggressive": {"buy_threshold": 4.0, "sell_threshold": -1.5, "volatility_penalty": 0.0,
                   "earnings_blackout_days": 0, "rsi_upper": 80, "sl_atr_mult": 1.2, "tp_atr_mult": 2.0},
    "Balanced":   {"buy_threshold": 4.5, "sell_threshold": -1.0, "volatility_penalty": 0.5,
                   "earnings_blackout_days": 2, "rsi_upper": 75, "sl_atr_mult": 1.5, "tp_atr_mult": 2.5},
    "Conservative":{"buy_threshold": 5.0, "sell_threshold": -0.5, "volatility_penalty": 1.0,
                   "earnings_blackout_days": 5, "rsi_upper": 70, "sl_atr_mult": 1.8, "tp_atr_mult": 3.0},
}

# ---------------- Helpers ----------------
def sma(series: pd.Series, w: int) -> pd.Series: return series.rolling(w).mean()
def ema(series: pd.Series, span: int) -> pd.Series: return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0] if series.shape[1] else pd.Series(dtype=float)
    else:
        series = pd.Series(series)
    series = pd.to_numeric(series, errors="coerce")
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close  = (df['Low']  - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _pick_series(df: pd.DataFrame, keys: List[str]) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        for key in keys:
            try:
                if key in df.columns.get_level_values(0):
                    s = df[key]
                    if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
                    return pd.to_numeric(s, errors="coerce")
            except Exception: pass
            try:
                if key in df.columns.get_level_values(1):
                    s = df.xs(key, level=1, axis=1)
                    if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
                    return pd.to_numeric(s, errors="coerce")
            except Exception: pass
        flat = df.copy()
        flat.columns = ["|".join([str(x) for x in c]) if isinstance(c, tuple) else str(c) for c in flat.columns]
        for key in keys:
            for col in flat.columns:
                if key.lower() in col.lower():
                    return pd.to_numeric(flat[col], errors="coerce")
    else:
        for key in keys:
            exact = [c for c in df.columns if str(c).lower() == key.lower()]
            if exact: return pd.to_numeric(df[exact[0]], errors="coerce")
        for key in keys:
            part = [c for c in df.columns if key.lower() in str(c).lower()]
            if part: return pd.to_numeric(df[part[0]], errors="coerce")
    return pd.Series(dtype=float)

def _extract_ohlcv_1d(df: pd.DataFrame) -> pd.DataFrame:
    o = _pick_series(df, ["Open"])
    h = _pick_series(df, ["High"])
    l = _pick_series(df, ["Low"])
    c = _pick_series(df, ["Close", "Adj Close"])
    v = _pick_series(df, ["Volume"])
    out = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v})
    return out.dropna()

# ---------------- Data ----------------
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_history(ticker: str, days: int = MAX_LOOKBACK_DAYS) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days * 2)
    df = pd.DataFrame()
    for _ in range(2):
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
        except Exception:
            df = pd.DataFrame()
        if df is not None and not df.empty: break
    if df is None or df.empty: return pd.DataFrame()
    df = df.rename(columns=lambda c: c.title() if isinstance(c, str) else c)
    return df.dropna()

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_info(ticker: str) -> Dict[str, Any]:
    for _ in range(2):
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}
            try:
                earnings_dates = tk.get_earnings_dates(limit=1)
            except Exception:
                earnings_dates = None
            next_earnings = None
            if earnings_dates is not None and not earnings_dates.empty:
                next_earnings = pd.to_datetime(earnings_dates.index[0]).date().isoformat()
            return {"info": info, "next_earnings": next_earnings}
        except Exception:
            pass
    return {"info": {}, "next_earnings": None}

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_spy() -> pd.DataFrame:
    try:
        raw = fetch_history("SPY")
        return _extract_ohlcv_1d(raw) if raw is not None and not raw.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# Fetch weekly historical data for backtesting and multi‑timeframe analysis.
# The period parameter is expressed in years to ensure enough data for long
# moving averages. Using Streamlit's cache to avoid repeated downloads.
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_history_weekly(ticker: str, years: int = 3) -> pd.DataFrame:
    try:
        # Period must be specified as string e.g. '3y' for yfinance
        period_str = f"{years}y"
        df = yf.download(ticker, period=period_str, interval="1wk", progress=False)
        return df.dropna() if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ---------------- Scoring ----------------
def analyze_ticker(
    ticker: str,
    profile: str,
    optional_ibkr_price: float | None = None,
    relax_guards: bool = False,
    market_guard: bool = True,
    spy_df: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    """
    Compute a swing‑trading signal for a given stock ticker.

    This function pulls historical price data, computes a variety of technical
    indicators (moving averages, RSI, MACD, ATR, Bollinger bands, etc.) and
    derives a numeric score based on the configured risk profile. The score is
    then mapped to a discrete trading signal (BUY/SELL/HOLD). A list of human
    readable explanations is also generated to aid transparency.

    Parameters
    ----------
    ticker : str
        The stock symbol to analyse.
    profile : str
        Which risk profile to use (keys of PROFILE_CONFIG).
    optional_ibkr_price : float, optional
        If provided, compares the current price to this value and reports the delta.
    relax_guards : bool, optional
        When True, disables some strict buy conditions (for backtesting).
    market_guard : bool, optional
        When True, enforces that the overall market regime (SPY) is healthy for buy signals.
    spy_df : pd.DataFrame, optional
        Pre‑fetched SPY OHLCV data. When provided, avoids calling fetch_spy multiple times.

    Returns
    -------
    dict
        A dictionary containing the computed signal, numeric score, metrics and explanations.
    """

    # initialize result structure
    res: Dict[str, Any] = {
        "ticker": ticker,
        "error": None,
        "explanations": [],
        "score": 0.0,
        "signal": "HOLD",
        "metrics": {},
        "df": None,
    }
    raw = fetch_history(ticker)
    if raw is None or raw.empty:
        res["error"] = "No data"; return res
    df = _extract_ohlcv_1d(raw)
    if df is None or df.empty:
        res["error"] = "No data"; return res

    df["SMA50"] = sma(df["Close"], 50)
    df["SMA200"] = sma(df["Close"], 200)
    df["RSI14"] = rsi(df["Close"], 14)
    macd_line, signal_line, hist = macd(df["Close"], 12, 26, 9)
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd_line, signal_line, hist
    df["ATR14"] = atr(df, 14)
    df["ATR_pct"] = (df["ATR14"] / df["Close"]) * 100
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    df["VolSurge"] = (df["Volume"] / (df["VolAvg20"] + 1e-9))
    df["High20"] = df["High"].rolling(20).max()
    df["Low20"]  = df["Low"].rolling(20).min()

    # Bollinger Bands (20‑period moving average ± 2 standard deviations).
    # These are used later to gauge overbought/oversold conditions. We
    # calculate them once here to avoid recalculating on each evaluation. Note:
    # ddof=0 for population standard deviation.
    window_bb = 20
    bb_mid = df["Close"].rolling(window_bb).mean()
    bb_std = df["Close"].rolling(window_bb).std(ddof=0)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    price = float(latest["Close"]) if not np.isnan(latest["Close"]) else None
    change_pct = ((latest["Close"] / prev["Close"]) - 1) * 100 if len(df) > 1 else 0.0
    gap_pct = (latest["Open"] / prev["Close"] - 1) * 100 if len(df) > 1 and not np.isnan(latest.get("Open", np.nan)) and not np.isnan(prev.get("Close", np.nan)) else None

    info_blob = fetch_info(ticker)
    next_earn = info_blob.get("next_earnings")
    days_to_earn = None
    if next_earn:
        try:
            d = pd.to_datetime(next_earn)
            days_to_earn = (d - pd.Timestamp.now(tz="UTC").normalize()).days
        except Exception:
            days_to_earn = None

    cfg = PROFILE_CONFIG.get(profile, PROFILE_CONFIG[DEFAULT_PROFILE])
    score = 0.0
    reasons: List[str] = []

    # ---------------------------------------------------------------------
    # Fundamental analysis
    # Pull select fundamentals such as trailing P/E, forward P/E, profit margin
    # and return on equity (ROE).  These values are used to provide context on
    # how expensive or profitable the company is compared to peers.  Some
    # heuristic scoring is applied: low P/E or high profit margins add to
    # confidence, while extremely high P/E or negative margins detract.  All
    # values are recorded in the metrics for display in the UI.
    fundamentals = info_blob.get("info", {}) if isinstance(info_blob, dict) else {}
    pe_ratio: float | None = None
    forward_pe: float | None = None
    profit_margin: float | None = None
    roe: float | None = None
    try:
        pe_ratio = float(fundamentals.get("trailingPE")) if fundamentals.get("trailingPE") is not None else None
    except Exception:
        pe_ratio = None
    try:
        forward_pe = float(fundamentals.get("forwardPE")) if fundamentals.get("forwardPE") is not None else None
    except Exception:
        forward_pe = None
    try:
        pm = fundamentals.get("profitMargins")
        # yfinance may return None or nan; ensure numeric
        if pm is not None and not isinstance(pm, (list, dict)):
            profit_margin = float(pm)
    except Exception:
        profit_margin = None
    try:
        ro = fundamentals.get("returnOnEquity")
        if ro is not None and not isinstance(ro, (list, dict)):
            roe = float(ro)
    except Exception:
        roe = None

    # Apply fundamental heuristics to score
    if profit_margin is not None:
        # Profit margins expressed as decimal (e.g. 0.25 = 25%).  High margins suggest
        # strong competitive position and pricing power, while negative margins
        # signal an unprofitable business.
        if profit_margin > 0.2:
            score += 0.5
            reasons.append(f"High profit margin {profit_margin*100:.1f}%")
        elif profit_margin > 0:
            score += 0.25
            reasons.append(f"Positive profit margin {profit_margin*100:.1f}%")
        else:
            score -= 0.25
            reasons.append(f"Negative profit margin {profit_margin*100:.1f}%")
    if pe_ratio is not None:
        # P/E ratio: lower values (relative to broad market/sector) imply a cheaper
        # valuation.  Moderate P/E around 15‑25 is neutral to mildly positive.
        if pe_ratio < 15:
            score += 0.5
            reasons.append(f"Low P/E {pe_ratio:.1f}")
        elif pe_ratio < 25:
            score += 0.25
            reasons.append(f"Moderate P/E {pe_ratio:.1f}")
        elif pe_ratio > 40:
            score -= 0.5
            reasons.append(f"Very high P/E {pe_ratio:.1f}")
        elif pe_ratio > 30:
            score -= 0.25
            reasons.append(f"Elevated P/E {pe_ratio:.1f}")
    if forward_pe is not None and pe_ratio is not None:
        # Compare forward P/E to trailing P/E.  A lower forward P/E suggests expected
        # earnings growth and may be favourable.
        try:
            if forward_pe < pe_ratio:
                score += 0.25
                reasons.append(f"Forward P/E ({forward_pe:.1f}) < trailing P/E")
            elif forward_pe > pe_ratio * 1.2:
                score -= 0.25
                reasons.append(f"Forward P/E ({forward_pe:.1f}) > trailing P/E")
        except Exception:
            pass
    if roe is not None:
        # Return on equity indicates how effectively management is using shareholders'
        # capital.  Values above ~15% are generally considered strong.
        if roe > 0.15:
            score += 0.3
            reasons.append(f"ROE strong {roe*100:.1f}%")
        elif roe < 0.05:
            score -= 0.3
            reasons.append(f"ROE weak {roe*100:.1f}%")
    # ---------------------------------------------------------------------

    # Trend
    if latest["Close"] > latest["SMA50"]: score += 1.0; reasons.append("Price > SMA50 (trend up)")
    else: score -= 0.5; reasons.append("Price < SMA50 (trend weak)")
    if latest["Close"] > latest["SMA200"]: score += 1.0; reasons.append("Price > SMA200 (long-term up)")
    else: score -= 0.5; reasons.append("Price < SMA200 (long-term weak)")
    if latest["SMA50"] > latest["SMA200"]: score += 0.75; reasons.append("SMA50 > SMA200 (golden trend)")
    else: score -= 0.25; reasons.append("SMA50 <= SMA200")

    # Slope 50d
    if len(df) >= 5 and not np.isnan(df["SMA50"].iloc[-5]):
        slope50 = df["SMA50"].iloc[-1] - df["SMA50"].iloc[-5]
        if slope50 > 0: score += 0.5; reasons.append("SMA50 rising")
        else: score -= 0.5; reasons.append("SMA50 falling")

    # Multi‑timeframe confirmation: compare weekly SMA10 vs SMA40
    # This helps gauge the broader trend and reduces false positives.
    try:
        # Fetch weekly history via cached helper (approx. 3 years).  Using our
        # caching layer avoids repeated API calls when analyzing multiple tickers.
        raw_weekly = fetch_history_weekly(ticker, years=3)
        if raw_weekly is not None and not raw_weekly.empty:
            # Use _pick_series to gracefully handle MultiIndex columns
            weekly_close = _pick_series(raw_weekly, ["Adj Close", "Close"]).dropna()
            if len(weekly_close) >= 40:
                sma10_w = weekly_close.rolling(10).mean()
                sma40_w = weekly_close.rolling(40).mean()
                # Determine if weekly 10w MA is above 40w MA (bullish)
                weekly_trend_ok = bool(sma10_w.iloc[-1] > sma40_w.iloc[-1])
                if weekly_trend_ok:
                    score += 0.5
                    reasons.append("Weekly SMA10 > SMA40 (long-term uptrend)")
                else:
                    score -= 0.5
                    reasons.append("Weekly SMA10 < SMA40 (long-term downtrend)")
    except Exception:
        pass

    # Momentum
    macd_ok = bool(latest["MACD"] > latest["MACD_signal"] and latest["MACD_hist"] > 0)
    if macd_ok: score += 1.0; reasons.append("MACD > Signal (positive momentum)")
    else: score -= 0.25; reasons.append("MACD momentum not confirmed")

    # RSI
    rsi_val = float(latest["RSI14"]) if not np.isnan(latest["RSI14"]) else 50.0
    if rsi_val >= cfg["rsi_upper"]: score -= 0.5; reasons.append(f"RSI {rsi_val:.1f} hot (>{cfg['rsi_upper']})")
    elif rsi_val >= 50: score += 0.75; reasons.append(f"RSI {rsi_val:.1f} in healthy range")
    elif rsi_val >= 40: score -= 0.25; reasons.append(f"RSI {rsi_val:.1f} below 50 (weakish)")
    else: score -= 0.75; reasons.append(f"RSI {rsi_val:.1f} weak (<40)")

    # Breakout + volume (dynamic volume threshold)
    vol_surge = float(latest["VolSurge"]) if latest["VolSurge"] == latest["VolSurge"] else 1.0
    near_high = latest["Close"] >= latest["High20"] * 0.997 if not np.isnan(latest["High20"]) else False
    # Determine a dynamic volume threshold based on the 80th percentile of the last 20 days
    vol_threshold_dynamic: float = 1.2
    try:
        vol_window = df["VolSurge"].dropna().iloc[-20:]
        if len(vol_window) >= 5:
            vol_threshold_dynamic = float(vol_window.quantile(0.8))
    except Exception:
        vol_threshold_dynamic = 1.2
    # True breakout when price closes above the recent high and volume exceeds dynamic threshold
    true_breakout = (latest["Close"] > latest["High20"]) and (vol_surge >= vol_threshold_dynamic if not np.isnan(vol_surge) else False)
    if true_breakout:
        score += 1.5
        reasons.append(f"Breakout with volume (x{vol_surge:.2f}, thr {vol_threshold_dynamic:.2f})")
    elif near_high:
        score += 0.5
        reasons.append("Near 20d high")
    elif latest["Close"] <= latest["Low20"] * 1.003 if not np.isnan(latest["Low20"]) else False:
        score -= 1.0
        reasons.append("Near 20d low (risk)")
    # Additional weight when volume surge on its own exceeds threshold
    if vol_surge >= vol_threshold_dynamic:
        score += 0.75
        reasons.append(f"Volume surge x{vol_surge:.2f} (thr {vol_threshold_dynamic:.2f})")

    # Volatility penalty
    atr_pct = float(latest["ATR_pct"]) if latest["ATR_pct"] == latest["ATR_pct"] else 0.0
    if atr_pct > 4.0:
        score -= cfg["volatility_penalty"]
        if cfg["volatility_penalty"] > 0: reasons.append(f"Volatility high (ATR {atr_pct:.1f}%)")

    # Extension damp
    ext_pct = None
    if latest["SMA50"] and not np.isnan(latest["SMA50"]):
        ext_pct = (latest["Close"] / latest["SMA50"] - 1) * 100
        if ext_pct > 8: score -= 0.5; reasons.append(f"Extended {ext_pct:.1f}% above 50d — score dampened")

    # Relative strength vs SPY 20d + market regime
    rel20d_pp = None
    spy_regime_ok = True
    try:
        # Use pre‑fetched SPY data when available to avoid redundant API calls
        spy = spy_df if spy_df is not None else fetch_spy()
        if len(df) > 21 and spy is not None and len(spy) > 21:
            r_t = df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1
            r_s = spy["Close"].iloc[-1] / spy["Close"].iloc[-21] - 1
            rel20d_pp = (r_t - r_s) * 100
            if rel20d_pp > 0:
                score += 0.5
                reasons.append(f"Outperforming SPY by {rel20d_pp:.1f}pp (20d)")
            else:
                reasons.append(f"Underperforming SPY by {abs(rel20d_pp):.1f}pp (20d)")
            spy["SMA50"] = sma(spy["Close"], 50)
            spy_ok_price = spy["Close"].iloc[-1] > spy["SMA50"].iloc[-1]
            spy_slope = spy["SMA50"].iloc[-1] - spy["SMA50"].iloc[-5]
            spy_ok_slope = spy_slope > 0 if not np.isnan(spy_slope) else True
            spy_regime_ok = bool(spy_ok_price and spy_ok_slope)
            if not spy_regime_ok:
                reasons.append("Market regime weak (SPY < 50d or slope↓)")
    except Exception:
        # Silently ignore SPY comparison errors (e.g., missing data)
        pass

    # Earnings blackout
    if days_to_earn is not None and days_to_earn <= cfg["earnings_blackout_days"]:
        allowed = False
        if profile == "Aggressive" and days_to_earn == 0 and gap_pct is not None and gap_pct >= 2 and vol_surge >= 1.5:
            allowed = True; score += 0.5; reasons.append(f"Earnings today: gap +{gap_pct:.1f}% with volume — aggressive override")
        if not allowed:
            score -= 1.0; reasons.append(f"Earnings in {days_to_earn}d (blackout)")

    # Cooldown
    if ext_pct is not None and ext_pct > 8 and len(df) >= 4:
        last3 = df["Close"].pct_change().iloc[-3:] > 0
        if last3.sum() == 3:
            score -= 0.25
            reasons.append("3 green days while extended — cooldown")

    # Bollinger band position scoring
    bb_pos = None
    try:
        latest_mid = bb_mid.iloc[-1]
        latest_upper = bb_upper.iloc[-1]
        latest_lower = bb_lower.iloc[-1]
        if (not np.isnan(latest_mid)) and (not np.isnan(latest_upper)) and (not np.isnan(latest_lower)) and (latest_upper != latest_lower):
            # calculate relative position within the Bollinger channel (0 = at lower band, 1 = at upper band)
            bb_pos = (price - latest_lower) / (latest_upper - latest_lower)
            # clamp to [0, 1]
            bb_pos = max(0.0, min(1.0, float(bb_pos)))
            # scoring: oversold near the lower band is favourable; overbought near upper band is unfavourable
            if bb_pos <= 0.2:
                score += 0.5
                reasons.append(f"Bollinger: price near lower band (BB pos {bb_pos:.2f})")
            elif bb_pos >= 0.8:
                score -= 0.5
                reasons.append(f"Bollinger: price near upper band (BB pos {bb_pos:.2f})")
    except Exception:
        bb_pos = None

    # Bollinger band width / squeeze detection
    bb_width_ratio = None
    try:
        # Current band width ratio (upper-lower relative to mid)
        if not np.isnan(bb_mid.iloc[-1]) and bb_mid.iloc[-1] != 0:
            current_width_ratio = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_mid.iloc[-1]
        else:
            current_width_ratio = None
        width_series = ((bb_upper - bb_lower) / bb_mid).dropna()
        width_window = width_series.iloc[-100:] if len(width_series) >= 100 else width_series
        if current_width_ratio is not None:
            bb_width_ratio = float(current_width_ratio)
            # Determine a squeeze when current width ratio is in the bottom 20% of the last 100 values
            if len(width_window) >= 20:
                width_threshold = float(width_window.quantile(0.2))
                if bb_width_ratio < width_threshold:
                    score += 0.25
                    reasons.append(f"Bollinger squeeze (band width ratio {bb_width_ratio:.3f} < {width_threshold:.3f})")
    except Exception:
        bb_width_ratio = None

    # ---------------------------------------------------------------------
    # Simple historical backtesting
    # To gauge how the strategy would have performed recently, run a simplified
    # backtest over the last ~120 days.  A buy signal is generated when
    # price is above both the 50d and 200d SMAs, the 50d SMA is above the
    # 200d SMA (bullish structure), MACD line exceeds its signal, and RSI
    # remains below the configured upper bound.  Returns are measured after
    # a fixed holding period (10 days).  We compute the win rate and
    # average return of these trades and apply modest adjustments to the
    # overall score.  These metrics are stored in the result for display.
    backtest_win = None
    backtest_avg = None
    backtest_count = 0
    try:
        hold_days = 10
        back_returns: List[float] = []
        # Limit lookback to the most recent 120 days to emphasize recent market
        # conditions.  Skip earlier rows when the dataset is longer.
        lookback_window = 120
        for idx in range(max(0, len(df) - lookback_window), len(df) - hold_days):
            row = df.iloc[idx]
            # Ensure SMA columns are present and not NaN
            if (not np.isnan(row.get("SMA50", np.nan)) and
                not np.isnan(row.get("SMA200", np.nan)) and
                row["Close"] > row["SMA50"] and
                row["Close"] > row["SMA200"] and
                row["SMA50"] > row["SMA200"] and
                row["MACD"] > row["MACD_signal"] and
                row["RSI14"] < cfg.get("rsi_upper", 70)):
                future_index = idx + hold_days
                if future_index < len(df):
                    ret = (df["Close"].iloc[future_index] / row["Close"]) - 1
                    back_returns.append(float(ret))
        backtest_count = len(back_returns)
        if back_returns:
            backtest_win = float(sum(1 for x in back_returns if x > 0) / len(back_returns))
            backtest_avg = float(np.mean(back_returns))
            reasons.append(
                f"Backtest win rate {backtest_win*100:.0f}% (avg {backtest_avg*100:.2f}%)"
            )
            # Score adjustments based on backtest success metrics
            if backtest_win > 0.6:
                score += 0.5
            elif backtest_win > 0.5:
                score += 0.25
            elif backtest_win < 0.4:
                score -= 0.5
            elif backtest_win < 0.5:
                score -= 0.25
            if backtest_avg > 0.05:
                score += 0.5
            elif backtest_avg > 0.02:
                score += 0.25
            elif backtest_avg < 0:
                score -= 0.25
    except Exception:
        backtest_win = None
        backtest_avg = None
        backtest_count = 0

    # ATR-based SL/TP
    sl_mult = cfg.get("sl_atr_mult", 1.5)
    tp_mult = cfg.get("tp_atr_mult", 2.5)
    sl_level = tp_level = None
    if not np.isnan(latest["ATR14"]) and price is not None:
        atr_abs = float(latest["ATR14"])
        sl_level = round(price - sl_mult * atr_abs, 2)
        tp_level = round(price + tp_mult * atr_abs, 2)
        reasons.append(f"ATR-based SL/TP: x{sl_mult} / x{tp_mult}")

    # Map to signal + guard with adaptive thresholds
    buy_th = cfg["buy_threshold"]
    sell_th = cfg["sell_threshold"]
    # Adapt buy/sell thresholds based on market regime when guard is enabled
    if market_guard:
        try:
            if spy_regime_ok:
                # In a strong market, slightly lower the buy threshold and raise the sell threshold
                buy_th -= 0.25
                sell_th += 0.25
            else:
                # In a weak market, require higher confidence to buy and more lenient to sell
                buy_th += 0.25
                sell_th -= 0.25
        except Exception:
            pass
    signal = "HOLD"
    if score >= buy_th:
        signal = "BUY"
    elif score <= sell_th:
        signal = "SELL"

    # Guard conditions rely on dynamic volume threshold and relative strength
    rel_ok = (rel20d_pp is not None and rel20d_pp > 0)
    breakout_vol_ok = (latest["Close"] > latest["High20"] and (float(latest["VolSurge"]) if latest["VolSurge"]==latest["VolSurge"] else 0) >= vol_threshold_dynamic)
    cond2 = breakout_vol_ok or near_high
    guard_ok = (macd_ok and cond2 and rel_ok and (spy_regime_ok if market_guard else True))
    if signal == "BUY" and not (relax_guards or guard_ok):
        signal = "HOLD"
        reasons.append("Buy guard failed: need MACD>Signal, RelStr>0, (breakout vol≥thr or near-high), and healthy SPY regime")

    ibkr_delta = None
    if optional_ibkr_price is not None and optional_ibkr_price > 0 and price:
        ibkr_delta = ((optional_ibkr_price / price) - 1) * 100
        reasons.append(f"IBKR vs YF delta: {ibkr_delta:+.2f}%")

    res.update({
        "df": df,
        "score": round(float(score), 2),
        "signal": signal,
        "metrics": {
            "price": round(price, 2) if price is not None else None,
            "change_pct": round(float(change_pct), 2),
            "sma50": round(float(latest["SMA50"]), 2) if not np.isnan(latest["SMA50"]) else None,
            "SMA200": round(float(latest["SMA200"]), 2) if not np.isnan(latest["SMA200"]) else None,
            "rsi14": round(rsi_val, 1),
            "macd": round(float(latest["MACD"]), 4) if not np.isnan(latest["MACD"]) else None,
            "macd_signal": round(float(latest["MACD_signal"]), 4) if not np.isnan(latest["MACD_signal"]) else None,
            "vol_surge": round(vol_surge, 2) if not np.isnan(vol_surge) else None,
            "atr_pct": round(atr_pct, 2) if not np.isnan(atr_pct) else None,
            "atr14": round(float(latest["ATR14"]), 2) if not np.isnan(latest["ATR14"]) else None,
            "days_to_earnings": int(days_to_earn) if days_to_earn is not None else None,
            "ibkr_delta_pct": round(float(ibkr_delta), 2) if ibkr_delta is not None else None,
            "ext_pct": round(float(ext_pct), 2) if ext_pct is not None else None,
            "rel20d_pp": round(float(rel20d_pp), 2) if rel20d_pp is not None else None,
            "bb_pos": round(float(bb_pos), 2) if bb_pos is not None else None,
            "bb_width": round(float(bb_width_ratio), 3) if bb_width_ratio is not None else None,
            "bb_width_pct": round(float(bb_width_ratio) * 100, 2) if bb_width_ratio is not None else None,
            # Fundamental metrics
            "pe_ratio": round(float(pe_ratio), 2) if pe_ratio is not None else None,
            "forward_pe": round(float(forward_pe), 2) if forward_pe is not None else None,
            # Profit margin and ROE are expressed as percentages for readability
            "profit_margin_pct": round(profit_margin * 100, 2) if profit_margin is not None else None,
            "roe_pct": round(roe * 100, 2) if roe is not None else None,
            # Backtest summary
            "backtest_win_rate_pct": round(backtest_win * 100, 1) if backtest_win is not None else None,
            "backtest_avg_return_pct": round(backtest_avg * 100, 2) if backtest_avg is not None else None,
            "backtest_signals": backtest_count,
            "sl": sl_level, "tp": tp_level,
            "sl_mult": sl_mult, "tp_mult": tp_mult,
            "buy_guard_ok": bool(guard_ok),
            "flags": {"macd_ok": bool(macd_ok), "rel_ok": bool(rel_ok),
                      "near_high": bool(near_high), "breakout_vol_ok": bool(breakout_vol_ok),
                      "spy_regime_ok": bool(spy_regime_ok)},
        },
        "explanations": reasons,
    })
    return res

def build_chart(df: pd.DataFrame, title: str, sl: float | None = None, tp: float | None = None) -> 'go.Figure':
    window = min(CHART_WINDOW_DAYS, len(df))
    sub = df.iloc[-window:].copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=sub.index, open=sub['Open'], high=sub['High'], low=sub['Low'], close=sub['Close'], name='Price'))
    if 'SMA50' in sub.columns:  fig.add_trace(go.Scatter(x=sub.index, y=sub['SMA50'], name='SMA50', mode='lines'))
    if 'SMA200' in sub.columns: fig.add_trace(go.Scatter(x=sub.index, y=sub['SMA200'], name='SMA200', mode='lines'))
    if 'High20' in sub.columns and 'Low20' in sub.columns:
        fig.add_trace(go.Scatter(x=sub.index, y=sub['High20'], name='20d High', mode='lines'))
        fig.add_trace(go.Scatter(x=sub.index, y=sub['Low20'],  name='20d Low',  mode='lines'))
    try:
        if sl is not None: fig.add_hline(y=sl, line_dash="dot", annotation_text="SL", annotation_position="top left")
        if tp is not None: fig.add_hline(y=tp, line_dash="dot", annotation_text="TP", annotation_position="top left")
    except Exception: pass
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False,
                      height=520, legend_orientation="h", margin=dict(l=10, r=10, t=40, b=10))
    return fig

# ---------------- AUTH (username + password via Supabase) ----------------
def _username_to_email(u: str) -> str:
    base = slugify((u or "").strip().lower()) or "user"
    return f"{base}@users.local"

def _logout():
    try:
        if sb: sb.auth.sign_out()
    except Exception:
        pass
    for k in ["auth_ok", "auth_user", "auth_uid", "login_fail_count", "login_lock_until"]:
        st.session_state.pop(k, None)
    st.rerun()  # <- fixed

def _ensure_profile(uid: str, username: str):
    try:
        sb.table("profiles").upsert({"id": uid, "username": username}).execute()
    except Exception:
        pass

def _auth_gate() -> bool:
    if not sb:
        st.error("Missing Supabase config. Add SUPABASE_URL and SUPABASE_ANON_KEY in Render → Environment.")
        st.stop()

    now = time.time()
    if now < st.session_state.get("login_lock_until", 0):
        st.warning("Too many failed attempts. Try again later.")
        return False

    # already signed in?
    try:
        u = sb.auth.get_user()
        if u and u.user:
            st.session_state["auth_ok"] = True
            st.session_state["auth_uid"] = u.user.id
            st.session_state.setdefault("auth_user", u.user.user_metadata.get("username", "user"))
    except Exception:
        pass

    if st.session_state.get("auth_ok"):
        with st.sidebar:
            st.markdown(f"**Signed in:** {st.session_state.get('auth_user')}")
            if st.button("Logout"):
                _logout()
        return True

    tabs = st.tabs(["Sign in", "Create profile"])

    # -------- LOGIN FORM (prevents double-click) --------
    with tabs[0]:
        with st.form("login_form"):
            u = st.text_input("Username", key="login_user")
            p = st.text_input("Password", type="password", key="login_pass")
            submitted_login = st.form_submit_button("Sign in")
        if submitted_login:
            try:
                email = _username_to_email(u)
                sess = sb.auth.sign_in_with_password({"email": email, "password": p})
                if sess and sess.user:
                    st.session_state["auth_ok"] = True
                    st.session_state["auth_uid"] = sess.user.id
                    st.session_state["auth_user"] = (u or "").strip()
                    st.session_state.pop("login_fail_count", None)
                    st.session_state.pop("login_lock_until", None)
                    _ensure_profile(sess.user.id, (u or "").strip())
                    st.rerun()  # <- fixed
                else:
                    raise Exception("no session")
            except Exception:
                fails = st.session_state.get("login_fail_count", 0) + 1
                st.session_state["login_fail_count"] = fails
                if fails >= 5:
                    st.session_state["login_lock_until"] = time.time() + 10 * 60
                st.error("Invalid username or password.")

    # -------- REGISTER FORM --------
    with tabs[1]:
        with st.form("register_form"):
            nu = st.text_input("Username (min 3 chars)", key="reg_user")
            npw = st.text_input("Password (min 8 chars)", type="password", key="reg_pass")
            nc  = st.text_input("Confirm password", type="password", key="reg_conf")
            submitted_reg = st.form_submit_button("Create profile")
        if submitted_reg:
            uname = (nu or "").strip()
            if len(slugify(uname)) < 3:
                st.error("Username too short."); st.stop()
            if len(npw) < 8:
                st.error("Password too short."); st.stop()
            if npw != nc:
                st.error("Passwords do not match."); st.stop()
            email = _username_to_email(uname)
            try:
                res = sb.auth.sign_up({"email": email, "password": npw, "options": {"data": {"username": uname}}})
                if not res or not res.user:
                    st.error("Could not create profile."); st.stop()
                sess = sb.auth.sign_in_with_password({"email": email, "password": npw})
                st.session_state["auth_ok"] = True
                st.session_state["auth_uid"] = sess.user.id
                st.session_state["auth_user"] = uname
                _ensure_profile(sess.user.id, uname)
                st.success("Profile created.")
                st.rerun()  # <- fixed
            except Exception:
                st.error("Username may be taken. Try another.")
                st.stop()

    st.stop()

# ---------------- Persistence via Supabase ----------------
def _session_uid() -> str | None:
    return st.session_state.get("auth_uid")

def load_watchlist() -> Dict[str, Any]:
    default = {"tickers": DEFAULT_TICKERS, "profile": DEFAULT_PROFILE,
               "auto_refresh_minutes": 15, "acct_size": 10000.0, "risk_pct": 1.0}
    uid = _session_uid()
    if not uid: return default
    skey = f"watchlist_state__{uid}"
    if skey in st.session_state:
        return st.session_state[skey]
    try:
        res = sb.table("watchlists").select("data").eq("user_id", uid).single().execute()
        data = (res.data or {}).get("data", None)
        st.session_state[skey] = data if isinstance(data, dict) else default
    except Exception:
        st.session_state[skey] = default
    return st.session_state[skey]

def save_watchlist(state: Dict[str, Any]) -> None:
    uid = _session_uid()
    if not uid: return
    skey = f"watchlist_state__{uid}"
    st.session_state[skey] = state
    try:
        sb.table("watchlists").upsert({
            "user_id": uid,
            "data": state,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }).execute()
    except Exception:
        pass

# ---------------- UI ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# Gate: Supabase auth first
if not _auth_gate():
    st.stop()

# Guards for missing libs
if yf is None:
    st.error("Missing dependency: 'yfinance'. Add it to requirements.txt.")
    st.stop()
if go is None:
    st.error("Missing dependency: 'plotly'. Add it to requirements.txt.")
    st.stop()

state = load_watchlist()

# Sidebar
with st.sidebar:
    st.subheader("⚙️ Settings")
    profile = st.selectbox("Risk profile", list(PROFILE_CONFIG.keys()),
                           index=list(PROFILE_CONFIG.keys()).index(state.get("profile", DEFAULT_PROFILE)))
    auto_minutes = st.number_input("Auto-refresh (minutes)", min_value=0, max_value=60,
                                   value=int(state.get("auto_refresh_minutes", 15)), step=1,
                                   help="0 disables auto refresh")

    st.markdown("---")
    st.subheader("👀 Watchlist")
    tickers_text = st.text_input("Add ticker(s) (comma-separated)", placeholder="e.g. AAPL, MSFT, NVDA")
    if st.button("➕ Add to watchlist"):
        if tickers_text.strip():
            new = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
            merged = sorted(list(set(state.get("tickers", []) + new)))
            state["tickers"] = merged
            save_watchlist(state)
            st.success(f"Added: {', '.join(new)}")

    current = state.get("tickers", [])
    if current:
        remove_choice = st.multiselect("Remove tickers", options=current)
        if st.button("🗑️ Remove selected") and remove_choice:
            state["tickers"] = [t for t in current if t not in remove_choice]
            save_watchlist(state)
            st.warning(f"Removed: {', '.join(remove_choice)}")

    st.markdown("---")
    st.subheader("📏 Position sizing")
    acct_size = st.number_input("Account size ($)", min_value=0.0, value=float(state.get("acct_size", 10000.0)), step=100.0)
    risk_pct  = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=float(state.get("risk_pct", 1.0)), step=0.1)

    st.markdown("---")
    st.subheader("🔒 Buy-guard & stops")
    relax_guards = st.checkbox("Relax guards (test mode)", value=bool(state.get("relax_guards", False)))
    market_guard = st.checkbox("Enforce SPY regime", value=bool(state.get("market_guard", True)))
    ts_enabled   = st.checkbox("Use trailing stop instead of fixed SL", value=bool(state.get("ts_enabled", False)))
    ts_mult      = st.number_input("Trailing stop ATR multiple", min_value=0.5, max_value=5.0, value=float(state.get("ts_mult", 1.5)), step=0.1)

    if st.button("💾 Save settings"):
        state["profile"] = profile
        state["auto_refresh_minutes"] = int(auto_minutes)
        state["acct_size"] = float(acct_size)
        state["risk_pct"]  = float(risk_pct)
        state["relax_guards"] = bool(relax_guards)
        state["market_guard"] = bool(market_guard)
        state["ts_enabled"] = bool(ts_enabled)
        state["ts_mult"] = float(ts_mult)
        save_watchlist(state)
        st.toast("Settings saved", icon="✅")

# Auto refresh
if state.get("auto_refresh_minutes", 15) > 0 and HAS_AUTOR:
    st_autorefresh(interval=state["auto_refresh_minutes"] * 60 * 1000, key="autorefresh")

with st.sidebar:
    if st.button("🔄 Refresh now"):
        st.rerun()  # <- fixed

# Main
watch = state.get("tickers", [])
if not watch:
    st.info("Add tickers to your watchlist from the sidebar to begin.")
    st.stop()

st.write(f"**Profile:** {profile}  •  **Tickers:** {', '.join(watch)}")

acct_size = float(state.get("acct_size", 10000.0))
risk_pct  = float(state.get("risk_pct", 1.0))
relax_guards = bool(state.get("relax_guards", False))
market_guard = bool(state.get("market_guard", True))
ts_enabled   = bool(state.get("ts_enabled", False))
ts_mult      = float(state.get("ts_mult", 1.5))

with st.expander("Optional: IBKR Last Prices (manual to compare deltas)"):
    ibkr_inputs: Dict[str, float | None] = {}
    cols = st.columns(min(4, len(watch))) if len(watch) > 0 else [st]
    for i, t in enumerate(watch):
        with cols[i % len(cols)]:
            val = st.text_input(f"{t}", placeholder="e.g. 198.23")
            try:
                ibkr_inputs[t] = float(val) if val else None
            except Exception:
                ibkr_inputs[t] = None
                st.caption("Enter a numeric price")

results: List[Dict[str, Any]] = []
with st.spinner("Analyzing tickers…"):
    # Fetch SPY data once and reuse for all tickers to avoid repeated API calls
    pre_spy = None
    try:
        pre_spy = fetch_spy()
    except Exception:
        pre_spy = None
    for t in watch:
        res = analyze_ticker(
            t,
            profile,
            optional_ibkr_price=ibkr_inputs.get(t),
            relax_guards=relax_guards,
            market_guard=market_guard,
            spy_df=pre_spy,
        )
        results.append(res)

# Summary table
rows: List[Dict[str, Any]] = []
for r in results:
    if r.get("error"):
        rows.append({"Ticker": r["ticker"], "Error": r.get("error")})
        continue
    m = r.get("metrics", {})
    rows.append({
        "Ticker": r["ticker"],
        "Signal": r.get("signal"),
        "Score": r.get("score"),
        "Price": m.get("price"),
        "Δ 1D %": m.get("change_pct"),
        "RSI14": m.get("rsi14"),
        "MACD>Sig": 1 if (m.get("macd") or 0) > (m.get("macd_signal") or 0) else 0,
        "Vol xAvg20": m.get("vol_surge"),
        "ATR %": m.get("atr_pct"),
        "Ext%50d": m.get("ext_pct"),
        "BB pos": m.get("bb_pos"),
        "BB width (%)": m.get("bb_width_pct"),
        "RelStr20d(pp)": m.get("rel20d_pp"),
        "PE": m.get("pe_ratio"),
        "PM %": m.get("profit_margin_pct"),
        "ROE %": m.get("roe_pct"),
        "BT Win %": m.get("backtest_win_rate_pct"),
        "BT Avg %": m.get("backtest_avg_return_pct"),
        "BT Count": m.get("backtest_signals"),
        "BuyGuard": "OK" if m.get("buy_guard_ok") else "Fail",
        "SL": m.get("sl"),
        "TP": m.get("tp"),
        "Days→Earn": m.get("days_to_earnings"),
        "IBKR Δ%": m.get("ibkr_delta_pct"),
    })

summary_df = pd.DataFrame(rows)
numeric_scores = [r.get("score", 0) for r in results if not r.get("error")]
watchlist_score = round(float(np.nanmean(numeric_scores)) if numeric_scores else 0.0, 2)

st.markdown(f"### 🧮 Watchlist Score: **{watchlist_score}**")
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# Downloads
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button("⬇️ Download table (CSV)",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="watchlist_signals.csv", mime="text/csv")
with col_dl2:
    diag = [{"ticker": r["ticker"], "score": r.get("score"),
             "signal": r.get("signal"), "metrics": r.get("metrics"),
             "explanations": r.get("explanations")} for r in results]
    st.download_button("⬇️ Download diagnostics (JSON)",
        data=json.dumps(diag, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="watchlist_diagnostics.json", mime="application/json")

# Per-ticker details
st.markdown("---")
for r in results:
    t = r["ticker"]
    st.markdown(f"## {t} · Signal: **{r['signal']}** · Score: **{r['score']}**")
    if r.get("error"):
        st.warning(r.get("error")); continue
    m = r.get("metrics", {})
    left, right = st.columns([2, 1])
    with left:
        fig = build_chart(r["df"], title=f"{t} (last {CHART_WINDOW_DAYS}d)", sl=m.get("sl"), tp=m.get("tp"))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Key metrics")
        st.metric(
            "Price",
            f"{m.get('price')}",
            delta=f"{m.get('change_pct')}%" if m.get('change_pct') is not None else None,
        )
        # Display a concise selection of technical indicators and computed values
        st.text(f"RSI14: {m.get('rsi14')}")
        st.text(f"ATR%: {m.get('atr_pct')}")
        st.text(f"ATR: {m.get('atr14')}")
        st.text(f"Vol xAvg20: {m.get('vol_surge')}")
        st.text(f"Ext% vs 50d: {m.get('ext_pct')}")
        st.text(f"BB pos (0-1): {m.get('bb_pos')}")
        # Show the width of the Bollinger channel as a percentage of the midline.  Lower values
        # indicate compressed volatility, potentially preceding a breakout.
        st.text(f"BB width (%): {m.get('bb_width_pct')}")
        st.text(f"RelStr 20d vs SPY (pp): {m.get('rel20d_pp')}")
        st.text(f"SL (x{m.get('sl_mult')} ATR): {m.get('sl')}")
        st.text(f"TP (x{m.get('tp_mult')} ATR): {m.get('tp')}")
        st.text(f"Days→Earnings: {m.get('days_to_earnings')}")
        if m.get("ibkr_delta_pct") is not None:
            st.text(f"IBKR Δ%: {m.get('ibkr_delta_pct')}%")

        # Display select fundamental metrics and backtest summary
        if m.get("pe_ratio") is not None:
            st.text(f"P/E ratio: {m.get('pe_ratio')}")
        if m.get("profit_margin_pct") is not None:
            st.text(f"Profit margin: {m.get('profit_margin_pct')}%")
        if m.get("roe_pct") is not None:
            st.text(f"ROE: {m.get('roe_pct')}%")
        if m.get("backtest_win_rate_pct") is not None:
            st.text(f"BT win rate: {m.get('backtest_win_rate_pct')}%")
        if m.get("backtest_avg_return_pct") is not None:
            st.text(f"BT avg return: {m.get('backtest_avg_return_pct')}%")
        if m.get("backtest_signals") is not None and int(m.get("backtest_signals")) > 0:
            st.text(f"BT signals: {m.get('backtest_signals')}")

    # --------- FIXED BLOCK (correct indentation + safe formatting) ----------
    with st.expander("📋 IBKR order (copy)"):
        side_default = 0 if r.get("signal") == "BUY" else 1
        side = st.selectbox("Side", ["BUY", "SELL"], index=side_default, key=f"side_{t}")

        entry = m.get("price")
        atr_abs = m.get("atr14")
        sl_mult = m.get("sl_mult")
        tp_mult = m.get("tp_mult")

        # Outer IF — ELSE must align with this IF
        if entry is not None and atr_abs is not None and sl_mult is not None and tp_mult is not None:
            ord_tp = round(entry + tp_mult * atr_abs, 2) if side == "BUY" else round(entry - tp_mult * atr_abs, 2)

            # Trailing stop variant
            if state.get("ts_enabled", False):
                trail_amt = round(state.get("ts_mult", 1.5) * atr_abs, 2)
                risk_per_share = trail_amt
                qty = int(max(1, np.floor(
                    (state.get("acct_size", 10000.0) * state.get("risk_pct", 1.0) / 100.0) / risk_per_share
                ))) if risk_per_share > 0 else 0

                order_text = f"""IBKR Bracket Order
Symbol: {t}
Side: {side}
Quantity: {qty}
Parent: LIMIT {side} @ {entry:.2f}
Child 1: TRAILING STOP {'SELL' if side=='BUY' else 'BUY TO COVER'} trail_amt ${trail_amt:.2f}
Child 2: TAKE-PROFIT LIMIT {'SELL' if side=='BUY' else 'BUY TO COVER'} @ {ord_tp:.2f}
Risk/share (≈trail): ${risk_per_share:.2f}
Risk amount ({state.get('risk_pct',1.0):.2f}% of ${state.get('acct_size',10000.0):.2f}): ${state.get('acct_size',10000.0) * state.get('risk_pct',1.0) / 100.0:.2f}
Note: R:R is approximate with trailing stops.
"""
            # Fixed SL/TP variant
            else:
                ord_sl = round(entry - sl_mult * atr_abs, 2) if side == "BUY" else round(entry + sl_mult * atr_abs, 2)
                risk_per_share = abs(entry - ord_sl)
                qty = int(max(1, np.floor(
                    (state.get("acct_size", 10000.0) * state.get("risk_pct", 1.0) / 100.0) / risk_per_share
                ))) if risk_per_share > 0 else 0

                rr = (abs(ord_tp - entry) / risk_per_share) if risk_per_share and risk_per_share > 0 else None
                rr_text = f"{rr:.2f}" if isinstance(rr, (int, float)) else "n/a"

                order_text = f"""IBKR Bracket Order
Symbol: {t}
Side: {side}
Quantity: {qty}
Parent: LIMIT {side} @ {entry:.2f}
Child 1: STOP {'SELL' if side=='BUY' else 'BUY TO COVER'} @ {ord_sl:.2f}
Child 2: TAKE-PROFIT LIMIT {'SELL' if side=='BUY' else 'BUY TO COVER'} @ {ord_tp:.2f}
Risk/share: ${risk_per_share:.2f}
Risk amount ({state.get('risk_pct',1.0):.2f}% of ${state.get('acct_size',10000.0):.2f}): ${state.get('acct_size',10000.0) * state.get('risk_pct',1.0) / 100.0:.2f}
Approx. R:R: {rr_text}
"""

            st.code(order_text, language="text")
        else:
            st.info("Not enough data to compute ATR-based SL/TP.")
    # -----------------------------------------------------------------------

    with st.expander("🔍 BuyGuard checks"):
        flags = m.get("flags", {})
        def b(v): return "✅" if v else "❌"
        st.write(f"MACD>Signal: {b(flags.get('macd_ok'))}")
        st.write(f"RelStr20d>0: {b(flags.get('rel_ok'))} (val: {m.get('rel20d_pp')})")
        st.write(f"Breakout vol or NearHigh: {b(flags.get('breakout_vol_ok') or flags.get('near_high'))} (vol xAvg20: {m.get('vol_surge')})")
        st.write(f"SPY regime OK: {b(flags.get('spy_regime_ok'))}")

    with st.expander("🤖 Why this signal?"):
        for reason in r.get("explanations", []):
            st.write("• ", reason)

st.caption("Educational use only — not financial advice.")
