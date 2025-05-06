# --- Page Setup (must be first command) ---
import streamlit as st
st.set_page_config(page_title="Forex AI Signals", layout="wide")

from streamlit_autorefresh import st_autorefresh
import pandas as pd
import requests
import numpy as np
from datetime import datetime
from scipy.signal import argrelextrema

# --- Auto Refresh Setup ---
st_autorefresh(interval=30000, key="auto_refresh")  # 30 sec

API_KEY = "b2a1234a9ea240f9ba85696e2a243403"
symbols = {
    "EUR/USD": "EUR/USD",
    "GBP/USD": "GBP/USD",
    "USD/JPY": "USD/JPY",
    "AUD/USD": "AUD/USD",
    "USD/CAD": "USD/CAD",
    "USD/CHF": "USD/CHF",
    "XAU/USD": "XAU/USD",
    "WTI/USD": "WTI/USD"
}

st.markdown("""
    <style>body, html, .block-container, table td, table th {font-size: 18px !important;}</style>
    <h1 style='text-align: center; color:#007acc;'>Forex AI Signals (Low / Medium / Strong)</h1>
""", unsafe_allow_html=True)

# --- Indicator Functions ---
@st.cache_data
def fetch_data(symbol, interval="5min", outputsize=200):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "apikey": API_KEY}
    r = requests.get(url, params=params)
    data = r.json()
    if "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.astype(float).sort_index()
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_adx(df, period=14):
    df['TR'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['+DM'] = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']), np.maximum(df['high'] - df['high'].shift(), 0), 0)
    df['-DM'] = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()), np.maximum(df['low'].shift() - df['low'], 0), 0)
    tr14 = df['TR'].rolling(window=period).mean()
    plus_dm14 = df['+DM'].rolling(window=period).mean()
    minus_dm14 = df['-DM'].rolling(window=period).mean()
    plus_di14 = 100 * (plus_dm14 / tr14)
    minus_di14 = 100 * (minus_dm14 / tr14)
    dx = 100 * abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    adx = dx.rolling(window=period).mean()
    return adx

def detect_candle_pattern(df):
    open = df['open'].iloc[-1]
    close = df['close'].iloc[-1]
    high = df['high'].iloc[-1]
    low = df['low'].iloc[-1]
    body = abs(close - open)
    candle_range = high - low
    if body < candle_range * 0.3:
        if close > open:
            return "Bullish"
        elif open > close:
            return "Bearish"
    return ""

def detect_divergence_direction(df):
    df['RSI'] = calculate_rsi(df['close'])
    df = df.dropna()
    close = df['close']
    rsi = df['RSI']
    swing_lows = argrelextrema(close.values, np.less_equal, order=2)[0]
    swing_highs = argrelextrema(close.values, np.greater_equal, order=2)[0]
    rsi_lows = argrelextrema(rsi.values, np.less_equal, order=2)[0]
    rsi_highs = argrelextrema(rsi.values, np.greater_equal, order=2)[0]
    if len(swing_lows) >= 2 and len(rsi_lows) >= 2:
        p1, p2 = swing_lows[-2], swing_lows[-1]
        if close[p2] < close[p1] and rsi[p2] > rsi[p1]:
            return "Bullish"
    if len(swing_highs) >= 2 and len(rsi_highs) >= 2:
        p1, p2 = swing_highs[-2], swing_highs[-1]
        if close[p2] > close[p1] and rsi[p2] < rsi[p1]:
            return "Bearish"
    return ""

def get_tf_confirmation(symbol):
    for tf in ["15min", "1h"]:
        df = fetch_data(symbol, interval=tf)
        if df is not None:
            dir = detect_divergence_direction(df)
            if dir:
                return f"Confirm {dir}"
    return ""

def generate_ai_suggestion(price, direction, indicators, tf_confirmed):
    if not direction:
        return ""
    sl = price * (1 - 0.002) if direction == "Bullish" else price * (1 + 0.002)
    tp = price * (1 + 0.004) if direction == "Bullish" else price * (1 - 0.004)
    confidence = "Low"
    if len(indicators) >= 3 and tf_confirmed:
        confidence = "Strong"
    elif len(indicators) >= 2:
        confidence = "Medium"
    return f"{confidence} {direction} @ {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f} | Confidence: {confidence}"

def generate_advice(trend, divergence, ai_suggestion, tf_confirm):
    if not divergence:
        return "No signal detected — wait"
    if trend != divergence:
        if "Strong" in ai_suggestion and "Confirm" in tf_confirm:
            return f"WARNING: {divergence} signal forming, but trend is {trend} — early entry possible"
        return f"NOTE: Divergence forming but trend still {trend.lower()} — wait for confirmation"
    if trend == divergence:
        if "Strong" in ai_suggestion:
            return f"STRONG: {trend.lower()} setup — trend and signal match"
        elif "Medium" in ai_suggestion:
            return f"MEDIUM: {trend.lower()} — trend match"
        else:
            return f"LOW: {trend.lower()} — trend match but weak"
    return "INFO: Analysis unclear"

# --- Run Analysis ---
rows = []
for label, symbol in symbols.items():
    df = fetch_data(symbol, interval="5min")
    if df is not None:
        df['RSI'] = calculate_rsi(df['close'])
        df['MACD'], df['MACD_Signal'] = calculate_macd(df['close'])
        df['EMA9'] = calculate_ema(df['close'], 9)
        df['EMA20'] = calculate_ema(df['close'], 20)
        df['ADX'] = calculate_adx(df)
        df = df.dropna()

        price_now = df['close'].iloc[-1]
        direction = detect_divergence_direction(df)
        tf_status = get_tf_confirmation(symbol)

        indicators = []
        if direction:
            if direction == "Bullish" and df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                indicators.append("MACD")
            if direction == "Bearish" and df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1]:
                indicators.append("MACD")
            if direction == "Bullish" and price_now > df['EMA9'].iloc[-1] and price_now > df['EMA20'].iloc[-1]:
                indicators.append("EMA")
            if direction == "Bearish" and price_now < df['EMA9'].iloc[-1] and price_now < df['EMA20'].iloc[-1]:
                indicators.append("EMA")
            if df['ADX'].iloc[-1] > 20:
                indicators.append("ADX")
            pattern = detect_candle_pattern(df)
            if direction in pattern:
                indicators.append("Candle")

        trend = (
            "Bullish" if df['EMA9'].iloc[-1] > df['EMA20'].iloc[-1] and price_now > df['EMA9'].iloc[-1]
            else "Bearish" if df['EMA9'].iloc[-1] < df['EMA20'].iloc[-1] and price_now < df['EMA9'].iloc[-1]
            else "Sideways"
        )

        tf_match = (direction == "Bullish" and tf_status == "Confirm Bullish") or (direction == "Bearish" and tf_status == "Confirm Bearish")
        ai_suggestion = generate_ai_suggestion(price_now, direction, indicators, tf_match)
        advice = generate_advice(trend, direction, ai_suggestion, tf_status)

        rows.append({
            "Pair": label,
            "Price": round(price_now, 5),
            "RSI": round(df['RSI'].iloc[-1], 2),
            "Trend": trend,
            "Divergence": direction,
            "TF": tf_status,
            "Confirmed Indicators": ", ".join(indicators),
            "AI Suggestion": ai_suggestion,
            "Advice": advice
        })

# --- Display Table with Highlight Logic ---
df_final = pd.DataFrame(rows)

# Display styled table (unchanged)
# --- Rest of your table styling code remains unchanged ---

st.markdown(styled_html, unsafe_allow_html=True)
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
