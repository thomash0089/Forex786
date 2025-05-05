# --- Signals with H & I (15-Min Timeframe | With Volume + Candle Pattern) ---
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import requests
import numpy as np
from datetime import datetime
from scipy.signal import argrelextrema

st.set_page_config(page_title="Forex AI Signals", layout="wide")
st_autorefresh(interval=120000, key="auto_refresh")  # 2 min

API_KEY = "b2a1234a9ea240f9ba85696e2a243403"
symbols = {
    "EUR/USD": "EUR/USD", "GBP/USD": "GBP/USD", "USD/JPY": "USD/JPY",
    "AUD/USD": "AUD/USD", "USD/CAD": "USD/CAD", "USD/CHF": "USD/CHF",
    "XAU/USD": "XAU/USD", "WTI/USD": "WTI/USD", "EUR/JPY": "EUR/JPY", "NZD/USD": "NZD/USD"
}


def fetch_data(symbol, interval="15min", outputsize=200):
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
    df['TR'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['+DM'] = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']),
                         np.maximum(df['high'] - df['high'].shift(), 0), 0)
    df['-DM'] = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()),
                         np.maximum(df['low'].shift() - df['low'], 0), 0)
    tr14 = df['TR'].rolling(window=period).mean()
    plus_dm14 = df['+DM'].rolling(window=period).mean()
    minus_dm14 = df['-DM'].rolling(window=period).mean()
    plus_di14 = 100 * (plus_dm14 / tr14)
    minus_di14 = 100 * (minus_dm14 / tr14)
    dx = 100 * abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    adx = dx.rolling(window=period).mean()
    return adx


def detect_candle_pattern(df):
    o1, h1, l1, c1 = df['open'].iloc[-2], df['high'].iloc[-2], df['low'].iloc[-2], df['close'].iloc[-2]
    o2, h2, l2, c2 = df['open'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]

    # Bullish Engulfing
    if c2 > o2 and o2 < c1 and c2 > o1 and c1 < o1:
        return "Bullish Engulfing"

    # Bearish Engulfing
    if o2 > c2 and o2 > c1 and c2 < o1 and c1 > o1:
        return "Bearish Engulfing"

    # Hammer
    body = abs(c2 - o2)
    candle_range = h2 - l2
    lower_shadow = min(o2, c2) - l2
    if body < candle_range * 0.3 and lower_shadow > body * 2:
        return "Hammer"

    # Shooting Star
    upper_shadow = h2 - max(o2, c2)
    if body < candle_range * 0.3 and upper_shadow > body * 2:
        return "Shooting Star"

    return ""


def detect_divergence_direction(df):
    df['RSI'] = calculate_rsi(df['close'])
    df = df.dropna()
    close = df['close']
    rsi = df['RSI']
    lows = argrelextrema(close.values, np.less_equal, order=3)[0]
    highs = argrelextrema(close.values, np.greater_equal, order=3)[0]
    if len(lows) >= 2:
        p1, p2 = lows[-2], lows[-1]
        if close.iloc[p2] < close.iloc[p1] and rsi.iloc[p2] > rsi.iloc[p1]:
            return "Bullish"
    if len(highs) >= 2:
        p1, p2 = highs[-2], highs[-1]
        if close.iloc[p2] > close.iloc[p1] and rsi.iloc[p2] < rsi.iloc[p1]:
            return "Bearish"
    return ""


def get_tf_confirmation(symbol):
    for tf in ["5min", "15min", "1h"]:
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
    count = len(indicators)
    if count >= 5:
        confidence = "Strong"
    elif count == 4:
        confidence = "Medium"
    elif count == 3:
        confidence = "Low"
    else:
        return ""
    return f"{confidence} {direction} @ {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f} | Confidence: {confidence}"


def detect_trend_reversal(df):
    if len(df) < 3:
        return ""
    e9 = df['EMA9'].iloc[-3:]
    e20 = df['EMA20'].iloc[-3:]
    if (e9.iloc[0] < e20.iloc[0]) and (e9.iloc[1] > e20.iloc[1]) and (e9.iloc[2] > e20.iloc[2]):
        return "Reversal Confirmed Bullish"
    elif (e9.iloc[0] > e20.iloc[0]) and (e9.iloc[1] < e20.iloc[1]) and (e9.iloc[2] < e20.iloc[2]):
        return "Reversal Confirmed Bearish"
    elif e9.iloc[-2] < e20.iloc[-2] and e9.iloc[-1] > e20.iloc[-1]:
        return "Reversal Forming Bullish"
    elif e9.iloc[-2] > e20.iloc[-2] and e9.iloc[-1] < e20.iloc[-1]:
        return "Reversal Forming Bearish"
    return ""


rows = []
for label, symbol in symbols.items():
    df = fetch_data(symbol, interval="15min")
    if df is not None:
        df['RSI'] = calculate_rsi(df['close'])
        df['MACD'], df['MACD_Signal'] = calculate_macd(df['close'])
        df['EMA9'] = calculate_ema(df['close'], 9)
        df['EMA20'] = calculate_ema(df['close'], 20)
        df['ADX'] = calculate_adx(df)
    if 'volume' in df.columns:
    df['Volume_EMA20'] = df['volume'].ewm(span=20).mean()
else:
    df['Volume_EMA20'] = np.nan
    df['Volume_EMA20'] = np.nan  # or use 0 if preferred
        df = df.dropna()

        price_now = df['close'].iloc[-1]
        direction = detect_divergence_direction(df)
        tf_status = get_tf_confirmation(symbol)
        reversal = detect_trend_reversal(df)

        indicators = []
        if direction:
            indicators.append("RSI")
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
            if df['volume'].iloc[-1] > df['Volume_EMA20'].iloc[-1]:
                indicators.append("Volume Spike")
            pattern = detect_candle_pattern(df)
            if direction in pattern:
                indicators.append("Candle")

        trend = (
            "Bullish" if df['EMA9'].iloc[-1] > df['EMA20'].iloc[-1] and price_now > df['EMA9'].iloc[-1]
            else "Bearish" if df['EMA9'].iloc[-1] < df['EMA20'].iloc[-1] and price_now < df['EMA9'].iloc[-1]
            else "Sideways"
        )

        tf_match = (direction == "Bullish" and "Confirm Bullish" in tf_status) or (
                    direction == "Bearish" and "Confirm Bearish" in tf_status)
        ai_suggestion = generate_ai_suggestion(price_now, direction, indicators, tf_match)
        advice = ai_suggestion if ai_suggestion else "No suggestion"

        rows.append({
            "Pair": label, "Price": round(price_now, 5), "RSI": round(df['RSI'].iloc[-1], 2),
            "Trend": trend, "Divergence": direction, "TF": tf_status,
            "Reversal Signal": reversal,
            "Confirmed Indicators": ", ".join(indicators),
            "AI Suggestion": ai_suggestion, "Advice": advice
        })

df_sorted = pd.DataFrame(rows).sort_values(by="Pair")
st.dataframe(df_sorted, use_container_width=True)
st.caption(f"Updated with Volume + Candle Pattern | Time: {datetime.now().strftime('%H:%M:%S')}")
