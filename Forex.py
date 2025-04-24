# --- Forex Divergence Scanner by Imtiaz ---
import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime
from scipy.signal import argrelextrema

# --- API Key ---
API_KEY = "b2a1234a9ea240f9ba85696e2a243403"

# --- Supported Pairs ---
symbols = {
    "EUR/USD": "EUR/USD",
    "GBP/USD": "GBP/USD",
    "USD/JPY": "USD/JPY",
    "AUD/USD": "AUD/USD",
    "USD/CAD": "USD/CAD"
}

# --- English News Events ---
news_events = {
    "EUR/USD": [
        {"time": "5:30 AM", "title": "German 10Y Bond Auction"},
        {"time": "7:00 AM", "title": "Eurozone Consumer Confidence"},
        {"time": "All Day", "title": "IMF Meetings (Day 2)"}
    ],
    "GBP/USD": [{"time": "2:00 PM", "title": "BoE Governor Speech"}],
    "USD/JPY": [{"time": "7:00 AM", "title": "Richmond Manufacturing Index"}],
    "AUD/USD": [{"time": "4:30 AM", "title": "RBA Meeting Minutes"}],
    "USD/CAD": [{"time": "8:30 AM", "title": "CAD CPI Report"}]
}

# --- Streamlit Setup ---
st.set_page_config(page_title="Forex By Imtiaz & Haris", layout="wide")
st.markdown("""
    <style>body, html, .block-container, .stMarkdown, .stTable, table td, table th {font-size: 20px !important;}</style>
    <h1 style='text-align: center; color:#007acc;'>📊 <b>Forex By Imtiaz & Haris (5min)</b></h1>
""", unsafe_allow_html=True)
st.caption("Includes RSI + Price Divergence (Always), and confirms with: MACD, EMA, Volume, Candlestick, ADX")

# --- Fetch Candle Data ---
def fetch_data_twelvedata(symbol="EUR/USD", interval="5min", outputsize=100):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "apikey": API_KEY}
    response = requests.get(url, params=params)
    data = response.json()
    if "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.astype(float).sort_index()
    return df

# --- Indicator Calculations ---
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
            return "Hammer (Bullish)"
        elif open > close:
            return "Shooting Star (Bearish)"
    return ""

# --- Divergence Detection + AI Suggestion ---
def detect_divergence(df):
    if len(df) < 30:
        return "", "", ""

    close = df['close']
    rsi = df['RSI']
    macd = df['MACD']
    signal = df['MACD_Signal']
    ema9 = df['EMA9']
    ema20 = df['EMA20']
    volume = df['volume']
    adx = df['ADX']
    candle_pattern = detect_candle_pattern(df)

    swing_lows = argrelextrema(close.values, np.less_equal, order=3)[0]
    swing_highs = argrelextrema(close.values, np.greater_equal, order=3)[0]
    rsi_lows = argrelextrema(rsi.values, np.less_equal, order=3)[0]
    rsi_highs = argrelextrema(rsi.values, np.greater_equal, order=3)[0]

    direction = None
    price_now = close.iloc[-1]
    rsi_now = rsi.iloc[-1]

    if len(swing_lows) >= 2 and len(rsi_lows) >= 2:
        p1, p2 = swing_lows[-2], swing_lows[-1]
        if p2 > p1 and close.iloc[p2] < close.iloc[p1] and rsi.iloc[p2] > rsi.iloc[p1]:
            direction = "Bullish"

    if len(swing_highs) >= 2 and len(rsi_highs) >= 2:
        p1, p2 = swing_highs[-2], swing_highs[-1]
        if p2 > p1 and close.iloc[p2] > close.iloc[p1] and rsi.iloc[p2] < rsi.iloc[p1]:
            direction = "Bearish"

    if not direction:
        return "", "", ""

    indicators = ["✅ RSI + Price Divergence"]
    macd_now = macd.iloc[-1]
    macd_signal = signal.iloc[-1]
    ema9_now = ema9.iloc[-1]
    ema20_now = ema20.iloc[-1]
    vol_now = volume.iloc[-1]
    vol_avg = volume.rolling(window=14).mean().iloc[-1]
    adx_now = adx.iloc[-1]

    if direction == "Bullish":
        if macd_now > macd_signal:
            indicators.append("✅ MACD Bullish")
        if price_now > ema9_now and price_now > ema20_now:
            indicators.append("✅ Price > EMA9 & EMA20")
        if vol_now > vol_avg:
            indicators.append("✅ Volume > Avg")
        if adx_now > 20:
            indicators.append(f"✅ ADX: {adx_now:.2f}")
        if "Bullish" in candle_pattern:
            indicators.append(f"✅ Pattern: {candle_pattern}")
    else:
        if macd_now < macd_signal:
            indicators.append("✅ MACD Bearish")
        if price_now < ema9_now and price_now < ema20_now:
            indicators.append("✅ Price < EMA9 & EMA20")
        if vol_now > vol_avg:
            indicators.append("✅ Volume > Avg")
        if adx_now > 20:
            indicators.append(f"✅ ADX: {adx_now:.2f}")
        if "Bearish" in candle_pattern:
            indicators.append(f"✅ Pattern: {candle_pattern}")

    confidence = "High" if len(indicators) >= 4 else "Medium"
    if direction == "Bullish":
        sl = price_now - (price_now * 0.002)
        tp = price_now + (price_now * 0.004)
        suggestion = f"📈 Buy @ {price_now:.5f} | SL: {sl:.5f} | TP: {tp:.5f} | Confidence: {confidence}"
    else:
        sl = price_now + (price_now * 0.002)
        tp = price_now - (price_now * 0.004)
        suggestion = f"📉 Sell @ {price_now:.5f} | SL: {sl:.5f} | TP: {tp:.5f} | Confidence: {confidence}"

    signal_text = f"{'🟢' if direction == 'Bullish' else '🔴'} {direction} Divergence @ {price_now:.5f} (RSI {rsi_now:.2f})"
    return signal_text, ", ".join(indicators), suggestion

# --- Build Table ---
rows = []
for label, symbol in symbols.items():
    df = fetch_data_twelvedata(symbol)
    if df is not None:
        if "volume" not in df:
            df["volume"] = 0.0
        df['RSI'] = calculate_rsi(df['close'])
        df['MACD'], df['MACD_Signal'] = calculate_macd(df['close'])
        df['EMA9'] = calculate_ema(df['close'], 9)
        df['EMA20'] = calculate_ema(df['close'], 20)
        df['ADX'] = calculate_adx(df)
        df = df.dropna()

        if not df.empty:
            price_now = df['close'].iloc[-1]
            price_prev = df['close'].iloc[-2]
            price = f"<span style='color:green'>{price_now:.5f}</span>" if price_now > price_prev else f"<span style='color:red'>{price_now:.5f}</span>"
            rsi = round(df['RSI'].iloc[-1], 2)
            high = df['high'].max()
            low = df['low'].min()
            trend = "⬆️" if df['close'].iloc[-1] > df['close'].iloc[-5] else "⬇️"
            trend_colored = f"<span style='color:green'>{trend} Up</span>" if trend == "⬆️" else f"<span style='color:red'>{trend} Down</span>"

            div, indicators, suggestion = detect_divergence(df)
            support = df['close'].iloc[-5:].min()
            resistance = df['close'].iloc[-5:].max()
            news_list = news_events.get(label, [])
            news_html = "<br>".join([f"🕒 {n['time']} - {n['title']}" for n in news_list])

            rows.append({
                "Pair": f"<b style='font-size:16px'>{label}</b>",
                "Current Price": price,
                "RSI": rsi,
                "High": round(high, 4),
                "Low": round(low, 4),
                "Trend": trend_colored,
                "Divergence": div,
                "Indicators": indicators,
                "Support": f"{support:.5f}",
                "Resistance": f"{resistance:.5f}",
                "AI Suggestion": suggestion,
                "News": news_html
            })

# --- Display Table ---
df_result = pd.DataFrame(rows)
st.write(df_result.to_html(escape=False, index=False), unsafe_allow_html=True)
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (TF: 5min)")
