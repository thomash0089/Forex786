# --- Signals with H & I (15-Min Timeframe | Cleaned Table) ---
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta, timezone
from scipy.signal import argrelextrema

st.set_page_config(page_title="Forex AI Signals", layout="wide")
st_autorefresh(interval=120000, key="auto_refresh")  # 2 min

API_KEY = "b2a1234a9ea240f9ba85696e2a243403"
symbols = {
    "EUR/USD": "EUR/USD", "GBP/USD": "GBP/USD", "USD/JPY": "USD/JPY",
    "AUD/USD": "AUD/USD", "USD/CAD": "USD/CAD", "USD/CHF": "USD/CHF",
    "XAU/USD": "XAU/USD", "WTI/USD": "WTI/USD", "EUR/JPY": "EUR/JPY", "NZD/USD": "NZD/USD"
}

news_events = {
    "EUR/USD": [{"time": "10:30", "title": "Euro CPI Data"}],
    "GBP/USD": [{"time": "11:00", "title": "BoE Governor Speech"}],
    "USD/JPY": [{"time": "13:00", "title": "US Jobless Claims"}],
    "AUD/USD": [{"time": "08:00", "title": "RBA Statement"}],
    "USD/CAD": [{"time": "15:00", "title": "Canada Trade Balance"}],
    "USD/CHF": [{"time": "14:00", "title": "US Fed Chair Remarks"}],
    "XAU/USD": [{"time": "13:30", "title": "Gold Reserve Report"}],
    "WTI/USD": [{"time": "12:30", "title": "Crude Oil Inventory"}],
    "EUR/JPY": [{"time": "09:00", "title": "ECB Bulletin"}],
    "NZD/USD": [{"time": "07:30", "title": "NZ Employment Report"}],
}

st.markdown("""
    <style>body, html, .block-container, table td, table th {font-size: 18px !important;}</style>
    <h1 style='text-align: center; color:#007acc;'>📊 Signals with H & I (15-Min Timeframe)</h1>
""", unsafe_allow_html=True)

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

divergence_timestamps = {}

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
    if count >= 4:
        confidence = "Strong"
    elif count == 3:
        confidence = "Medium"
    elif count == 2:
        confidence = "Low"
    else:
        return ""
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

def check_news_alert(pair):
    now = datetime.now()
    alert_list = []
    for event in news_events.get(pair, []):
        try:
            event_time = datetime.strptime(event["time"], "%H:%M").replace(year=now.year, month=now.month, day=now.day)
            if timedelta(0) <= (event_time - now) <= timedelta(minutes=30):
                alert_list.append(f"{event['title']} @ {event['time']}")
        except:
            continue
    return " | ".join(alert_list) if alert_list else ""

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
            pattern = detect_candle_pattern(df)
            if direction in pattern:
                indicators.append("Candle")

        trend = (
            "Bullish" if df['EMA9'].iloc[-1] > df['EMA20'].iloc[-1] and price_now > df['EMA9'].iloc[-1]
            else "Bearish" if df['EMA9'].iloc[-1] < df['EMA20'].iloc[-1] and price_now < df['EMA9'].iloc[-1]
            else "Sideways"
        )

        tf_match = (direction == "Bullish" and "Confirm Bullish" in tf_status) or (direction == "Bearish" and "Confirm Bearish" in tf_status)
        ai_suggestion = generate_ai_suggestion(price_now, direction, indicators, tf_match)
        advice = generate_advice(trend, direction, ai_suggestion, tf_status)

        rows.append({
            "Pair": label, "Price": round(price_now, 5), "RSI": round(df['RSI'].iloc[-1], 2),
            "Trend": trend, "Divergence": direction, "TF": tf_status,
            "Reversal Signal": reversal,
            "Confirmed Indicators": ", ".join(indicators),
            "AI Suggestion": ai_suggestion, "Advice": advice,
            "News Alert": check_news_alert(label)
        })

import streamlit.components.v1 as components
column_order = ["Pair", "Price", "RSI", "Trend", "Divergence", "TF", "Reversal Signal", "Confirmed Indicators", "AI Suggestion", "Advice", "News Alert"]

styled_html = "<table style='width:100%; border-collapse: collapse;'>"
styled_html += "<tr>" + "".join([
    f"<th style='border: 1px solid #ccc; padding: 6px; background-color:#e0e0e0'>{col}</th>"
    for col in column_order
]) + "</tr>"

def style_row(row):
    ai = row['AI Suggestion']
    tf = row['TF']
    trend = row['Trend']
    div = row['Divergence']
    if (
        pd.notna(ai) and "Confidence: Strong" in ai and trend == div
        and ((div == "Bullish" and "Confirm Bullish" in tf) or (div == "Bearish" and "Confirm Bearish" in tf))
    ):
        return 'background-color: #add8e6;'
    if (
        pd.notna(ai) and "Confidence: Medium" in ai and trend == div
        and ((div == "Bullish" and "Confirm Bullish" in tf) or (div == "Bearish" and "Confirm Bearish" in tf))
    ):
        return 'background-color: #ccffcc;'
    if "Reversal" in row['Reversal Signal']:
        return 'background-color: #fff0b3;'
    return ''

def trend_color_text(trend):
    color = "green" if trend == "Bullish" else "red" if trend == "Bearish" else "gray"
    return f"<span style='color:{color}; font-weight:bold;'>{trend}</span>"

df_sorted = pd.DataFrame(rows)
df_sorted = df_sorted.sort_values(by="Pair", na_position='last')

for _, row in df_sorted.iterrows():
    style = style_row(row)
    styled_html += f"<tr style='{style}'>"
    for col in column_order:
        val = row[col]
        if col == "Pair":
            val = f"<strong style='font-size: 18px;'>{val}</strong>"
        elif col == "Trend":
            val = trend_color_text(val)
        styled_html += f"<td style='border: 1px solid #ccc; padding: 6px;'>{val}</td>"
    styled_html += "</tr>"
styled_html += "</table>"

st.markdown(styled_html, unsafe_allow_html=True)
st.caption(f"Timeframe: 15-Min | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.text(f"Scanned Pairs: {len(rows)}")
strongs = [r for r in rows if "Confidence: Strong" in r["AI Suggestion"]]
st.text(f"Strong Signals Found: {len(strongs)}")
