# --- Signals with H & I (15-Min Timeframe | Enhanced Version with Confirmation Filters) ---
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import argrelextrema

st.set_page_config(page_title="Forex AI Signals", layout="wide")
st_autorefresh(interval=120000, key="auto_refresh")

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

def detect_volume_spike(df):
    return df['volume'].iloc[-1] > df['volume'].rolling(window=10).mean().iloc[-1] * 1.5

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

def check_news_block(pair):
    now = datetime.now()
    for event in news_events.get(pair, []):
        try:
            event_time = datetime.strptime(event["time"], "%H:%M").replace(year=now.year, month=now.month, day=now.day)
            if timedelta(0) <= (event_time - now) <= timedelta(minutes=30):
                return True
        except:
            continue
    return False

def generate_ai_suggestion(price, direction, indicators, tf_confirmed):
    sl = price * (1 - 0.002) if direction == "Bullish" else price * (1 + 0.002)
    tp = price * (1 + 0.004) if direction == "Bullish" else price * (1 - 0.004)
    count = len(indicators)
    if count >= 4:
        confidence = "Strong"
    elif count == 3:
        confidence = "Medium"
    else:
        return ""
    return f"{confidence} {direction} @ {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f} | Confidence: {confidence}"

def detect_trend(df):
    if df['EMA9'].iloc[-1] > df['EMA20'].iloc[-1] and df['close'].iloc[-1] > df['EMA9'].iloc[-1]:
        return "Bullish"
    elif df['EMA9'].iloc[-1] < df['EMA20'].iloc[-1] and df['close'].iloc[-1] < df['EMA9'].iloc[-1]:
        return "Bearish"
    return "Sideways"
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
        trend = detect_trend(df)
        tf_status = get_tf_confirmation(symbol)

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
            if detect_candle_pattern(df) == direction:
                indicators.append("Candle")
            if detect_volume_spike(df):
                indicators.append("Volume")

        ai_suggestion = ""
        if direction and trend == direction and len(indicators) >= 3 and not check_news_block(label):
            ai_suggestion = generate_ai_suggestion(price_now, direction, indicators, tf_status)

        rows.append({
            "Pair": label, "Price": round(price_now, 5), "RSI": round(df['RSI'].iloc[-1], 2),
            "Trend": trend, "Divergence": direction, "TF": tf_status,
            "Confirmed Indicators": ", ".join(indicators),
            "AI Suggestion": ai_suggestion,
            "News Alert": "Yes" if check_news_block(label) else ""
        })

# --- Display Table ---
st.markdown("### 📊 AI Signal Table")
df_sorted = pd.DataFrame(rows)
def color_confidence(val):
    if isinstance(val, str) and "Strong" in val:
        return 'background-color: #b3ffd9'
    elif isinstance(val, str) and "Medium" in val:
        return 'background-color: #ffffcc'
    return ''

st.dataframe(df_sorted.style.applymap(color_confidence, subset=["AI Suggestion"]), use_container_width=True)
st.caption(f"Timeframe: 15-Min | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.text(f"Total Pairs Scanned: {len(rows)}")
strong_signals = [r for r in rows if "Confidence: Strong" in r["AI Suggestion"]]
st.text(f"Strong Signals: {len(strong_signals)}")
