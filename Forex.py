# --- Forex AI Signal with RSI Sound Alert ---
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import requests
import numpy as np
from datetime import datetime
from pytz import timezone
import xml.etree.ElementTree as ET
from dateutil import parser as date_parser
import streamlit.components.v1 as components
import yfinance as yf

st.set_page_config(page_title="Signals", layout="wide")
st.markdown("<h1 style='text-align:center; color:#007acc;'>📊 Signals + News</h1>", unsafe_allow_html=True)
st_autorefresh(interval=120000, key="ai_refresh")

API_KEY = "b2a1234a9ea240f9ba85696e2a243403"

symbols = {
    "EUR/USD": "EUR/USD", "GBP/USD": "GBP/USD", "USD/JPY": "USD/JPY",
    "AUD/USD": "AUD/USD", "USD/CAD": "USD/CAD", "USD/CHF": "USD/CHF",
    "XAU/USD": "XAU/USD", "WTI/USD": "WTI/USD", "EUR/JPY": "EUR/JPY", "NZD/USD": "NZD/USD",
    "EUR/GBP": "EUR/GBP", "EUR/CAD": "EUR/CAD", "GBP/JPY": "GBP/JPY",
    "EUR/AUD": "EUR/AUD", "AUD/JPY": "AUD/JPY", "GBP/NZD": "GBP/NZD",
    "EUR/NZD": "EUR/NZD", "XAG/USD": "XAG/USD",
}

# --- Sound Alert ---
def play_rsi_alert():
    components.html("""
    <audio autoplay>
        <source src="https://www.soundjay.com/button/beep-07.wav" type="audio/wav">
    </audio>
    """, height=0)

# --- DXY Data ---
def fetch_dxy_data():
    try:
        dxy = yf.Ticker("DX-Y.NYB")
        data = dxy.history(period="1d", interval="1m")
        if data.empty:
            raise ValueError("No DXY data")
        current = data["Close"].iloc[-1]
        previous = data["Close"].iloc[0]
        change = current - previous
        percent = (change / previous) * 100
        return current, percent
    except Exception as e:
        print("⚠️ yfinance DXY failed:", e)
        fallback = 100.237
        fallback_prev = 100.40
        change = fallback - fallback_prev
        percent = (change / fallback_prev) * 100
        return fallback, percent

# --- News from Forex Factory ---
def fetch_forex_factory_news():
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    try:
        root = ET.fromstring(requests.get(url).content)
    except ET.ParseError:
        return []

    news = []
    for item in root.findall("./channel/item"):
        try:
            title = item.find("title").text
            time = date_parser.parse(item.find("pubDate").text)
            currency = item.find("{http://www.forexfactory.com/rss}currency").text.upper()
            if time.date() == datetime.utcnow().date():
                news.append({"title": title, "time": time, "currency": currency})
        except:
            continue
    return news

news_events = fetch_forex_factory_news()
dxy_price, dxy_change = fetch_dxy_data()
rows = []

# --- Indicator Functions ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
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

def calculate_atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

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
    return dx.rolling(window=period).mean()

def detect_candle_pattern(df):
    o, c, h, l = df['open'].iloc[-2:], df['close'].iloc[-2:], df['high'].iloc[-2:], df['low'].iloc[-2:]
    co, cc, ch, cl = o.iloc[-1], c.iloc[-1], h.iloc[-1], l.iloc[-1]
    po, pc = o.iloc[-2], c.iloc[-2]
    body, rng = abs(cc - co), ch - cl
    if body < rng * 0.1: return "Doji"
    if pc < po and cc > co and cc > po and co < pc: return "Bullish Engulfing"
    if pc > po and cc < co and cc < po and co > pc: return "Bearish Engulfing"
    if body < rng * 0.3 and cl < co and cl < cc: return "Hammer"
    if body < rng * 0.3 and ch > co and ch > cc: return "Shooting Star"
    return ""

def detect_trend_reversal(df):
    e9, e20 = df['EMA9'].iloc[-3:], df['EMA20'].iloc[-3:]
    if e9[0] < e20[0] and e9[1] > e20[1] and e9[2] > e20[2]: return "Reversal Confirmed Bullish"
    if e9[0] > e20[0] and e9[1] < e20[1] and e9[2] < e20[2]: return "Reversal Confirmed Bearish"
    if e9[-2] < e20[-2] and e9[-1] > e20[-1]: return "Reversal Forming Bullish"
    if e9[-2] > e20[-2] and e9[-1] < e20[-1]: return "Reversal Forming Bearish"
    return ""

def generate_ai_suggestion(price, indicators, atr, signal_type):
    if not indicators: return ""
    sl = price - atr * 1.2 if signal_type == "Bullish" else price + atr * 1.2
    tp = price + atr * 2.5 if signal_type == "Bullish" else price - atr * 2.5
    count = len(indicators)
    if count >= 4:
        conf = "Strong"
    elif count == 3:
        conf = "Medium"
    else:
        return ""
    color = "green" if signal_type == "Bullish" else "red"
    return f"{conf} <span style='color:{color}'>{signal_type}</span> Signal @ {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f} | Confidence: {conf}"

# --- MAIN SIGNAL SCANNER LOOP ---
for label, symbol in symbols.items():
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=5min&outputsize=200&apikey={API_KEY}"
        data = requests.get(url).json()
        if "values" not in data:
            continue
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df = df.astype(float).sort_index()

        df["RSI"] = calculate_rsi(df["close"])
        df["MACD"], df["MACD_Signal"] = calculate_macd(df["close"])
        df["EMA9"] = calculate_ema(df["close"], 9)
        df["EMA20"] = calculate_ema(df["close"], 20)
        df["ADX"] = calculate_adx(df)
        df["ATR"] = calculate_atr(df)
        df.dropna(inplace=True)

        price = df["close"].iloc[-1]
        atr = df["ATR"].iloc[-1]
        rsi_val = df["RSI"].iloc[-1]

        # Optional RSI Alert
        # if rsi_val > 70 or rsi_val < 20:
        #     play_rsi_alert()
        #     st.warning(f"🔔 RSI Alert for {label}: RSI = {rsi_val:.2f}")

        trend = "Bullish" if df["EMA9"].iloc[-1] > df["EMA20"].iloc[-1] and price > df["EMA9"].iloc[-1] else \
                "Bearish" if df["EMA9"].iloc[-1] < df["EMA20"].iloc[-1] and price < df["EMA9"].iloc[-1] else "Sideways"

        indicators, signal_type = [], ""
        if rsi_val > 50:
            indicators.append("Bullish"); signal_type = "Bullish"
        elif rsi_val < 50:
            indicators.append("Bearish"); signal_type = "Bearish"
        if df["MACD"].iloc[-1] > df["MACD_Signal"].iloc[-1]: indicators.append("MACD")
        if df["EMA9"].iloc[-1] > df["EMA20"].iloc[-1] and price > df["EMA9"].iloc[-1]: indicators.append("EMA")
        if df["ADX"].iloc[-1] > 20: indicators.append("ADX")
        pattern = detect_candle_pattern(df)
        if pattern: indicators.append("Candle")

        suggestion = generate_ai_suggestion(price, indicators, atr, signal_type)
        if not suggestion: continue

        rows.append({
            "Pair": label,
            "Price": round(price, 5),
            "RSI 5M": round(rsi_val, 2),
            "RSI 15 M": "—",
            "RSI 1H": "—",
            "RSI 4H": "—",
            "ATR": round(atr, 5),
            "ATR Status": "🔴 Low" if atr < 0.0004 else "🟡 Normal" if atr < 0.0009 else "🟢 High",
            "Trend 5m": trend,
            "Trend Daily": "—",
            "Reversal Signal": detect_trend_reversal(df),
            "Signal Type": signal_type,
            "Confirmed Indicators": ", ".join(indicators),
            "Candle Pattern": pattern or "—",
            "AI Suggestion": suggestion,
            "DXY Impact": f"{dxy_price:.2f} ({dxy_change:+.2f}%)" if "USD" in label else "—"
        })
    except Exception as e:
        print(f"Error with {label}: {e}")
        continue

# --- Display Results ---
column_order = ["Pair", "Price", "RSI 5M", "RSI 15 M", "RSI 1H", "RSI 4H", "ATR", "ATR Status",
                "Trend 5m", "Trend Daily", "Reversal Signal", "Signal Type",
                "Confirmed Indicators", "Candle Pattern", "AI Suggestion", "DXY Impact"]

df_result = pd.DataFrame(rows)
for col in column_order:
    if col not in df_result.columns:
        df_result[col] = "—"
df_result = df_result[column_order]
df_result["Score"] = df_result["AI Suggestion"].apply(lambda x: 3 if "Strong" in x else 2 if "Medium" in x else 0)
df_sorted = df_result.sort_values(by="Score", ascending=False).drop(columns=["Score"])

st.dataframe(df_sorted, use_container_width=True)
st.caption(f"Timeframe: 5-Min | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.text(f"Scanned Pairs: {len(rows)}")
st.text(f"Strong Signals Found: {len([r for r in rows if 'Strong' in r['AI Suggestion']])}")
