# --- Forex AI Signal with RSI Sound Alert (Part 1/2) ---
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

def play_rsi_alert():
    components.html("""
    <audio autoplay>
        <source src="https://www.soundjay.com/button/beep-07.wav" type="audio/wav">
    </audio>
    """, height=0)

def fetch_dxy_data():
    try:
        dxy = yf.Ticker("DX-Y.NYB")
        data = dxy.history(period="1d", interval="1m")
        if data.empty:
            raise ValueError("No data received from yfinance")
        current = data["Close"].iloc[-1]
        previous = data["Close"].iloc[0]
        change = current - previous
        percent = (change / previous) * 100
        return current, percent
    except Exception as e:
        dxy_price = 100.237
        dxy_previous = 100.40
        change = dxy_price - dxy_previous
        percent = (change / dxy_previous) * 100
        return dxy_price, percent

def fetch_forex_factory_news():
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    try:
        response = requests.get(url)
        root = ET.fromstring(response.content)
    except:
        return []

    news_data = []
    for item in root.findall("./channel/item"):
        try:
            title = item.find("title").text
            pub_time = date_parser.parse(item.find("pubDate").text)
            currency = item.find("{http://www.forexfactory.com/rss}currency").text.strip().upper()
            if pub_time.date() == datetime.utcnow().date():
                news_data.append({
                    "title": title,
                    "time": pub_time,
                    "currency": currency
                })
        except:
            continue
    return news_data

def analyze_impact(title):
    title = title.lower()
    if any(x in title for x in ["cpi", "gdp", "employment", "retail", "core", "inflation", "interest rate"]):
        if any(w in title for w in ["increase", "higher", "rises", "strong", "beats"]):
            return "🟢 Positive"
        elif any(w in title for w in ["decrease", "lower", "falls", "weak", "misses"]):
            return "🔴 Negative"
        else:
            return "🟡 Mixed"
    return "⚪ Neutral"

def get_today_news_with_impact(pair):
    base, quote = pair.split('/')
    today_events = []
    for n in news_events:
        if n["currency"] == quote:
            impact = analyze_impact(n["title"])
            time_str = n["time"].strftime("%H:%M")
            today_events.append(f"{n['title']} ({impact}) @ {time_str}")
    return today_events or ["—"]

def get_next_news(pair):
    base, quote = pair.split('/')
    mapping = {
        "USD": ["USD", "United States", "US", "U.S."],
        "EUR": ["EUR", "Eurozone", "Germany", "France"],
        "GBP": ["GBP", "UK", "Britain", "England"],
        "JPY": ["JPY", "Japan"],
        "AUD": ["AUD", "Australia"],
        "CAD": ["CAD", "Canada"],
        "CHF": ["CHF", "Switzerland"],
        "NZD": ["NZD", "New Zealand"],
        "XAU": ["Gold"],
        "XAG": ["Silver"],
        "WTI": ["Oil", "Crude"]
    }
    keywords = mapping.get(base, []) + mapping.get(quote, [])
    upcoming = [n for n in news_events if any(k.lower() in n["title"].lower() for k in keywords) and n["time"] > datetime.utcnow()]
    if upcoming:
        next_event = sorted(upcoming, key=lambda x: x["time"])[0]
        return f"{next_event['title']} @ {next_event['time'].strftime('%H:%M')}"
    return "—"

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

def calculate_atr(df, period=14):
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
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
# --- Forex AI Signal with RSI Sound Alert (Part 2/2) ---

def detect_candle_reversal(df):
    body = abs(df['close'] - df['open'])
    range_ = df['high'] - df['low']
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    long_tail = lower_wick > (2 * body)
    pinbar = (long_tail) & (upper_wick < body)
    engulfing = (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
    reversal = pinbar | engulfing
    return reversal.shift().fillna(False)

def fetch_forex_data(pair):
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={pair}&interval=5min&outputsize=100&apikey={API_KEY}"
        response = requests.get(url).json()
        df = pd.DataFrame(response["values"])
        df.columns = df.columns.str.lower()
        df = df.rename(columns={"datetime": "time"})
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
        return df
    except:
        return pd.DataFrame()

def compute_signal(df):
    df['rsi'] = calculate_rsi(df['close'])
    macd, signal = calculate_macd(df['close'])
    df['macd'] = macd
    df['macd_signal'] = signal
    df['ema_20'] = calculate_ema(df['close'], 20)
    df['ema_50'] = calculate_ema(df['close'], 50)
    df['ema_200'] = calculate_ema(df['close'], 200)
    df['adx'] = calculate_adx(df.copy())

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    score = 0
    reasons = []

    # RSI Signals
    if latest['rsi'] < 30:
        score += 2
        reasons.append("RSI Oversold")
        play_rsi_alert()
    elif latest['rsi'] > 70:
        score -= 2
        reasons.append("RSI Overbought")
        play_rsi_alert()

    # MACD
    if latest['macd'] > latest['macd_signal'] and previous['macd'] < previous['macd_signal']:
        score += 1
        reasons.append("MACD Bullish Cross")
    elif latest['macd'] < latest['macd_signal'] and previous['macd'] > previous['macd_signal']:
        score -= 1
        reasons.append("MACD Bearish Cross")

    # EMA trend
    if latest['ema_20'] > latest['ema_50'] > latest['ema_200']:
        score += 1
        reasons.append("EMA Bullish Alignment")
    elif latest['ema_20'] < latest['ema_50'] < latest['ema_200']:
        score -= 1
        reasons.append("EMA Bearish Alignment")

    # ADX
    if latest['adx'] > 25:
        score += 1
        reasons.append("Strong Trend (ADX)")

    # Reversal
    reversal = detect_candle_reversal(df)
    if reversal.iloc[-1]:
        score += 2
        reasons.append("Candle Reversal")

    signal = "🟢 Buy" if score >= 3 else "🔴 Sell" if score <= -3 else "⚪ Hold"
    return signal, reasons, latest['rsi']

# Fetch all news only once
news_events = fetch_forex_factory_news()

# Display DXY
dxy, dxy_change = fetch_dxy_data()
dxy_col = st.columns(1)[0]
dxy_col.metric("💵 DXY Index", f"{dxy:.2f}", f"{dxy_change:.2f}%")

# Signal Grid
cols = st.columns(4)
for idx, (symbol, name) in enumerate(symbols.items()):
    df = fetch_forex_data(symbol.replace("/", ""))
    if df.empty:
        continue

    signal, reasons, rsi = compute_signal(df)
    next_news = get_next_news(symbol)
    today_news = get_today_news_with_impact(symbol)

    with cols[idx % 4]:
        st.subheader(f"📈 {symbol}")
        st.metric("📍 Signal", signal)
        st.text(f"RSI: {rsi:.1f}")
        st.text("Reasons:")
        for r in reasons:
            st.markdown(f"- {r}")
        st.text("📅 Today News:")
        for event in today_news:
            st.markdown(f"- {event}")
        st.markdown(f"⏳ Next: {next_news}")
