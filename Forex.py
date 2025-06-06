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
    "EUR/NZD": "EUR/NZD", "AUD/JPY": "AUD/JPY", "GBP/NZD": "GBP/NZD",
    "EUR/NZD": "EUR/NZD", "XAG/USD": "XAG/USD",
}

# --- Play Sound Alert if RSI exceeds threshold ---
def play_rsi_alert():
    components.html("""
    <audio autoplay>
        <source src="https://www.soundjay.com/button/beep-07.wav" type="audio/wav">
    </audio>
    """, height=0)

# --- Fetch DXY Data ---
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
        print("⚠️ yfinance failed, fallback to static DXY", e)
        dxy_price = 100.237
        dxy_previous = 100.40
        change = dxy_price - dxy_previous
        percent = (change / dxy_previous) * 100
        return dxy_price, percent

# --- Fetch Forex Factory News ---
def fetch_forex_factory_news():
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    response = requests.get(url)
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
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

news_events = fetch_forex_factory_news()
dxy_price, dxy_change = fetch_dxy_data()
rows = []

# --- RSI Divergence Detection Functions (ADDED) ---
def find_local_extrema(series, order=3):
    minima, maxima = [], []
    for i in range(order, len(series) - order):
        window = series[i - order:i + order + 1]
        if series[i] == min(window):
            minima.append(i)
        if series[i] == max(window):
            maxima.append(i)
    return minima, maxima

def check_rsi_divergence(df, lookback=30):
    if len(df) < lookback:
        return ""
    prices = df['close'].iloc[-lookback:].reset_index(drop=True)
    rsi = df['RSI'].iloc[-lookback:].reset_index(drop=True)

    min_idx, max_idx = find_local_extrema(prices)

    # Bullish divergence: price lower low, RSI higher low
    for i in range(len(min_idx) - 1):
        idx1, idx2 = min_idx[i], min_idx[i + 1]
        if idx2 <= idx1:
            continue
        if prices[idx2] < prices[idx1] and rsi[idx2] > rsi[idx1]:
            return "Bullish Divergence"

    # Bearish divergence: price higher high, RSI lower high
    for i in range(len(max_idx) - 1):
        idx1, idx2 = max_idx[i], max_idx[i + 1]
        if idx2 <= idx1:
            continue
        if prices[idx2] > prices[idx1] and rsi[idx2] < rsi[idx1]:
            return "Bearish Divergence"

    return ""

# --- Continued: Forex AI Signal with RSI Sound Alert (Part 2/2) ---

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
    quote = quote.upper()
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

def detect_candle_pattern(df):
    o, c, h, l = df['open'].iloc[-2:], df['close'].iloc[-2:], df['high'].iloc[-2:], df['low'].iloc[-2:]
    co, cc, ch, cl = o.iloc[-1], c.iloc[-1], h.iloc[-1], l.iloc[-1]
    po, pc = o.iloc[-2], c.iloc[-2]
    body, range_ = abs(cc - co), ch - cl
    if body < range_ * 0.1: return "Doji"
    if pc < po and cc > co and cc > po and co < pc: return "Bullish Engulfing"
    if pc > po and cc < co and cc < po and co > pc: return "Bearish Engulfing"
    if body < range_ * 0.3 and cl < co and cl < cc and (ch - max(co, cc)) < body: return "Hammer"
    if body < range_ * 0.3 and ch > co and ch > cc and (min(co, cc) - cl) < body: return "Shooting Star"
    return ""

def process_symbol(symbol):
    url = f"https://api.metatrader5.com/v1/timeseries/{symbol.replace('/', '')}?interval=15m&count=150"
    try:
        response = requests.get(url, headers={"X-Api-Key": API_KEY})
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

    df = pd.DataFrame(data)
    if df.empty or 'close' not in df.columns:
        return None
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df.astype(float)

    df['RSI'] = calculate_rsi(df['close'])
    df['MACD'], df['MACD_SIGNAL'] = calculate_macd(df['close'])
    df['EMA12'] = calculate_ema(df['close'], 12)
    df['EMA26'] = calculate_ema(df['close'], 26)
    df['ATR'] = calculate_atr(df)
    df['ADX'] = calculate_adx(df)
    df['CANDLE_PATTERN'] = detect_candle_pattern(df)

    rsi_last = df['RSI'].iloc[-1]
    macd_last = df['MACD'].iloc[-1]
    macd_signal_last = df['MACD_SIGNAL'].iloc[-1]

    # RSI alert sound on thresholds
    if rsi_last < 30 or rsi_last > 70:
        play_rsi_alert()

    # RSI divergence detection (ADDED)
    rsi_div = check_rsi_divergence(df)

    # Build signal and notes based on conditions (your existing logic can be extended)
    signal = "Neutral"
    notes = []
    if rsi_last < 30:
        signal = "Buy"
        notes.append("RSI oversold")
    elif rsi_last > 70:
        signal = "Sell"
        notes.append("RSI overbought")
    if macd_last > macd_signal_last:
        notes.append("MACD bullish")
    elif macd_last < macd_signal_last:
        notes.append("MACD bearish")
    if df['ADX'].iloc[-1] > 25:
        notes.append("Strong trend")
    if df['CANDLE_PATTERN'].iloc[-1]:
        notes.append(df['CANDLE_PATTERN'].iloc[-1])
    if rsi_div:
        notes.append(rsi_div)

    return {
        "Symbol": symbol,
        "Price": round(df['close'].iloc[-1], 5),
        "RSI": round(rsi_last, 2),
        "MACD": round(macd_last, 5),
        "Signal": signal,
        "Notes": ", ".join(notes),
        "RSI Divergence": rsi_div,
        "News (Today)": get_today_news_with_impact(symbol),
        "Next News": get_next_news(symbol)
    }

for sym in symbols:
    result = process_symbol(sym)
    if result:
        rows.append(result)

df_signals = pd.DataFrame(rows)

# --- Display Section ---
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown(f"<h3>US Dollar Index (DXY): {dxy_price:.3f} ({dxy_change:+.2f}%)</h3>", unsafe_allow_html=True)
    st.write("Note: Data refreshed every 2 minutes.")

with col2:
    st.dataframe(df_signals.style.format({
        "Price": "{:.5f}",
        "RSI": "{:.2f}",
        "MACD": "{:.5f}",
    }), height=500)

# Display news for selected pair (optional)
if not df_signals.empty:
    selected_pair = st.selectbox("Select pair to view today's news", df_signals['Symbol'])
    news_list = df_signals.loc[df_signals['Symbol'] == selected_pair, "News (Today)"].values[0]
    st.write(f"### News Today for {selected_pair}")
    for n in news_list:
        st.write(f"- {n}")
