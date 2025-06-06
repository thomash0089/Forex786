# --- Forex AI Signal with RSI Divergence and Sound Alert (Part 1/2) ---
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

# --- Indicator Calculation Functions ---

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

# --- Candle Pattern Detection ---
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

# --- Trend Reversal Detection ---
def detect_trend_reversal(df):
    e9, e20 = df['EMA9'].iloc[-3:], df['EMA20'].iloc[-3:]
    if e9[0] < e20[0] and e9[1] > e20[1] and e9[2] > e20[2]: return "Reversal Confirmed Bullish"
    if e9[0] > e20[0] and e9[1] < e20[1] and e9[2] < e20[2]: return "Reversal Confirmed Bearish"
    if e9[-2] < e20[-2] and e9[-1] > e20[-1]: return "Reversal Forming Bullish"
    if e9[-2] > e20[-2] and e9[-1] < e20[-1]: return "Reversal Forming Bearish"
    return ""

# --- AI Suggestion Generator ---
def generate_ai_suggestion(price, indicators, atr, signal_type):
    if not indicators: return ""
    sl = price - (atr * 1.2) if signal_type == "Bullish" else price + (atr * 1.2)
    tp = price + (atr * 2.5) if signal_type == "Bullish" else price - (atr * 2.5)
    count = len(indicators)
    if count >= 4:
        conf = "Strong"
    elif count == 3:
        conf = "Medium"
    else:
        return ""
    color = "green" if signal_type == "Bullish" else "red"
    signal_txt = f"{conf} <span style='color:{color}'>{signal_type}</span> Signal @ {price:.5f}"
    return f"{signal_txt} | SL: {sl:.5f} | TP: {tp:.5f} | Confidence: {conf}"

# --- News Impact & Fetch Helpers ---
def analyze_impact(title):
    title = title.lower()
    if any(x in title for x in ["cpi", "gdp", "employment", "retail", "core", "inflation", "interest rate"]):
        if any(w in title for w in ["increase", "higher", "rises", "expands", "beats"]):
            return "Positive"
        if any(w in title for w in ["decrease", "lower", "falls", "contracts", "misses"]):
            return "Negative"
    return "Neutral"

def currency_news(currency):
    return [news for news in news_events if news['currency'] == currency]

# --- RSI Divergence Detection Helpers ---

def find_local_extrema(series, order=3):
    """
    Find local minima and maxima indices in the series.
    """
    minima, maxima = [], []
    for i in range(order, len(series) - order):
        window = series[i - order:i + order + 1]
        if series[i] == min(window):
            minima.append(i)
        if series[i] == max(window):
            maxima.append(i)
    return minima, maxima

def check_rsi_divergence(df, lookback=30):
    """
    Detect RSI bullish and bearish divergence on 5-min data.
    Returns a string: "Bullish Divergence", "Bearish Divergence", or ""
    """
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
        price_low_1 = prices.iloc[idx1]
        price_low_2 = prices.iloc[idx2]
        rsi_low_1 = rsi.iloc[idx1]
        rsi_low_2 = rsi.iloc[idx2]
        if price_low_2 < price_low_1 and rsi_low_2 > rsi_low_1:
            return "Bullish Divergence"

    # Bearish divergence: price higher high, RSI lower high
    for i in range(len(max_idx) - 1):
        idx1, idx2 = max_idx[i], max_idx[i + 1]
        if idx2 <= idx1:
            continue
        price_high_1 = prices.iloc[idx1]
        price_high_2 = prices.iloc[idx2]
        rsi_high_1 = rsi.iloc[idx1]
        rsi_high_2 = rsi.iloc[idx2]
        if price_high_2 > price_high_1 and rsi_high_2 < rsi_high_1:
            return "Bearish Divergence"

    return ""
# --- Main Data Processing Loop and Display (Part 2/2) ---

column_order = [
    "Pair", "Price", "RSI", "RSI Divergence", "ATR", "ATR Status", "Trend",
    "Reversal Signal", "Signal Type", "Confirmed Indicators", "Candle Pattern",
    "AI Suggestion", "DXY Impact", "News", "Today's News"
]

for pair, symbol in symbols.items():
    try:
        # Fetch 5-min data from API (replace YOUR_ENDPOINT with actual)
        url = f"https://api.example.com/v1/forex/{symbol}/5min?apikey={API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data["prices"])  # Adjust to actual API data structure
        if df.empty or len(df) < 50:
            continue
        # Expected columns: timestamp, open, high, low, close, volume
        df['close'] = df['close'].astype(float)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp').reset_index(drop=True)

        # Indicators
        df['RSI'] = calculate_rsi(df['close'])
        df['MACD'], df['MACD_signal'] = calculate_macd(df['close'])
        df['EMA9'] = calculate_ema(df['close'], 9)
        df['EMA20'] = calculate_ema(df['close'], 20)
        df['ADX'] = calculate_adx(df)
        df['ATR'] = calculate_atr(df)

        # Latest values
        price = df['close'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        atr = df['ATR'].iloc[-1]

        # RSI alert if RSI > 75 or RSI < 25
        if rsi > 75 or rsi < 25:
            play_rsi_alert()

        # ATR Status
        atr_status = "High Volatility" if atr > 0.0015 else "Low Volatility"

        # Trend based on EMA9 and EMA20
        trend = "Bullish" if df['EMA9'].iloc[-1] > df['EMA20'].iloc[-1] else "Bearish"

        # Detect reversal
        reversal_signal = detect_trend_reversal(df)

        # Candle Pattern
        candle_pattern = detect_candle_pattern(df)

        # Confirmed indicators for signal
        confirmed_indicators = []
        if rsi < 30:
            confirmed_indicators.append("RSI Oversold")
        if rsi > 70:
            confirmed_indicators.append("RSI Overbought")
        if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]:
            confirmed_indicators.append("MACD Bullish")
        else:
            confirmed_indicators.append("MACD Bearish")
        if trend == "Bullish":
            confirmed_indicators.append("EMA Bullish")
        else:
            confirmed_indicators.append("EMA Bearish")
        if reversal_signal:
            confirmed_indicators.append(reversal_signal)

        # RSI Divergence
        rsi_divergence = check_rsi_divergence(df)
        if rsi_divergence:
            confirmed_indicators.append(rsi_divergence)

        # Signal type based on confirmed indicators
        if any(x in confirmed_indicators for x in ["RSI Oversold", "MACD Bullish", "EMA Bullish", "Reversal Confirmed Bullish", "Bullish Divergence"]):
            signal_type = "Bullish"
        elif any(x in confirmed_indicators for x in ["RSI Overbought", "MACD Bearish", "EMA Bearish", "Reversal Confirmed Bearish", "Bearish Divergence"]):
            signal_type = "Bearish"
        else:
            signal_type = "Neutral"

        # AI suggestion
        ai_suggestion = generate_ai_suggestion(price, confirmed_indicators, atr, signal_type)

        # DXY Impact
        dxy_impact = "Positive" if dxy_change > 0 else "Negative"

        # News related to pair's currencies
        primary_currency = pair.split("/")[0]
        related_news = currency_news(primary_currency)
        news_titles = [f"{n['time'].strftime('%H:%M')} {n['title']}" for n in related_news]
        todays_news = "<br>".join(news_titles) if news_titles else "—"

        rows.append({
            "Pair": pair,
            "Price": f"{price:.5f}",
            "RSI": f"{rsi:.2f}",
            "RSI Divergence": rsi_divergence or "—",
            "ATR": f"{atr:.6f}",
            "ATR Status": atr_status,
            "Trend": trend,
            "Reversal Signal": reversal_signal or "—",
            "Signal Type": signal_type,
            "Confirmed Indicators": ", ".join(confirmed_indicators),
            "Candle Pattern": candle_pattern or "—",
            "AI Suggestion": ai_suggestion or "—",
            "DXY Impact": dxy_impact,
            "News": "<br>".join(news_titles[:3]) if news_titles else "—",
            "Today's News": todays_news
        })

    except Exception as e:
        print(f"Error processing {pair}: {e}")
        continue

# --- Display Table ---
def style_table(df):
    def color_rsi_divergence(val):
        if "Bullish" in val:
            return "color: green; font-weight: bold"
        elif "Bearish" in val:
            return "color: red; font-weight: bold"
        return ""

    def color_signal_type(val):
        if val == "Bullish":
            return "color: green; font-weight: bold"
        elif val == "Bearish":
            return "color: red; font-weight: bold"
        return ""

    styled = df.style.applymap(color_rsi_divergence, subset=["RSI Divergence"])\
                      .applymap(color_signal_type, subset=["Signal Type"])\
                      .set_properties(**{'white-space': 'pre-wrap'})\
                      .hide(axis="index")
    return styled

df_display = pd.DataFrame(rows, columns=column_order)
df_display_sorted = df_display.sort_values(by="Signal Type", ascending=False).reset_index(drop=True)

st.dataframe(style_table(df_display_sorted), height=750)
