# --- Forex AI Signal with RSI Sound Alert + Divergence Detection (Full Updated) ---
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone
import xml.etree.ElementTree as ET
from dateutil import parser as date_parser
import streamlit.components.v1 as components
import yfinance as yf

st.set_page_config(page_title="Signals", layout="wide")
st.markdown("<h1 style='text-align:center; color:#007acc;'>📊 Signals + News + Divergence</h1>", unsafe_allow_html=True)
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

# --- ANALYSIS FUNCTIONS ---
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

def detect_trend_reversal(df):
    e9, e20 = df['EMA9'].iloc[-3:], df['EMA20'].iloc[-3:]
    if e9[0] < e20[0] and e9[1] > e20[1] and e9[2] > e20[2]: return "Reversal Confirmed Bullish"
    if e9[0] > e20[0] and e9[1] < e20[1] and e9[2] < e20[2]: return "Reversal Confirmed Bearish"
    if e9[-2] < e20[-2] and e9[-1] > e20[-1]: return "Reversal Forming Bullish"
    if e9[-2] > e20[-2] and e9[-1] < e20[-1]: return "Reversal Forming Bearish"
    return ""

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

# --- Divergence Detection ---
def detect_rsi_divergence(df, lookback=14):
    """
    Detect bullish or bearish divergence on the last few bars.
    Returns 'Bullish Divergence', 'Bearish Divergence' or ''.
    """

    # Find local lows/highs of price and RSI in lookback window
    prices = df['close'][-lookback:]
    rsi = df['RSI'][-lookback:]

    # Find local minima and maxima indexes for price
    def local_minima(series):
        return [i for i in range(1, len(series) - 1) if series[i] < series[i-1] and series[i] < series[i+1]]
    def local_maxima(series):
        return [i for i in range(1, len(series) - 1) if series[i] > series[i-1] and series[i] > series[i+1]]

    price_lows = local_minima(prices.values)
    price_highs = local_maxima(prices.values)
    rsi_lows = local_minima(rsi.values)
    rsi_highs = local_maxima(rsi.values)

    # Bullish divergence: price makes lower low, RSI makes higher low
    for i in range(len(price_lows)-1):
        p1, p2 = price_lows[i], price_lows[i+1]
        if prices.iloc[p2] < prices.iloc[p1]:  # Lower low in price
            # Find RSI lows near those points
            rsi_low_1 = min((r for r in rsi_lows if abs(r - p1) <= 2), default=None)
            rsi_low_2 = min((r for r in rsi_lows if abs(r - p2) <= 2), default=None)
            if rsi_low_1 is not None and rsi_low_2 is not None:
                if rsi.iloc[rsi_low_2] > rsi.iloc[rsi_low_1]:
                    return "Bullish Divergence"

    # Bearish divergence: price makes higher high, RSI makes lower high
    for i in range(len(price_highs)-1):
        p1, p2 = price_highs[i], price_highs[i+1]
        if prices.iloc[p2] > prices.iloc[p1]:  # Higher high in price
            # Find RSI highs near those points
            rsi_high_1 = min((r for r in rsi_highs if abs(r - p1) <= 2), default=None)
            rsi_high_2 = min((r for r in rsi_highs if abs(r - p2) <= 2), default=None)
            if rsi_high_1 is not None and rsi_high_2 is not None:
                if rsi.iloc[rsi_high_2] < rsi.iloc[rsi_high_1]:
                    return "Bearish Divergence"

    return ""

# Store divergence signals with timestamp for persistence (20 min)
if "divergences" not in st.session_state:
    st.session_state.divergences = {}

current_time = datetime.utcnow()

def update_divergences(pair, divergence):
    """Update divergence store and cleanup old entries."""
    # Clean old divergence signals (> 20 min old)
    to_delete = []
    for key, val in st.session_state.divergences.items():
        if current_time - val["time"] > timedelta(minutes=20):
            to_delete.append(key)
    for key in to_delete:
        del st.session_state.divergences[key]
    
    # Update current divergence
    if divergence:
        st.session_state.divergences[pair] = {"signal": divergence, "time": current_time}

def get_divergence_signal(pair):
    """Return stored divergence signal or empty string."""
    if pair in st.session_state.divergences:
        return st.session_state.divergences[pair]["signal"]
    return ""

# --- Main loop over symbols ---
for label, symbol in symbols.items():
    try:
        url = f"https://fcsapi.com/api-v3/forex/history?symbol={symbol}&period=5min&access_key={API_KEY}"
        data = requests.get(url).json()

        if data['status'] != True or 'response' not in data:
            st.error(f"API error for {label}: {data.get('error', 'Unknown error')}")
            continue

        df = pd.DataFrame(data['response'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime')
        df.set_index('datetime', inplace=True)
        df = df.astype(float)

        df['RSI'] = calculate_rsi(df['close'])
        df['EMA9'] = calculate_ema(df['close'], 9)
        df['EMA20'] = calculate_ema(df['close'], 20)
        df['ATR'] = calculate_atr(df)
        df['ADX'] = calculate_adx(df)

        macd, signal = calculate_macd(df['close'])
        df['MACD'] = macd
        df['Signal'] = signal

        price = df['close'][-1]
        rsi_val = df['RSI'][-1]
        atr = df['ATR'][-1]
        adx = df['ADX'][-1]

        # Trend based on EMA and MACD
        trend = "Bullish" if df['EMA9'][-1] > df['EMA20'][-1] and df['MACD'][-1] > df['Signal'][-1] else "Bearish"

        # Detect candle pattern
        pattern = detect_candle_pattern(df)

        # Detect trend reversal signal
        reversal_signal = detect_trend_reversal(df)

        # RSI signal alert
        rsi_alert_signal = ""
        if rsi_val > 70:
            rsi_alert_signal = "RSI Overbought (Sell)"
            play_rsi_alert()
        elif rsi_val < 30:
            rsi_alert_signal = "RSI Oversold (Buy)"
            play_rsi_alert()

        # Detect divergence
        divergence = detect_rsi_divergence(df)
        update_divergences(label, divergence)
        divergence_display = get_divergence_signal(label) or "—"

        # Build signal type and confirmed indicators
        signal_type = ""
        indicators = []

        if rsi_val < 30: 
            signal_type = "Bullish"
            indicators.append("RSI Oversold")
        elif rsi_val > 70: 
            signal_type = "Bearish"
            indicators.append("RSI Overbought")

        if pattern in ["Bullish Engulfing", "Hammer"]:
            signal_type = "Bullish"
            indicators.append(pattern)
        elif pattern in ["Bearish Engulfing", "Shooting Star"]:
            signal_type = "Bearish"
            indicators.append(pattern)

        if reversal_signal.startswith("Reversal Confirmed Bullish"):
            signal_type = "Bullish"
            indicators.append("Trend Reversal Confirmed")
        elif reversal_signal.startswith("Reversal Confirmed Bearish"):
            signal_type = "Bearish"
            indicators.append("Trend Reversal Confirmed")

        # ATR status (low, normal, high)
        if atr < 0.0004:
            atr_status = "🔴 Low"
        elif atr < 0.0009:
            atr_status = "🟡 Normal"
        else:
            atr_status = "🟢 High"

        # AI Suggestion
        suggestion = generate_ai_suggestion(price, indicators, atr, signal_type)

        # Append row data
        rows.append({
            "Pair": label,
            "Price": round(price, 5),
            "RSI": round(rsi_val, 2),
            "ATR": round(atr, 5),
            "ATR Status": atr_status,
            "Trend": trend,
            "Reversal Signal": reversal_signal or "—",
            "Signal Type": signal_type or "—",
            "Confirmed Indicators": ", ".join(indicators) or "—",
            "Candle Pattern": pattern or "—",
            "Divergence": divergence_display,
            "AI Suggestion": suggestion or "—",
            "DXY Impact": f"{dxy_price:.2f} ({dxy_change:+.2f}%)" if "USD" in label and dxy_price and dxy_change else "—",
            "News": get_next_news(label),
            "Today's News": "<br>".join(get_today_news_with_impact(label))
        })

    except Exception as e:
        st.warning(f"Failed for {label}: {e}")

# --- Display the table ---
df_table = pd.DataFrame(rows)

column_order = ["Pair", "Price", "RSI", "ATR", "ATR Status", "Trend", "Reversal Signal",
                "Signal Type", "Confirmed Indicators", "Candle Pattern", "Divergence", "AI Suggestion",
                "DXY Impact", "News", "Today's News"]

df_table = df_table[column_order]

def color_divergence(val):
    if "Bullish" in val:
        return 'color: green; font-weight: bold;'
    elif "Bearish" in val:
        return 'color: red; font-weight: bold;'
    else:
        return ''

def style_df(df):
    styles = [
        dict(selector="th", props=[("font-size", "14px"), ("text-align", "center")]),
        dict(selector="td", props=[("font-size", "13px")]),
    ]
    return df.style.set_table_styles(styles).applymap(color_divergence, subset=["Divergence"]).set_properties(**{'text-align': 'center'})

st.dataframe(style_df(df_table), height=700)

