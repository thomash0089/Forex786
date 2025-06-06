# --- Forex AI Signal with RSI Sound Alert + RSI Divergence (Part 1/2) ---
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
    except Exception:
        # Fallback static data if API fails
        dxy_price = 100.237
        dxy_previous = 100.40
        change = dxy_price - dxy_previous
        percent = (change / dxy_previous) * 100
        return dxy_price, percent

def fetch_forex_factory_news():
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    response = requests.get(url)
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
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
# --- Forex AI Signal with RSI Sound Alert + RSI Divergence (Part 2/2) ---

def fetch_candles(symbol, interval='15m', limit=100):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol.replace('/','')}&interval={interval}&limit={limit}"
    try:
        data = requests.get(url).json()
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['open'] = df['open'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def detect_rsi_divergence(df, rsi_period=14):
    df['rsi'] = calculate_rsi(df['close'], rsi_period)
    divergences = []
    # Very simple divergence detection:
    # Find local lows in price & check if RSI forms higher low (bullish divergence)
    # Find local highs in price & check if RSI forms lower high (bearish divergence)
    for i in range(2, len(df)-2):
        price_low = df['low'][i] < df['low'][i-1] and df['low'][i] < df['low'][i+1]
        rsi_low = df['rsi'][i] < df['rsi'][i-1] and df['rsi'][i] < df['rsi'][i+1]
        price_high = df['high'][i] > df['high'][i-1] and df['high'][i] > df['high'][i+1]
        rsi_high = df['rsi'][i] > df['rsi'][i-1] and df['rsi'][i] > df['rsi'][i+1]

        # Bullish divergence
        if price_low and (df['rsi'][i] > df['rsi'][i-2]):
            divergences.append((df.index[i], 'Bullish Divergence'))

        # Bearish divergence
        if price_high and (df['rsi'][i] < df['rsi'][i-2]):
            divergences.append((df.index[i], 'Bearish Divergence'))
    return divergences

def generate_signal(df):
    if df.empty:
        return "No data"

    df['rsi'] = calculate_rsi(df['close'])
    df['ema12'] = calculate_ema(df['close'], 12)
    df['ema26'] = calculate_ema(df['close'], 26)
    df['macd'], df['macd_signal'] = calculate_macd(df['close'])
    df['adx'] = calculate_adx(df)
    latest = df.iloc[-1]

    signal = "Hold"
    if latest['rsi'] < 30 and latest['macd'] > latest['macd_signal']:
        signal = "Buy"
    elif latest['rsi'] > 70 and latest['macd'] < latest['macd_signal']:
        signal = "Sell"

    # Additional signals from divergence
    divergences = detect_rsi_divergence(df)
    for _, div in divergences:
        if div == 'Bullish Divergence':
            signal = "Buy"
        elif div == 'Bearish Divergence':
            signal = "Sell"

    return signal, divergences

def main():
    st.write(f"### DXY Index: {dxy_price:.2f} ({dxy_change:+.2f}%)")

    cols = st.columns(len(symbols))
    for i, (pair, symbol) in enumerate(symbols.items()):
        with cols[i % len(cols)]:
            st.markdown(f"#### {pair}")

            df = fetch_candles(symbol)
            if df.empty:
                st.write("No data")
                continue

            signal, divergences = generate_signal(df)
            rsi_val = calculate_rsi(df['close']).iloc[-1]

            # Play alert if RSI crosses thresholds
            if rsi_val < 30 or rsi_val > 70:
                play_rsi_alert()

            news = get_today_news_with_impact(pair)
            st.write(f"**Signal:** {signal}")
            st.write(f"**RSI:** {rsi_val:.1f}")
            if divergences:
                st.write("Divergences detected:")
                for _, div in divergences:
                    st.write(f"- {div}")
            st.write("**News:**")
            for n in news:
                st.write(f"- {n}")
            st.markdown("---")

if __name__ == "__main__":
    main()
