import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from pytz import timezone

# Make sure st.set_page_config is the first Streamlit command
st.set_page_config(page_title="Forex AI Signals", layout="wide")

# Streamlit app heading
st.markdown("<h1 style='text-align:center; color:#007acc;'>📊 Forex AI Signals (15-Min Timeframe)</h1>", unsafe_allow_html=True)
st_autorefresh(interval=120000, key="auto_refresh")

# API key for fetching data from TwelveData
API_KEY = "b2a1234a9ea240f9ba85696e2a243403"

# Currency symbols to track
symbols = {
    "EUR/USD": "EUR/USD", "GBP/USD": "GBP/USD", "USD/JPY": "USD/JPY",
    "AUD/USD": "AUD/USD", "USD/CAD": "USD/CAD", "USD/CHF": "USD/CHF",
    # Add other pairs as needed
}

# Function to fetch data from TwelveData API
@st.cache
def fetch_data(symbol, interval="15min", outputsize=200):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "apikey": API_KEY}
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()  # Check if request was successful
        data = r.json()
        if "values" in data:
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['close'] = pd.to_numeric(df['close'])
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            return df
        else:
            st.error(f"Error fetching data for {symbol}")
            return None
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

# Function to detect divergence direction (Bullish/Bearish)
def detect_divergence_direction(df):
    # Example of a divergence detection logic based on price and RSI
    rsi = df['RSI'].iloc[-1]  # Get the last RSI value
    price = df['close'].iloc[-1]  # Get the last closing price
    # Add your custom divergence logic here
    if rsi > 70 and price > df['close'].iloc[-2]:  # Example condition for bullish divergence
        return "Bullish"
    elif rsi < 30 and price < df['close'].iloc[-2]:  # Example condition for bearish divergence
        return "Bearish"
    else:
        return "No Divergence"

# Function to detect trend reversal based on simple price action or indicators
def detect_trend_reversal(df):
    # Example of trend reversal logic
    if df['close'].iloc[-1] > df['open'].iloc[-1]:
        return "Bullish Reversal"
    else:
        return "Bearish Reversal"

# Function to detect volume spike (Example logic)
def detect_volume_spike(df):
    volume = df['volume'].iloc[-1]  # Get the last volume value
    avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]  # Average volume over last 20 periods
    if volume > 2 * avg_volume:  # Example of volume spike detection
        return "Volume Spike"
    return "Normal Volume"

# Function to calculate RSI for the given dataframe
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    return df

# Main function to display data and analysis
def display_signals():
    for symbol in symbols:
        df = fetch_data(symbol)
        if df is not None:
            # Calculate RSI
            df = calculate_rsi(df)

            # Detect divergence, trend reversal, and volume spikes
            direction = detect_divergence_direction(df)
            reversal = detect_trend_reversal(df)
            volume_spike = detect_volume_spike(df)

            # Display the results in Streamlit
            st.write(f"### {symbol}")
            st.write(f"Direction: {direction}")
            st.write(f"Reversal: {reversal}")
            st.write(f"Volume: {volume_spike}")
            st.write(f"RSI: {df['RSI'].iloc[-1]}")

# Run the display function
display_signals()
