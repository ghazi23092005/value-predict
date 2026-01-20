import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
from xgboost import XGBRegressor
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# --- CONFIG & CYBER-STYLING ---
st.set_page_config(page_title="NEO-BTC ORACLE", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050505; color: #00f2ff; font-family: 'Segoe UI', sans-serif; }
    [data-testid="stMetricValue"] { color: #00f2ff; text-shadow: 0 0 10px #00f2ff; font-size: 1.8rem !important; }
    .stMetric { background: rgba(0, 242, 255, 0.05); border: 1px solid #00f2ff; border-radius: 5px; padding: 10px; }
    h1 { text-align: center; background: linear-gradient(to right, #00f2ff, #ff00ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .countdown-box { text-align: center; color: #ff00ff; font-family: 'Courier New'; font-size: 1.1rem; margin-bottom: 20px; }
    .stSpinner > div { border-top-color: #00f2ff !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("Ghazi BTC Predict Model v3.2")

@st.cache_data(ttl=3600)
def fetch_cyber_data():
    btc_raw = yf.download('BTC-USD', period='2y', interval='1d', progress=False)
    dxy_raw = yf.download('DX-Y.NYB', period='2y', interval='1d', progress=False)
    idr_raw = yf.download('IDR=X', period='1d', interval='1m', progress=False)
    
    for raw in [btc_raw, dxy_raw]:
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

    kurs_idr = idr_raw['Close'].iloc[-1] if not idr_raw.empty else 16250
    btc = btc_raw[['Close', 'Volume']].copy()
    dxy = dxy_raw[['Close']].copy()
    dxy.columns = ['DXY']
    btc.index = pd.to_datetime(btc.index).tz_localize(None)
    dxy.index = pd.to_datetime(dxy.index).tz_localize(None)
    
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1000", timeout=10).json()
        fng = pd.DataFrame(r['data'])
        fng['timestamp'] = pd.to_numeric(fng['timestamp'])
        fng['timestamp'] = pd.to_datetime(fng['timestamp'], unit='s').dt.tz_localize(None)
        fng = fng.set_index('timestamp')['value'].astype(int).to_frame('FNG')
    except:
        fng = pd.DataFrame(index=btc.index); fng['FNG'] = 50

    df = pd.concat([btc, dxy, fng], axis=1).ffill().dropna(subset=['Close']) 
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['Target'] = df['Close'].shift(-1)
    return df.dropna(), kurs_idr

with st.spinner('Calibrating Neural Matrix...'):
    df, kurs_idr = fetch_cyber_data()
    
    if df is not None and not df.empty:
        features = ['Close', 'Volume', 'DXY', 'FNG', 'RSI', 'EMA_20']
        X = df[features].iloc[:-1]
        y = df['Target'].iloc[:-1]
        
        model = XGBRegressor(learning_rate=0.05, max_depth=5, n_estimators=500)
        model.fit(X, y)
        
        # --- MULTI-DAY PREDICTION (REC-FORECAST) ---
        current_data = df[features].iloc[[-1]].copy()
        predictions = []
        dates = []
        
        for i in range(1, 4): # Prediksi 3 hari ke depan
            pred = model.predict(current_data)[0]
            predictions.append(pred)
            next_date = df.index[-1] + timedelta(days=i)
            dates.append(next_date)
            # Update data untuk loop berikutnya (prediksi kasar)
            current_data['Close'] = pred 
            
        pred_tomorrow = predictions[0]
        current_price = float(df['Close'].iloc[-1])
        data_ready = True
    else:
        data_ready = False

if data_ready:
    # Header & Countdown
    now = datetime.now()
    target = now.replace(hour=7, minute=0, second=0, microsecond=0)
    if now >= target: target += timedelta(days=1)
    diff = target - now
    st.markdown(f"<div class='countdown-box'>Next Prophecy in: {diff.seconds//3600}h {(diff.seconds//60)%60}m</div>", unsafe_allow_html=True)

    def format_idr(usd_val):
        idr_val = usd_val * kurs_idr
        return f"Rp{idr_val/1e9:.2f} M" if idr_val >= 1e9 else f"Rp{idr_val:,.0f}"

    # Metrik Utama
    c1, c2, c3 = st.columns(3)
    c1.metric("NETWORK PRICE", f"${current_price:,.0f}", format_idr(current_price))
    c2.metric("TOMORROW PROJECTION", f"${pred_tomorrow:,.0f}", f"{format_idr(pred_tomorrow)} (Est.)")
    c3.metric("SENTIMENT CORE", f"{df['FNG'].iloc[-1]}%", "Market Psychology")

    # --- ENHANCED CHART ---
    st.markdown("### MULTI-DAY VISION (ESTIMATION)")
    fig = go.Figure()
    
    # Data Historis (30 hari terakhir biar jelas)
    hist_df = df.tail(30)
    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Close'], name="History", line=dict(color='#00f2ff', width=2)))
    
    # Data Proyeksi (Garis Putus-putus)
    proj_dates = [df.index[-1]] + dates
    proj_values = [current_price] + predictions
    fig.add_trace(go.Scatter(x=proj_dates, y=proj_values, name="Projection", line=dict(color='#ff00ff', width=2, dash='dot')))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='#00f2ff', xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(0, 242, 255, 0.1)'),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("⚠️ Pink dotted line is a rough estimation based on recursive neural patterns.")
