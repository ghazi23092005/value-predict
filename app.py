import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
from xgboost import XGBRegressor
import plotly.graph_objects as go

# --- CONFIG & CYBER-STYLING ---
st.set_page_config(page_title="NEO-BTC ORACLE", layout="wide")

# CSS untuk tampilan Hologram Modern
st.markdown("""
    <style>
    .main {
        background-color: #050505;
        color: #00f2ff;
        font-family: 'Segoe UI', sans-serif;
    }
    [data-testid="stMetricValue"] {
        color: #00f2ff;
        text-shadow: 0 0 10px #00f2ff, 0 0 20px #00f2ff;
    }
    .stMetric {
        background: rgba(0, 242, 255, 0.05);
        border: 1px solid #00f2ff;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.2);
        border-radius: 5px;
    }
    h1 {
        text-align: center;
        background: linear-gradient(to right, #00f2ff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 10px rgba(255, 0, 255, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ NEO-BTC CYBER ORACLE v2.5")
st.markdown("<p style='text-align: center; color: #ff00ff;'>Neural Core: Online | Timezone Sync: Active</p>", unsafe_allow_html=True)

# --- DATA ENGINE (FIXED MERGE) ---
@st.cache_data(ttl=3600)
def fetch_cyber_data():
    # 1. Download BTC & DXY
    btc = yf.download('BTC-USD', start='2023-01-01')
    dxy = yf.download('DX-Y.NYB', start='2023-01-01')[['Close']]
    dxy.columns = ['DXY']
    
    # Hilangkan timezone agar tidak error saat merge
    btc.index = btc.index.tz_localize(None)
    dxy.index = dxy.index.tz_localize(None)
    
    # 2. Fear & Greed Index
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1000").json()
        fng = pd.DataFrame(r['data'])
        fng['timestamp'] = pd.to_datetime(fng['timestamp'], unit='s').dt.tz_localize(None)
        fng = fng.set_index('timestamp')['value'].astype(int).to_frame('FNG')
    except:
        # Fallback jika API FNG down
        fng = pd.DataFrame(index=btc.index)
        fng['FNG'] = 50

    # 3. Secure Merge
    df = btc.copy()
    df = df.merge(dxy, left_index=True, right_index=True, how='left')
    df = df.merge(fng, left_index=True, right_index=True, how='left')
    
    # Fill weekend gaps (DXY tutup sabtu-minggu)
    df = df.ffill().dropna()
    
    # 4. Indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['Target'] = df['Close'].shift(-1)
    return df.dropna()

with st.spinner("Synchronizing Neural Networks..."):
    df = fetch_cyber_data()

# --- SIDEBAR CONTROL ---
st.sidebar.markdown("### 🎛️ CORE CALIBRATION")
lr = st.sidebar.slider("Synapse Sensitivity (LR)", 0.01, 0.1, 0.05)
depth = st.sidebar.slider("Neural Depth", 3, 10, 5)

# --- MODELING ---
features = ['Close', 'Volume', 'DXY', 'FNG', 'RSI', 'EMA_20']
X = df[features].iloc[:-1]
y = df['Target'].iloc[:-1]

model = XGBRegressor(learning_rate=lr, max_depth=depth, n_estimators=500)
model.fit(X, y)

# Prediction
last_row = df[features].iloc[[-1]]
pred_tomorrow = model.predict(last_row)[0]
current_price = df['Close'].iloc[-1]

# --- DASHBOARD ---
c1, c2, c3 = st.columns(3)
c1.metric("NETWORK PRICE", f"${current_price:,.0f}")
c2.metric("NEURAL PROJECTION", f"${pred_tomorrow:,.0f}", f"{pred_tomorrow-current_price:,.0f}")
c3.metric("SENTIMENT CORE", f"{df['FNG'].iloc[-1]}%")

st.markdown("### 📊 DATA STREAM PROJECTION")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="History", line=dict(color='#00f2ff', width=2)))
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='#00f2ff',
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='rgba(0, 242, 255, 0.1)')
)
st.plotly_chart(fig, use_container_width=True)

st.caption("v2.5 System Stable | Cyberpunk Interface Active")
