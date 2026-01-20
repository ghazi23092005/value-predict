import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
from xgboost import XGBRegressor
import plotly.graph_objects as go

# --- CONFIG & CYBER-STYLING ---
st.set_page_config(page_title="NEO-BTC ORACLE", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050505; color: #00f2ff; font-family: 'Segoe UI', sans-serif; }
    [data-testid="stMetricValue"] { color: #00f2ff; text-shadow: 0 0 10px #00f2ff; }
    .stMetric { background: rgba(0, 242, 255, 0.05); border: 1px solid #00f2ff; border-radius: 5px; }
    h1 { text-align: center; background: linear-gradient(to right, #00f2ff, #ff00ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    </style>
    """, unsafe_allow_html=True)

st.title("BTC Predict Model v2.9")

# --- DATA ENGINE (STABLE v2.9) ---
@st.cache_data(ttl=3600)
def fetch_cyber_data():
    try:
        # Download data
        btc_raw = yf.download('BTC-USD', period='2y', interval='1d', progress=False)
        dxy_raw = yf.download('DX-Y.NYB', period='2y', interval='1d', progress=False)
        
        # Perbaikan Multi-Index
        if isinstance(btc_raw.columns, pd.MultiIndex):
            btc_raw.columns = btc_raw.columns.get_level_values(0)
        if isinstance(dxy_raw.columns, pd.MultiIndex):
            dxy_raw.columns = dxy_raw.columns.get_level_values(0)

        btc = btc_raw[['Close', 'Volume']].copy()
        dxy = dxy_raw[['Close']].copy()
        dxy.columns = ['DXY']
        
        # Sync index
        btc.index = pd.to_datetime(btc.index).tz_localize(None)
        dxy.index = pd.to_datetime(dxy.index).tz_localize(None)
        
        # Fear & Greed (Fixing the FutureWarning)
        try:
            r = requests.get("https://api.alternative.me/fng/?limit=1000", timeout=10).json()
            fng = pd.DataFrame(r['data'])
            # FIX: Cast to numeric first to avoid Warning
            fng['timestamp'] = pd.to_numeric(fng['timestamp'])
            fng['timestamp'] = pd.to_datetime(fng['timestamp'], unit='s').dt.tz_localize(None)
            fng = fng.set_index('timestamp')['value'].astype(int).to_frame('FNG')
        except:
            fng = pd.DataFrame(index=btc.index)
            fng['FNG'] = 50

        # Concat & Clean
        df = pd.concat([btc, dxy, fng], axis=1)
        df = df.ffill().dropna(subset=['Close']) 
        
        # Technicals
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['EMA_20'] = ta.ema(df['Close'], length=20)
        df['Target'] = df['Close'].shift(-1)
        return df.dropna()
    except Exception as e:
        st.error(f"Oracle Connection Error: {e}")
        return None

# Trigger Data
df = fetch_cyber_data()

if df is not None and not df.empty:
    st.markdown("<p style='text-align: center; color: #ff00ff;'>Neural Core: Online | Projection: Stabilized</p>", unsafe_allow_html=True)
    
    # --- MODELING ---
    features = ['Close', 'Volume', 'DXY', 'FNG', 'RSI', 'EMA_20']
    X = df[features].iloc[:-1]
    y = df['Target'].iloc[:-1]

    model = XGBRegressor(learning_rate=0.05, max_depth=5, n_estimators=500)
    model.fit(X, y)

    last_row = df[features].iloc[[-1]]
    pred_tomorrow = model.predict(last_row)[0]
    current_price = float(df['Close'].iloc[-1])

    # --- DASHBOARD LAYOUT ---
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
else:
    st.info("🔮 Re-calibrating Neural Links... Mohon tunggu sebentar (Data sedang ditarik).")
    st.button("Force Re-sync Data")
