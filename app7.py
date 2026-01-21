import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
import time

# --- CONFIG ---
st.set_page_config(page_title="BTC PREDICT V3.4", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050505; color: #00f2ff; }
    [data-testid="stMetricValue"] { color: #00f2ff; }
    .stMetric { background: rgba(0, 242, 255, 0.05); border: 1px solid #333; border-radius: 5px; padding: 10px; }
    .ghazi-text { text-align: right; color: #ff00ff; font-weight: bold; font-family: 'Courier New', monospace; margin-bottom: -20px; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER: GHAZI PROJECT ---
st.markdown('<p class="ghazi-text">Ghazi Project v3.4</p>', unsafe_allow_html=True)
st.title("BTC Predictor: Neural Pipeline")

# --- FUNCTIONS ---
def get_crypto_news():
    try:
        url = "https://cryptopanic.com/api/v1/posts/?kind=news&currencies=BTC"
        news_data = requests.get(url, timeout=5).json()
        return news_data.get('results', [])[:5]
    except:
        return []

# --- EXECUTION PIPELINE ---
with st.status("🔮 Memulai Neural Pipeline...", expanded=True) as status:
    
    st.write("📡 Menarik data pasar dari Yahoo Finance...")
    try:
        # Penarikan data pasar
        btc = yf.download('BTC-USD', period='2y', interval='1d', progress=False)
        dxy = yf.download('DX-Y.NYB', period='2y', interval='1d', progress=False)
        
        if isinstance(btc.columns, pd.MultiIndex): btc.columns = btc.columns.get_level_values(0)
        if isinstance(dxy.columns, pd.MultiIndex): dxy.columns = dxy.columns.get_level_values(0)
        
        df = btc[['Close', 'High', 'Low', 'Volume']].copy()
        df['DXY'] = dxy['Close'].reindex(df.index).ffill()
        
        st.write("📊 Mengintegrasikan Fear & Greed Index...")
        r_fng = requests.get("https://api.alternative.me/fng/?limit=1000", timeout=10).json()
        fng = pd.DataFrame(r_fng['data'])
        fng['timestamp'] = pd.to_numeric(fng['timestamp'])
        fng['timestamp'] = pd.to_datetime(fng['timestamp'], unit='s').dt.tz_localize(None)
        fng = fng.set_index('timestamp')['value'].astype(int).to_frame('FNG')
        df = df.join(fng).ffill()
        
        st.write("🛠️ Melakukan Feature Engineering (RSI, EMA, ATR)...")
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['EMA_20'] = ta.ema(df['Close'], length=20)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['Target_Diff'] = df['Close'].shift(-1) - df['Close']
        
        st.write("🧠 Melatih Model XGBoost & Backtesting...")
        features = ['Close', 'Volume', 'DXY', 'FNG', 'RSI', 'EMA_20', 'ATR']
        train_df = df.dropna(subset=['Target_Diff'])
        X = train_df[features]
        y = train_df['Target_Diff']
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        model = XGBRegressor(learning_rate=0.03, max_depth=6, n_estimators=500)
        model.fit(X_train, y_train)
        
        # Kalkulasi MAE
        test_preds_diff = model.predict(X_test)
        test_actual_prices = train_df['Close'].iloc[split+1:].values
        test_pred_prices = train_df['Close'].iloc[split:-1].values + test_preds_diff[:-1]
        mae = mean_absolute_error(test_actual_prices, test_pred_prices)
        
        st.write("📰 Mengambil sentimen berita global...")
        news_list = get_crypto_news()
        
        # Persiapan Data Live
        live_data = df.iloc[[-1]]
        price_live = float(live_data['Close'].iloc[0])
        price_yesterday = float(df.iloc[[-2]]['Close'].iloc[0])
        pred_diff = model.predict(live_data[features])[0]
        price_pred = price_live + pred_diff
        
        status.update(label="✅ Sinkronisasi Berhasil!", state="complete", expanded=False)
        data_ready = True
    except Exception as e:
        status.update(label=f"❌ Kegagalan Sistem: {e}", state="error")
        data_ready = False

# --- OUTPUT DASHBOARD ---
if data_ready:
    # Row 1: Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LIVE PRICE (TODAY)", f"${price_live:,.0f}")
    c2.metric("CLOSE (YESTERDAY)", f"${price_yesterday:,.0f}", f"{price_live - price_yesterday:,.2f}")
    c3.metric("PREDICTION (TOMORROW)", f"${price_pred:,.0f}", f"{pred_diff:,.2f}")
    c4.metric("ERROR MARGIN (MAE)", f"${mae:,.2f}")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### 📈 Projection Graph")
        plot_df = df.tail(30)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], name="History", line=dict(color='#00f2ff')))
        fig.add_trace(go.Scatter(
            x=[plot_df.index[-1], plot_df.index[-1] + pd.Timedelta(days=1)], 
            y=[price_live, price_pred],
            name="Forecast", line=dict(color='#ff00ff', dash='dot'), mode='lines+markers'
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#00f2ff', margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🔍 Backtest Result"):
            comp_df = pd.DataFrame({'Actual': test_actual_prices, 'Predicted': test_pred_prices}, index=train_df.index[split+1:])
            st.line_chart(comp_df)

    with col_right:
        st.markdown("### 📰 News Stream")
        if news_list:
            for n in news_list:
                st.markdown(f"**[{n['title']}]({n['url']})**")
                st.caption(f"Source: {n['domain']} | 👍 {n['votes']['positive']}")
                st.divider()
        else:
            st.info("No recent news found.")
