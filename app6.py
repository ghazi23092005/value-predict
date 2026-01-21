import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go

# --- CONFIG ---
st.set_page_config(page_title="BTC PREDICT V3.0", layout="wide")

# CSS Styling (Tetap mempertahankan tema cyber agar scannable)
st.markdown("""
    <style>
    .main { background-color: #050505; color: #00f2ff; }
    [data-testid="stMetricValue"] { color: #00f2ff; }
    .stMetric { background: rgba(0, 242, 255, 0.05); border: 1px solid #333; border-radius: 5px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("BTC Predictor: Validation & Backtest Mode")

@st.cache_data(ttl=3600)
def fetch_enhanced_data():
    try:
        # Download data
        btc = yf.download('BTC-USD', period='2y', interval='1d', progress=False)
        dxy = yf.download('DX-Y.NYB', period='2y', interval='1d', progress=False)
        
        # Multi-index fix
        if isinstance(btc.columns, pd.MultiIndex): btc.columns = btc.columns.get_level_values(0)
        if isinstance(dxy.columns, pd.MultiIndex): dxy.columns = dxy.columns.get_level_values(0)

        df = btc[['Close', 'High', 'Low', 'Volume']].copy()
        df['DXY'] = dxy['Close'].reindex(df.index).ffill()
        
        # Fear & Greed API
        r = requests.get("https://api.alternative.me/fng/?limit=1000", timeout=10).json()
        fng = pd.DataFrame(r['data'])
        fng['timestamp'] = pd.to_numeric(fng['timestamp'])
        fng['timestamp'] = pd.to_datetime(fng['timestamp'], unit='s').dt.tz_localize(None)
        fng = fng.set_index('timestamp')['value'].astype(int).to_frame('FNG')
        
        df = df.join(fng).ffill()

        # --- FEATURE ENGINEERING (Poin 3) ---
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['EMA_20'] = ta.ema(df['Close'], length=20)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # --- TARGET LOGIC: PRICE CHANGE (Poin 1) ---
        # Kita memprediksi PERUBAHAN harga, bukan harga absolut untuk menghindari bias
        df['Target_Diff'] = df['Close'].shift(-1) - df['Close']
        
        return df.dropna()
    except Exception as e:
        st.error(f"Error: {e}")
        return None

df = fetch_enhanced_data()

if df is not None:
    # --- MODELING & BACKTESTING (Poin 2) ---
    features = ['Close', 'Volume', 'DXY', 'FNG', 'RSI', 'EMA_20', 'ATR']
    X = df[features].iloc[:-1]
    y = df['Target_Diff'].iloc[:-1]

    # Split Data (80% Train, 20% Test)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBRegressor(learning_rate=0.03, max_depth=6, n_estimators=500, objective='reg:absoluteerror')
    model.fit(X_train, y_train)

    # Validasi Akurasi
    test_preds_diff = model.predict(X_test)
    # Kembalikan ke harga absolut untuk hitung MAE
    test_actual_prices = df['Close'].iloc[split+1:].values
    test_pred_prices = df['Close'].iloc[split:-1].values + test_preds_diff
    mae = mean_absolute_error(test_actual_prices, test_pred_prices)

    # Prediksi Masa Depan (Besok)
    last_row = df[features].iloc[[-1]]
    pred_diff_tomorrow = model.predict(last_row)[0]
    current_price = float(df['Close'].iloc[-1])
    pred_tomorrow = current_price + pred_diff_tomorrow

    # --- DASHBOARD ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CURRENT PRICE", f"${current_price:,.0f}")
    col2.metric("PREDICTION (NEXT DAY)", f"${pred_tomorrow:,.0f}", f"{pred_diff_tomorrow:,.2f}")
    col3.metric("MODEL ERROR (MAE)", f"${mae:,.2f}")
    col4.metric("SENTIMENT", f"{df['FNG'].iloc[-1]}%")

    # --- CHART WITH PROJECTION (Poin 4) ---
    st.markdown("### 📈 Price Action & Neural Projection")
    
    # Ambil 60 hari terakhir saja agar chart bersih
    plot_df = df.tail(60)
    future_date = plot_df.index[-1] + pd.Timedelta(days=1)

    fig = go.Figure()
    # Garis Histori
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], name="History", line=dict(color='#00f2ff', width=2)))
    # Garis Proyeksi
    fig.add_trace(go.Scatter(
        x=[plot_df.index[-1], future_date], 
        y=[current_price, pred_tomorrow],
        name="Projection",
        line=dict(color='#ff00ff', width=3, dash='dot'),
        mode='lines+markers'
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='#00f2ff', hovermode="x unified",
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- BACKTEST INSIGHT ---
    with st.expander("Lihat Detail Validasi Model"):
        st.write("Perbandingan Harga Aktual vs Prediksi pada 20% Data Terakhir (Test Set):")
        comparison_df = pd.DataFrame({
            'Actual': test_actual_prices,
            'Predicted': test_pred_prices
        }, index=df.index[split+1:])
        st.line_chart(comparison_df)
        st.info(f"Model XGBoost dilatih pada {len(X_train)} hari dan diuji pada {len(X_test)} hari data yang belum pernah dilihat.")
