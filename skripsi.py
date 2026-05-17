import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# --- CONFIG ---
st.set_page_config(page_title="Prediksi Saham Tech BEI", layout="wide")
st.title("📈 Optimasi Prediksi Harga Saham Teknologi (BEI)")
st.markdown("Aplikasi prediksi harga saham H+1 menggunakan **Multiple Linear Regression (MLR)** berbasis **SMA** dan **RSI**.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Parameter Model")
tickers = {"GOTO": "GOTO.JK", "BUKA": "BUKA.JK", "EMTK": "EMTK.JK"}
selected_ticker = st.sidebar.selectbox("Pilih Saham", list(tickers.keys()))
start_date = st.sidebar.date_input("Tanggal Awal", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("Tanggal Akhir", pd.to_datetime("today"))

# --- FUNCTION: FETCH & ENGINEER DATA ---
@st.cache_data(ttl=3600) # TTL 1 jam agar tidak menyimpan error kelamaan
def load_and_prep_data(ticker, start, end):
    try:
        # 1. Fetch Data menggunakan metode Ticker().history() yang lebih stabil di Cloud
        stock = yf.Ticker(tickers[ticker])
        
        # Konversi format tanggal ke string agar yfinance tidak bingung
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        
        df = stock.history(start=start_str, end=end_str)
        
        # Jika Yahoo Finance benar-benar memblokir atau tidak ada data
        if df.empty:
            return df
            
        # Standardisasi nama kolom (mengantisipasi perubahan versi yfinance)
        df.columns = df.columns.str.capitalize()
        
        # Pilih kolom utama dan buang baris kosong
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        
        # 2. Feature Engineering
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # RSI Calculation (Wilder's Smoothing)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Target Variable: Shift Close price by 1 day upwards (H+1)
        df['Target_Next_Close'] = df['Close'].shift(-1)
        
        # Drop rows with NaN (from rolling and shifting)
        df = df.dropna()
        return df
        
    except Exception as e:
        st.error(f"Terjadi kesalahan teknis saat mengambil data: {e}")
        return pd.DataFrame() # Kembalikan dataframe kosong agar ditangkap oleh error handler di bawah

# --- MAIN EXECUTION ---
if start_date < end_date:
    data = load_and_prep_data(selected_ticker, start_date, end_date)
    
    if data.empty:
        st.error("Data tidak ditemukan atau rentang waktu terlalu singkat.")
        st.info("💡 **Tips:** Jika error ini muncul di Streamlit Cloud, ini biasanya karena Yahoo Finance membatasi koneksi (Rate Limit) dari server publik secara sementara. Silakan tunggu beberapa menit dan refresh/muat ulang halaman ini.")
    else:
        st.subheader(f"Dataset Observasi: {selected_ticker}")
        st.dataframe(data.tail())

        # --- MODELING (MLR) ---
        # Features and Target
        X = data[['Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI_14']]
        y = data['Target_Next_Close']
        
        # Time Series Split (80% Train, 20% Test)
        split_idx = int(len(data) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scaling Features (Crucial for MLR with large Volume data)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # --- UI: DISPLAY METRICS & PREDICTION ---
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        # Predict Tomorrow's Price using the VERY LAST row of data
        latest_features = scaler.transform([X.iloc[-1]])
        tomorrow_pred = model.predict(latest_features)[0]
        current_price = data['Close'].iloc[-1]
        price_diff = tomorrow_pred - current_price
        
        col1.metric("Harga Saat Ini (Close Terakhir)", f"Rp {current_price:,.2f}")
        col2.metric("Prediksi H+1 (Besok)", f"Rp {tomorrow_pred:,.2f}", f"{price_diff:,.2f} Rupiah")
        col3.metric("Model RMSE", f"Rp {rmse:,.2f}")
        col4.metric("Model MAE", f"Rp {mae:,.2f}")
        
        # --- UI: VISUALIZATION (Plotly) ---
        st.markdown("---")
        st.subheader("Visualisasi Prediksi vs Aktual (Data Testing)")
        
        # Align dates for plotting
        test_dates = data.index[split_idx:]
        
        fig = go.Figure()
        # Actual Close Price (Testing Period)
        fig.add_trace(go.Scatter(x=test_dates, y=y_test, mode='lines', name='Harga Aktual (H+1)', line=dict(color='blue')))
        # Predicted Close Price
        fig.add_trace(go.Scatter(x=test_dates, y=y_pred, mode='lines', name='Harga Prediksi (H+1)', line=dict(color='red', dash='dash')))
        
        fig.update_layout(title="Perbandingan Harga Aktual vs Prediksi Regresi",
                          xaxis_title="Tanggal",
                          yaxis_title="Harga (Rupiah)",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Candlestick chart
        st.subheader(f"Pergerakan Harga Saham {selected_ticker} & Indikator")
        fig_candle = go.Figure(data=[go.Candlestick(x=data.index,
                        open=data['Open'], high=data['High'],
                        low=data['Low'], close=data['Close'], name="Candlestick")])
        fig_candle.add_trace(go.Scatter(x=data.index, y=data['SMA_5'], mode='lines', name='SMA 5', line=dict(color='orange')))
        fig_candle.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20', line=dict(color='purple')))
        
        fig_candle.update_layout(xaxis_rangeslider_visible=False, template="plotly_white", height=500)
        st.plotly_chart(fig_candle, use_container_width=True)

else:
    st.error("Tanggal Akhir harus lebih besar dari Tanggal Awal.")