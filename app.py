import streamlit as st
import pandas as pd
import datetime
from utils import fetch_data, add_indicators, fetch_data_for_prophet, forecast_with_prophet
from model import train_model

# App Configuration
st.set_page_config(page_title="📊 Stock Predictor", layout="wide")
st.title("📈 Stock Price Prediction & Forecasting App")
st.markdown("Predict stock prices using Machine Learning & Prophet Forecasting.")

# --- USER INPUT ---
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, RELIANCE.NS):", "RELIANCE.NS")
freq_option = st.selectbox("Select Frequency", ["Daily", "Hourly"])
forecast_days = st.slider("Days to Forecast (Prophet)", 1, 30, 7)

is_hourly = freq_option == "Hourly"
today = datetime.datetime.now()
default_start = today - datetime.timedelta(days=30 if is_hourly else 365)

start_date = st.date_input("Start Date", default_start.date())
end_date = st.date_input("End Date", today.date())

if start_date >= end_date:
    st.warning("⚠️ Start date must be before end date. Auto-correcting...")
    start_date = default_start.date()
    end_date = today.date()

# --- FETCH & CLEAN DATA ---
with st.spinner("📡 Fetching data..."):
    df = fetch_data(ticker, start_date, end_date, interval="1h" if is_hourly else "1d")

if df.empty:
    st.error("❌ Could not fetch stock data. Check the ticker or date range.")
    st.stop()

# --- RENAME & VALIDATE ---
df.columns = [str(col) for col in df.columns]  # Flatten column names if needed
if 'Date' in df.columns:
    df.rename(columns={'Date': 'ds'}, inplace=True)
elif 'Datetime' in df.columns:
    df.rename(columns={'Datetime': 'ds'}, inplace=True)

if 'Close' in df.columns:
    df.rename(columns={'Close': 'y'}, inplace=True)

# ✅ Remove timezone from ds (required by Prophet)
df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

if not {'ds', 'y'}.issubset(df.columns):
    st.error(f"❌ Required columns 'ds' and 'y' are missing. Available columns: {list(df.columns)}")
    st.stop()

df.dropna(subset=['ds', 'y'], inplace=True)

# --- FORECASTING SECTION ---
st.subheader("📆 Prophet Forecast")
try:
    forecast_df = forecast_with_prophet(df[['ds', 'y']].copy(), forecast_days, freq="H" if is_hourly else "D")
    st.line_chart(forecast_df.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])
except Exception as e:
    st.error(f"Forecast Error: {e}")

# --- MACHINE LEARNING SECTION ---
try:
    df_ml = df.copy()
    df_ml.rename(columns={'y': 'Close'}, inplace=True)  # Required for indicators
    df_ml = add_indicators(df_ml)
    features = ['MA10', 'MA50', 'RSI', 'MACD', 'Signal']

    preds, actuals, mse, train_len, test_len = train_model(df_ml.copy(), features)

    st.subheader("📉 ML Prediction vs Actual")
    st.write(f"📚 Training Samples: {train_len} | 🧪 Test Samples: {test_len} | MSE: `{mse:.2f}`")

    result_df = df_ml.tail(test_len).copy()
    result_df['Predicted'] = preds
    result_df = result_df[['Close', 'Predicted']]
    result_df.index = pd.to_datetime(df['ds'].tail(test_len).values)
    st.line_chart(result_df)
except Exception as e:
    st.error(f"ML Model Error: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption("🚀 Built using Python, Streamlit, scikit-learn & Prophet")
