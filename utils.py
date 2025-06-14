import pandas as pd
import yfinance as yf

def fetch_data(ticker, start, end, interval="1d"):
    data = yf.download(ticker, start=start, end=end, interval=interval)

    if data.empty:
        return pd.DataFrame()

    data.reset_index(inplace=True)

    # Flatten columns if MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if col[0] != '' else col[1] for col in data.columns]

    return data

def add_indicators(df):
    if 'Close' not in df.columns:
        raise KeyError("❌ 'Close' column not found in dataframe.")

    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df

def fetch_data_for_prophet(ticker, start, end, interval="1d"):
    df = fetch_data(ticker, start, end, interval)
    df.columns = [str(col) for col in df.columns]
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'ds'}, inplace=True)
    elif 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'ds'}, inplace=True)
    if 'Close' in df.columns:
        df.rename(columns={'Close': 'y'}, inplace=True)
    return df[['ds', 'y']].dropna()

from prophet import Prophet

def forecast_with_prophet(df, periods=7, freq='D'):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast
