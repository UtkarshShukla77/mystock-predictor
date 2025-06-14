import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def train_model(df, features):
    # Ensure required features and 'Close' are present
    missing_features = set(features + ['Close']) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    df.dropna(subset=features + ['Close'], inplace=True)

    X = df[features].values
    y = df['Close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return predictions, y_test, mse, len(y_train), len(y_test)
