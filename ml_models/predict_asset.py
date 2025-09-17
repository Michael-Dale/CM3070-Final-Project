import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def load_asset_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df["Close"].values

def create_dataset(series, window_size=10):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

def generate_portfolio_weights(predictions, risk_level):
    raw = {p['asset']: p['growth_pct'] for p in predictions}

    # Risk sensitivity modifiers
    risk_profile = {
        "low": {"negative_factor": 0.0, "positive_amp": 1.0},
        "medium": {"negative_factor": 0.5, "positive_amp": 1.2},
        "high": {"negative_factor": 1.0, "positive_amp": 1.5},
    }

    profile = risk_profile[risk_level]
    adjusted = {}

    for asset, growth in raw.items():
        if growth < 0:
            adjusted_value = growth * profile["negative_factor"]
        else:
            adjusted_value = growth * profile["positive_amp"]
        adjusted[asset] = max(adjusted_value, 0)

    total = sum(adjusted.values())
    if total == 0:
        # fallback to equal weights
        n = len(adjusted)
        return {k: 1 / n for k in adjusted}

    weights = {k: v / total for k, v in adjusted.items()}
    return weights



def train_and_predict(asset_name, path):
    print(f"\nTraining model for {asset_name}...")

    # Load and prepare data
    prices = load_asset_data(path)
    X, y = create_dataset(prices)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE for {asset_name}: {mse:.4f}")

    # Predict next day price
    latest_window = prices[-10:].reshape(1, -1)
    next_price = model.predict(latest_window)[0]
    print(f"Predicted next price: R{next_price:.2f}")
    print(f"Latest known price: R{prices[-1]:.2f}")
    growth = (next_price - prices[-1]) / prices[-1] * 100
    print(f"Expected change: {growth:.2f}%\n")

    return {
        "asset": asset_name,
        "latest": prices[-1],
        "predicted": next_price,
        "growth_pct": growth,
        "mse": mse
    }


if __name__ == "__main__":
    results = []
    results.append(train_and_predict("AAPL", "data/cleaned_data/AAPL_clean.csv"))
    results.append(train_and_predict("ETH", "data/cleaned_data/ETH_clean.csv"))
    results.append(train_and_predict("BTC", "data/cleaned_data/BTC_clean.csv"))

    # Print summary
    for r in results:
        print(f"{r['asset']}: Predicted {r['growth_pct']:.2f}% change")
