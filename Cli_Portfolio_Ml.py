import pandas as pd
import matplotlib.pyplot as plt
from ml_models.predict_asset import train_and_predict
from functools import reduce
import os

# Classify assets by volatility
def classify_assets_by_volatility(asset_list):
    vol_dict = {}
    for asset in asset_list:
        df = pd.read_csv(f"cleaned_data/{asset}_clean.csv")
        df["returns"] = df["Close"].pct_change()
        vol_dict[asset] = df["returns"].std()

    sorted_assets = sorted(vol_dict.items(), key=lambda x: x[1])
    n = len(sorted_assets)
    return {
        "low": [a for a, _ in sorted_assets[:n//3]],
        "medium": [a for a, _ in sorted_assets[n//3:2*n//3]],
        "high": [a for a, _ in sorted_assets[2*n//3:]]
    }

# CLI Input
def get_user_input():
    print("Welcome to the Financial Advisor Bot CLI\n")
    while True:
        risk = input("Enter your risk level (low, medium, high): ").strip().lower()
        if risk in ["low", "medium", "high"]:
            break
        print("Invalid input. Please type: low, medium, or high.")
    
    while True:
        try:
            amount = float(input("Enter the amount you want to invest (in Rands): ").strip())
            if amount > 0:
                break
            print("Please enter a positive amount.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return {
        "risk_level": risk,
        "investment_amount": amount
    }

# Load CSV asset data
def load_asset_data(symbol):
    path = f"cleaned_data/{symbol}_clean.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df["returns"] = df["Close"].pct_change()
    return df.dropna()

# Backtest
def backtest_portfolio(portfolio_alloc, user_profile):
    asset_names = list(portfolio_alloc.keys())
    dfs = []

    for asset in asset_names:
        df = load_asset_data(asset)[["Date", "Close"]].copy()
        df["Normalized"] = df["Close"] / df["Close"].iloc[0]
        df = df[["Date", "Normalized"]].rename(columns={"Normalized": asset})
        dfs.append(df)

    merged = reduce(lambda left, right: pd.merge(left, right, on="Date", how="outer"), dfs)
    merged = merged.sort_values("Date").ffill().dropna()

    for asset, info in portfolio_alloc.items():
        merged[asset + "_weighted"] = merged[asset] * info["weight"]
    merged["Recommended"] = merged[[a + "_weighted" for a in asset_names]].sum(axis=1)

    equal_weight = 1 / len(asset_names)
    for asset in asset_names:
        merged[asset + "_equal"] = merged[asset] * equal_weight
    merged["Baseline"] = merged[[a + "_equal" for a in asset_names]].sum(axis=1)

    start = merged.index[0]
    end = merged.index[-1]
    rec_start = merged.loc[start, "Recommended"]
    rec_end = merged.loc[end, "Recommended"]
    base_start = merged.loc[start, "Baseline"]
    base_end = merged.loc[end, "Baseline"]

    rec_growth = (rec_end - rec_start) / rec_start * 100
    base_growth = (base_end - base_start) / base_start * 100
    rec_final = user_profile["investment_amount"] * rec_end

    print("\ Backtest Results")
    print(f"Recommended Portfolio: {rec_end:.2f}x | Growth: {rec_growth:.2f}% → R{rec_final:,.2f}")
    print(f"Baseline Portfolio:    {base_end:.2f}x | Growth: {base_growth:.2f}%")

    output_df = merged[["Date", "Recommended", "Baseline"]]
    output_df.to_csv("backtest_results.csv", index=False)
    print("Results saved to backtest_results.csv")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(merged["Date"], merged["Recommended"], label="Recommended", color="blue")
    plt.plot(merged["Date"], merged["Baseline"], label="Baseline", color="orange", linestyle='--')
    plt.title(f"Backtested Portfolio vs. Baseline ({user_profile['risk_level'].capitalize()} Risk)")
    plt.xlabel("Date")
    plt.ylabel("Growth (Normalized)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main CLI Execution
if __name__ == "__main__":
    all_assets = ["AAPL", "ETH", "BTC", "TSLA", "JNJ", "DOGE", "SOL", "TRX", "BNB", "LITE", "USD", "XRP"]
    user_profile = get_user_input()
    classified = classify_assets_by_volatility(all_assets)
    selected_assets = classified.get(user_profile["risk_level"], all_assets)

    print("\nGathering predictions...")
    predictions = [train_and_predict(asset, f"cleaned_data/{asset}_clean.csv") for asset in selected_assets]

    sorted_predictions = sorted(predictions, key=lambda x: x["growth_pct"], reverse=True)
    top_predictions = sorted_predictions[:3]

    total_growth = sum([max(p["growth_pct"], 0.01) for p in top_predictions])
    portfolio = {
        p["asset"]: {
            "weight": max(p["growth_pct"], 0.01) / total_growth,
            "amount": (max(p["growth_pct"], 0.01) / total_growth) * user_profile["investment_amount"]
        } for p in top_predictions
    }

    print("\ Portfolio Recommendation")
    for asset, info in portfolio.items():
        print(f"{asset}: {info['weight']*100:.1f}% → R{info['amount']:.2f}")

    print("\ ML Predictions")
    for p in top_predictions:
        print(f"{p['asset']}: Predicted {p['growth_pct']:.2f}% growth")

    backtest_portfolio(portfolio, user_profile)
