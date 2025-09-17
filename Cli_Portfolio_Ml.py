import pandas as pd
import matplotlib.pyplot as plt
from ml_models.predict_asset import train_and_predict

# Risk-sensitive dynamic weighting
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
        n = len(adjusted)
        return {k: 1 / n for k in adjusted}

    weights = {k: v / total for k, v in adjusted.items()}
    return weights

# CLI for user portfolio preferences
def get_user_input():
    print("Welcome to the Financial Advisor Bot Prototype!\n")

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
            else:
                print("Please enter a positive amount.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return {
        "risk_level": risk,
        "preferred_assets": ["AAPL", "BTC", "ETH"],
        "investment_amount": amount
    }

def load_asset_data(symbol):
    path = f"cleaned_data/{symbol}_clean.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df["returns"] = df["Close"].pct_change()
    return df.dropna()

def allocate_portfolio(predictions, user_profile):
    weights = generate_portfolio_weights(predictions, user_profile["risk_level"])
    total = user_profile["investment_amount"]
    allocation = {}
    for asset in user_profile["preferred_assets"]:
        weight = weights.get(asset, 0)
        allocation[asset] = {
            "weight": weight,
            "amount": weight * total
        }
    return allocation

def backtest_portfolio(portfolio_alloc, user_profile):
    from functools import reduce

    asset_names = list(portfolio_alloc.keys())
    dfs = []

    for asset in asset_names:
        df = load_asset_data(asset)[["Date", "Close"]].copy()
        df = df.sort_values("Date").reset_index(drop=True)
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

    plt.figure(figsize=(10, 5))
    plt.plot(merged["Date"], merged["Recommended"], label="Recommended Portfolio", color="blue")
    plt.plot(merged["Date"], merged["Baseline"], label="Equal Weight Baseline", color="orange", linestyle='--')
    plt.title(f"Backtested Portfolio vs. Baseline ({user_profile['risk_level'].capitalize()} Risk)")
    plt.xlabel("Date")
    plt.ylabel("Growth (Normalized)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    start = merged.index[0]
    end = merged.index[-1]
    rec_start = merged.loc[start, "Recommended"]
    rec_end = merged.loc[end, "Recommended"]
    base_start = merged.loc[start, "Baseline"]
    base_end = merged.loc[end, "Baseline"]

    rec_growth = (rec_end - rec_start) / rec_start * 100
    base_growth = (base_end - base_start) / base_start * 100
    rec_final = user_profile["investment_amount"] * rec_end
    base_final = user_profile["investment_amount"] * base_end

    print("\ Backtest Results")
    print(f"\nRecommended Portfolio: {rec_end:.2f}x | Growth: {rec_growth:.2f}% → R{rec_final:,.2f}")
    print(f"Baseline Portfolio:    {base_end:.2f}x | Growth: {base_growth:.2f}% → R{base_final:,.2f}")

    output_df = merged[["Date", "Recommended", "Baseline"]].copy()
    output_df.to_csv("backtest_results.csv", index=False)
    print("Results saved to backtest_results.csv")

if __name__ == "__main__":
    user_profile = get_user_input()

    predictions = [
        train_and_predict("AAPL", "cleaned_data/AAPL_clean.csv"),
        train_and_predict("ETH", "cleaned_data/ETH_clean.csv"),
        train_and_predict("BTC", "cleaned_data/BTC_clean.csv")
    ]

    portfolio = allocate_portfolio(predictions, user_profile)

    print("\ Portfolio Recommendation")
    for asset, info in portfolio.items():
        print(f"{asset}: {info['weight']*100:.0f}%  →  R{info['amount']:.2f}")

    print("\ ML Predictions")
    for p in predictions:
        print(f"{p['asset']}: Predicted {p['growth_pct']:.2f}% change")

    backtest_portfolio(portfolio, user_profile)
