from flask import Flask, render_template, request
import os
from ml_models.predict_asset import train_and_predict
from Cli_Portfolio_Ml import load_asset_data
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import reduce


app = Flask(__name__)

@app.route("/")
def form():
    return render_template("form.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    risk = request.form["risk"]
    amount = float(request.form["amount"])
    all_assets = ["AAPL", "ETH", "BTC", "TSLA", "JNJ", "DOGE", "SOL", "TRX", "BNB", "LITE", "USD", "XRP"]

    # Classify by volatility and select matching group
    classified_assets = classify_assets_by_volatility(all_assets)
    selected_assets = classified_assets.get(risk, all_assets)

    # Predict future growth
    predictions = [
        train_and_predict(asset, f"cleaned_data/{asset}_clean.csv")
        for asset in selected_assets
    ]

    # Sort all by predicted growth
    sorted_predictions = sorted(predictions, key=lambda x: x["growth_pct"], reverse=True)

    # Ensure always select top 3 assets
    top_predictions = sorted_predictions[:3]

    # Normalise weights
    total_growth = sum(p["growth_pct"] if p["growth_pct"] > 0 else 1 for p in top_predictions)
    portfolio = {
        p["asset"]: {
            "weight": (p["growth_pct"] if p["growth_pct"] > 0 else 1) / total_growth,
            "amount": ((p["growth_pct"] if p["growth_pct"] > 0 else 1) / total_growth) * amount
        } for p in top_predictions
    }

    # Only use top 3 predictions
    predictions = top_predictions

    user_profile = {
        "risk_level": risk,
        "preferred_assets": list(portfolio.keys()),
        "investment_amount": amount
    }

    results = backtest_portfolio(portfolio, user_profile)

    return render_template("results.html", portfolio=portfolio, predictions=predictions, results=results)


def backtest_portfolio(portfolio_alloc, user_profile):
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

    start = merged.index[0]
    end = merged.index[-1]
    rec_start = merged.loc[start, "Recommended"]
    rec_end = merged.loc[end, "Recommended"]
    base_start = merged.loc[start, "Baseline"]
    base_end = merged.loc[end, "Baseline"]

    rec_growth = (rec_end - rec_start) / rec_start * 100
    base_growth = (base_end - base_start) / base_start * 100
    rec_final = user_profile["investment_amount"] * rec_end

    # Save graph
    plt.figure(figsize=(10, 5))
    plt.plot(merged["Date"], merged["Recommended"], label="Recommended Portfolio", color="blue")
    plt.plot(merged["Date"], merged["Baseline"], label="Equal Weight Baseline", color="orange", linestyle='--')
    plt.title(f"Backtested Portfolio vs. Baseline ({user_profile['risk_level'].capitalize()} Risk)")
    plt.xlabel("Date")
    plt.ylabel("Growth (Normalized)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/portfolio_growth_comparison.png")
    plt.close()

    return {
        "recommended_growth": rec_growth,
        "baseline_growth": base_growth,
        "recommended_final": rec_final
    }



def classify_assets_by_volatility(asset_list):
    vol_dict = {}
    for asset in asset_list:
        df = pd.read_csv(f"cleaned_data/{asset}_clean.csv")
        df["returns"] = df["Close"].pct_change()
        vol = df["returns"].std()
        vol_dict[asset] = vol

    # Rank by volatility
    sorted_assets = sorted(vol_dict.items(), key=lambda x: x[1])
    n = len(sorted_assets)
    classified = {
        "low": [a for a, _ in sorted_assets[:n//3]],
        "medium": [a for a, _ in sorted_assets[n//3:2*n//3]],
        "high": [a for a, _ in sorted_assets[2*n//3:]]
    }
    return classified


if __name__ == "__main__":
    app.run(debug=True)
