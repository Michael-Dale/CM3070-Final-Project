import pandas as pd
import os

# Create output folder
os.makedirs("cleaned_data", exist_ok=True)

# AAPL CLEANING
aapl_df = pd.read_csv("data/Apple/aapl.csv")
aapl_df.columns = aapl_df.columns.str.strip()
aapl_df["Date"] = pd.to_datetime(aapl_df["Date"], format="%m/%d/%Y")
aapl_df["Close"] = aapl_df["Close/Last"].replace('[\$,]', '', regex=True).astype(float)
aapl_clean = aapl_df[["Date", "Close"]].sort_values("Date")
aapl_clean.to_csv("cleaned_data/AAPL_clean.csv", index=False)

# ETHEREUM CLEANING
eth_df = pd.read_csv("data/Crypto/coin_Ethereum.csv")
eth_df["Date"] = pd.to_datetime(eth_df["Date"])
eth_clean = eth_df[["Date", "Close"]].sort_values("Date")
eth_clean.to_csv("cleaned_data/ETH_clean.csv", index=False)

# BITCOIN CLEANING
btc_df = pd.read_csv("data/Crypto/coin_Bitcoin.csv")
btc_df["Date"] = pd.to_datetime(btc_df["Date"])
btc_clean = btc_df[["Date", "Close"]].sort_values("Date")
btc_clean.to_csv("cleaned_data/BTC_clean.csv", index=False)

# Dogecoin CLEANING
btc_df = pd.read_csv("data/Crypto/coin_Dogecoin.csv")
btc_df["Date"] = pd.to_datetime(btc_df["Date"])
btc_clean = btc_df[["Date", "Close"]].sort_values("Date")
btc_clean.to_csv("cleaned_data/DOGE_clean.csv", index=False)

# Solana CLEANING
btc_df = pd.read_csv("data/Crypto/coin_Solana.csv")
btc_df["Date"] = pd.to_datetime(btc_df["Date"])
btc_clean = btc_df[["Date", "Close"]].sort_values("Date")
btc_clean.to_csv("cleaned_data/SOL_clean.csv", index=False)

# TRON CLEANING
btc_df = pd.read_csv("data/Crypto/coin_Tron.csv")
btc_df["Date"] = pd.to_datetime(btc_df["Date"])
btc_clean = btc_df[["Date", "Close"]].sort_values("Date")
btc_clean.to_csv("cleaned_data/TRX_clean.csv", index=False)

# BinanceCoin CLEANING
btc_df = pd.read_csv("data/Crypto/coin_BinanceCoin.csv")
btc_df["Date"] = pd.to_datetime(btc_df["Date"])
btc_clean = btc_df[["Date", "Close"]].sort_values("Date")
btc_clean.to_csv("cleaned_data/BNB_clean.csv", index=False)

# Litecoin CLEANING
btc_df = pd.read_csv("data/Crypto/coin_Litecoin.csv")
btc_df["Date"] = pd.to_datetime(btc_df["Date"])
btc_clean = btc_df[["Date", "Close"]].sort_values("Date")
btc_clean.to_csv("cleaned_data/LITE_clean.csv", index=False)

# USDCoin CLEANING
btc_df = pd.read_csv("data/Crypto/coin_USDCoin.csv")
btc_df["Date"] = pd.to_datetime(btc_df["Date"])
btc_clean = btc_df[["Date", "Close"]].sort_values("Date")
btc_clean.to_csv("cleaned_data/USD_clean.csv", index=False)

# XRP CLEANING
btc_df = pd.read_csv("data/Crypto/coin_XRP.csv")
btc_df["Date"] = pd.to_datetime(btc_df["Date"])
btc_clean = btc_df[["Date", "Close"]].sort_values("Date")
btc_clean.to_csv("cleaned_data/XRP_clean.csv", index=False)

# JNJ CLEANING
jnj_df = pd.read_csv("data/Stocks/jnj.csv")
jnj_df["Date"] = pd.to_datetime(jnj_df["Date"])
jnj_clean = jnj_df[["Date", "Close"]].sort_values("Date")
jnj_clean.to_csv("cleaned_data/JNJ_clean.csv", index=False)

# TSLA CLEANING
tsla_df = pd.read_csv("data/Stocks/tsla.csv")
tsla_df["Date"] = pd.to_datetime(tsla_df["Date"])
tsla_clean = tsla_df[["Date", "Close"]].sort_values("Date")
tsla_clean.to_csv("cleaned_data/TSLA_clean.csv", index=False)

print("All datasets cleaned and saved to 'cleaned_data/' folder.")
