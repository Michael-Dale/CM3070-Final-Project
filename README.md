# Financial Advisor Bot

An intelligent, user-driven financial advisory system that generates personalized multi-asset portfolio recommendations based on risk preferences and historical data.

## Project Overview

The Financial Advisor Bot is a Python-based tool that allows users to receive optimized investment portfolio recommendations tailored to their risk appetite — **low**, **medium**, or **high**.

It uses a combination of:
- **Historical asset performance**
- **Volatility-based asset classification**
- **Machine learning price prediction**
- **Sharpe-ratio-informed allocation**
- **Interactive CLI and Flask-based web interface**

The system supports both novice and advanced investors in making informed portfolio decisions using real market data.

---

## Features

- CLI & Web UI input for risk level and investment amount  
- Dynamic portfolio allocation logic based on predicted growth and volatility  
- Backtesting engine with comparison against equal-weight baseline  
- Asset volatility classification automatically adjusted by risk  
- Interactive growth graphs and CSV output  
- Modular ML model predictions per asset 

---

## Technologies Used

- **Python 3.11**
- **Flask** (web server)
- **Pandas** & **NumPy** (data handling)
- **Matplotlib** (graphing)
- **TensorFlow/Keras** (prediction model)
- **Jinja2** (Flask templating)
- **HTML/CSS** (user interface)

---

## Folder Structure

```
├── cleaned_data/             # Preprocessed CSVs of each asset
├── data/                     # Raw asset data
├── ml_models/
│   └── predict_asset.py      # ML price prediction model
├── templates/
│   ├── form.html             # Web input form
│   └── results.html          # Portfolio results + graph
├── static/
│   └── portfolio_growth_comparison.png
├── Cli_Portfolio_Ml.py       # Core CLI & logic engine
├── Flask Server.py           # Flask web server
├── clean_data.py             # Preprocessing script
├── backtest_results.csv      # Output of last test
└── README.md
```

---

## Setup Instructions

1. **Clone this repository**
   ```bash
   git clone https://github.com/Michael-Dale/CM3070-Final-Project.git
   cd CM3070-Final-Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the CLI**
   ```bash
   python Cli_Portfolio_Ml.py
   ```

4. **Run the Flask Web App**
   ```bash
   python "Flask Server.py"
   ```

5. **Access in browser**
   Visit http://127.0.0.1:5000 to interact with the web interface.

---

## Example Assets Supported

- AAPL (Apple Inc.)
- BTC (Bitcoin)
- ETH (Ethereum)
- TSLA (Tesla Inc.)
- JNJ (Johnson & Johnson)
- DOGE (Dogecoin)
- SOL (Solana)
- TRX (Tron)

---

## Backtesting and Output

The system backtests all portfolios over historical data and generates:
- A comparative **growth graph** (saved in `static/`)
- CSV file showing recommended vs baseline performance
- Final investment value and % growth output

---

## Author

**Michael Dale**  
BSc Computer Science, University of London AI & ML
