# 📈 Stock AI — Claude-Powered Analysis System

A full ML + AI stock analysis pipeline:
- **45M+ data points** via yfinance (hourly/daily multi-ticker)
- **30+ technical indicators** (trend, momentum, volatility, volume)
- **XGBoost** feature selector — finds the most predictive meta-indicators
- **LSTM deep learning** — sequence-based price prediction
- **Claude AI** — interprets ML signals in plain English
- **Streamlit dashboard** — beautiful live interface

---

## 🚀 Quickstart (5 steps)

### 1. Clone / create the project folder
```bash
mkdir stock-ai && cd stock-ai
# (copy all project files here)
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key
Open `config.py` and replace `YOUR_API_KEY_HERE` with your key from
https://console.anthropic.com/

### 5. Train the models (one-time, ~15-30 min)
```bash
python ml_pipeline.py
```

This will:
- Download historical data for all tickers in `config.py`
- Compute 30+ technical indicators
- Run XGBoost to find the top 8 meta-indicators
- Train an LSTM on those features
- Save everything to the `models/` directory

### 6. Launch the dashboard
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser. 🎉

---

## 📁 Project Structure

```
stock-ai/
├── config.py           ← All settings (API key, tickers, hyperparams)
├── data_collector.py   ← Downloads OHLCV data via yfinance
├── indicators.py       ← 30+ technical indicators via `ta` library
├── ml_pipeline.py      ← XGBoost selector + LSTM trainer + inference
├── claude_analyzer.py  ← Claude API integration (analysis prompts)
├── app.py              ← Streamlit dashboard
├── requirements.txt    ← Python dependencies
├── data/               ← Auto-created: cached CSV data
└── models/             ← Auto-created: trained model files
```

---

## ⚙️ Customization

### Add more tickers
Edit `TICKERS` list in `config.py`.

### Get more data points (closer to 45M)
Change `INTERVAL = "1h"` in `config.py` for hourly data.
Note: yfinance only provides 2 years of hourly data.

### Tune the ML models
All hyperparameters are in `config.py`:
```python
XGB_N_ESTIMATORS = 500   # More = better but slower
LSTM_EPOCHS      = 50    # More = better but slower
TOP_N_FEATURES   = 8     # How many indicators to use
SEQUENCE_LENGTH  = 20    # How many days LSTM looks back
```

### Train on specific tickers only
```python
from data_collector import download_data
from indicators import add_all_indicators
from ml_pipeline import train_xgboost, train_lstm

raw   = download_data(tickers=["AAPL", "NVDA"])
feats = add_all_indicators(raw)
xgb, top_features = train_xgboost(feats)
lstm = train_lstm(feats, top_features)
```

---

## 🧠 How It Works

```
Raw Price Data (OHLCV)
        │
        ▼
30+ Technical Indicators
(RSI, MACD, ADX, Bollinger, ATR, OBV …)
        │
        ▼
XGBoost Feature Selection
(Finds top 8 most predictive indicators)
        │
        ▼
LSTM Deep Learning
(Learns sequence patterns in top indicators)
        │
        ├──── LSTM probability (60% weight)
        ├──── XGBoost probability (40% weight)
        └──── Ensemble Signal: BUY / HOLD / SELL
                        │
                        ▼
               Claude AI Analysis
       (Interprets signal + indicator context
        → structured trading recommendation)
                        │
                        ▼
            Streamlit Dashboard
         (Live charts + signal cards)
```

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.

- Backtested accuracy does NOT guarantee future performance
- Markets are non-stationary; models can fail in new regimes
- Never risk money you cannot afford to lose
- This is not financial advice
- Consult a licensed financial advisor before trading

---

## 📚 Libraries Used

| Library | Purpose |
|---------|---------|
| yfinance | Stock data download |
| ta | 30+ technical indicators |
| xgboost | Gradient boosting classifier |
| tensorflow | LSTM deep learning |
| anthropic | Claude AI API |
| streamlit | Dashboard UI |
| plotly | Interactive charts |
| scikit-learn | Preprocessing + metrics |
