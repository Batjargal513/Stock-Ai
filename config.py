# ============================================================
#  config.py  —  Central configuration for Stock AI project
# ============================================================

# ── Anthropic ────────────────────────────────────────────────
ANTHROPIC_API_KEY = "YOUR_API_KEY_HERE"   # ← Paste your key here
CLAUDE_MODEL      = "claude-opus-4-5"

# ── Stocks to track ─────────────────────────────────────────
TICKERS = [
    "AAPL", "TSLA", "MSFT", "NVDA", "GOOGL",
    "AMZN", "META", "SPY",  "QQQ",  "AMD",
]

# ── Data collection ──────────────────────────────────────────
START_DATE    = "2015-01-01"
END_DATE      = "2024-12-31"
INTERVAL      = "1d"          # "1d" | "1h" | "5m"
DATA_PATH     = "data/raw_stock_data.csv"
FEATURES_PATH = "data/features.csv"

# ── Model training ───────────────────────────────────────────
TEST_SPLIT       = 0.2
SEQUENCE_LENGTH  = 20          # Days of history fed into LSTM
XGB_N_ESTIMATORS = 500
LSTM_EPOCHS      = 50
LSTM_BATCH_SIZE  = 64
TOP_N_FEATURES   = 8           # Features selected by XGBoost
MODELS_DIR       = "models/"

# ── Streamlit dashboard ──────────────────────────────────────
REFRESH_SECONDS  = 60          # Auto-refresh interval
