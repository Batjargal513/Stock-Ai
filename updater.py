# ============================================================
#  updater.py  —  Daily data refresh + model retraining
#  Run manually:   python updater.py
#  Schedule it:    see README for cron / Task Scheduler setup
# ============================================================

import os
import sys
import logging
import joblib
import pandas as pd
from datetime import datetime, timedelta

# ── Logging setup ────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(f"logs/update_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def update_data():
    """Append latest price data to the existing CSV instead of re-downloading everything."""
    from data_collector import download_data
    from config import DATA_PATH, TICKERS

    log.info("=" * 50)
    log.info("DAILY UPDATE STARTED")
    log.info("=" * 50)

    # Load existing data to find the last date
    if os.path.exists(DATA_PATH):
        existing = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        last_date = existing.index.max()
        # Fetch only from last date onward (+ 1 day buffer)
        start = (last_date - timedelta(days=2)).strftime("%Y-%m-%d")
        log.info(f"Existing data found. Last date: {last_date.date()}. Fetching from {start}…")
    else:
        existing = pd.DataFrame()
        start = "2015-01-01"
        log.info("No existing data found. Doing full download…")

    # Download only the new chunk
    import yfinance as yf
    frames = []
    for ticker in TICKERS:
        try:
            raw = yf.Ticker(ticker).history(
                start=start,
                auto_adjust=True,
            )
            if raw.empty:
                log.warning(f"  {ticker} — no new data")
                continue
            raw.index = raw.index.tz_localize(None)  # strip timezone
            raw["Ticker"] = ticker
            frames.append(raw)
            log.info(f"  {ticker} — {len(raw)} new rows fetched")
        except Exception as e:
            log.error(f"  {ticker} — failed: {e}")

    if not frames:
        log.warning("No new data fetched. Market may be closed today.")
        return False

    new_data = pd.concat(frames)

    # Merge with existing, drop duplicates, sort
    if not existing.empty:
        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
    else:
        combined = new_data

    combined.to_csv(DATA_PATH)
    log.info(f"Data updated → {DATA_PATH}  ({len(combined):,} total rows)")
    return True


def update_features():
    """Recompute indicators on the updated data."""
    from data_collector import download_data
    from indicators import add_all_indicators

    log.info("Recomputing indicators…")
    raw = download_data()
    features = add_all_indicators(raw)
    log.info(f"Features updated → {len(features):,} rows, {len(features.columns)} columns")
    return features


def retrain_models(features):
    """Retrain XGBoost + LSTM on fresh data."""
    from ml_pipeline import train_xgboost, train_lstm

    log.info("Retraining XGBoost…")
    _, top_features = train_xgboost(features)

    log.info("Retraining LSTM…")
    train_lstm(features, top_features)

    log.info("Models retrained and saved.")


def run_update(retrain: bool = False):
    """
    Full update pipeline.
    retrain=False  → just refresh data (fast, daily)
    retrain=True   → refresh data + retrain models (slow, weekly)
    """
    try:
        fetched = update_data()

        if fetched:
            features = update_features()

            if retrain:
                log.info("Full retrain requested…")
                retrain_models(features)
            else:
                log.info("Skipping model retrain (data-only update). Pass --retrain to retrain.")
        else:
            log.info("Nothing to update today.")

        log.info("UPDATE COMPLETE ✅")

    except Exception as e:
        log.error(f"Update failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    retrain_flag = "--retrain" in sys.argv
    run_update(retrain=retrain_flag)
