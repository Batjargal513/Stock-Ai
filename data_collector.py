# ============================================================
#  data_collector.py  —  Downloads and caches stock OHLCV data
# ============================================================

import os
import pandas as pd
import yfinance as yf
from config import TICKERS, START_DATE, END_DATE, INTERVAL, DATA_PATH


def download_data(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    interval: str = INTERVAL,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download historical OHLCV data for a list of tickers.
    Results are cached to DATA_PATH so subsequent calls are instant.

    Returns a DataFrame with columns:
        Open, High, Low, Close, Volume, Ticker
    """
    os.makedirs("data", exist_ok=True)

    if os.path.exists(DATA_PATH) and not force_refresh:
        print(f"✅  Loading cached data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        print(f"    {len(df):,} rows  |  {df['Ticker'].nunique()} tickers")
        return df

    print(f"⬇️   Downloading data for {len(tickers)} tickers …")
    frames = []
    for i, ticker in enumerate(tickers, 1):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                print(f"  [{i}/{len(tickers)}] ⚠️  {ticker} – no data returned")
                continue

            raw.columns = raw.columns.get_level_values(0) if isinstance(
                raw.columns, pd.MultiIndex
            ) else raw.columns
            raw["Ticker"] = ticker
            frames.append(raw)
            print(f"  [{i}/{len(tickers)}] ✔  {ticker} – {len(raw):,} rows")
        except Exception as exc:
            print(f"  [{i}/{len(tickers)}] ✗  {ticker} – {exc}")

    if not frames:
        raise RuntimeError("No data was downloaded. Check your internet connection.")

    df = pd.concat(frames)
    df.sort_index(inplace=True)
    df.to_csv(DATA_PATH)
    print(f"\n💾  Saved {len(df):,} rows to {DATA_PATH}")
    return df


def get_latest_ohlcv(ticker: str) -> pd.Series:
    """Fetch the single latest daily bar for a ticker (live)."""
    raw = yf.download(ticker, period="5d", interval="1d",
                      auto_adjust=True, progress=False)
    raw.columns = raw.columns.get_level_values(0) if isinstance(
        raw.columns, pd.MultiIndex
    ) else raw.columns
    return raw.iloc[-1]


if __name__ == "__main__":
    data = download_data()
    print(data.tail())
