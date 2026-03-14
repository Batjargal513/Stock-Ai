# ============================================================
#  indicators.py  —  Computes 30+ technical indicators
# ============================================================

import pandas as pd
import numpy as np
import ta
from config import FEATURES_PATH
import os


# ── All indicator definitions ────────────────────────────────
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects df to have columns: Open, High, Low, Close, Volume, Ticker
    Returns df enriched with 30+ technical indicators and a binary Target.
    """
    results = []
    for ticker, group in df.groupby("Ticker"):
        group = group.copy().sort_index()
        c, h, l, v = group["Close"], group["High"], group["Low"], group["Volume"]

        # ── Trend ────────────────────────────────────────────
        group["EMA_9"]      = ta.trend.ema_indicator(c, window=9)
        group["EMA_21"]     = ta.trend.ema_indicator(c, window=21)
        group["EMA_50"]     = ta.trend.ema_indicator(c, window=50)
        group["SMA_200"]    = ta.trend.sma_indicator(c, window=200)
        group["MACD"]       = ta.trend.macd(c)
        group["MACD_signal"]= ta.trend.macd_signal(c)
        group["MACD_diff"]  = ta.trend.macd_diff(c)
        group["ADX"]        = ta.trend.adx(h, l, c)
        group["DI_pos"]     = ta.trend.adx_pos(h, l, c)
        group["DI_neg"]     = ta.trend.adx_neg(h, l, c)
        group["Ichimoku_a"] = ta.trend.ichimoku_a(h, l)
        group["Ichimoku_b"] = ta.trend.ichimoku_b(h, l)
        group["CCI"]        = ta.trend.cci(h, l, c)
        group["Aroon_up"]   = ta.trend.aroon_up(h, l)
        group["Aroon_down"] = ta.trend.aroon_down(h, l)

        # ── Momentum ─────────────────────────────────────────
        group["RSI"]        = ta.momentum.rsi(c, window=14)
        group["RSI_fast"]   = ta.momentum.rsi(c, window=7)
        group["Stoch_k"]    = ta.momentum.stoch(h, l, c)
        group["Stoch_d"]    = ta.momentum.stoch_signal(h, l, c)
        group["Williams_R"] = ta.momentum.williams_r(h, l, c)
        group["ROC"]        = ta.momentum.roc(c)
        group["TSI"]        = ta.momentum.tsi(c)

        # ── Volatility ───────────────────────────────────────
        group["BB_high"]    = ta.volatility.bollinger_hband(c)
        group["BB_low"]     = ta.volatility.bollinger_lband(c)
        group["BB_mid"]     = ta.volatility.bollinger_mavg(c)
        group["BB_width"]   = ta.volatility.bollinger_wband(c)
        group["BB_pct"]     = ta.volatility.bollinger_pband(c)
        group["ATR"]        = ta.volatility.average_true_range(h, l, c)
        group["KC_high"]    = ta.volatility.keltner_channel_hband(h, l, c)
        group["KC_low"]     = ta.volatility.keltner_channel_lband(h, l, c)

        # ── Volume ───────────────────────────────────────────
        group["OBV"]        = ta.volume.on_balance_volume(c, v)
        group["MFI"]        = ta.volume.money_flow_index(h, l, c, v)
        group["VWAP"]       = ta.volume.volume_weighted_average_price(h, l, c, v)
        group["CMF"]        = ta.volume.chaikin_money_flow(h, l, c, v)
        group["EOM"]        = ta.volume.ease_of_movement(h, l, v)

        # ── Derived price features ───────────────────────────
        group["Returns_1d"] = c.pct_change(1)
        group["Returns_5d"] = c.pct_change(5)
        group["Returns_10d"]= c.pct_change(10)
        group["High_Low_pct"]= (h - l) / l
        group["Close_Open_pct"] = (c - group["Open"]) / group["Open"]
        group["Volume_change"] = v.pct_change()
        group["Trend_EMA_cross"] = (group["EMA_9"] > group["EMA_21"]).astype(int)

        # ── Target: did price rise next day? ─────────────────
        group["Target"] = (c.shift(-1) > c).astype(int)

        results.append(group)

    out = pd.concat(results).dropna()
    os.makedirs("data", exist_ok=True)
    out.to_csv(FEATURES_PATH)
    print(f"💾  Features saved → {FEATURES_PATH}  ({len(out):,} rows, {len(out.columns)} columns)")
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all indicator column names (excluding meta columns)."""
    skip = {"Open", "High", "Low", "Close", "Volume", "Ticker", "Target"}
    return [c for c in df.columns if c not in skip]


def get_latest_indicators(ticker: str, raw_df: pd.DataFrame) -> pd.Series:
    """Return the latest row of indicators for one ticker."""
    df = add_all_indicators(raw_df[raw_df["Ticker"] == ticker])
    return df.iloc[-1]


if __name__ == "__main__":
    from data_collector import download_data
    data = download_data()
    features = add_all_indicators(data)
    print(features.tail(3).T)
