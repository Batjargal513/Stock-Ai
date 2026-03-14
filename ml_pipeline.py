# ============================================================
#  ml_pipeline.py  —  XGBoost feature selector + LSTM trainer
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import (
    TEST_SPLIT, SEQUENCE_LENGTH, XGB_N_ESTIMATORS,
    LSTM_EPOCHS, LSTM_BATCH_SIZE, TOP_N_FEATURES, MODELS_DIR,
)
from indicators import get_feature_columns


# ── Paths ────────────────────────────────────────────────────
os.makedirs(MODELS_DIR, exist_ok=True)
XGB_PATH    = os.path.join(MODELS_DIR, "xgb_selector.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
LSTM_PATH   = os.path.join(MODELS_DIR, "lstm_model.keras")
FEATURES_TXT= os.path.join(MODELS_DIR, "top_features.txt")


# ════════════════════════════════════════════════════════════
#  PHASE 1 – XGBoost Feature Selection
# ════════════════════════════════════════════════════════════

def train_xgboost(df: pd.DataFrame) -> tuple[XGBClassifier, list[str]]:
    """
    Train XGBoost on ALL indicators to find the TOP_N_FEATURES
    most predictive ones. Saves model to disk.
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["Target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, shuffle=False
    )

    print(f"\n🌲  Training XGBoost on {len(feature_cols)} features …")
    xgb = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = xgb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅  XGBoost accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Hold/Sell", "Buy"]))

    importance = pd.Series(xgb.feature_importances_, index=feature_cols)
    top_features = importance.nlargest(TOP_N_FEATURES).index.tolist()
    print(f"\n🏆  Top {TOP_N_FEATURES} meta-indicators:")
    for i, f in enumerate(top_features, 1):
        print(f"  {i:2}. {f:25s}  importance={importance[f]:.4f}")

    joblib.dump(xgb, XGB_PATH)
    with open(FEATURES_TXT, "w") as fh:
        fh.write("\n".join(top_features))
    print(f"\n💾  XGBoost saved → {XGB_PATH}")
    return xgb, top_features


def load_top_features() -> list[str]:
    if not os.path.exists(FEATURES_TXT):
        raise FileNotFoundError("Run train_xgboost() first.")
    with open(FEATURES_TXT) as fh:
        return [l.strip() for l in fh.readlines() if l.strip()]


# ════════════════════════════════════════════════════════════
#  PHASE 2 – LSTM Model
# ════════════════════════════════════════════════════════════

def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def build_lstm(n_features: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, n_features)),
        BatchNormalization(),
        Dropout(0.25),
        LSTM(64, return_sequences=True),
        Dropout(0.20),
        LSTM(32),
        Dropout(0.15),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_lstm(df: pd.DataFrame, top_features: list[str]) -> tf.keras.Model:
    """Train LSTM on the top meta-indicators from XGBoost."""
    X_raw = df[top_features].values
    y_raw = df["Target"].values

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    joblib.dump(scaler, SCALER_PATH)

    # Build sequences
    X_seq, y_seq = _make_sequences(X_scaled, y_raw, SEQUENCE_LENGTH)
    split = int(len(X_seq) * (1 - TEST_SPLIT))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    print(f"\n🧠  Training LSTM  (train={len(X_train):,}  test={len(X_test):,}) …")
    model = build_lstm(len(top_features))
    model.summary()

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    print(f"\n✅  LSTM accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Hold/Sell", "Buy"]))

    model.save(LSTM_PATH)
    print(f"💾  LSTM saved → {LSTM_PATH}")
    return model


# ════════════════════════════════════════════════════════════
#  INFERENCE – Predict on new data
# ════════════════════════════════════════════════════════════

def predict(df_ticker: pd.DataFrame) -> dict:
    """
    Given a DataFrame of one ticker (already feature-enriched),
    return the latest prediction probability and a BUY/HOLD/SELL label.
    """
    from indicators import get_feature_columns

    top_features = load_top_features()
    scaler: MinMaxScaler = joblib.load(SCALER_PATH)
    model: tf.keras.Model = load_model(LSTM_PATH)

    # LSTM uses only the top features (what it was trained on)
    X_top = df_ticker[top_features].values
    X_scaled = scaler.transform(X_top)

    if len(X_scaled) < SEQUENCE_LENGTH:
        return {"error": f"Need at least {SEQUENCE_LENGTH} rows of data."}

    seq = X_scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, len(top_features))
    prob = float(model.predict(seq, verbose=0)[0][0])

    # XGBoost uses ALL features (same 42 it was trained on)
    xgb: XGBClassifier = joblib.load(XGB_PATH)
    all_features = get_feature_columns(df_ticker)
    X_all = df_ticker[all_features].values
    xgb_prob = float(xgb.predict_proba(X_all[-1].reshape(1, -1))[0][1])

    # Ensemble average
    ensemble = (prob * 0.6) + (xgb_prob * 0.4)

    if ensemble >= 0.62:
        label, emoji = "BUY",  "🟢"
    elif ensemble <= 0.38:
        label, emoji = "SELL", "🔴"
    else:
        label, emoji = "HOLD", "🟡"

    return {
        "lstm_prob":  round(prob, 4),
        "xgb_prob":   round(xgb_prob, 4),
        "ensemble":   round(ensemble, 4),
        "signal":     label,
        "emoji":      emoji,
        "confidence": round(abs(ensemble - 0.5) * 2 * 100, 1),  # 0-100%
        "top_features": top_features,
        "latest_values": {
            f: round(float(df_ticker[f].iloc[-1]), 4) for f in top_features
        },
    }


# ════════════════════════════════════════════════════════════
#  MAIN – Full training pipeline
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data_collector import download_data
    from indicators import add_all_indicators

    print("=" * 60)
    print("  STOCK AI  –  Full Training Pipeline")
    print("=" * 60)

    raw = download_data()
    features_df = add_all_indicators(raw)

    xgb_model, top_features = train_xgboost(features_df)
    lstm_model = train_lstm(features_df, top_features)

    print("\n🎉  Training complete! Run `streamlit run app.py` to start the dashboard.")
