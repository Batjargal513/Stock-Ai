# ============================================================
#  app.py  —  Streamlit dashboard  (run: streamlit run app.py)
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import requests
from datetime import datetime, timedelta

# ── Patch yfinance to use browser-like headers ───────────────
# Render and many cloud providers get blocked by Yahoo Finance
# without proper User-Agent headers
_YF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}
yf.utils.user_agent_headers = _YF_HEADERS


def yf_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Reliable yfinance wrapper that works on Render and other cloud servers.
    Falls back to direct Yahoo Finance API call if yfinance fails.
    """
    # Try yfinance first
    try:
        t = yf.Ticker(ticker)
        t._session = requests.Session()
        t._session.headers.update(_YF_HEADERS)
        df = t.history(period=period, interval=interval, auto_adjust=True)
        if not df.empty:
            df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
            return df
    except Exception:
        pass

    # Fallback: direct Yahoo Finance API call
    try:
        period_map = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
                      "6mo": 180, "1y": 365, "2y": 730, "5y": 1825, "10y": 3650}
        days = period_map.get(period, 365)
        end_ts = int(datetime.now().timestamp())
        start_ts = int((datetime.now() - timedelta(days=days)).timestamp())
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?interval={interval}&period1={start_ts}&period2={end_ts}"
        )
        resp = requests.get(url, headers=_YF_HEADERS, timeout=15)
        data = resp.json()["chart"]["result"][0]
        ts = data["timestamp"]
        q  = data["indicators"]["quote"][0]
        adj = data["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
        df = pd.DataFrame({
            "Open":   q["open"],
            "High":   q["high"],
            "Low":    q["low"],
            "Close":  adj,
            "Volume": q["volume"],
        }, index=pd.to_datetime(ts, unit="s"))
        df.index.name = "Date"
        df.dropna(inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame()

from config import TICKERS, MODELS_DIR, ANTHROPIC_API_KEY
from data_collector import download_data
from indicators import add_all_indicators
from claude_analyzer import analyze_stock, quick_summary, explain_indicator

# ── Check if models exist ────────────────────────────────────
MODELS_READY = (
    os.path.exists(os.path.join(MODELS_DIR, "lstm_model.keras")) and
    os.path.exists(os.path.join(MODELS_DIR, "xgb_selector.joblib"))
)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Stock AI — Claude-Powered Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1a1a2e;
}

/* White background everywhere */
.stApp, .main, section[data-testid="stSidebar"] {
    background-color: #f8f9fc !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
div[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #1a1a2e !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* Tabs */
div[data-testid="stTabs"] button {
    color: #64748b !important;
    font-weight: 500;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2563eb !important;
    border-bottom-color: #2563eb !important;
    font-weight: 700;
}

/* Buttons */
div[data-testid="stButton"] button[kind="primary"] {
    background: #2563eb;
    color: white;
    border: none;
    font-weight: 600;
    border-radius: 8px;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: #1d4ed8;
}

/* Divider */
hr { border-color: #e2e8f0 !important; }

/* Title */
h1 { color: #1a1a2e !important; font-weight: 700 !important; }
h2, h3 { color: #1e293b !important; }

/* Dataframe */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Code blocks */
code { font-family: 'IBM Plex Mono', monospace; background: #f1f5f9; color: #1a1a2e; }

/* Info / warning / error boxes */
div[data-testid="stAlert"] { border-radius: 8px; }

/* Selectbox and inputs */
div[data-testid="stSelectbox"] > div,
div[data-testid="stTextInput"] > div > div {
    background: #ffffff !important;
    border-color: #e2e8f0 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def get_live_price_data(ticker: str) -> dict:
    try:
        hist = yf_history(ticker, period="5d", interval="1d")
        t = yf.Ticker(ticker)
        try:
            info = t.fast_info
        except Exception:
            info = None
        if len(hist) >= 2:
            prev_close = float(hist["Close"].iloc[-2])
            curr_close = float(hist["Close"].iloc[-1])
            change_pct = round((curr_close - prev_close) / prev_close * 100, 2)
        elif len(hist) == 1:
            curr_close = float(hist["Close"].iloc[-1])
            change_pct = 0.0
        else:
            curr_close = change_pct = 0.0

        # Safely get each field — fast_info can throw on some environments
        try:
            volume = f"{int(info.three_month_average_volume):,}" if info else "N/A"
        except Exception:
            vol = int(hist["Volume"].iloc[-1]) if len(hist) > 0 else 0
            volume = f"{vol:,}" if vol else "N/A"

        try:
            week_52_high = round(float(info.year_high), 2) if info else 0.0
        except Exception:
            # Use 1y history for 52W high
            hist_1y = yf_history(ticker, period="1y")
            week_52_high = round(float(hist_1y["High"].max()), 2) if not hist_1y.empty else 0.0

        try:
            week_52_low = round(float(info.year_low), 2) if info else 0.0
        except Exception:
            hist_1y = yf_history(ticker, period="1y")
            week_52_low = round(float(hist_1y["Low"].min()), 2) if not hist_1y.empty else 0.0

        try:
            market_cap = f"${info.market_cap/1e9:.1f}B" if (info and info.market_cap) else "N/A"
        except Exception:
            market_cap = "N/A"

        return {
            "price":        round(curr_close, 2),
            "change_pct":   change_pct,
            "volume":       volume,
            "week_52_high": week_52_high,
            "week_52_low":  week_52_low,
            "market_cap":   market_cap,
        }
    except Exception as e:
        return {
            "price": 0.0, "change_pct": 0.0,
            "volume": "N/A", "week_52_high": 0.0,
            "week_52_low": 0.0, "market_cap": "N/A",
        }


@st.cache_data(ttl=600)
def load_chart_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    # Use .history() — more reliable on server environments than yf.download()
    df = yf_history(ticker, period=period, interval="1d")
    return df


@st.cache_data(ttl=3600)
def fetch_ticker_history(ticker: str) -> pd.DataFrame:
    """
    Download 10 years of daily data for one ticker using .history()
    which works reliably on Render / server environments.
    """
    df = yf_history(ticker, period="10y", interval="1d")
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["Ticker"] = ticker
    return df


@st.cache_data(ttl=3600)
def get_prediction_cached(ticker: str):
    """Fetch data, compute features, and run ML prediction."""
    try:
        from ml_pipeline import predict
        raw = fetch_ticker_history(ticker)
        feat = add_all_indicators(raw)
        ticker_df = feat[feat["Ticker"] == ticker]
        return predict(ticker_df)
    except Exception as e:
        return {"error": str(e)}


def signal_color(signal: str) -> str:
    return {"BUY": "#00e5a0", "SELL": "#ff4b6e", "HOLD": "#ffd93d"}.get(signal, "#888")


def build_candlestick_chart(ticker: str, period: str = "6mo") -> go.Figure:
    df = load_chart_data(ticker, period)
    if df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.02,
    )

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="#00e5a0",
        decreasing_line_color="#ff4b6e",
    ), row=1, col=1)

    # EMA lines
    for w, col in [(20, "#4fc3f7"), (50, "#ff8a65"), (200, "#ce93d8")]:
        ema = df["Close"].ewm(span=w).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=ema, name=f"EMA {w}",
            line=dict(width=1, color=col), opacity=0.8,
        ), row=1, col=1)

    # Bollinger Bands
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    fig.add_trace(go.Scatter(
        x=df.index, y=sma20 + 2*std20,
        name="BB Upper", line=dict(width=0.5, color="#546e7a", dash="dot"),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=sma20 - 2*std20,
        name="BB Lower", line=dict(width=0.5, color="#546e7a", dash="dot"),
        fill="tonexty", fillcolor="rgba(84,110,122,0.06)", showlegend=False,
    ), row=1, col=1)

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi, name="RSI",
        line=dict(color="#ffd93d", width=1.5),
    ), row=2, col=1)
    for lvl, col in [(70, "rgba(255,75,110,0.3)"), (30, "rgba(0,229,160,0.3)")]:
        fig.add_hline(y=lvl, line_dash="dash", line_color=col, row=2, col=1)

    # Volume
    colors = ["#00e5a0" if c >= o else "#ff4b6e"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors, opacity=0.7,
    ), row=3, col=1)

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=580,
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    fig.update_yaxes(gridcolor="#f1f5f9", linecolor="#e2e8f0")
    fig.update_xaxes(gridcolor="#f1f5f9", linecolor="#e2e8f0")
    return fig


def build_watchlist_summary(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            d = get_live_price_data(t)
            rows.append({
                "Ticker": t,
                "Price": f"${d['price']}",
                "Change %": d["change_pct"],
                "Mkt Cap": d["market_cap"],
                "52W H": f"${d['week_52_high']}",
                "52W L": f"${d['week_52_low']}",
            })
        except Exception:
            rows.append({"Ticker": t, "Price": "–", "Change %": 0})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════
#  Sidebar
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📈 Stock AI")
    st.markdown("*Claude-Powered Analysis*")
    st.divider()

    selected_ticker = st.selectbox("Select ticker", TICKERS)
    chart_period    = st.selectbox("Chart period", ["1mo","3mo","6mo","1y","2y","5y"], index=3)
    st.divider()

    st.markdown("### ⚙️ Settings")
    api_key_input = st.text_input("Anthropic API Key", type="password",
                                  value=ANTHROPIC_API_KEY if ANTHROPIC_API_KEY != "YOUR_API_KEY_HERE" else "")
    if api_key_input:
        import anthropic as _ant
        _ant.Anthropic.api_key = api_key_input
        os.environ["ANTHROPIC_API_KEY"] = api_key_input

    st.divider()
    if not MODELS_READY:
        st.warning("⚠️ Models not trained yet.")
        st.code("python ml_pipeline.py", language="bash")

    st.markdown("### 📘 Quick Guide")
    st.markdown("""
1. **Train models** once via terminal
2. **Select a ticker** above
3. Hit **Run Full Analysis**
4. Read Claude's interpretation
""")


# ════════════════════════════════════════════════════════════
#  Main layout
# ════════════════════════════════════════════════════════════

st.title(f"📊 {selected_ticker} — AI Stock Analysis")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC")

# ── Live price strip ─────────────────────────────────────────
with st.spinner("Fetching live data …"):
    price_data = get_live_price_data(selected_ticker)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("💵 Price",       f"${price_data['price']}",   f"{price_data['change_pct']}%")
col2.metric("📦 Volume",      price_data["volume"])
col3.metric("🏢 Market Cap",  price_data["market_cap"])
col4.metric("📈 52W High",    f"${price_data['week_52_high']}")
col5.metric("📉 52W Low",     f"${price_data['week_52_low']}")

st.divider()

# ── Chart ────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📉 Chart", "🤖 ML Signal", "🧠 Claude Analysis"])

with tab1:
    fig = build_candlestick_chart(selected_ticker, chart_period)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if not MODELS_READY:
        st.error("Models not trained. Run `python ml_pipeline.py` first.")
    else:
        st.subheader("🤖 Machine Learning Prediction")
        with st.spinner("Running ML models …"):
            pred = get_prediction_cached(selected_ticker)

        if "error" in pred:
            st.error(f"Prediction error: {pred['error']}")
        else:
            signal = pred["signal"]
            color  = signal_color(signal)

            # ── Signal banner ─────────────────────────────────
            banner_bg = {"BUY": "#f0fdf4", "SELL": "#fff1f2", "HOLD": "#fefce8"}.get(signal, "#f8f9fc")
            border_col = {"BUY": "#16a34a", "SELL": "#dc2626", "HOLD": "#ca8a04"}.get(signal, "#94a3b8")
            text_col   = {"BUY": "#15803d", "SELL": "#b91c1c", "HOLD": "#92400e"}.get(signal, "#334155")
            st.markdown(f"""
<div style="background:{banner_bg};border:2px solid {border_col};
border-radius:16px;padding:28px;text-align:center;margin-bottom:24px;">
    <div style="font-size:3rem;">{pred['emoji']}</div>
    <div style="font-size:2.8rem;font-weight:800;color:{text_col};">{signal}</div>
    <div style="color:#475569;font-size:1rem;margin-top:6px;">Ensemble confidence: <strong>{pred['confidence']:.1f}%</strong></div>
</div>
""", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("LSTM Probability",    f"{pred['lstm_prob']*100:.1f}%")
            m2.metric("XGBoost Probability", f"{pred['xgb_prob']*100:.1f}%")
            m3.metric("Ensemble Score",      f"{pred['ensemble']:.4f}")

            st.divider()
            st.subheader("🏆 Top Meta-Indicators")

            vals = pred["latest_values"]
            cols = st.columns(4)
            for i, (feat, val) in enumerate(vals.items()):
                with cols[i % 4]:
                    st.metric(feat, round(val, 4))

            # Gauge chart for ensemble score
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred["ensemble"] * 100,
                title={"text": "Ensemble Score (0=Sell, 100=Buy)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": color},
                    "steps": [
                        {"range": [0,  38], "color": "rgba(220,38,38,0.12)"},
                        {"range": [38, 62], "color": "rgba(202,138,4,0.12)"},
                        {"range": [62,100], "color": "rgba(22,163,74,0.12)"},
                    ],
                    "threshold": {"line": {"color": color, "width": 4}, "value": pred["ensemble"]*100},
                },
                number={"suffix": "%"},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#ffffff", font_color="#1a1a2e", height=280,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

with tab3:
    st.subheader("🧠 Claude AI Deep Analysis")

    if not MODELS_READY:
        st.error("Train models first, then come back here.")
    elif not api_key_input and ANTHROPIC_API_KEY == "YOUR_API_KEY_HERE":
        st.warning("Add your Anthropic API key in the sidebar to enable Claude analysis.")
    else:
        run_btn = st.button("🚀 Run Full Claude Analysis", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Getting ML prediction …"):
                pred = get_prediction_cached(selected_ticker)

            if "error" in pred:
                st.error(f"ML error: {pred['error']}")
            else:
                with st.spinner("Claude is analyzing … (10-20s)"):
                    try:
                        analysis = analyze_stock(selected_ticker, pred, price_data)
                        st.markdown(analysis)

                        st.divider()
                        st.subheader("💬 Explain an Indicator")
                        chosen = st.selectbox("Pick an indicator to explain",
                                              list(pred["latest_values"].keys()))
                        if st.button("Explain it"):
                            val = pred["latest_values"][chosen]
                            with st.spinner("Asking Claude …"):
                                explanation = explain_indicator(chosen, val)
                            st.info(explanation)

                    except Exception as e:
                        st.error(f"Claude API error: {e}")

st.divider()

# ── Watchlist ────────────────────────────────────────────────
st.subheader("📋 Watchlist Overview")
with st.spinner("Loading watchlist …"):
    wl_df = build_watchlist_summary(TICKERS)

def color_change(val):
    if isinstance(val, (int, float)):
        return f"color: {'#00e5a0' if val >= 0 else '#ff4b6e'}"
    return ""

st.dataframe(
    wl_df.style.applymap(color_change, subset=["Change %"]),
    use_container_width=True,
    height=350,
)

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
---
<div style="text-align:center;color:#94a3b8;font-size:0.8rem;">
⚠️ This tool is for <strong>educational purposes only</strong>. 
Not financial advice. Always do your own research and consult a licensed advisor before trading.
</div>
""", unsafe_allow_html=True)
