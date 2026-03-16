# ============================================================
#  app.py  —  Stock AI Dashboard  (streamlit run app.py)
# ============================================================

import os, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
from datetime import datetime, timedelta

from config import TICKERS, MODELS_DIR, ANTHROPIC_API_KEY
from data_collector import download_data
from indicators import add_all_indicators
from claude_analyzer import analyze_stock, explain_indicator

MODELS_READY = (
    os.path.exists(os.path.join(MODELS_DIR, "lstm_model.keras")) and
    os.path.exists(os.path.join(MODELS_DIR, "xgb_selector.joblib"))
)

st.set_page_config(
    page_title="Stock AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Instrument Sans', sans-serif !important;
}
code, pre { font-family: 'JetBrains Mono', monospace !important; }

/* Background */
.stApp, .main { background: #f5f6fa !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8eaf0 !important;
}
section[data-testid="stSidebar"] .stTextInput input {
    background: #f5f6fa !important;
    border: 1.5px solid #e8eaf0 !important;
    border-radius: 10px !important;
    color: #1e2433 !important;
    font-size: 0.9rem !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #f5f6fa !important;
    border: 1.5px solid #e8eaf0 !important;
    border-radius: 10px !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1.5px solid #e8eaf0 !important;
    border-radius: 14px !important;
    padding: 18px 22px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
div[data-testid="metric-container"] label {
    color: #8892a4 !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #1e2433 !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

/* Tabs */
div[data-testid="stTabs"] {
    border-bottom: 2px solid #e8eaf0 !important;
}
div[data-testid="stTabs"] button {
    color: #8892a4 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 10px 20px !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2563eb !important;
    border-bottom: 2px solid #2563eb !important;
    font-weight: 700 !important;
}

/* Primary button */
div[data-testid="stButton"] button {
    background: #2563eb !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] button:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.25) !important;
    transform: translateY(-1px) !important;
}

/* Popular ticker buttons */
section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    background: #f0f4ff !important;
    color: #2563eb !important;
    font-size: 0.72rem !important;
    padding: 4px 6px !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
    background: #dbeafe !important;
    box-shadow: none !important;
    transform: none !important;
}

/* Divider */
hr { border-color: #e8eaf0 !important; margin: 16px 0 !important; }

/* Headings */
h1, h2, h3, h4 { color: #1e2433 !important; font-weight: 700 !important; }

/* Dataframe */
div[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1.5px solid #e8eaf0 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}

/* Expander */
div[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1.5px solid #e8eaf0 !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}

/* Alerts */
div[data-testid="stAlert"] { border-radius: 10px !important; }

/* Text area */
textarea {
    background: #f5f6fa !important;
    border: 1.5px solid #e8eaf0 !important;
    border-radius: 10px !important;
    color: #1e2433 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f5f6fa; }
::-webkit-scrollbar-thumb { background: #d1d5e0; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  yfinance helpers
# ══════════════════════════════════════════════════════════════

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def yf_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        t._session = requests.Session()
        t._session.headers.update(_HEADERS)
        df = t.history(period=period, interval=interval, auto_adjust=True)
        if not df.empty:
            df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
            return df
    except Exception:
        pass
    try:
        period_map = {"5d":5,"1mo":30,"3mo":90,"6mo":180,"1y":365,"2y":730,"5y":1825,"10y":3650}
        days = period_map.get(period, 365)
        end_ts   = int(datetime.now().timestamp())
        start_ts = int((datetime.now() - timedelta(days=days)).timestamp())
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
               f"?interval={interval}&period1={start_ts}&period2={end_ts}")
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        data = resp.json()["chart"]["result"][0]
        ts   = data["timestamp"]
        q    = data["indicators"]["quote"][0]
        adj  = data["indicators"].get("adjclose",[{}])[0].get("adjclose", q["close"])
        df   = pd.DataFrame({"Open":q["open"],"High":q["high"],"Low":q["low"],
                              "Close":adj,"Volume":q["volume"]},
                             index=pd.to_datetime(ts, unit="s"))
        df.index.name = "Date"
        return df.dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_live_price(ticker: str) -> dict:
    try:
        hist = yf_history(ticker, period="5d")
        t    = yf.Ticker(ticker)
        try:    info = t.fast_info
        except: info = None

        if len(hist) >= 2:
            prev = float(hist["Close"].iloc[-2])
            curr = float(hist["Close"].iloc[-1])
            chg  = round((curr - prev) / prev * 100, 2)
        elif len(hist) == 1:
            curr = float(hist["Close"].iloc[-1]); chg = 0.0
        else:
            curr = chg = 0.0

        try:    vol  = f"{int(info.three_month_average_volume):,}" if info else "N/A"
        except:
            vol_raw = int(hist["Volume"].iloc[-1]) if len(hist) > 0 else 0
            vol = f"{vol_raw:,}" if vol_raw else "N/A"
        try:    h52 = round(float(info.year_high), 2) if info else 0.0
        except: h52 = round(float(yf_history(ticker,"1y")["High"].max()), 2)
        try:    l52 = round(float(info.year_low), 2) if info else 0.0
        except: l52 = round(float(yf_history(ticker,"1y")["Low"].min()), 2)
        try:    mcap = f"${info.market_cap/1e9:.1f}B" if (info and info.market_cap) else "N/A"
        except: mcap = "N/A"

        return {"price":round(curr,2),"change_pct":chg,"volume":vol,
                "week_52_high":h52,"week_52_low":l52,"market_cap":mcap}
    except:
        return {"price":0.0,"change_pct":0.0,"volume":"N/A",
                "week_52_high":0.0,"week_52_low":0.0,"market_cap":"N/A"}


@st.cache_data(ttl=600)
def get_chart_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    return yf_history(ticker, period=period)


@st.cache_data(ttl=3600)
def get_prediction(ticker: str):
    try:
        from ml_pipeline import predict
        df = yf_history(ticker, period="10y")
        if df.empty: raise RuntimeError(f"No data for {ticker}")
        df = df[["Open","High","Low","Close","Volume"]].copy()
        df["Ticker"] = ticker
        feat = add_all_indicators(df)
        return predict(feat[feat["Ticker"] == ticker])
    except Exception as e:
        return {"error": str(e)}


def build_chart(ticker: str, period: str = "1y") -> go.Figure:
    df = get_chart_data(ticker, period)
    if df.empty: return go.Figure()

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing=dict(line=dict(color="#16a34a"), fillcolor="#16a34a"),
        decreasing=dict(line=dict(color="#dc2626"), fillcolor="#dc2626"),
    ), row=1, col=1)

    for w, c in [(20,"#6366f1"),(50,"#f59e0b"),(200,"#ec4899")]:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].ewm(span=w).mean(),
            name=f"EMA {w}", line=dict(width=1.5, color=c), opacity=0.85), row=1, col=1)

    sma = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    fig.add_trace(go.Scatter(x=df.index, y=sma+2*std,
        line=dict(width=0.5, color="#94a3b8", dash="dot"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sma-2*std,
        line=dict(width=0.5, color="#94a3b8", dash="dot"),
        fill="tonexty", fillcolor="rgba(99,102,241,0.05)", showlegend=False), row=1, col=1)

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - 100/(1 + gain/loss)
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI",
        line=dict(color="#f59e0b", width=1.5)), row=2, col=1)
    for lvl, c in [(70,"rgba(220,38,38,0.15)"),(30,"rgba(22,163,74,0.15)")]:
        fig.add_hline(y=lvl, line_dash="dash", line_color=c, row=2, col=1)

    colors = ["#16a34a" if c >= o else "#dc2626"
              for c,o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors, opacity=0.65), row=3, col=1)

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=600,
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0,
                    font=dict(color="#64748b", size=11)),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    fig.update_yaxes(gridcolor="#f1f5f9", tickfont=dict(color="#94a3b8", size=11),
                     linecolor="#e8eaf0")
    fig.update_xaxes(gridcolor="#f1f5f9", tickfont=dict(color="#94a3b8", size=11),
                     linecolor="#e8eaf0")
    return fig


def build_watchlist(tickers):
    rows = []
    for t in tickers:
        try:
            d = get_live_price(t)
            rows.append({"Ticker":t,"Price":f"${d['price']}","Change %":d["change_pct"],
                         "Mkt Cap":d["market_cap"],"52W H":f"${d['week_52_high']}",
                         "52W L":f"${d['week_52_low']}"})
        except:
            rows.append({"Ticker":t,"Price":"–","Change %":0})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding:4px 0 20px'>
      <div style='font-size:1.35rem;font-weight:800;color:#1e2433;letter-spacing:-0.02em'>
        📈 Stock AI
      </div>
      <div style='font-size:0.7rem;color:#2563eb;font-weight:700;
           text-transform:uppercase;letter-spacing:0.1em;margin-top:2px'>
        Claude-Powered Analysis
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Search any ticker**")
    ticker_input = st.text_input("Ticker", value=st.session_state.get("ticker","AAPL"),
                                  placeholder="AAPL, TSLA, BTC-USD…",
                                  label_visibility="collapsed").upper().strip()

    st.markdown('<div style="font-size:0.72rem;color:#8892a4;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin:12px 0 6px">Popular</div>', unsafe_allow_html=True)
    popular = ["AAPL","TSLA","NVDA","MSFT","GOOGL","AMZN","META","SPY",
               "AMD","NFLX","COIN","PLTR","BTC-USD","ETH-USD","GLD"]
    cols = st.columns(3)
    for i, t in enumerate(popular):
        if cols[i%3].button(t, key=f"p_{t}", use_container_width=True):
            st.session_state["ticker"] = t
            st.rerun()

    selected = ticker_input if ticker_input else st.session_state.get("ticker","AAPL")
    st.session_state["ticker"] = selected

    st.divider()
    chart_period = st.selectbox("Chart period",
                                 ["1mo","3mo","6mo","1y","2y","5y"], index=3)
    st.divider()
    st.markdown("**Anthropic API Key**")
    api_key = st.text_input("Key", type="password",
                             value=ANTHROPIC_API_KEY if ANTHROPIC_API_KEY != "YOUR_API_KEY_HERE" else "",
                             label_visibility="collapsed")
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    if not MODELS_READY:
        st.divider()
        st.warning("⚠️ Models not trained yet")
        st.code("python ml_pipeline.py")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

# Header
chg  = get_live_price(selected).get("change_pct", 0)
chg_color = "#16a34a" if chg >= 0 else "#dc2626"
chg_arrow = "▲" if chg >= 0 else "▼"

st.markdown(f"""
<div style='background:#ffffff;border:1.5px solid #e8eaf0;border-radius:16px;
     padding:20px 28px;margin-bottom:20px;
     box-shadow:0 1px 4px rgba(0,0,0,0.04)'>
  <div style='display:flex;align-items:center;gap:14px'>
    <div style='background:linear-gradient(135deg,#2563eb,#6366f1);width:52px;height:52px;
         border-radius:14px;display:flex;align-items:center;justify-content:center;
         font-size:1.5rem;flex-shrink:0'>📊</div>
    <div>
      <div style='font-size:2rem;font-weight:800;color:#1e2433;letter-spacing:-0.04em;
           line-height:1'>{selected}</div>
      <div style='font-size:0.78rem;color:#8892a4;margin-top:3px'>
        {datetime.now().strftime("%B %d, %Y")} &nbsp;·&nbsp;
        <span style='color:{chg_color};font-weight:600'>{chg_arrow} {abs(chg):.2f}% today</span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Metrics
with st.spinner(""):
    price_data = get_live_price(selected)

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("💵 Price",     f"${price_data['price']}",  f"{price_data['change_pct']}%")
c2.metric("📦 Volume",    price_data["volume"])
c3.metric("🏢 Mkt Cap",   price_data["market_cap"])
c4.metric("📈 52W High",  f"${price_data['week_52_high']}")
c5.metric("📉 52W Low",   f"${price_data['week_52_low']}")

st.divider()

tab1, tab2, tab3 = st.tabs(["📉  Chart", "🤖  ML Signal", "🧠  Claude Analysis"])

with tab1:
    fig = build_chart(selected, chart_period)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if not MODELS_READY:
        st.info("Train models first — run `python ml_pipeline.py`")
    else:
        with st.spinner("Running ML models…"):
            pred = get_prediction(selected)

        if "error" in pred:
            st.error(f"Error: {pred['error']}")
        else:
            sig   = pred["signal"]
            conf  = pred["confidence"]
            emoji = pred.get("emoji","")

            bg    = {"BUY":"#f0fdf4","SELL":"#fff1f2","HOLD":"#fefce8"}.get(sig,"#f8fafc")
            border= {"BUY":"#bbf7d0","SELL":"#fecdd3","HOLD":"#fef08a"}.get(sig,"#e8eaf0")
            color = {"BUY":"#15803d","SELL":"#b91c1c","HOLD":"#854d0e"}.get(sig,"#475569")

            st.markdown(f"""
            <div style='background:{bg};border:2px solid {border};border-radius:16px;
                 padding:32px;text-align:center;margin-bottom:24px'>
              <div style='font-size:3rem'>{emoji}</div>
              <div style='font-size:3rem;font-weight:900;color:{color};
                   letter-spacing:-0.04em;line-height:1.1'>{sig}</div>
              <div style='color:#64748b;font-size:0.9rem;margin-top:8px'>
                Confidence: <strong style='color:{color}'>{conf:.1f}%</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)

            m1,m2,m3 = st.columns(3)
            m1.metric("LSTM",     f"{pred['lstm_prob']*100:.1f}%")
            m2.metric("XGBoost",  f"{pred['xgb_prob']*100:.1f}%")
            m3.metric("Ensemble", f"{pred['ensemble']:.4f}")

            st.divider()
            st.markdown("#### 🏆 Top Meta-Indicators")
            vals = pred["latest_values"]
            cols = st.columns(4)
            for i,(feat,val) in enumerate(vals.items()):
                with cols[i%4]:
                    st.metric(feat, round(val,4))

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred["ensemble"]*100,
                title={"text":"Signal Score","font":{"color":"#64748b","size":13}},
                number={"suffix":"%","font":{"color":color,"size":36}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#94a3b8",
                            "tickfont":{"color":"#94a3b8"}},
                    "bar":{"color":color,"thickness":0.25},
                    "bgcolor":"#ffffff",
                    "bordercolor":"#e8eaf0",
                    "steps":[
                        {"range":[0,38],  "color":"rgba(220,38,38,0.08)"},
                        {"range":[38,62], "color":"rgba(245,158,11,0.08)"},
                        {"range":[62,100],"color":"rgba(22,163,74,0.08)"},
                    ],
                },
            ))
            fig_g.update_layout(paper_bgcolor="#ffffff", font_color="#64748b",
                                height=260, margin=dict(l=20,r=20,t=40,b=10))
            st.plotly_chart(fig_g, use_container_width=True)

with tab3:
    if not MODELS_READY:
        st.info("Train models first")
    elif not api_key and ANTHROPIC_API_KEY == "YOUR_API_KEY_HERE":
        st.markdown("""
        <div style='background:#eff6ff;border:1.5px solid #bfdbfe;border-radius:12px;
             padding:24px;text-align:center'>
          <div style='font-size:1.5rem'>🔑</div>
          <div style='color:#1e40af;font-weight:600;margin:8px 0'>API Key Required</div>
          <div style='color:#64748b;font-size:0.85rem'>Add your Anthropic key in the sidebar</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button("🚀 Run Full Claude Analysis", use_container_width=True):
            with st.spinner("Fetching ML predictions…"):
                pred = get_prediction(selected)
            if "error" in pred:
                st.error(f"Error: {pred['error']}")
            else:
                with st.spinner("Claude is analyzing…"):
                    try:
                        analysis = analyze_stock(selected, pred, price_data)
                        st.markdown(analysis)
                        st.divider()
                        st.markdown("#### 💬 Explain an Indicator")
                        chosen = st.selectbox("Pick indicator",
                                              list(pred["latest_values"].keys()))
                        if st.button("Explain"):
                            with st.spinner("Asking Claude…"):
                                st.info(explain_indicator(chosen,
                                        pred["latest_values"][chosen]))
                    except Exception as e:
                        st.error(f"Claude error: {e}")

st.divider()

# Watchlist
st.markdown("### 📋 Watchlist")
with st.expander("✏️ Edit watchlist"):
    wl_input = st.text_area("Tickers (comma separated)",
                             value=", ".join(st.session_state.get("watchlist", TICKERS)),
                             height=70)
    if st.button("Update Watchlist"):
        new_wl = [t.strip().upper() for t in wl_input.split(",") if t.strip()]
        st.session_state["watchlist"] = new_wl
        st.success(f"Updated — {len(new_wl)} tickers")

watchlist_tickers = st.session_state.get("watchlist", TICKERS)
with st.spinner("Loading…"):
    wl_df = build_watchlist(watchlist_tickers)

def color_change(v):
    if isinstance(v,(int,float)):
        return f"color:{'#16a34a' if v>=0 else '#dc2626'}"
    return ""

st.dataframe(wl_df.style.applymap(color_change, subset=["Change %"]),
             use_container_width=True, height=380)

st.markdown("""
<div style='text-align:center;color:#c4c9d4;font-size:0.75rem;padding:20px 0 8px'>
  ⚠️ Educational purposes only · Not financial advice
</div>
""", unsafe_allow_html=True)
