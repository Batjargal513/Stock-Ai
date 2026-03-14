# ============================================================
#  claude_analyzer.py  —  Claude-powered stock analysis
# ============================================================

import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL


client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ── Prompt templates ─────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert quantitative financial analyst with 20+ years of experience 
in algorithmic trading, technical analysis, and machine learning-based market prediction.

Your role is to synthesize ML model signals with technical indicators to produce 
clear, actionable trading analysis. Always:
- Be direct and specific about signals
- Quantify risks where possible
- Reference specific indicator values in your analysis
- Flag any conflicting signals clearly
- Never give generic advice; ground everything in the provided data

Format your responses with clear sections using markdown."""


def analyze_stock(
    ticker: str,
    prediction: dict,
    price_data: dict,
    news_context: str = "",
) -> str:
    """
    Send indicator data + ML predictions to Claude for deep analysis.

    Args:
        ticker:       Stock symbol
        prediction:   Output from ml_pipeline.predict()
        price_data:   Dict with keys: price, change_pct, volume, market_cap
        news_context: Optional recent news summary string

    Returns:
        Claude's full analysis as a markdown string
    """
    indicators = prediction.get("latest_values", {})
    signal     = prediction.get("signal", "N/A")
    ensemble   = prediction.get("ensemble", 0.5)
    confidence = prediction.get("confidence", 0)
    lstm_prob  = prediction.get("lstm_prob", 0.5)
    xgb_prob   = prediction.get("xgb_prob", 0.5)

    # Format indicator values nicely
    indicator_lines = "\n".join(
        f"    - {k}: {v}" for k, v in indicators.items()
    )

    news_section = f"\n\n**Recent News Context:**\n{news_context}" if news_context else ""

    prompt = f"""
Analyze the following stock data and provide a comprehensive trading analysis.

## Stock: {ticker}

### Current Price Data
- Current Price:  ${price_data.get('price', 'N/A')}
- Day Change:     {price_data.get('change_pct', 'N/A')}%
- Volume:         {price_data.get('volume', 'N/A')}
- 52-Week High:   ${price_data.get('week_52_high', 'N/A')}
- 52-Week Low:    ${price_data.get('week_52_low', 'N/A')}

### ML Model Signals
- LSTM Prediction:     {lstm_prob*100:.1f}% probability of price increase
- XGBoost Prediction:  {xgb_prob*100:.1f}% probability of price increase
- Ensemble Signal:     **{signal}** (confidence: {confidence:.1f}%)
- Raw Ensemble Score:  {ensemble:.4f}  (>0.62 = Buy, <0.38 = Sell)

### Top Technical Indicators (ranked by ML importance)
{indicator_lines}
{news_section}

---

Please provide your analysis in the following sections:

## 1. Signal Summary
One clear sentence: what is the primary signal and why?

## 2. Technical Analysis Breakdown
Interpret each indicator value. Flag any divergences or conflicts between indicators.

## 3. Strength of Signal
- Are the LSTM and XGBoost models in agreement?
- How do the technical indicators support or contradict the ML signal?
- Rate overall signal strength: Weak / Moderate / Strong / Very Strong

## 4. Key Risks
List 2-4 specific risks that could invalidate this signal (with indicator-based reasoning).

## 5. Suggested Action
- Signal: BUY / HOLD / SELL
- Time horizon: short-term (1-5 days) / swing (1-4 weeks) / positional (1-3 months)
- Entry consideration: what price level or condition to watch
- Stop-loss suggestion: based on ATR or support levels
- Target: realistic upside/downside based on Bollinger Bands / historical volatility

## 6. Confidence Level
Rate your confidence: Low / Medium / High / Very High
Explain your rating in 1-2 sentences.
"""

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def quick_summary(ticker: str, signal: str, confidence: float, price: float) -> str:
    """
    Generate a short 2-sentence summary card for dashboard tiles.
    Much cheaper API call than the full analysis.
    """
    prompt = (
        f"In exactly 2 sentences, give a sharp analyst take on {ticker} "
        f"given a {signal} signal at ${price:.2f} with {confidence:.1f}% confidence. "
        f"Be specific and reference likely price movement. No fluff."
    )
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def explain_indicator(indicator_name: str, value: float) -> str:
    """Plain-English explanation of a single indicator value for beginners."""
    prompt = (
        f"Explain what a {indicator_name} value of {value:.4f} means for a stock. "
        f"Use plain English, 2-3 sentences max. Say what it implies for near-term price action."
    )
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


if __name__ == "__main__":
    # Quick test
    dummy_prediction = {
        "lstm_prob":  0.73,
        "xgb_prob":   0.68,
        "ensemble":   0.71,
        "signal":     "BUY",
        "confidence": 42.0,
        "latest_values": {
            "RSI": 58.3,
            "MACD": 1.24,
            "ADX": 32.1,
            "ATR": 3.45,
            "BB_pct": 0.72,
            "OBV": 152000000,
            "EMA_9": 182.50,
            "Volume_change": 0.18,
        },
    }
    dummy_price = {
        "price": 183.50,
        "change_pct": 1.2,
        "volume": "85M",
        "week_52_high": 199.62,
        "week_52_low": 124.17,
    }
    analysis = analyze_stock("AAPL", dummy_prediction, dummy_price)
    print(analysis)
