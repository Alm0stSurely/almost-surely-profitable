# Almost Surely Profitable

An LLM-powered paper trading agent that makes daily investment decisions on ETFs and equities.

## How it works

```
Every day (after US market close):

1. Python fetches market data (yfinance)
   → Prices, volumes, returns, volatility

2. Python computes technical indicators
   → Moving averages, RSI, Bollinger Bands, drawdowns, correlations

3. Python sends context to an LLM via API:
   - Market state (indicators for all assets)
   - Portfolio state (positions, cash, P&L)
   - Recent decision history
   - System prompt with risk management principles

4. LLM analyzes and responds with structured JSON:
   { "actions": [{"ticker": "SPY", "action": "hold"},
                  {"ticker": "GLD", "action": "buy", "pct": 10}],
     "reasoning": "..." }

5. Python executes paper orders, updates portfolio, logs performance
```

The LLM is the decision-making brain. Python handles infrastructure (data, portfolio, logging, reporting).

Risk management concepts from [Behavioral_RL](https://github.com/Alm0stSurely/Behavioral_RL) (prospect theory, loss aversion, CVaR) are injected into the LLM system prompt to make it risk-aware.

## Target assets

**ETFs & Indices:**

| Asset | Ticker | Type |
|-------|--------|------|
| S&P 500 | SPY | ETF |
| Nasdaq 100 | QQQ | ETF |
| CAC 40 | CAC.PA | Index |
| Gold | GLD | Commodity ETF |
| US Bonds | TLT | Bond ETF |
| Euro Stoxx 50 | FEZ | ETF |

**French Equities (Euronext Paris):**

| Company | Ticker |
|---------|--------|
| LVMH | MC.PA |
| TotalEnergies | TTE.PA |
| Sanofi | SAN.PA |
| L'Oreal | OR.PA |
| Airbus | AIR.PA |
| Schneider Electric | SU.PA |
| Air Liquide | AI.PA |
| BNP Paribas | BNP.PA |
| AXA | CS.PA |
| Hermes | RMS.PA |
| Safran | SAF.PA |
| Dassault Systemes | DSY.PA |
| Vinci | DG.PA |
| Saint-Gobain | SGO.PA |
| Kering | KER.PA |

## Architecture

```
src/
  data/          # Market data fetching (yfinance) + technical indicators
  portfolio/     # Paper portfolio management (positions, cash, P&L, order execution)
  llm/           # LLM integration (prompt construction, API calls, response parsing)
  risk/          # Risk concepts for system prompt (prospect theory, CVaR)
  backtest/      # Backtesting engine
notebooks/       # Research notebooks
results/         # Daily logs, weekly reports, plots
docs/            # Research notes
```

## Parameters

- **Starting capital:** EUR 10,000 (paper money)
- **Decision frequency:** Daily (after US market close, 21:00 UTC)
- **Weekly report:** Every Friday — P&L, best/worst positions, comparison vs buy-and-hold SPY & CAC 40
- **Benchmark:** Buy-and-hold SPY, buy-and-hold CAC 40

## Technology choices

**Current phase: Python (research)**
- Python for infrastructure (data, portfolio, indicators, reporting)
- LLM API for trading decisions
- yfinance for free market data
- No GPU required

**Future phase: Rust (live trading)**
- If results prove viable → rewrite to Rust for live execution
- Microsecond latency, zero-cost abstractions, memory safety
- Architecture reference: [dprc-autotrader-v2](https://github.com/affaan-m/dprc-autotrader-v2)
- The Python codebase serves as a **functional specification** for the Rust rewrite

## Dependencies

- Python 3.11+
- yfinance (market data)
- pandas, numpy (data manipulation)
- matplotlib, plotly (visualization)
- scipy, scikit-learn (stats)
- python-dotenv (config)

## References

- Tversky & Kahneman (1992) — Prospect Theory
- Rockafellar & Uryasev (2000) — CVaR optimization
- Bechara et al. (1994) — Iowa Gambling Task
- [Behavioral_RL](https://github.com/Alm0stSurely/Behavioral_RL) — RL framework with behavioral biases

## Status

**Phase 1 — Research.** Paper trading only, no real money.

## License

MIT
