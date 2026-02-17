# Almost Surely Profitable

> *"In probability theory, an event is said to happen almost surely if it happens with probability 1. Whether this portfolio converges to profit with probability 1 remains an open conjecture."*

An experiment in letting a large language model manage a paper trading portfolio. The hypothesis: an LLM injected with prospect theory and CVaR principles can make non-trivial daily allocation decisions across ETFs and equities. The null hypothesis: it cannot.

We are testing the alternative.

## The idea

Most quantitative trading systems optimize for expected returns. Humans don't -- we feel losses roughly 2.25x more than equivalent gains (Kahneman & Tversky, 1992), we anchor to recent prices, and we panic-sell at drawdowns that a rational agent would ignore.

So instead of building a perfectly rational agent, I'm building one that is *deliberately risk-averse* in a behaviorally-informed way. The LLM receives market data, computes nothing itself (Python handles that), and makes allocation decisions with a system prompt that encodes:

- **Loss aversion** from prospect theory
- **CVaR** (Conditional Value at Risk) as the primary risk measure
- **Position sizing** constraints inspired by Kelly criterion intuitions
- **Drawdown sensitivity** that increases non-linearly with losses

The theoretical foundation comes from [Behavioral_RL](https://github.com/Alm0stSurely/Behavioral_RL), a framework for risk-sensitive reinforcement learning on the Iowa Gambling Task.

## How it works

```
Every day, after US market close:

1. Python fetches 30 days of data for 21 assets via yfinance
2. Python computes indicators: SMA(20,50,200), RSI, Bollinger, volatility, drawdowns, correlations
3. Everything is sent to the LLM along with current portfolio state
4. The LLM returns a JSON decision: buy X% of cash, sell Y, hold Z
5. Python executes paper orders and logs everything
```

The LLM is the decision-maker. Python is the infrastructure. The separation is deliberate -- if the LLM makes a bad call, the logs will show exactly what it saw and what it decided. Reproducibility matters, even in research that involves stochastic processes.

## Universe

**ETFs:** SPY, QQQ, GLD (gold), TLT (bonds), FEZ (Euro Stoxx 50), CAC 40

**Euronext Paris:** LVMH, TotalEnergies, Sanofi, L'Oreal, Airbus, Schneider Electric, Air Liquide, BNP Paribas, AXA, Hermes, Safran, Dassault Systemes, Vinci, Saint-Gobain, Kering

21 assets. Enough to test diversification, not enough to drown in noise.

## Parameters

- **Starting capital:** EUR 10,000 (paper)
- **Decisions:** Daily, after US close (21:00 UTC)
- **Monitoring:** Every 2 hours during market hours -- alerts on moves > 2%
- **Weekly report:** Fridays -- P&L, best/worst, comparison vs buy-and-hold SPY & CAC 40
- **Benchmark:** The simplest strategy that does nothing. If I can't beat "buy SPY and go to sleep", this project has negative alpha.

## Technology

**Now: Python.** Fast to iterate, rich ecosystem, good enough for research. The goal is to validate the approach, not to optimize latency.

**Later (maybe): Rust.** If the results suggest this is worth running with real capital, a rewrite becomes necessary. Microsecond latency, memory safety, no GC pauses. The Python code would serve as the functional spec. Reference: [dprc-autotrader-v2](https://github.com/affaan-m/dprc-autotrader-v2).

## Project structure

```
src/
  data/          Market data + technical indicators
  portfolio/     Paper portfolio (positions, cash, P&L, order execution)
  llm/           LLM integration (prompts, API, response parsing)
  risk/          Risk concepts for the system prompt
  backtest/      Backtesting engine
results/         Daily logs, weekly reports
```

## Results

Updated weekly at [alm0stsurely.github.io/trading](https://alm0stsurely.github.io/trading/).

## References

- Tversky & Kahneman (1992) -- *Advances in Prospect Theory*
- Rockafellar & Uryasev (2000) -- *Optimization of Conditional Value-at-Risk*
- Bechara et al. (1994) -- *Insensitivity to future consequences following damage to human prefrontal cortex* (Iowa Gambling Task)

## Disclaimer

This is a research project. No real money, no broker, no real orders. Market data is read-only via public APIs. Nothing here is financial advice. The name is aspirational, not predictive.

## License

MIT

---

*Almost surely, this portfolio will converge. The question is to what.*
