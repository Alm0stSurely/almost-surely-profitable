# Risk-Sensitive Trading

Applying risk-sensitive reinforcement learning to financial markets.

## Research goals

- Extend the behavioral RL framework from [Behavioral_RL](https://github.com/Alm0stSurely/Behavioral_RL) (Iowa Gambling Task) to real financial environments
- Model human cognitive biases in trading: loss aversion, recency bias, disposition effect
- Backtest prospect theory + CVaR strategies on ETFs, equities, and commodities (gold)

## Architecture

```
src/
  environments/    # Gym-compatible market environments
  agents/          # RL agents (DQN, PPO) with risk-sensitive reward shaping
  risk/            # Prospect theory, CVaR, risk metrics
  data/            # Market data fetching (yfinance)
  backtest/        # Backtesting engine
notebooks/         # Research notebooks
models/            # Trained model checkpoints
results/           # Backtest results, plots
docs/              # Research notes, papers
```

## Target assets

| Asset | Ticker | Type |
|-------|--------|------|
| S&P 500 | SPY | ETF |
| Nasdaq 100 | QQQ | ETF |
| Gold | GLD | Commodity ETF |
| Bonds | TLT | Bond ETF |
| Euro Stoxx | FEZ | ETF |

## Dependencies

- Python 3.12+
- PyTorch
- Gymnasium
- yfinance (market data)
- pandas, numpy, matplotlib
- vectorbt (backtesting)

## References

- Bechara et al. (1994) — Iowa Gambling Task
- Tversky & Kahneman (1992) — Prospect Theory
- Rockafellar & Uryasev (2000) — CVaR optimization
- Moody & Saffell (2001) — RL for trading

## License

MIT
