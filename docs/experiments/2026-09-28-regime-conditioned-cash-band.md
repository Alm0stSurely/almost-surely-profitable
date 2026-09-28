# Experiment Design: Regime-Conditioned Cash Band (H3)

**Status:** designed, awaiting implementation
**Date:** 2026-09-28
**Author:** P. Clawmogorov
**Branch (future):** `feat/backtest-cash-band-regime`

## 1. Problem Statement

The strategy trails SPY buy-and-hold by **-14.65 pp** (evaluation 2026-09-25).
Root-cause attribution (research sessions 09-23 → 09-25):

- The gap is dominated by the **Feb–Jun high-cash era**, not by poor trade selection.
- `cash_drag_report.py`: **54 cumulative drag days** vs **11 cap-binding days**.
- Current cash band is a function of a **single 1-D regime signal** (volatility
  percentile) with fixed bounds:

| Regime   | Cash band  | Basis                  |
|----------|------------|------------------------|
| HIGH vol | 30–50 %    | capital preservation   |
| NORMAL   | 15–30 %    | balanced               |
| LOW vol  | 10–20 %    | fully deployed         |

- Observation: during long NORMAL-regime stretches (like the current one), the
  band stays 15–30 % regardless of trend or correlation context. Cash sits at
  ~27.7 % for weeks while the equal-weight benchmark compounds.

## 2. Hypothesis

> Conditioning the cash band on a **composite regime score** (vol × trend ×
> correlation) instead of volatility alone reduces cash drag in favorable
> regimes without increasing max drawdown beyond its current -1.6 % level.

Favorable = low/median vol AND non-trending-down AND low/median correlation.
Unfavorable = any leg strongly negative → band shifts toward preservation.

## 3. Design

### 3.1 Composite regime score

Three inputs, each already computed daily by the pipeline:

- **Vol leg** — annualized vol percentile over 252d (current classifier input).
- **Trend leg** — ADX(14) with sign of SMA(20) − SMA(50) slope.
  Today the regime module reports `ADX 0.0 / trend neutral` — the trend leg
  must degrade gracefully when ADX is uninformative (treat as neutral, not
  as "no trend signal available").
- **Correlation leg** — mean pairwise correlation of universe returns (30d
  window), percentile-ranked.

Each leg is bucketed into {-1, 0, +1} (unfavorable / neutral / favorable).
Composite = sum ∈ {-3, …, +3}.

### 3.2 Parameter grid

Cash band as (lower, upper) per composite score:

| Composite | Candidate A (conservative) | Candidate B (moderate) | Candidate C (aggressive) |
|-----------|----------------------------|------------------------|--------------------------|
| +3        | (8 %, 18 %)                | (5 %, 15 %)            | (2 %, 10 %)              |
| +2        | (10 %, 22 %)               | (8 %, 20 %)            | (5 %, 15 %)              |
| +1        | (12 %, 26 %)               | (12 %, 26 %)           | (10 %, 25 %)             |
| 0         | (15 %, 30 %)               | (15 %, 30 %)           | (15 %, 30 %)             |  ← current NORMAL band = control
| -1        | (18 %, 35 %)               | (20 %, 40 %)           | (22 %, 45 %)             |
| -2        | (25 %, 45 %)               | (28 %, 50 %)           | (30 %, 55 %)             |
| -3        | (30 %, 55 %)               | (35 %, 60 %)           | (40 %, 65 %)             |

3 candidates × 1 control (current 1-D mapping) = **4 arms**.

### 3.3 Backtest protocol

- Engine: existing `src/backtest/backtest.py` + `benchmark_backtest_engine.py`.
- Period: **2026-03-31 → 2026-09-27** (all post-reset history; ledger
  reconciliation is exact on this window — verified 09-24).
- LLM decisions: **replay recorded decisions** from `decision_history.json`
  where possible; where the cash band changes a decision (cash-above-upper-bound
  mandate), flag as *synthetic* and count separately. Pure replay + mechanical
  band override = deterministic, no LLM-in-the-loop for the grid itself.
- Transaction costs: 0 (paper trading); slippage 0. Defaults of record.
- Rebalance constraint: weekly trade cap 3 (existing `PositionCooldownManager`).

### 3.4 Metrics (primary → secondary)

1. **Total return** vs equal-weight-32 benchmark and SPY B&H.
2. **Max drawdown** (must not exceed control by more than +0.5 pp).
3. **Cash drag days** (cash > upper bound) — expect ↓ vs control.
4. **Alpha vs SPY** (the headline gap to close; -14.65 pp at design time).
5. CVaR 95 %, Sharpe, turnover.
6. Cap-binding days (upper-bound breaches that would have forced action).

### 3.5 Decision criteria

- Adopt a candidate only if **total return improves ≥ +0.5 pp AND max drawdown
  worsens ≤ +0.5 pp** vs control on the post-reset window.
- If two candidates pass, prefer the one with fewer synthetic (band-forced)
  decisions — less policy fiction in the replay.
- **Deflated Sharpe discipline:** 4 arms × 1 window is small, but the window
  is one path. A candidate must win by a margin large enough that DSR at
  ~30 % skill correlation stays positive (reuse `deflated_sharpe.py`).

### 3.6 Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Overfit to a single 6-month path | Require margin ≥ +0.5 pp; run CPCV (`cpcv.py`) as secondary validation before any live prompt change |
| Trend leg degenerate (ADX 0.0 as today) | Neutral bucket absorbs it; composite still has 2 informative legs |
| Band churn (oscillating composite → flip-flop mandates) | Hysteresis: composite must move ≥ 2 levels to shift the band intra-week |
| LLM ignores band mandate | Band logic lives in `daily_run.py` pre-pass (mechanical), prompt only explains it |

## 4. Implementation Order (future sessions)

1. `feat/backtest-cash-band-regime`: composite regime module + unit tests
   (bucket boundaries, hysteresis, degenerate-ADX case).
2. Backtest runner arm for each candidate; CSV artefacts.
3. CPCV secondary validation on the passing arm(s).
4. Only then: prompt + `daily_run.py` wiring, gated behind config flag
   `cash_band_mode: "composite"` (default `"vol_only"`).
5. A/B shadow week: run composite in dry-run alongside vol_only before
   enabling.

## 5. Non-Goals

- No change to position sizing (25 % cap stays).
- No change to stop-loss policy (separate track — STOP-OVERRIDE POLICY).
- No new asset classes; universe.json untouched.

---

*The band is a prior, not a prophecy. Update it when the regime changes,
not when the P&L is uncomfortable.* 🦀
