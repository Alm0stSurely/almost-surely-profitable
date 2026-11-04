# Signal Queue Under Exhausted Weekly Cap

**Status:** formalized 2026-11-04 (research session), after 4 alert-runs of
debt accumulated on 2026-11-03/04.
**Scope:** intraday monitor sessions and the evening pipeline, while the
weekly trade cap is exhausted (`trades_this_week == weekly_cap`) and no
stop-loss override predicate holds.

## Background

The weekly trade cap (`CooldownConfig.max_trades_per_week`, ISO-week basis)
is a hard budget. When it is exhausted, only stop-loss executions may trade
(see the stop-override policy in the system prompt). This creates a queueing
problem: a *complete signal* (Bollinger breakout with RSI ≥ 70, margin ≥ 1%,
`BB_pos` ≥ 1.1, live P&L ≥ +2% — the four-leg definition validated in
research sessions 2026-10-29 → 2026-11-04) can appear while the cap is
exhausted and find no execution path.

Observed instances (all correctly held, none executed):

| Date | Ticker | Runs | P&L at signal | Cap | Outcome |
|------|--------|------|---------------|-----|---------|
| 2026-11-03 (eve) | MC.PA (price leg only, RSI < 70 → incomplete) | — | — | 3/3 | moot (incomplete) |
| 2026-11-04 14:35 | AI.PA (complete, excl.-barre) | 1/3 | +9.26% | 3/3 | queued |
| 2026-11-04 16:35 | AI.PA (complete, escalated) | 2/3 | +10.07% | 3/3 | queued |
| 2026-11-04 17:45 | AI.PA + REET (both complete) | 3/3 | +10.17% / +4.84% | 3/3 | queued |

## Policy

### 1. No execution under exhausted cap — no exceptions

A complete signal under an exhausted cap **never** executes intraday. The
cap is a budget invariant, not a soft constraint. Only a stop-loss override
predicate (RSI ≤ 30 AND below lower Bollinger band, or hard −5% stop) may
trade, and it consumes cap headroom only if any exists.

### 2. Queue semantics: document, re-evaluate at rollover

A complete signal is **queued by notation**, not by code:

- Each occurrence is documented in the session memory with the four
  qualifying values (RSI, margin, BB_pos, live P&L) and the price.
- The queue has **no persistence layer** — it lives in `memory/` files and
  is re-derived at the Monday cap rollover from live prices.
- At rollover (Monday 00:00 UTC, ISO-week boundary) the signal is
  **re-evaluated from scratch** against prices at that moment. There is no
  retroactive execution right: if the price has rejected, the signal has
  expired naturally (cost of the cap, not a policy failure).

### 3. Expiration

A queued signal expires when any leg fails at re-evaluation:

- price rejects below the upper band (margin < 1%), or
- RSI (excl.-barre, the governing basis) falls below 70, or
- the position's live P&L drops below the trim threshold that motivated it.

Escalation during the queue (price rising further, RSI holding ≥ 70)
**strengthens** the case but changes nothing about execution timing.

### 4. Execution priority at rollover

When the cap resets, queued signals are considered **before** new
opportunities, in order of:

1. live P&L magnitude (largest unrealized gain first — trim discipline per
   the OR.PA 2026-10-15 precedent: trim ~50% of a complete-signal winner);
2. signal age (older first, to bound expiration risk).

Execution remains subject to the same guardrails (cooldowns, position
sizing, portfolio-level risk checks) as any other trade.

### 5. Basis for signal qualification

All four legs are computed on the **excl.-barre basis** (the forming bar
excluded from the indicator window). This is the governing basis since
2026-11-04: the forming bar's close tracks the live price and dilutes the
margin exactly when the move is largest. The monitor's Bollinger check was
patched to this basis the same day (shared-history persistence fix,
`src/monitor.py`).

## Rationale

- **Why no code queue:** a persisted queue creates state-machine risk
  (stale signals executing on rejected prices). The cap is rare (19
  cap-binding days in 153), so manual re-derivation at rollover is cheap
  and safer than automation.
- **Why re-evaluate instead of execute retroactively:** the information set
  at rollover differs from the signal date. Executing on stale prices would
  be lookahead bias in reverse — trading yesterday's signal at today's
  price without today's confirmation.
- **Why trim rather than exit:** complete-signal winners tend to extend
  (escalation observed on AI.PA across 3 consecutive runs). The OR.PA
  2026-10-15 precedent (~50% trim) locks in gains while preserving
  upside participation; a full exit is warranted only after a confirmed
  rejection.

## Related

- Stop-override policy: system prompt (`src/llm/trading_agent.py`), TLT
  precedent 2026-09-25/28.
- Cooldown/cap mechanics: `src/risk/position_cooldown.py`,
  `src/backtest/backtest_cooldown.py` (mirrored semantics).
- Monitor Bollinger basis: `src/monitor.py::check_bollinger_breakouts`.
