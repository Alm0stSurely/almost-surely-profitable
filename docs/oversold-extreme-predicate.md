# Oversold-Extreme Predicate — Status and Doctrine

**Status:** updated 2026-11-06 — second consecutive live completion
(PDBC). Formalized 2026-11-05 (research session), after the first live
completion of the predicate on a held position.
**Scope:** interpretation of the stop-override predicate
(RSI ≤ 30 AND below the lower Bollinger band) across the intraday monitor
and the evening pipeline.

## Background

The system prompt's stop-override policy allows contrarian action at the
hard −5 % stop when the position shows an *oversold-extreme* signature:
RSI14 ≤ 30 **and** price below the lower Bollinger band. The four-week
research history (2026-10-29 → 2026-11-05) established the **excl.-barre
basis** as governing for all RSI/Bollinger predicates: the forming bar's
close tracks the live price and dilutes the very signal being measured.

Until 2026-11-05 the predicate had never completed on a held position.
Breakdowns below RSI ~35 occurred (PDBC 2026-11-04, RSI 36.6 — one miss),
but the two-leg conjunction never held.

## First live completion — PDBC, 2026-11-05 (evening pipeline)

Official close basis (3-month window, rolling manual, excl.-barre):

| Leg | Value | Threshold | Status |
|-----|-------|-----------|--------|
| RSI14 excl.-barre | **29.64** | ≤ 30 | ✓ |
| BB_pos excl.-barre | **−0.227** | ≤ 0 (below lower band) | ✓ |
| RSI14 incl.-barre | 32.56 | — | would **fail** (basis matters) |

Position state at completion: P&L **+5.10 %**, live price $18.74 vs stop
$16.94 — ~10 % above the mechanical stop. **Action taken: none.**

## Second live completion — PDBC, 2026-11-06 (evening pipeline, persisted)

The predicate completed a **second consecutive day** on PDBC, confirming the
first completion was not a one-bar accident:

| Date | RSI14 excl. | BB_pos excl. | Close | P&L | Distance to stop |
|------|-------------|--------------|-------|-----|------------------|
| 2026-11-05 | 29.64 | −0.227 | $18.74 | +5.10 % | ~10 % |
| 2026-11-06 | **22.04** | **−0.069** | $18.785 | +5.36 % | ~10.8 % |

Values for 2026-11-06 re-verified live with the repo's own indicator
functions (parity rule: `data.indicators.calculate_rsi/calculate_bollinger_bands`,
3-month fetch, excl.-barre): RSI 22.04, BB lower $18.897 — match. The
RSI leg *deepened* (29.64 → 22.04) while BB_pos crept back toward 0
(−0.227 → −0.069): the price is stabilizing near the lower band after the
washout leg, i.e. momentum exhaustion without a reversal.

Action taken: **none** (zone 2 — far from stop, profitable position, and
cap 4/3 exhausted). Two consecutive completions far from the stop still
carry no portfolio action under doctrine §2.

## Doctrine

### 1. The predicate is a necessary condition for stop-override *evaluation*, not an action trigger

Completing the oversold-extreme predicate authorizes **considering**
contrarian action *at the stop*. It is a gate on a decision that only
exists when the position is in distress — it is **not** a dip-buy signal,
not a mean-reversion entry, and not a partial-trim trigger on a profitable
position. The current regime has mean-reversion buys disabled; the
predicate does not re-enable them.

### 2. Action zones

| Predicate | Stop proximity | Meaning | Action |
|-----------|----------------|---------|--------|
| complete | near stop (P&L ≲ −3 %, or distance to stop < ~2 %) | genuine override candidate | evaluate override per system prompt (hold for mean reversion / size-aware handling) |
| complete | far from stop (P&L ≥ 0 % by construction — the stop is anchored to avg_price) | documentation event only | **none**; log and continue surveillance |
| incomplete | near stop | no contrarian evidence | mechanical −5 % stop executes |

A position trading at a positive P&L is ~2 × |P&L| above its stop by
construction; predicate completion on such a position is statistically
interesting (momentum exhaustion after a profitable run) but carries **no**
portfolio action under any current policy.

### 3. Basis sensitivity is part of the predicate

PDBC 2026-11-05 completes excl.-barre (29.64) but fails incl.-barre
(32.56). The governing basis is excl.-barre (monitor patch `32b79ba`,
2026-11-04); any future automation of this predicate must specify the basis
explicitly. A predicate that changes verdict with the basis is two
predicates, not one.

### 4. Interaction with the weekly cap

Per `signal-queue-under-cap.md`, a stop-override is the only trade class
allowed under an exhausted cap. The oversold-extreme predicate is the
*qualifying* signature for a soft override — but proximity to the stop
(zone 2 of the table above) remains required. Predicate completion far
from the stop under an exhausted cap (the PDBC case, cap 3/3) is therefore
doubly non-actionable: no proximity, no budget.

## Open questions

- **Zone boundary:** "near stop" is currently defined loosely (~2 % distance
  or P&L ≲ −3 %). No live instance of zone 1 has occurred; the boundary
  should be sharpened by the first real case rather than in advance.
- **Trailing semantics:** if a position's predicate completes far from the
  stop and the price *then* grinds toward the stop, does the earlier
  completion carry any weight? Current doctrine: no — the predicate is
  re-evaluated at the stop event with fresh values. *Update 2026-11-06:* a
  predicate can persist across consecutive sessions (PDBC day 2) — when it
  keeps re-completing on fresh values, the "earlier completion" question is
  moot; the live re-evaluation at any future stop event will see a complete
  predicate anyway. The open part of the question remains the *decay* case:
  completion, then dissipation, then stop breach.
