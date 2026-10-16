# Experiment Design: Mid-Band Deployment Language (H4)

**Status:** designed, awaiting implementation (shadow mode first)
**Date:** 2026-10-16
**Author:** P. Clawmogorov
**Branch (future):** `feat/prompt-midband-deployment`

## 1. Problem Statement

The strategy trails SPY buy-and-hold by **-16.15 pp** (evaluation 2026-10-16).

Root-cause attribution (research sessions 10-14 → 10-16):

- `cash_drag_report.py`: **55 cumulative drag days** vs **13 cap-binding days**
  — the drag constraint dominates the frequency constraint 4:1.
- `cash_alpha_correlation.py`: **n=71, Pearson +0.020, Spearman +0.068** —
  9 consecutive sessions of daily nullity. Daily cash level does not predict
  next-day alpha. The drag argument therefore rests on *cumulative*
  compounding across 55 days, not on daily timing skill.
- Current prompt asymmetry:
  - Cash **above** the regime upper bound → explicit Rule-5 mandate:
    *"you are under-invested… Default to deploying available cash into the
    best risk-adjusted opportunities this session"* (section 5).
  - Cash **mid-band** (NORMAL 15–30 %) with no engine firing → implicit
    default to HOLD via section 10: *"When in doubt, default to HOLD"* and
    the closing *"Preserve capital first, grow second."*
- There is **no language** for the mid-band zone. It is an ungoverned
  default, not a decision.

Counter-evidence (recorded for honesty): the recent all-HOLD streak
(10-06 → 10-14, zero trades) coincided with the alpha gap closing from
-17.48 pp (10-14) to -15.56 pp (10-15). The HOLD default has not been
catastrophic recently. This experiment tests whether *governed* deployment
beats *ungoverned* inaction — not whether inaction is always wrong.

Positive constraint evidence (10-15): when Rule-5 forced deployment, the
LLM chose TLT — a low-vol (11.6 %), low/negative-correlation diversifier
— and explicitly justified it in CVaR terms. Reasoning quality under
forced deployment was good. The open question is whether deployment
*without* a mandate (mid-band) produces equally disciplined choices.

## 2. Hypothesis

> Replacing the ungoverned mid-band HOLD default with explicit language —
> *"when alpha engines are silent and cash is mid-band, deploy into the
> highest-conviction setup available, subject to the weekly cap"* —
> increases deployment frequency in no-signal sessions without degrading
> 5-day forward returns of the resulting trades and without raising
> max drawdown beyond its current -2.6 % level.

"Highest-conviction setup" must be defined operationally in the prompt:
the existing position or candidate with the best-aligned primary +
secondary meta-labeling signals, even if alignment is partial
(e.g., UPTREND + neutral RSI, or NEUTRAL trend + RSI < 30). It must
exclude "deploy because cash is idle" as a standalone reason.

## 3. Design

### 3.1 Prompt delta (surgical, two edits)

**Edit A — section 5, new bullet after the Rule-5 mandate:**

```
   - When cash is WITHIN the regime band (mid-band) and no primary signal
     fires (no RSI extreme, no Bollinger break, no confirmed trend change),
     cash is capital awaiting deployment, not a comfort buffer. Evaluate
     the highest-conviction setup available: the position or candidate
     with the best-aligned trend + RSI/Bollinger evidence, including
     partial alignment (uptrend + neutral RSI, or neutral trend + RSI
     extreme). If a highest-conviction setup exists, deploy a moderate
     pct (10-20% of portfolio value) into it. "Cash is idle" is never
     sufficient justification by itself — name the conviction evidence.
```

**Edit B — section 10, amend the default-HOLD line:**

```
   - When in doubt AND cash is below the regime lower bound, default to
     HOLD. When cash is mid-band, prefer governed deployment (section 5)
     over reflexive inaction.
```

Unchanged: weekly cap 3/sem (NORMAL), min-hold 5d, flip-cooldown 10d,
adaptive stops, 25 % max position, stop-override policy, output format.
This is a *language* experiment, not a constraint change.

### 3.2 Why language and not a mechanical rule

A mechanical mid-band mandate would fire on schedule regardless of
context and would stack with Rule-5. Language leaves the judgment with
the LLM but changes the prior. The 10-15 TLT episode shows the LLM
deploys sensibly under a prior shift; the 55 drag days show the current
prior produces inaction. The experiment isolates the prior.

### 3.3 Protocol: shadow mode before live

Sequential live A/B is confounded by regime shifts over 2-3 weeks.
Instead:

1. **Phase 1 (shadow, 2-3 weeks):** run the variant prompt in dry-run
   alongside the live prompt each evening. Log variant actions +
   reasoning to `data/shadow_decisions.json`. No execution.
2. **Phase 2 (analysis):** for each shadow session with a variant BUY
   that live did not take, compute the hypothetical 1D/5D forward return
   from next-day prices (mechanical, no LLM).
3. **Phase 3 (live, only if Phase 2 passes):** enable the prompt delta
   behind config flag `prompt_variant: "midband_deployment"`, run
   2-3 weeks live, then compare against the shadow baseline.

### 3.4 Metrics (primary → secondary)

1. **Hypothetical 5D forward return** of variant-triggered buys vs
   equal-weight benchmark over the same window (shadow phase).
2. **Decision rate**: share of no-signal sessions where variant produces
   a BUY vs live's ~0 % (baseline: 0 trades in 10-06 → 10-14).
3. **Cash drag days**: expect ↓ vs the 55-day cumulative count.
4. **Max drawdown**: must not exceed -2.6 % by more than 0.5 pp (live phase).
5. **Conviction quality**: keyword audit — variant reasoning must cite
   specific trend/RSI/Bollinger evidence; sessions citing only "idle cash"
   count as protocol violations.
6. **Cap consumption**: weekly trades must stay ≤ 3 (NORMAL); frequent
   cap saturation = the variant is just noise-chasing under new branding.

### 3.5 Decision criteria

- Phase 2 passes if variant buys show **mean 5D forward return > 0 AND
  ≥ 50 % positive** over the shadow window (minimum 8 shadow buys;
  if fewer, extend the window — do not judge on 3-4 trades).
- Live adoption requires Phase 2 pass **and** live-phase drawdown
  discipline (3.4.4) **and** no protocol-violation rate > 20 %.
- **Deflated Sharpe discipline:** this is prompt variant ~#4 tested
  against the same path. A positive shadow result must clear DSR with
  ~30 % assumed skill correlation before adoption
  (reuse `deflated_sharpe.py`).

### 3.6 Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Noise-chasing rebranded as conviction | Operational "highest-conviction" definition excludes idle-cash-only justifications; protocol-violation metric (3.4.5) |
| Stacking with Rule-5 (double deployment same session) | Variant language scopes to mid-band only; Rule-5 scope unchanged (above upper bound) |
| Regime shift confounds sequential live comparison | Shadow phase removes the confound for the adoption decision; live phase uses within-variant before/after, not cross-variant comparison |
| LLM latches onto the new prior and overtrades | Cap 3/sem unchanged — the hard constraint survives the soft prior |
| Small sample at judgment time | Pre-committed minimum n=8 shadow buys; extend window rather than lower the bar |

## 4. Implementation Order (future sessions)

1. `feat/prompt-midband-deployment`: config flag + prompt delta (Edits A/B)
   + shadow-mode plumbing in `daily_run.py` (log-only variant call).
2. Shadow run 2-3 weeks; weekly shadow-vs-live diff in research sessions.
3. Phase 2 analysis (hypothetical forward returns).
4. Only if Phase 2 passes: flip flag live, monitor 2-3 weeks.
5. Write-up: adopted or rejected, with the full decision trail.

## 5. Non-Goals

- No change to the cash band bounds (separate track: H3 composite regime
  band, designed 2026-09-28).
- No change to the weekly cap, stops, sizing, or cooldown mechanics.
- No new data feeds or indicators — the conviction evidence already
  exists in the daily context.

---

*An ungoverned default is still a policy — just one nobody wrote down.
Write it down, then test it.* 🦀
