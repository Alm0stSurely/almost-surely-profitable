"""One-off audit: stop-override condition basis sensitivity for DBA 2026-11-02.

Context (memory/trading-2026-11-02.md): the pipeline's governing basis
(includes today's bar) computed DBA BB_position +0.01 -> override condition
(RSI<30 AND BB<0) FAILED -> stop executed mechanically. The intraday monitor's
excl.-today's-bar basis read BB_position -0.077 -> condition HELD all day.
The stop-override policy text (SYSTEM_PROMPT, llm/trading_agent.py L149-162)
does not specify whether the live bar is included in RSI(14)/Bollinger.

This script re-fetches DBA/SAN.PA daily bars, recomputes both bases using the
REPO'S OWN indicator functions (src/data/indicators.py) for exact parity,
verifies the live-quoted indicator values, and writes an artefact.
It changes no code and no prompt.
"""
import sys
from pathlib import Path

REPO = Path("/home/deploy/hermes-workspaces/clawmogorov/repos/almost-surely-profitable")
sys.path.insert(0, str(REPO / "src"))

import yfinance as yf
import pandas as pd
import json

from data.indicators import calculate_rsi, calculate_bollinger_bands

OUT = REPO / "results/analysis/stop_override_basis_20261102.txt"

TICKER = "DBA"
TARGET_DATE = "2026-11-02"


def load_closes(ticker: str, target: pd.Timestamp) -> pd.Series:
    df = yf.download(ticker, period="6mo", interval="1d", progress=False,
                     auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.loc[df.index <= target, "Close"]


def indicators_at(close: pd.Series):
    rsi = calculate_rsi(close, 14)
    upper, middle, lower = calculate_bollinger_bands(close, 20, 2.0)
    span = (upper - lower).replace(0.0, float("nan"))
    bb_pos = ((close - lower) / span).fillna(0.5)
    return rsi, bb_pos, lower


def main() -> int:
    lines = []
    lines.append("=" * 70)
    lines.append("STOP-OVERRIDE CONDITION BASIS AUDIT - DBA 2026-11-02")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Question: does the stop-override condition RSI(14)<30 AND BB_pos<0")
    lines.append("flip when today's bar is included vs excluded?")
    lines.append("")
    lines.append("Policy text (SYSTEM_PROMPT stop-override clause (a)) does NOT")
    lines.append("specify the measurement basis. Pipeline context includes today's")
    lines.append("bar; the intraday monitor recomputes excluding today's bar.")
    lines.append("")
    lines.append("Indicators recomputed with the repo's own functions")
    lines.append("(src/data/indicators.py: calculate_rsi / calculate_bollinger_bands)")
    lines.append("for exact parity with the pipeline and the monitor.")
    lines.append("")

    target = pd.Timestamp(TARGET_DATE)
    close = load_closes(TICKER, target)
    if target not in set(close.index):
        lines.append(f"ERROR: {TARGET_DATE} not in downloaded range "
                     f"({close.index.min().date()}..{close.index.max().date()})")
        OUT.write_text("\n".join(lines) + "\n")
        return 1

    price = float(close.loc[target])

    # Included-bar basis (pipeline): indicators at target bar, incl. bar
    rsi_incl_s, bb_incl_s, _ = indicators_at(close)
    rsi_incl, bb_incl = float(rsi_incl_s.loc[target]), float(bb_incl_s.loc[target])

    # Excluded-bar basis (monitor): indicators at last bar BEFORE target
    last_excl = close.index[close.index < target].max()
    close_excl = close.loc[close.index <= last_excl]
    rsi_excl_s, bb_excl_s, lower_excl_s = indicators_at(close_excl)
    rsi_excl = float(rsi_excl_s.loc[last_excl])
    bb_excl = float(bb_excl_s.loc[last_excl])
    lower_excl = float(lower_excl_s.loc[last_excl])

    cond_incl = (rsi_incl < 30) and (bb_incl < 0)
    cond_excl = (rsi_excl < 30) and (bb_excl < 0)

    lines.append(f"Ticker: {TICKER}   Target date: {TARGET_DATE}")
    lines.append(f"Excl.-basis anchor (last close before target): {last_excl.date()}")
    lines.append(f"Close on target date: {price:.4f}")
    lines.append(f"Excl.-basis lower Bollinger band at anchor: {lower_excl:.4f}")
    lines.append("")
    lines.append(f"{'Basis':<30}{'RSI(14)':>10}{'BB_pos':>10}{'cond holds?':>14}")
    lines.append("-" * 64)
    lines.append(f"{'incl. today (pipeline)':<30}{rsi_incl:>10.3f}{bb_incl:>10.3f}"
                 f"{str(cond_incl):>14}")
    lines.append(f"{'excl. today (monitor)':<30}{rsi_excl:>10.3f}{bb_excl:>10.3f}"
                 f"{str(cond_excl):>14}")
    lines.append("")
    flip = cond_incl != cond_excl
    lines.append(f"Condition flip on basis alone: {'YES' if flip else 'no'}")
    lines.append("")

    lines.append("Cross-check vs live-quoted values:")
    lines.append(f"  live reasoning quotes DBA RSI 4.5, BB_pos 0.01 (incl. basis)")
    lines.append(f"  recomputed incl. basis: RSI {rsi_incl:.3f}, BB_pos {bb_incl:.3f}")
    lines.append(f"  monitor (excl. basis, 17:45 UTC): RSI 4.68, BB_pos -0.077")
    lines.append(f"  recomputed excl. basis: RSI {rsi_excl:.3f}, BB_pos {bb_excl:.3f}")
    lines.append("")

    # Verify SAN.PA sell-discipline quotes (RSI 85.7, BB_pos 1.12)
    san = load_closes("SAN.PA", target)
    rsi_san_s, bb_san_s, _ = indicators_at(san)
    rsi_san = float(rsi_san_s.loc[target])
    bb_san = float(bb_san_s.loc[target])
    lines.append(f"  SAN.PA incl. basis: RSI {rsi_san:.2f} (live quote 85.7), "
                 f"BB_pos {bb_san:.3f} (live quote 1.12)")
    lines.append("")

    with open(REPO / "results/daily/2026-11-02.json") as f:
        res = json.load(f)
    for t in res.get("executed_trades", []):
        lines.append(f"  executed {t['ticker']} {t['action']} @ {t['price']:.4f} "
                     f"(realized {t.get('realized_pnl', 0):+.2f})")
    lines.append("")
    lines.append("Implication:")
    lines.append("  The stop-override policy needs a basis clause. Candidate:")
    lines.append("  'RSI(14) and Bollinger bands are computed on daily closes")
    lines.append("  EXCLUDING the current (partial) bar, consistent with the")
    lines.append("  intraday monitor's governing basis.'")
    lines.append("  LOCKED while H4 midband_deployment is in shadow mode")
    lines.append("  (shared SYSTEM_PROMPT / user-prompt construction must not")
    lines.append("  change mid-experiment); route via a new variant or post-")
    lines.append("  experiment prompt revision.")
    lines.append("")
    lines.append("=" * 70)

    text = "\n".join(lines)
    OUT.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
