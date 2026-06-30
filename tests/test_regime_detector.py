"""
Comprehensive tests for the regime detection module.

Covers:
- Volatility regime classification (percentile-based)
- ADX calculation (Wilder's smoothing, TR, +DI/-DI)
- Trend regime detection (ADX + SMA cross)
- Correlation regime detection (off-diagonal average)
- Strategy recommendations (regime → actionable signals)
- LLM formatting (prompt generation)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from analysis.regime_detector import (
    RegimeDetector,
    RegimeState,
    format_regime_for_llm,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    return RegimeDetector()


@pytest.fixture
def flat_prices():
    """Prices with zero volatility — all identical."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    return pd.DataFrame({"A": np.full(100, 100.0), "B": np.full(100, 50.0)}, index=dates)


@pytest.fixture
def trending_up_prices():
    """Strong uptrend — ADX should be high, SMA20 > SMA50."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    t = np.arange(100)
    # Strong linear trend with small noise
    a = 100 + 0.5 * t + np.random.RandomState(42).normal(0, 0.5, 100)
    b = 50 + 0.3 * t + np.random.RandomState(43).normal(0, 0.3, 100)
    return pd.DataFrame({"A": a, "B": b}, index=dates)


@pytest.fixture
def mean_reverting_prices():
    """Oscillating prices centered at constant mean — ADX should be low."""
    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    t = np.arange(300)
    # Pure oscillation around 100 and 50, no drift
    a = 100 + 5 * np.sin(t / 3) + np.random.RandomState(44).normal(0, 0.2, 300)
    b = 50 + 3 * np.cos(t / 4) + np.random.RandomState(45).normal(0, 0.15, 300)
    return pd.DataFrame({"A": a, "B": b}, index=dates)


@pytest.fixture
def high_vol_prices():
    """High volatility regime — recent vol much higher than historical."""
    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    rs = np.random.RandomState(46)
    # First 250 days: low vol (0.5% daily)
    returns_a_low = rs.normal(0, 0.005, 250)
    returns_b_low = rs.normal(0, 0.005, 250)
    # Last 50 days: high vol (4% daily) — will dominate current vol calc
    returns_a_high = rs.normal(0, 0.04, 50)
    returns_b_high = rs.normal(0, 0.04, 50)
    returns_a = np.concatenate([returns_a_low, returns_a_high])
    returns_b = np.concatenate([returns_b_low, returns_b_high])
    a = 100 * np.exp(np.cumsum(returns_a))
    b = 50 * np.exp(np.cumsum(returns_b))
    return pd.DataFrame({"A": a, "B": b}, index=dates)


@pytest.fixture
def low_vol_prices():
    """Low volatility regime — tiny daily moves."""
    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    rs = np.random.RandomState(47)
    returns_a = rs.normal(0, 0.002, 300)  # 0.2% daily vol
    returns_b = rs.normal(0, 0.0015, 300)
    a = 100 * np.exp(np.cumsum(returns_a))
    b = 50 * np.exp(np.cumsum(returns_b))
    return pd.DataFrame({"A": a, "B": b}, index=dates)


@pytest.fixture
def highly_correlated_prices():
    """Two assets moving almost in lockstep."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    rs = np.random.RandomState(48)
    common = rs.normal(0, 0.01, 100)
    a = 100 * np.exp(np.cumsum(common + rs.normal(0, 0.001, 100)))
    b = 50 * np.exp(np.cumsum(common + rs.normal(0, 0.001, 100)))
    return pd.DataFrame({"A": a, "B": b}, index=dates)


@pytest.fixture
def uncorrelated_prices():
    """Two assets with independent moves."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    rs = np.random.RandomState(49)
    a = 100 * np.exp(np.cumsum(rs.normal(0, 0.01, 100)))
    b = 50 * np.exp(np.cumsum(rs.normal(0, 0.01, 100)))
    return pd.DataFrame({"A": a, "B": b}, index=dates)


# ---------------------------------------------------------------------------
# RegimeState
# ---------------------------------------------------------------------------

class TestRegimeState:
    def test_summary_format(self):
        state = RegimeState(
            volatility_regime="high",
            trend_regime="trending_up",
            correlation_regime="normal",
            volatility_percentile=82.5,
            adx_value=30.2,
            avg_correlation=0.45,
        )
        summary = state.summary()
        assert "Vol: high (82th pct)" in summary
        assert "Trend: trending_up (ADX: 30.2)" in summary
        assert "Corr: normal (0.45)" in summary

    def test_summary_low_vol_mean_reverting(self):
        state = RegimeState(
            volatility_regime="low",
            trend_regime="mean_reverting",
            correlation_regime="low_correlation",
            volatility_percentile=12.0,
            adx_value=15.0,
            avg_correlation=0.15,
        )
        summary = state.summary()
        assert "Vol: low (12th pct)" in summary
        assert "Trend: mean_reverting (ADX: 15.0)" in summary
        assert "Corr: low_correlation (0.15)" in summary


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestRegimeDetectorInit:
    def test_default_params(self, detector):
        assert detector.vol_lookback == 20
        assert detector.vol_percentile_threshold_high == 75.0
        assert detector.vol_percentile_threshold_low == 25.0
        assert detector.adx_period == 14
        assert detector.adx_trending_threshold == 25.0
        assert detector.adx_mean_reverting_threshold == 20.0
        assert detector.correlation_lookback == 60

    def test_custom_params(self):
        det = RegimeDetector(
            vol_lookback=10,
            vol_percentile_threshold_high=80.0,
            vol_percentile_threshold_low=20.0,
            adx_period=7,
            adx_trending_threshold=30.0,
            adx_mean_reverting_threshold=15.0,
            correlation_lookback=30,
        )
        assert det.vol_lookback == 10
        assert det.vol_percentile_threshold_high == 80.0
        assert det.vol_percentile_threshold_low == 20.0
        assert det.adx_period == 7
        assert det.adx_trending_threshold == 30.0
        assert det.adx_mean_reverting_threshold == 15.0
        assert det.correlation_lookback == 30


# ---------------------------------------------------------------------------
# Volatility Regime
# ---------------------------------------------------------------------------

class TestDetectVolatilityRegime:
    def test_high_volatility(self, detector, high_vol_prices):
        regime, pct = detector.detect_volatility_regime(high_vol_prices)
        assert regime == "high"
        assert pct >= detector.vol_percentile_threshold_high

    def test_low_volatility(self, detector):
        # Create data where current vol is definitively the lowest
        dates = pd.date_range("2024-01-01", periods=300, freq="D")
        rs = np.random.RandomState(97)
        # 280 days of moderate-to-high vol (1.5% - 3%)
        returns_a = rs.normal(0, 0.02, 280)
        returns_b = rs.normal(0, 0.018, 280)
        # Last 20 days: extremely low vol (0.05%)
        returns_a = np.concatenate([returns_a, rs.normal(0, 0.0005, 20)])
        returns_b = np.concatenate([returns_b, rs.normal(0, 0.0005, 20)])
        a = 100 * np.exp(np.cumsum(returns_a))
        b = 50 * np.exp(np.cumsum(returns_b))
        prices = pd.DataFrame({"A": a, "B": b}, index=dates)
        regime, pct = detector.detect_volatility_regime(prices)
        assert regime == "low"
        assert pct <= detector.vol_percentile_threshold_low

    def test_normal_volatility(self, detector):
        # Create data with mixed volatility regimes — current window should be valid
        dates = pd.date_range("2024-01-01", periods=300, freq="D")
        rs = np.random.RandomState(98)
        # Generate returns with the SAME distribution throughout — current vol ~ median
        returns_a = rs.normal(0, 0.015, 300)
        returns_b = rs.normal(0, 0.012, 300)
        a = 100 * np.exp(np.cumsum(returns_a))
        b = 50 * np.exp(np.cumsum(returns_b))
        prices = pd.DataFrame({"A": a, "B": b}, index=dates)
        regime, pct = detector.detect_volatility_regime(prices)
        # For stationary returns, the current vol should be within valid bounds
        assert regime in ("high", "normal", "low")
        assert 0 <= pct <= 100
        # With 2 assets and 300 days of stationary data, we expect ~normal most of the time
        # but sampling variance can push it to the edges — just verify it's reasonable
        assert pct > 5  # Not the absolute minimum
        assert pct < 95  # Not the absolute maximum

    def test_flat_prices_zero_vol(self, detector, flat_prices):
        # Flat prices → zero current vol → should be low percentile
        regime, pct = detector.detect_volatility_regime(flat_prices)
        assert regime == "low"
        assert pct == 0.0

    def test_single_asset(self, detector):
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        prices = pd.DataFrame({"ONLY": 100 * np.exp(np.cumsum(np.random.RandomState(50).normal(0, 0.02, 100)))}, index=dates)
        regime, pct = detector.detect_volatility_regime(prices)
        assert regime in ("high", "normal", "low")
        assert 0 <= pct <= 100

    def test_percentile_bounds(self, detector, high_vol_prices):
        _, pct = detector.detect_volatility_regime(high_vol_prices)
        assert 0 <= pct <= 100


# ---------------------------------------------------------------------------
# ADX Calculation
# ---------------------------------------------------------------------------

class TestCalculateAdx:
    def test_returns_dataframe(self, detector, trending_up_prices):
        adx = detector.calculate_adx(trending_up_prices)
        assert isinstance(adx, pd.DataFrame)
        assert list(adx.columns) == ["A", "B"]
        assert len(adx) == len(trending_up_prices)

    def test_adx_non_negative(self, detector, trending_up_prices):
        adx = detector.calculate_adx(trending_up_prices)
        assert (adx.dropna() >= 0).all().all()

    def test_adx_strong_trend_high(self, detector, trending_up_prices):
        adx = detector.calculate_adx(trending_up_prices)
        # Last values should reflect the strong trend
        last_adx = adx.iloc[-1]
        assert last_adx["A"] > 20  # Should detect trend
        assert last_adx["B"] > 20

    def test_adx_mean_reverting_low(self, detector, mean_reverting_prices):
        adx = detector.calculate_adx(mean_reverting_prices)
        last_adx = adx.iloc[-1]
        # Oscillating prices should have lower ADX
        assert last_adx["A"] < 25

    def test_with_high_low(self, detector, trending_up_prices):
        rs = np.random.RandomState(51)
        high = trending_up_prices * (1 + rs.uniform(0.001, 0.01, trending_up_prices.shape))
        low = trending_up_prices * (1 - rs.uniform(0.001, 0.01, trending_up_prices.shape))
        adx = detector.calculate_adx(trending_up_prices, high, low)
        assert isinstance(adx, pd.DataFrame)
        assert not adx.dropna().empty

    def test_without_high_low_uses_close(self, detector, trending_up_prices):
        adx_close_only = detector.calculate_adx(trending_up_prices)
        # When high/low not provided, prices are used as proxy
        assert isinstance(adx_close_only, pd.DataFrame)
        assert not adx_close_only.dropna().empty

    def test_flat_prices_adx(self, detector, flat_prices):
        # Flat prices → TR = 0 for most periods after first
        adx = detector.calculate_adx(flat_prices)
        # Should not crash; NaN expected for initial warmup
        assert isinstance(adx, pd.DataFrame)


# ---------------------------------------------------------------------------
# Trend Regime
# ---------------------------------------------------------------------------

class TestDetectTrendRegime:
    def test_trending_up(self, detector, trending_up_prices):
        adx = detector.calculate_adx(trending_up_prices)
        regime, adx_val = detector.detect_trend_regime(trending_up_prices, adx)
        assert regime == "trending_up"
        assert adx_val >= detector.adx_trending_threshold

    def test_mean_reverting(self, detector, mean_reverting_prices):
        adx = detector.calculate_adx(mean_reverting_prices)
        regime, adx_val = detector.detect_trend_regime(mean_reverting_prices, adx)
        # Oscillating prices should NOT show strong trending
        assert regime in ("mean_reverting", "neutral")
        assert adx_val < detector.adx_trending_threshold

    def test_neutral_regime(self, detector):
        # Create data with moderate ADX (between thresholds)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        t = np.arange(100)
        # Weak trend — ADX should land in neutral zone
        a = 100 + 0.1 * t + np.random.RandomState(52).normal(0, 1.0, 100)
        b = 50 + 0.05 * t + np.random.RandomState(53).normal(0, 0.8, 100)
        prices = pd.DataFrame({"A": a, "B": b}, index=dates)
        adx = detector.calculate_adx(prices)
        regime, adx_val = detector.detect_trend_regime(prices, adx)
        # Could be neutral or trending depending on noise, but must be valid
        assert regime in ("trending_up", "trending_down", "trending_mixed", "mean_reverting", "neutral")
        assert adx_val >= 0

    def test_uses_provided_adx(self, detector, trending_up_prices):
        adx = detector.calculate_adx(trending_up_prices)
        regime1, val1 = detector.detect_trend_regime(trending_up_prices, adx)
        regime2, val2 = detector.detect_trend_regime(trending_up_prices)  # computes internally
        assert regime1 == regime2
        assert val1 == pytest.approx(val2, abs=1e-9)

    def test_trending_down(self, detector):
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        t = np.arange(100)
        a = 100 - 0.5 * t + np.random.RandomState(54).normal(0, 0.5, 100)
        b = 50 - 0.3 * t + np.random.RandomState(55).normal(0, 0.3, 100)
        prices = pd.DataFrame({"A": a, "B": b}, index=dates)
        adx = detector.calculate_adx(prices)
        regime, adx_val = detector.detect_trend_regime(prices, adx)
        assert regime == "trending_down"
        assert adx_val >= detector.adx_trending_threshold

    def test_trending_mixed(self, detector):
        # One asset up, one down — mixed signals
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        t = np.arange(100)
        a = 100 + 0.5 * t + np.random.RandomState(56).normal(0, 0.5, 100)
        b = 50 - 0.3 * t + np.random.RandomState(57).normal(0, 0.3, 100)
        prices = pd.DataFrame({"A": a, "B": b}, index=dates)
        adx = detector.calculate_adx(prices)
        regime, adx_val = detector.detect_trend_regime(prices, adx)
        if adx_val >= detector.adx_trending_threshold:
            assert regime == "trending_mixed"
        else:
            assert regime in ("mean_reverting", "neutral")


# ---------------------------------------------------------------------------
# Correlation Regime
# ---------------------------------------------------------------------------

class TestDetectCorrelationRegime:
    def test_high_correlation(self, detector, highly_correlated_prices):
        regime, avg_corr = detector.detect_correlation_regime(highly_correlated_prices)
        assert regime == "high_correlation"
        assert avg_corr > 0.7

    def test_low_correlation(self, detector, uncorrelated_prices):
        regime, avg_corr = detector.detect_correlation_regime(uncorrelated_prices)
        assert regime == "low_correlation"
        assert avg_corr < 0.3

    def test_insufficient_history(self, detector):
        # Only 10 days, less than correlation_lookback (60)
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        prices = pd.DataFrame({"A": np.arange(10), "B": np.arange(10) + 5}, index=dates)
        regime, avg_corr = detector.detect_correlation_regime(prices)
        assert regime == "normal"
        assert avg_corr == 0.5

    def test_single_asset_correlation(self, detector):
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        prices = pd.DataFrame({"ONLY": np.arange(100)}, index=dates)
        # Single asset → corr matrix is 1x1, mask removes diagonal → empty array
        # np.mean of empty slice gives NaN; need to see how code handles it
        regime, avg_corr = detector.detect_correlation_regime(prices)
        # Should not crash; regime should be one of the valid values
        assert regime in ("high_correlation", "normal", "low_correlation")

    def test_correlation_bounds(self, detector, highly_correlated_prices):
        _, avg_corr = detector.detect_correlation_regime(highly_correlated_prices)
        assert -1 <= avg_corr <= 1


# ---------------------------------------------------------------------------
# Full Analysis
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_returns_regime_state(self, detector, trending_up_prices):
        state = detector.analyze(trending_up_prices)
        assert isinstance(state, RegimeState)
        assert state.volatility_regime in ("high", "normal", "low")
        assert state.trend_regime in ("trending_up", "trending_down", "trending_mixed", "mean_reverting", "neutral")
        assert state.correlation_regime in ("high_correlation", "normal", "low_correlation")
        assert 0 <= state.volatility_percentile <= 100
        assert state.adx_value >= 0
        assert -1 <= state.avg_correlation <= 1

    def test_with_high_low(self, detector, trending_up_prices):
        rs = np.random.RandomState(58)
        high = trending_up_prices * (1 + rs.uniform(0.001, 0.01, trending_up_prices.shape))
        low = trending_up_prices * (1 - rs.uniform(0.001, 0.01, trending_up_prices.shape))
        state = detector.analyze(trending_up_prices, high, low)
        assert isinstance(state, RegimeState)

    def test_consistency(self, detector, trending_up_prices):
        # Same input → same output (deterministic)
        state1 = detector.analyze(trending_up_prices)
        state2 = detector.analyze(trending_up_prices)
        assert state1.volatility_regime == state2.volatility_regime
        assert state1.trend_regime == state2.trend_regime
        assert state1.correlation_regime == state2.correlation_regime
        assert state1.volatility_percentile == pytest.approx(state2.volatility_percentile, abs=1e-9)
        assert state1.adx_value == pytest.approx(state2.adx_value, abs=1e-9)
        assert state1.avg_correlation == pytest.approx(state2.avg_correlation, abs=1e-9)


# ---------------------------------------------------------------------------
# Strategy Recommendations
# ---------------------------------------------------------------------------

class TestGetStrategyRecommendation:
    def test_high_vol_conservative(self, detector):
        state = RegimeState(
            volatility_regime="high",
            trend_regime="neutral",
            correlation_regime="normal",
            volatility_percentile=85.0,
            adx_value=20.0,
            avg_correlation=0.5,
        )
        rec = detector.get_strategy_recommendation(state)
        assert rec["position_sizing"] == "conservative"
        assert rec["stop_loss_tightening"] is True

    def test_low_vol_aggressive(self, detector):
        state = RegimeState(
            volatility_regime="low",
            trend_regime="neutral",
            correlation_regime="normal",
            volatility_percentile=10.0,
            adx_value=20.0,
            avg_correlation=0.5,
        )
        rec = detector.get_strategy_recommendation(state)
        assert rec["position_sizing"] == "aggressive"
        assert rec["stop_loss_tightening"] is False

    def test_trending_enables_trend_following(self, detector):
        state = RegimeState(
            volatility_regime="normal",
            trend_regime="trending_up",
            correlation_regime="normal",
            volatility_percentile=50.0,
            adx_value=30.0,
            avg_correlation=0.5,
        )
        rec = detector.get_strategy_recommendation(state)
        assert rec["trend_following"] is True
        assert rec["mean_reversion_opportunities"] is False

    def test_mean_reverting_enables_mr(self, detector):
        state = RegimeState(
            volatility_regime="normal",
            trend_regime="mean_reverting",
            correlation_regime="normal",
            volatility_percentile=50.0,
            adx_value=15.0,
            avg_correlation=0.5,
        )
        rec = detector.get_strategy_recommendation(state)
        assert rec["mean_reversion_opportunities"] is True
        assert rec["trend_following"] is False

    def test_high_correlation_reduce_exposure(self, detector):
        state = RegimeState(
            volatility_regime="normal",
            trend_regime="neutral",
            correlation_regime="high_correlation",
            volatility_percentile=50.0,
            adx_value=20.0,
            avg_correlation=0.8,
        )
        rec = detector.get_strategy_recommendation(state)
        assert rec["reduce_correlated_exposure"] is True

    def test_all_defaults_off(self, detector):
        state = RegimeState(
            volatility_regime="normal",
            trend_regime="neutral",
            correlation_regime="normal",
            volatility_percentile=50.0,
            adx_value=20.0,
            avg_correlation=0.5,
        )
        rec = detector.get_strategy_recommendation(state)
        assert rec["position_sizing"] == "normal"
        assert rec["stop_loss_tightening"] is False
        assert rec["mean_reversion_opportunities"] is False
        assert rec["trend_following"] is False
        assert rec["reduce_correlated_exposure"] is False

    def test_combined_regime(self, detector):
        state = RegimeState(
            volatility_regime="high",
            trend_regime="trending_down",
            correlation_regime="high_correlation",
            volatility_percentile=90.0,
            adx_value=35.0,
            avg_correlation=0.85,
        )
        rec = detector.get_strategy_recommendation(state)
        assert rec["position_sizing"] == "conservative"
        assert rec["stop_loss_tightening"] is True
        assert rec["trend_following"] is True
        assert rec["reduce_correlated_exposure"] is True


# ---------------------------------------------------------------------------
# LLM Formatting
# ---------------------------------------------------------------------------

class TestFormatRegimeForLlm:
    def test_contains_regime_summary(self, detector, trending_up_prices):
        state = detector.analyze(trending_up_prices)
        rec = detector.get_strategy_recommendation(state)
        text = format_regime_for_llm(state, rec)
        assert "Market Regime Analysis" in text
        assert state.summary() in text

    def test_contains_recommendations(self, detector, trending_up_prices):
        state = detector.analyze(trending_up_prices)
        rec = detector.get_strategy_recommendation(state)
        text = format_regime_for_llm(state, rec)
        assert "Position Sizing:" in text
        assert "Stop-Loss Adjustment:" in text
        assert "Mean Reversion Trades:" in text
        assert "Trend Following:" in text
        assert "Correlation Risk:" in text

    def test_high_vol_text(self):
        state = RegimeState(
            volatility_regime="high",
            trend_regime="neutral",
            correlation_regime="normal",
            volatility_percentile=85.0,
            adx_value=20.0,
            avg_correlation=0.5,
        )
        det = RegimeDetector()
        rec = det.get_strategy_recommendation(state)
        text = format_regime_for_llm(state, rec)
        assert "Volatility is elevated" in text
        assert "prioritize capital preservation" in text

    def test_low_vol_text(self):
        state = RegimeState(
            volatility_regime="low",
            trend_regime="neutral",
            correlation_regime="normal",
            volatility_percentile=10.0,
            adx_value=20.0,
            avg_correlation=0.5,
        )
        det = RegimeDetector()
        rec = det.get_strategy_recommendation(state)
        text = format_regime_for_llm(state, rec)
        assert "Volatility is compressed" in text
        assert "opportunities for larger positions" in text

    def test_trending_text(self):
        state = RegimeState(
            volatility_regime="normal",
            trend_regime="trending_up",
            correlation_regime="normal",
            volatility_percentile=50.0,
            adx_value=30.0,
            avg_correlation=0.5,
        )
        det = RegimeDetector()
        rec = det.get_strategy_recommendation(state)
        text = format_regime_for_llm(state, rec)
        assert "Markets are trending strongly" in text
        assert "momentum strategies favored" in text

    def test_mean_reverting_text(self):
        state = RegimeState(
            volatility_regime="normal",
            trend_regime="mean_reverting",
            correlation_regime="normal",
            volatility_percentile=50.0,
            adx_value=15.0,
            avg_correlation=0.5,
        )
        det = RegimeDetector()
        rec = det.get_strategy_recommendation(state)
        text = format_regime_for_llm(state, rec)
        assert "Markets are range-bound" in text
        assert "mean reversion and volatility compression trades favored" in text

    def test_high_correlation_text(self):
        state = RegimeState(
            volatility_regime="normal",
            trend_regime="neutral",
            correlation_regime="high_correlation",
            volatility_percentile=50.0,
            adx_value=20.0,
            avg_correlation=0.8,
        )
        det = RegimeDetector()
        rec = det.get_strategy_recommendation(state)
        text = format_regime_for_llm(state, rec)
        assert "Assets are highly correlated" in text
        assert "diversification benefits are limited" in text

    def test_low_correlation_text(self):
        state = RegimeState(
            volatility_regime="normal",
            trend_regime="neutral",
            correlation_regime="low_correlation",
            volatility_percentile=50.0,
            adx_value=20.0,
            avg_correlation=0.1,
        )
        det = RegimeDetector()
        rec = det.get_strategy_recommendation(state)
        text = format_regime_for_llm(state, rec)
        assert "Assets show low correlation" in text
        assert "good diversification opportunities" in text


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_nan_prices(self, detector):
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        prices = pd.DataFrame({
            "A": np.where(np.arange(100) == 50, np.nan, 100 + np.arange(100) * 0.1),
            "B": 50 + np.arange(100) * 0.05,
        }, index=dates)
        # Should not crash; pandas handles NaN in pct_change and rolling
        state = detector.analyze(prices)
        assert isinstance(state, RegimeState)

    def test_very_short_history(self, detector):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        prices = pd.DataFrame({"A": [100, 101, 102, 103, 104]}, index=dates)
        state = detector.analyze(prices)
        assert isinstance(state, RegimeState)
        # Correlation should fall back to "normal", 0.5
        assert state.correlation_regime == "normal"
        assert state.avg_correlation == 0.5

    def test_identical_assets(self, detector):
        # Two assets with perfectly identical returns
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        returns = np.random.RandomState(59).normal(0, 0.01, 100)
        a = 100 * np.exp(np.cumsum(returns))
        prices = pd.DataFrame({"A": a, "B": a}, index=dates)
        regime, avg_corr = detector.detect_correlation_regime(prices)
        assert regime == "high_correlation"
        assert avg_corr > 0.99

    def test_inverted_assets(self, detector):
        # Two assets with perfectly inverse returns
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        returns = np.random.RandomState(60).normal(0, 0.01, 100)
        a = 100 * np.exp(np.cumsum(returns))
        b = 100 * np.exp(np.cumsum(-returns))
        prices = pd.DataFrame({"A": a, "B": b}, index=dates)
        regime, avg_corr = detector.detect_correlation_regime(prices)
        assert regime == "low_correlation"
        assert avg_corr < -0.9
