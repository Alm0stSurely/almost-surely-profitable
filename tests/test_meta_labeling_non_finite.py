"""Non-finite input guards for the meta-labeling pipeline.

Pins the PR contract:
- _extract_features emits only finite features (per-element drop, loud log,
  n-1 healthy siblings survive).
- fit() drops non-finite training rows loudly instead of silently learning
  NaN as a split direction (sklearn >= 1.4 RF behavior) or crashing the
  whole call on inf.
- predict() skips signals missing any trained feature (never zero-fills a
  hole in the feature vector).
- size_positions() refuses None/NaN probabilities loudly and validates the
  Kelly win/loss ratio at the boundary.
- signal.confidence == 0.0 survives as 0.0 (no falsy-sentinel laundering).
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.backtest.meta_labeling import (
    MetaLabel,
    MetaLabeler,
    MetaLabelingConfig,
    PrimarySignal,
    SignalType,
)

MODULE_LOGGER = "src.backtest.meta_labeling"

HEALTHY_KEYS = {
    "returns_mean", "returns_std", "returns_skew", "cumulative_return",
    "price_vs_sma20", "volatility_trend", "volume_vs_mean", "volume_trend",
    "rsi", "rsi_trend", "bb_position", "primary_signal", "signal_confidence",
    "hour", "day_of_week", "is_month_start", "is_month_end",
}


@pytest.fixture
def config():
    return MetaLabelingConfig()


@pytest.fixture
def price_data():
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "close": 100 + np.cumsum(np.random.randn(60) * 0.5),
            "volume": np.random.randint(1_000_000, 5_000_000, 60),
            "rsi": 30 + np.random.rand(60) * 40,
            "bb_position": np.random.rand(60),
        },
        index=dates,
    )


@pytest.fixture
def healthy_signal(price_data):
    return PrimarySignal(
        timestamp=price_data.index[30], ticker="T",
        signal=SignalType.BUY, confidence=0.8,
    )


def _make_training_set(price_data, n=120):
    signals = [
        PrimarySignal(
            timestamp=price_data.index[min(20 + (i % 35), len(price_data) - 1)],
            ticker="T", signal=SignalType.BUY, confidence=0.6,
        )
        for i in range(n)
    ]
    outcomes = [i % 2 for i in range(n)]
    return signals, outcomes


class TestExtractFeaturesFiniteGuards:
    def test_nan_tail_close_omits_price_level_features(self, config, price_data, caplog):
        data = price_data.copy()
        ts = data.index[-1]
        data.loc[ts, "close"] = np.nan
        labeler = MetaLabeler(config)
        with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
            feats = labeler._extract_features(
                PrimarySignal(timestamp=ts, ticker="T",
                              signal=SignalType.BUY, confidence=0.8),
                data,
            )
        assert "cumulative_return" not in feats
        assert "price_vs_sma20" not in feats
        # n-1 siblings property: returns/indicator families still measured
        assert "returns_mean" in feats
        assert "rsi" in feats
        assert "volume_vs_mean" in feats
        assert any("Dropped non-finite features" in r.getMessage()
                   for r in caplog.records)

    def test_zero_window_start_omits_inf_candidates(self, config, price_data):
        data = price_data.copy()
        data.loc[data.index[-20], "close"] = 0.0
        labeler = MetaLabeler(config)
        with np.errstate(invalid="ignore", divide="ignore"):
            feats = labeler._extract_features(
                PrimarySignal(timestamp=data.index[-1], ticker="T",
                              signal=SignalType.BUY, confidence=0.8),
                data,
            )
        # pct_change divides by the zero tick -> inf returns; the ratio
        # divides by the zero window edge -> inf candidate. All dropped.
        assert "cumulative_return" not in feats
        assert "returns_mean" not in feats
        # unrelated families survive
        assert "volume_vs_mean" in feats
        assert "rsi" in feats

    def test_healthy_extraction_emits_no_warnings(self, config, price_data, healthy_signal, caplog):
        """Bounded healthy-path pin: normal windows stay quiet."""
        labeler = MetaLabeler(config)
        with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
            feats = labeler._extract_features(healthy_signal, price_data)
        assert caplog.records == []
        assert set(feats.keys()) == HEALTHY_KEYS

    def test_zero_confidence_survives(self, config, price_data):
        labeler = MetaLabeler(config)
        feats = labeler._extract_features(
            PrimarySignal(timestamp=price_data.index[30], ticker="T",
                          signal=SignalType.BUY, confidence=0.0),
            price_data,
        )
        assert feats["signal_confidence"] == 0.0

    def test_none_confidence_defaults_to_half(self, config, price_data):
        labeler = MetaLabeler(config)
        feats = labeler._extract_features(
            PrimarySignal(timestamp=price_data.index[30], ticker="T",
                          signal=SignalType.BUY, confidence=None),
            price_data,
        )
        assert feats["signal_confidence"] == 0.5

    def test_nan_confidence_dropped_loudly(self, config, price_data, caplog):
        labeler = MetaLabeler(config)
        with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
            feats = labeler._extract_features(
                PrimarySignal(timestamp=price_data.index[30], ticker="T",
                              signal=SignalType.BUY, confidence=float("nan")),
                price_data,
            )
        assert "signal_confidence" not in feats
        assert any("signal_confidence" in r.getMessage() for r in caplog.records)


class TestFitFiniteGuards:
    def test_fit_drops_single_degenerate_sample_loudly(self, config, price_data, caplog):
        signals, outcomes = _make_training_set(price_data)
        data = price_data.copy()
        data.loc[data.index[-1], "close"] = np.nan
        signals[7] = PrimarySignal(
            timestamp=data.index[-1], ticker="T",
            signal=SignalType.BUY, confidence=0.6,
        )

        labeler = MetaLabeler(config)
        with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
            labeler.fit(signals, data, outcomes)

        assert labeler.is_fitted, "119/120 healthy samples must still train"
        assert any("Dropping 1/120" in r.getMessage() for r in caplog.records)

    def test_fit_with_inf_feature_trains_after_drop(self, config, price_data, caplog):
        """Pre-fix this raised ValueError on the whole call (all-or-nothing).

        The zero tick sits at the second-to-last row so that only the moved
        signal's window (ending at the last row) contains it; all other
        training windows stay healthy.
        """
        signals, outcomes = _make_training_set(price_data)
        data = price_data.copy()
        data.loc[data.index[-2], "close"] = 0.0
        signals[7] = PrimarySignal(
            timestamp=data.index[-1], ticker="T",
            signal=SignalType.BUY, confidence=0.6,
        )

        labeler = MetaLabeler(config)
        with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
            labeler.fit(signals, data, outcomes)
        assert labeler.is_fitted
        assert any("Dropping 1/120" in r.getMessage() for r in caplog.records)


class TestPredictFiniteGuards:
    def _fit_labeler(self, config, price_data):
        signals, outcomes = _make_training_set(price_data)
        return MetaLabeler(config).fit(signals, price_data, outcomes)

    def test_degenerate_signal_skipped_and_sibling_scored(
        self, config, price_data, healthy_signal, caplog
    ):
        labeler = self._fit_labeler(config, price_data)
        data = price_data.copy()
        ts = data.index[-1]
        data.loc[ts, "close"] = np.nan
        dirty = PrimarySignal(timestamp=ts, ticker="T",
                              signal=SignalType.BUY, confidence=0.8)

        with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
            preds = labeler.predict([dirty, healthy_signal], data)

        assert preds[0].predicted_proba == 0.0, "degenerate signal must be skipped"
        assert 0.0 < preds[1].predicted_proba < 1.0, "healthy sibling must be scored"
        assert len(caplog.records) >= 1, "skip must be loud"

    def test_missing_column_family_skipped_not_zero_filled(
        self, config, price_data, healthy_signal, caplog
    ):
        """Model trained with volume features must not score a signal whose
        price_data has no volume column: a 0-fill would fabricate a reading
        on a coordinate the data never occupied."""
        labeler = self._fit_labeler(config, price_data)
        no_volume = price_data.drop(columns=["volume"])

        with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
            preds = labeler.predict([healthy_signal], no_volume)

        assert preds[0].predicted_proba == 0.0
        assert any("missing features" in r.getMessage() for r in caplog.records)


class TestSizePositionsGuards:
    @pytest.fixture
    def signal(self, price_data):
        return PrimarySignal(timestamp=price_data.index[30], ticker="T",
                             signal=SignalType.BUY, confidence=0.8)

    @pytest.mark.parametrize("bad_proba", [float("nan"), None])
    def test_refuses_missing_proba_loudly(self, config, signal, bad_proba, caplog):
        labeler = MetaLabeler(config)
        label = MetaLabel(signal=signal, features={}, actual_outcome=0,
                          predicted_proba=bad_proba)
        with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
            sized = labeler.size_positions([label])
        assert sized[0].position_size == 0.0
        assert any("Refusing to size" in r.getMessage() for r in caplog.records)

    def test_healthy_proba_sizes_normally(self, config, signal):
        labeler = MetaLabeler(config)
        label = MetaLabel(signal=signal, features={}, actual_outcome=0,
                          predicted_proba=0.9)
        sized = labeler.size_positions([label])
        assert sized[0].position_size > 0.0

    @pytest.mark.parametrize("bad_ratio", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_win_loss_ratio_raises(self, config, signal, bad_ratio):
        labeler = MetaLabeler(config)
        label = MetaLabel(signal=signal, features={}, actual_outcome=0,
                          predicted_proba=0.9)
        with pytest.raises(ValueError, match="avg_win_loss_ratio"):
            labeler.size_positions([label], avg_win_loss_ratio=bad_ratio)
