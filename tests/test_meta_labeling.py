"""
Comprehensive tests for meta-labeling module.

Covers:
- MetaLabelingConfig defaults and factory methods
- Feature extraction with various data shapes and columns
- Model training, prediction, and sizing
- Edge cases: empty data, insufficient history, missing columns
- Kelly Criterion mathematical correctness
- Filtering and threshold behavior
- Triple barrier outcome conversion
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import MagicMock

from src.backtest.meta_labeling import (
    SignalType,
    PrimarySignal,
    MetaLabel,
    MetaLabelingConfig,
    MetaLabeler,
    create_meta_labels_from_triple_barrier,
    apply_meta_labeling,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return MetaLabelingConfig()


@pytest.fixture
def conservative_config():
    return MetaLabelingConfig.conservative()


@pytest.fixture
def aggressive_config():
    return MetaLabelingConfig.aggressive()


@pytest.fixture
def sample_price_data():
    """Typical OHLCV DataFrame with indicators (60 rows to allow lookback=20)."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    np.random.seed(42)
    closes = 100 + np.cumsum(np.random.randn(60) * 0.5)
    data = pd.DataFrame({
        "close": closes,
        "volume": np.random.randint(1_000_000, 5_000_000, 60),
        "rsi": 30 + np.random.rand(60) * 40,
        "bb_position": np.random.rand(60),
    }, index=dates)
    return data


@pytest.fixture
def minimal_price_data():
    """Bare-minimum DataFrame with only close prices (60 rows)."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    return pd.DataFrame({"close": np.linspace(100, 110, 60)}, index=dates)


@pytest.fixture
def empty_price_data():
    return pd.DataFrame()


@pytest.fixture
def signal_at_day_15(sample_price_data):
    return PrimarySignal(
        timestamp=sample_price_data.index[25],
        ticker="TEST",
        signal=SignalType.BUY,
        confidence=0.8,
    )


@pytest.fixture
def many_signals(sample_price_data):
    """List of 120 signals for training."""
    signals = []
    for i in range(120):
        ts = sample_price_data.index[min(20 + (i % 35), len(sample_price_data) - 1)]
        st = SignalType.BUY if i % 3 == 0 else SignalType.SELL if i % 3 == 1 else SignalType.HOLD
        signals.append(PrimarySignal(timestamp=ts, ticker="TEST", signal=st, confidence=0.6))
    return signals


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestMetaLabelingConfig:
    def test_default_values(self, config):
        assert config.n_estimators == 100
        assert config.max_depth == 5
        assert config.min_samples_leaf == 50
        assert config.random_state == 42
        assert config.lookback_window == 20
        assert config.min_probability == 0.5
        assert config.max_position_pct == 0.25
        assert config.kelly_fraction == 0.5

    def test_conservative_factory(self, conservative_config):
        assert conservative_config.n_estimators == 200
        assert conservative_config.max_depth == 3
        assert conservative_config.min_samples_leaf == 100
        assert conservative_config.min_probability == 0.6
        assert conservative_config.max_position_pct == 0.15
        assert conservative_config.kelly_fraction == 0.3

    def test_aggressive_factory(self, aggressive_config):
        assert aggressive_config.n_estimators == 100
        assert aggressive_config.max_depth == 10
        assert aggressive_config.min_samples_leaf == 25
        assert aggressive_config.min_probability == 0.45
        assert aggressive_config.max_position_pct == 0.35
        assert aggressive_config.kelly_fraction == 0.7

    def test_independence_of_factories(self):
        c1 = MetaLabelingConfig.conservative()
        c2 = MetaLabelingConfig.aggressive()
        assert c1.min_probability != c2.min_probability
        assert c1.max_position_pct != c2.max_position_pct


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_basic_features(self, config, sample_price_data, signal_at_day_15):
        labeler = MetaLabeler(config)
        features = labeler._extract_features(signal_at_day_15, sample_price_data)

        assert isinstance(features, dict)
        assert len(features) > 0
        assert "returns_mean" in features
        assert "returns_std" in features
        assert "cumulative_return" in features
        assert "price_vs_sma20" in features
        assert "volume_vs_mean" in features
        assert "volume_trend" in features
        assert "rsi" in features
        assert "rsi_trend" in features
        assert "bb_position" in features
        assert "primary_signal" in features
        assert "signal_confidence" in features
        assert "hour" in features
        assert "day_of_week" in features
        assert "is_month_start" in features
        assert "is_month_end" in features

    def test_minimal_data_only_close(self, config, minimal_price_data):
        labeler = MetaLabeler(config)
        signal = PrimarySignal(
            timestamp=minimal_price_data.index[25],
            ticker="TEST",
            signal=SignalType.BUY,
        )
        features = labeler._extract_features(signal, minimal_price_data)
        assert "returns_mean" in features
        assert "rsi" not in features
        assert "bb_position" not in features
        assert "volume_vs_mean" not in features

    def test_empty_dataframe(self, config):
        labeler = MetaLabeler(config)
        signal = PrimarySignal(
            timestamp=pd.Timestamp("2024-01-15"),
            ticker="TEST",
            signal=SignalType.BUY,
        )
        empty_df = pd.DataFrame()
        features = labeler._extract_features(signal, empty_df)
        assert features == {}

    def test_insufficient_data_returns_empty(self, config, sample_price_data):
        labeler = MetaLabeler(config)
        # Timestamp too early — less than lookback_window available
        signal = PrimarySignal(
            timestamp=sample_price_data.index[2],
            ticker="TEST",
            signal=SignalType.BUY,
        )
        features = labeler._extract_features(signal, sample_price_data)
        assert features == {}

    def test_confidence_defaults_to_half(self, config, sample_price_data):
        labeler = MetaLabeler(config)
        signal = PrimarySignal(
            timestamp=sample_price_data.index[25],
            ticker="TEST",
            signal=SignalType.SELL,
            confidence=None,
        )
        features = labeler._extract_features(signal, sample_price_data)
        assert features["signal_confidence"] == 0.5

    def test_primary_signal_value_mapping(self, config, sample_price_data):
        labeler = MetaLabeler(config)
        for st, expected in [(SignalType.BUY, 1), (SignalType.SELL, -1), (SignalType.HOLD, 0)]:
            signal = PrimarySignal(
                timestamp=sample_price_data.index[25],
                ticker="TEST",
                signal=st,
            )
            features = labeler._extract_features(signal, sample_price_data)
            assert features["primary_signal"] == expected

    def test_rsi_column_variants(self, config):
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        for col_name in ["rsi", "rsi_14"]:
            data = pd.DataFrame({
                "close": np.linspace(100, 110, 60),
                col_name: np.linspace(20, 80, 60),
            }, index=dates)
            signal = PrimarySignal(timestamp=dates[25], ticker="T", signal=SignalType.BUY)
            labeler = MetaLabeler(config)
            features = labeler._extract_features(signal, data)
            assert "rsi" in features
            assert features["rsi"] == data[col_name].iloc[25]

    def test_bollinger_column_variants(self, config):
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        for col_name in ["bb_position", "bollinger_position"]:
            data = pd.DataFrame({
                "close": np.linspace(100, 110, 60),
                col_name: np.linspace(0, 1, 60),
            }, index=dates)
            signal = PrimarySignal(timestamp=dates[25], ticker="T", signal=SignalType.BUY)
            labeler = MetaLabeler(config)
            features = labeler._extract_features(signal, data)
            assert "bb_position" in features

    def test_volatility_trend_with_short_returns(self, config):
        """When <20 returns but >=10, older_vol should equal recent_vol."""
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        data = pd.DataFrame({"close": 100 + np.random.randn(60)}, index=dates)
        signal = PrimarySignal(timestamp=dates[25], ticker="T", signal=SignalType.BUY)
        labeler = MetaLabeler(config)
        features = labeler._extract_features(signal, data)
        # With 10 returns, recent_vol = older_vol -> volatility_trend ~ 0
        assert abs(features["volatility_trend"]) < 1e-6


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestFit:
    def test_train_success(self, config, sample_price_data, many_signals):
        labeler = MetaLabeler(config)
        outcomes = [1 if i % 2 == 0 else 0 for i in range(len(many_signals))]
        result = labeler.fit(many_signals, sample_price_data, outcomes)
        assert result is labeler
        assert labeler.is_fitted
        assert len(labeler.metrics) > 0
        assert "accuracy" in labeler.metrics
        assert "precision" in labeler.metrics
        assert "recall" in labeler.metrics
        assert "roc_auc" in labeler.metrics

    def test_mismatched_lengths_raises(self, config, sample_price_data, many_signals):
        labeler = MetaLabeler(config)
        with pytest.raises(ValueError, match="same length"):
            labeler.fit(many_signals, sample_price_data, [1, 0])

    def test_too_few_samples_warns_and_does_not_fit(self, config, sample_price_data):
        labeler = MetaLabeler(config)
        signals = [
            PrimarySignal(timestamp=sample_price_data.index[15], ticker="T", signal=SignalType.BUY)
            for _ in range(5)
        ]
        outcomes = [1, 0, 1, 0, 1]
        result = labeler.fit(signals, sample_price_data, outcomes)
        assert result is labeler
        assert not labeler.is_fitted

    def test_feature_names_populated(self, config, sample_price_data, many_signals):
        labeler = MetaLabeler(config)
        outcomes = [1 if i % 2 == 0 else 0 for i in range(len(many_signals))]
        labeler.fit(many_signals, sample_price_data, outcomes)
        assert len(labeler.feature_names) > 0
        assert all(isinstance(f, str) for f in labeler.feature_names)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class TestPredict:
    def test_unfitted_returns_uniform(self, config, sample_price_data):
        labeler = MetaLabeler(config)
        signals = [
            PrimarySignal(timestamp=sample_price_data.index[15], ticker="T", signal=SignalType.BUY)
        ]
        results = labeler.predict(signals, sample_price_data)
        assert len(results) == 1
        assert results[0].predicted_proba == 0.5
        assert results[0].actual_outcome == 0

    def test_fitted_predicts_proba(self, config, sample_price_data, many_signals):
        labeler = MetaLabeler(config)
        outcomes = [1 if i % 2 == 0 else 0 for i in range(len(many_signals))]
        labeler.fit(many_signals, sample_price_data, outcomes)

        new_signals = [
            PrimarySignal(timestamp=sample_price_data.index[20], ticker="T", signal=SignalType.BUY)
            for _ in range(5)
        ]
        results = labeler.predict(new_signals, sample_price_data)
        assert len(results) == 5
        for r in results:
            assert 0.0 <= r.predicted_proba <= 1.0
            assert r.actual_outcome == 0

    def test_insufficient_data_gets_zero_proba(self, config, sample_price_data):
        labeler = MetaLabeler(config)
        # Fitted with dummy data to bypass unfitted path
        labeler.is_fitted = True
        labeler.feature_names = ["returns_mean"]
        signal = PrimarySignal(
            timestamp=sample_price_data.index[2],  # Too early
            ticker="T",
            signal=SignalType.BUY,
        )
        results = labeler.predict([signal], sample_price_data)
        assert results[0].predicted_proba == 0.0
        assert results[0].features == {}

    def test_empty_signal_list(self, config, sample_price_data):
        labeler = MetaLabeler(config)
        assert labeler.predict([], sample_price_data) == []


# ---------------------------------------------------------------------------
# Position Sizing (Kelly Criterion)
# ---------------------------------------------------------------------------

class TestSizePositions:
    def test_kelly_formula(self, config):
        """Kelly: f = p - (1-p)/b, sized = f * kelly_fraction, capped at max."""
        labeler = MetaLabeler(config)
        meta_labels = [
            MetaLabel(
                signal=PrimarySignal(
                    timestamp=pd.Timestamp("2024-01-15"), ticker="T", signal=SignalType.BUY
                ),
                features={},
                actual_outcome=0,
                predicted_proba=0.7,
            )
        ]
        sized = labeler.size_positions(meta_labels, avg_win_loss_ratio=2.0)
        # Kelly = 0.7 - 0.3/2 = 0.55
        # Sized = 0.55 * 0.5 = 0.275
        # Capped at 0.25
        assert sized[0].position_size == pytest.approx(0.25, abs=1e-9)

    def test_kelly_below_max(self, config):
        labeler = MetaLabeler(config)
        meta_labels = [
            MetaLabel(
                signal=PrimarySignal(
                    timestamp=pd.Timestamp("2024-01-15"), ticker="T", signal=SignalType.BUY
                ),
                features={},
                actual_outcome=0,
                predicted_proba=0.6,
            )
        ]
        sized = labeler.size_positions(meta_labels, avg_win_loss_ratio=2.0)
        # Kelly = 0.6 - 0.4/2 = 0.4
        # Sized = 0.4 * 0.5 = 0.2
        assert sized[0].position_size == pytest.approx(0.2, abs=1e-9)

    def test_below_min_probability_gets_zero(self, config):
        labeler = MetaLabeler(config)
        meta_labels = [
            MetaLabel(
                signal=PrimarySignal(
                    timestamp=pd.Timestamp("2024-01-15"), ticker="T", signal=SignalType.BUY
                ),
                features={},
                actual_outcome=0,
                predicted_proba=0.4,  # Below 0.5 threshold
            )
        ]
        sized = labeler.size_positions(meta_labels)
        assert sized[0].position_size == 0.0

    def test_exact_min_probability(self, config):
        labeler = MetaLabeler(config)
        meta_labels = [
            MetaLabel(
                signal=PrimarySignal(
                    timestamp=pd.Timestamp("2024-01-15"), ticker="T", signal=SignalType.BUY
                ),
                features={},
                actual_outcome=0,
                predicted_proba=0.5,
            )
        ]
        sized = labeler.size_positions(meta_labels, avg_win_loss_ratio=1.0)
        # Kelly = 0.5 - 0.5/1 = 0.0 → sized = 0
        assert sized[0].position_size == 0.0

    def test_negative_kelly_clamped_to_zero(self, config):
        labeler = MetaLabeler(config)
        meta_labels = [
            MetaLabel(
                signal=PrimarySignal(
                    timestamp=pd.Timestamp("2024-01-15"), ticker="T", signal=SignalType.BUY
                ),
                features={},
                actual_outcome=0,
                predicted_proba=0.51,
            )
        ]
        sized = labeler.size_positions(meta_labels, avg_win_loss_ratio=1.0)
        # Kelly = 0.51 - 0.49 = 0.02 → sized = 0.01
        assert sized[0].position_size > 0

        meta_labels[0].predicted_proba = 0.49
        sized = labeler.size_positions(meta_labels, avg_win_loss_ratio=1.0)
        # Kelly = 0.49 - 0.51 = -0.02 → clamped to 0
        assert sized[0].position_size == 0.0

    def test_multiple_labels(self, config):
        labeler = MetaLabeler(config)
        meta_labels = [
            MetaLabel(
                signal=PrimarySignal(
                    timestamp=pd.Timestamp("2024-01-15"), ticker="T", signal=SignalType.BUY
                ),
                features={},
                actual_outcome=0,
                predicted_proba=p,
            )
            for p in [0.3, 0.6, 0.9]
        ]
        sized = labeler.size_positions(meta_labels, avg_win_loss_ratio=2.0)
        assert sized[0].position_size == 0.0  # Below threshold
        assert sized[1].position_size > 0
        assert sized[2].position_size > sized[1].position_size


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestFilterSignals:
    def test_default_threshold(self, config):
        labeler = MetaLabeler(config)
        meta_labels = [
            MetaLabel(
                signal=PrimarySignal(
                    timestamp=pd.Timestamp("2024-01-15"), ticker="T", signal=SignalType.BUY
                ),
                features={},
                actual_outcome=0,
                predicted_proba=p,
            )
            for p in [0.4, 0.5, 0.6]
        ]
        filtered = labeler.filter_signals(meta_labels)
        assert len(filtered) == 2  # 0.5 and 0.6
        assert all(l.predicted_proba >= 0.5 for l in filtered)

    def test_custom_threshold(self, config):
        labeler = MetaLabeler(config)
        meta_labels = [
            MetaLabel(
                signal=PrimarySignal(
                    timestamp=pd.Timestamp("2024-01-15"), ticker="T", signal=SignalType.BUY
                ),
                features={},
                actual_outcome=0,
                predicted_proba=p,
            )
            for p in [0.4, 0.5, 0.6]
        ]
        filtered = labeler.filter_signals(meta_labels, min_probability=0.55)
        assert len(filtered) == 1
        assert filtered[0].predicted_proba == 0.6

    def test_empty_list(self, config):
        labeler = MetaLabeler(config)
        assert labeler.filter_signals([]) == []


# ---------------------------------------------------------------------------
# Triple Barrier Outcome Conversion
# ---------------------------------------------------------------------------

class TestCreateMetaLabelsFromTripleBarrier:
    def test_upper_barrier_success(self):
        result = MagicMock()
        result.label = 1
        assert create_meta_labels_from_triple_barrier([], [result]) == [1]

    def test_lower_barrier_failure(self):
        result = MagicMock()
        result.label = -1
        assert create_meta_labels_from_triple_barrier([], [result]) == [0]

    def test_barrier_type_upper(self):
        from src.backtest.triple_barrier import BarrierType
        result = MagicMock(spec=['barrier_type'])
        result.barrier_type = BarrierType.UPPER
        assert create_meta_labels_from_triple_barrier([], [result]) == [1]

    def test_barrier_type_lower(self):
        from src.backtest.triple_barrier import BarrierType
        result = MagicMock(spec=['barrier_type'])
        result.barrier_type = BarrierType.LOWER
        assert create_meta_labels_from_triple_barrier([], [result]) == [0]

    def test_unknown_attribute_defaults_to_zero(self):
        result = MagicMock(spec=[])
        assert create_meta_labels_from_triple_barrier([], [result]) == [0]

    def test_mixed_results(self):
        from src.backtest.triple_barrier import BarrierType
        results = [
            MagicMock(label=1),
            MagicMock(label=-1),
            MagicMock(spec=['barrier_type'], barrier_type=BarrierType.UPPER),
            MagicMock(spec=['barrier_type'], barrier_type=BarrierType.LOWER),
            MagicMock(spec=[]),
        ]
        outcomes = create_meta_labels_from_triple_barrier([], results)
        assert outcomes == [1, 0, 1, 0, 0]

    def test_empty_results(self):
        assert create_meta_labels_from_triple_barrier([], []) == []


# ---------------------------------------------------------------------------
# End-to-end Pipeline
# ---------------------------------------------------------------------------

class TestApplyMetaLabeling:
    def test_full_pipeline(self, sample_price_data, many_signals):
        historical = many_signals[:100]
        new = many_signals[100:]
        outcomes = [1 if i % 2 == 0 else 0 for i in range(len(historical))]

        sized, metrics = apply_meta_labeling(
            primary_signals=historical,
            price_data=sample_price_data,
            historical_outcomes=outcomes,
            new_signals=new,
        )
        assert len(sized) == len(new)
        assert "accuracy" in metrics
        for s in sized:
            assert s.position_size is not None
            assert 0 <= s.position_size <= 0.25

    def test_pipeline_with_insufficient_data(self, minimal_price_data):
        """When all signals have insufficient history, fit warns and returns uniform."""
        signals = [
            PrimarySignal(
                timestamp=minimal_price_data.index[2],
                ticker="T",
                signal=SignalType.BUY,
            )
            for _ in range(5)
        ]
        outcomes = [1, 0, 1, 0, 1]
        sized, metrics = apply_meta_labeling(
            primary_signals=signals,
            price_data=minimal_price_data,
            historical_outcomes=outcomes,
            new_signals=signals[:1],
        )
        # fit returns early (too few samples), predict returns uniform 0.5
        assert sized[0].predicted_proba == 0.5
