"""
Meta-Labeling for Trading Strategy Optimization.

Implementation of Marcos Lopez de Prado's Meta-Labeling method.

The concept:
1. Primary Model (e.g., LLM agent) generates trading signals (buy/sell)
2. Meta Model predicts the probability of success for each primary signal
3. Position sizing is based on the meta-model's probability

This approach:
- Filters out low-confidence trades from the primary model
- Sizes positions proportionally to expected success probability
- Reduces false positives without sacrificing recall

References:
- Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Ch. 8
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Type of trading signal."""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class PrimarySignal:
    """A signal from the primary model."""
    timestamp: pd.Timestamp
    ticker: str
    signal: SignalType
    confidence: Optional[float] = None  # Primary model confidence if available


@dataclass
class MetaLabel:
    """Result of meta-labeling for a primary signal."""
    signal: PrimarySignal
    features: Dict[str, float]
    actual_outcome: int  # 1 for success, 0 for failure
    predicted_proba: Optional[float] = None  # Meta-model prediction
    position_size: Optional[float] = None  # Recommended position size


@dataclass
class MetaLabelingConfig:
    """Configuration for meta-labeling."""
    # Model parameters
    n_estimators: int = 100
    max_depth: int = 5
    min_samples_leaf: int = 50
    random_state: int = 42
    
    # Feature parameters
    lookback_window: int = 20  # For calculating features
    
    # Position sizing parameters
    min_probability: float = 0.5  # Minimum prob to take a trade
    max_position_pct: float = 0.25  # Maximum position size (25%)
    kelly_fraction: float = 0.5  # Half-Kelly for safety
    
    @classmethod
    def conservative(cls) -> "MetaLabelingConfig":
        """Conservative config: higher thresholds, smaller positions."""
        return cls(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=100,
            min_probability=0.6,
            max_position_pct=0.15,
            kelly_fraction=0.3
        )
    
    @classmethod
    def aggressive(cls) -> "MetaLabelingConfig":
        """Aggressive config: lower thresholds, larger positions."""
        return cls(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=25,
            min_probability=0.45,
            max_position_pct=0.35,
            kelly_fraction=0.7
        )


class MetaLabeler:
    """
    Meta-labeling model for filtering and sizing primary model signals.
    
    Usage:
        1. Train: labeler.fit(primary_signals, price_data, outcomes)
        2. Predict: predictions = labeler.predict(primary_signals, price_data)
        3. Size: sized_trades = labeler.size_positions(predictions)
    """
    
    def __init__(self, config: Optional[MetaLabelingConfig] = None):
        self.config = config or MetaLabelingConfig()
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.metrics: Dict[str, float] = {}
        
    def _extract_features(
        self,
        signal: PrimarySignal,
        price_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Extract features for meta-model at signal timestamp.
        
        Features include:
        - Technical indicators (RSI, volatility, momentum)
        - Primary signal characteristics
        - Temporal features
        - Market regime indicators
        """
        ts = signal.timestamp
        window = self.config.lookback_window
        
        # Guard against empty or trivial DataFrames
        if price_data.empty or len(price_data) == 0:
            logger.warning(f"Empty price data for features at {ts}")
            return {}
        
        # Get data up to signal time
        mask = price_data.index <= ts
        available_data = price_data[mask]
        
        if len(available_data) < window:
            logger.warning(f"Insufficient data for features at {ts}")
            return {}
        
        recent = available_data.tail(window)
        
        features = {}
        
        # Price-based features
        if 'close' in recent.columns:
            closes = recent['close']
            
            # Returns
            returns = closes.pct_change().dropna()
            features['returns_mean'] = returns.mean()
            features['returns_std'] = returns.std()
            features['returns_skew'] = returns.skew()
            features['cumulative_return'] = (closes.iloc[-1] / closes.iloc[0]) - 1
            
            # Trend
            features['price_vs_sma20'] = (closes.iloc[-1] / closes.mean()) - 1
            
            # Volatility regime
            if len(returns) >= 10:
                recent_vol = returns.tail(10).std()
                older_vol = returns.head(10).std() if len(returns) >= 20 else recent_vol
                features['volatility_trend'] = recent_vol / (older_vol + 1e-8) - 1
            else:
                features['volatility_trend'] = 0
        
        # Volume features
        if 'volume' in recent.columns:
            volumes = recent['volume']
            features['volume_vs_mean'] = (volumes.iloc[-1] / volumes.mean()) - 1
            features['volume_trend'] = volumes.tail(5).mean() / (volumes.head(5).mean() + 1e-8) - 1
        
        # Technical indicators if available
        for col in ['rsi', 'rsi_14']:
            if col in recent.columns:
                features['rsi'] = recent[col].iloc[-1]
                features['rsi_trend'] = recent[col].tail(5).mean() - recent[col].head(5).mean()
                break
        
        # Bollinger position
        for col in ['bb_position', 'bollinger_position']:
            if col in recent.columns:
                features['bb_position'] = recent[col].iloc[-1]
                break
        
        # Primary signal features
        features['primary_signal'] = signal.signal.value
        features['signal_confidence'] = signal.confidence or 0.5
        
        # Temporal features
        features['hour'] = ts.hour
        features['day_of_week'] = ts.dayofweek
        features['is_month_start'] = int(ts.is_month_start)
        features['is_month_end'] = int(ts.is_month_end)
        
        return features
    
    def fit(
        self,
        signals: List[PrimarySignal],
        price_data: pd.DataFrame,
        outcomes: List[int]
    ) -> "MetaLabeler":
        """
        Train the meta-labeling model.
        
        Args:
            signals: List of primary model signals
            price_data: OHLCV data with indicators
            outcomes: Binary outcomes (1=success, 0=failure) for each signal
        
        Returns:
            self (fitted model)
        """
        if len(signals) != len(outcomes):
            raise ValueError("signals and outcomes must have same length")
        
        logger.info(f"Training meta-labeler on {len(signals)} signals")
        
        # Extract features for all signals
        X_list = []
        y_list = []
        valid_indices = []
        
        for i, (signal, outcome) in enumerate(zip(signals, outcomes)):
            features = self._extract_features(signal, price_data)
            if features:
                X_list.append(features)
                y_list.append(outcome)
                valid_indices.append(i)
        
        if len(X_list) < 100:
            logger.warning(f"Too few valid samples: {len(X_list)}. Need at least 100.")
            return self
        
        # Convert to DataFrame for consistent column ordering
        X_df = pd.DataFrame(X_list)
        self.feature_names = list(X_df.columns)
        X = X_df.values
        y = np.array(y_list)
        
        logger.info(f"Features: {self.feature_names}")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        # Train/test split (time-based would be better, but random is OK for first pass)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.config.random_state, stratify=y
        )
        
        # Fit model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Calculate metrics
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        logger.info(f"Meta-model metrics: {self.metrics}")
        
        # Feature importance
        importances = dict(zip(self.feature_names, self.model.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"Top 5 features: {top_features}")
        
        return self
    
    def predict(
        self,
        signals: List[PrimarySignal],
        price_data: pd.DataFrame
    ) -> List[MetaLabel]:
        """
        Predict success probability for new signals.
        
        Args:
            signals: List of primary signals to evaluate
            price_data: Current OHLCV data
        
        Returns:
            List of MetaLabel with predicted probabilities
        """
        if not self.is_fitted:
            logger.warning("Meta-model not fitted. Returning uniform probabilities.")
            return [
                MetaLabel(
                    signal=signal,
                    features=self._extract_features(signal, price_data),
                    actual_outcome=0,
                    predicted_proba=0.5
                )
                for signal in signals
            ]
        
        results = []
        
        for signal in signals:
            features = self._extract_features(signal, price_data)
            
            if not features:
                # Insufficient data, skip
                results.append(MetaLabel(
                    signal=signal,
                    features={},
                    actual_outcome=0,
                    predicted_proba=0.0
                ))
                continue
            
            # Ensure consistent feature ordering
            X = np.array([[features.get(f, 0) for f in self.feature_names]])
            proba = self.model.predict_proba(X)[0, 1]
            
            results.append(MetaLabel(
                signal=signal,
                features=features,
                actual_outcome=0,  # Unknown at prediction time
                predicted_proba=proba
            ))
        
        return results
    
    def size_positions(
        self,
        meta_labels: List[MetaLabel],
        avg_win_loss_ratio: float = 1.5
    ) -> List[MetaLabel]:
        """
        Calculate position sizes based on Kelly Criterion.
        
        Args:
            meta_labels: Predictions with probabilities
            avg_win_loss_ratio: Average win/loss ratio (for Kelly calculation)
        
        Returns:
            MetaLabels with position_size field populated
        """
        sized_labels = []
        
        for label in meta_labels:
            p = label.predicted_proba
            
            # Filter low-probability trades
            if p < self.config.min_probability:
                label.position_size = 0.0
                sized_labels.append(label)
                continue
            
            # Kelly Criterion: f = p - (1-p)/b
            # where b = win/loss ratio
            b = avg_win_loss_ratio
            kelly = p - (1 - p) / b
            
            # Apply Kelly fraction and max position limit
            position = kelly * self.config.kelly_fraction
            position = min(position, self.config.max_position_pct)
            position = max(0, position)  # No negative sizes
            
            label.position_size = position
            sized_labels.append(label)
        
        return sized_labels
    
    def filter_signals(
        self,
        meta_labels: List[MetaLabel],
        min_probability: Optional[float] = None
    ) -> List[MetaLabel]:
        """
        Filter signals based on predicted probability threshold.
        
        Args:
            meta_labels: Predictions with probabilities
            min_probability: Override threshold (uses config if None)
        
        Returns:
            Filtered list of high-confidence signals
        """
        threshold = min_probability or self.config.min_probability
        filtered = [label for label in meta_labels if label.predicted_proba >= threshold]
        logger.info(f"Filtered {len(meta_labels)} signals to {len(filtered)} (threshold={threshold})")
        return filtered


def create_meta_labels_from_triple_barrier(
    signals: List[PrimarySignal],
    triple_barrier_results: List[Any]
) -> List[int]:
    """
    Convert triple barrier results to binary outcomes for meta-labeling.
    
    Success (1): Upper barrier touched (profit)
    Failure (0): Lower barrier (stop loss) or vertical barrier (time exit)
    
    Args:
        signals: Primary signals
        triple_barrier_results: Results from TripleBarrierLabeler
    
    Returns:
        Binary outcomes list
    """
    outcomes = []
    
    for result in triple_barrier_results:
        # Success = upper barrier (profit taking)
        if hasattr(result, 'label'):
            outcome = 1 if result.label == 1 else 0
        elif hasattr(result, 'barrier_type'):
            from .triple_barrier import BarrierType
            outcome = 1 if result.barrier_type == BarrierType.UPPER else 0
        else:
            outcome = 0
        
        outcomes.append(outcome)
    
    return outcomes


# Convenience function for quick usage
def apply_meta_labeling(
    primary_signals: List[PrimarySignal],
    price_data: pd.DataFrame,
    historical_outcomes: List[int],
    new_signals: List[PrimarySignal],
    config: Optional[MetaLabelingConfig] = None
) -> Tuple[List[MetaLabel], Dict[str, float]]:
    """
    End-to-end meta-labeling pipeline.
    
    Args:
        primary_signals: Historical signals for training
        price_data: OHLCV data
        historical_outcomes: Binary outcomes for training
        new_signals: Signals to predict on
        config: Optional configuration
    
    Returns:
        (sized_labels, metrics)
    """
    labeler = MetaLabeler(config)
    
    # Train
    labeler.fit(primary_signals, price_data, historical_outcomes)
    
    # Predict
    predictions = labeler.predict(new_signals, price_data)
    
    # Size
    sized = labeler.size_positions(predictions)
    
    return sized, labeler.metrics
