"""
Combinatorial Purged Cross-Validation (CPCV)
Implementation based on Lopez de Prado (2018) "Advances in Financial Machine Learning"

CPCV addresses the problem of leakage in financial cross-validation by:
1. Purging: Removing observations between train and test to prevent overlap
2. Embargoing: Additional buffer after test set
3. Combinatorial: Testing all possible combinations of train/test splits
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Tuple, Optional, Generator
import logging
import math

logger = logging.getLogger(__name__)


class PurgedKFold:
    """
    K-Fold cross-validation with purging between train and test sets.
    
    Purging removes observations from the training set that overlap with the test set
    in terms of information (e.g., if using overlapping returns).
    """
    
    def __init__(self, n_splits: int = 5, purge_gap: int = 1):
        """
        Args:
            n_splits: Number of folds
            purge_gap: Number of observations to purge between train and test
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
    
    def split(self, X: pd.DataFrame, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test sets.
        
        Yields:
            (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Define test set
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)
            test_indices = indices[test_start:test_end]
            
            # Define train set with purging
            train_indices = np.concatenate([
                indices[:max(0, test_start - self.purge_gap)],
                indices[min(n_samples, test_end + self.purge_gap):]
            ])
            
            yield train_indices, test_indices


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV)
    
    Generates all combinations of train/test splits with purging and embargo.
    This creates more robust backtests by testing strategies on multiple paths.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_gap: int = 1,
        embargo_pct: float = 0.01
    ):
        """
        Args:
            n_splits: Total number of splits to create
            n_test_splits: Number of splits to use for testing in each combination
            purge_gap: Number of observations to purge between train and test
            embargo_pct: Percentage of test set to embargo after test (prevents leakage)
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        
        # Calculate number of combinations
        self.n_combinations = int(math.comb(n_splits, n_test_splits))
        logger.info(f"CPCV will generate {self.n_combinations} train/test combinations")
    
    def split(
        self,
        X: pd.DataFrame,
        y=None,
        groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray, dict], None, None]:
        """
        Generate all combinatorial splits with purging and embargo.
        
        Yields:
            (train_indices, test_indices, metadata) tuples
            metadata contains: combination_id, train_splits, test_splits
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits
        
        # Create all split indices
        splits = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = min((i + 1) * fold_size, n_samples)
            splits.append((start, end))
        
        # Generate all combinations of test splits
        for combo_id, test_split_indices in enumerate(combinations(range(self.n_splits), self.n_test_splits)):
            # Get test indices
            test_indices_list = []
            for split_idx in test_split_indices:
                start, end = splits[split_idx]
                test_indices_list.append(indices[start:end])
            test_indices = np.concatenate(test_indices_list)
            
            # Calculate embargo
            embargo_size = max(1, int(len(test_indices) * self.embargo_pct))
            
            # Get train indices (all splits not in test, with purging and embargo)
            train_indices_list = []
            for split_idx in range(self.n_splits):
                if split_idx not in test_split_indices:
                    start, end = splits[split_idx]
                    
                    # Check if this split needs purging (adjacent to test splits)
                    needs_purge_before = any(
                        abs(split_idx - test_idx) == 1 and test_idx < split_idx
                        for test_idx in test_split_indices
                    )
                    needs_purge_after = any(
                        abs(split_idx - test_idx) == 1 and test_idx > split_idx
                        for test_idx in test_split_indices
                    )
                    
                    if needs_purge_before:
                        start = min(start + self.purge_gap, end)
                    if needs_purge_after:
                        end = max(end - self.purge_gap - embargo_size, start)
                    
                    if start < end:
                        train_indices_list.append(indices[start:end])
            
            if train_indices_list:
                train_indices = np.concatenate(train_indices_list)
            else:
                train_indices = np.array([], dtype=int)
            
            metadata = {
                'combination_id': combo_id,
                'train_splits': [i for i in range(self.n_splits) if i not in test_split_indices],
                'test_splits': list(test_split_indices),
                'train_size': len(train_indices),
                'test_size': len(test_indices),
                'purge_gap': self.purge_gap,
                'embargo_size': embargo_size
            }
            
            yield train_indices, test_indices, metadata
    
    def get_n_splits(self) -> int:
        """Return the number of splits (combinations)"""
        return self.n_combinations


def apply_purged_cv(
    data: pd.DataFrame,
    features: List[str],
    target: str,
    model,
    cv: CombinatorialPurgedCV,
    sample_weights: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Apply purged cross-validation and return predictions.
    
    Args:
        data: DataFrame with features and target
        features: List of feature column names
        target: Target column name
        model: Scikit-learn compatible model
        cv: Cross-validator (PurgedKFold or CombinatorialPurgedCV)
        sample_weights: Optional sample weights
        
    Returns:
        DataFrame with predictions and fold information
    """
    results = []
    
    X = data[features]
    y = data[target]
    
    for fold_idx, (train_idx, test_idx, metadata) in enumerate(cv.split(X, y)):
        if len(train_idx) == 0 or len(test_idx) == 0:
            logger.warning(f"Skipping empty fold {fold_idx}")
            continue
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit model
        if sample_weights is not None:
            w_train = sample_weights[train_idx]
            model.fit(X_train, y_train, sample_weight=w_train)
        else:
            model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_test)
        
        # Store results
        fold_results = pd.DataFrame({
            'actual': y_test.values,
            'predicted': predictions,
            'fold': fold_idx,
            'combination_id': metadata.get('combination_id', fold_idx)
        }, index=y_test.index)
        
        results.append(fold_results)
    
    return pd.concat(results, axis=0)


def calculate_purged_cv_score(
    results: pd.DataFrame,
    metric_func
) -> dict:
    """
    Calculate cross-validation scores accounting for purged structure.
    
    Args:
        results: DataFrame from apply_purged_cv
        metric_func: Function that takes (y_true, y_pred) and returns score
        
    Returns:
        Dictionary with mean, std, and per-fold scores
    """
    fold_scores = []
    
    for fold in results['fold'].unique():
        fold_data = results[results['fold'] == fold]
        score = metric_func(fold_data['actual'], fold_data['predicted'])
        fold_scores.append(score)
    
    return {
        'mean': np.mean(fold_scores),
        'std': np.std(fold_scores),
        'min': np.min(fold_scores),
        'max': np.max(fold_scores),
        'scores': fold_scores
    }


# Example usage and testing
if __name__ == "__main__":
    # Create sample time series data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'target': np.random.randn(1000)
    }, index=dates)
    
    print("=" * 70)
    print("Combinatorial Purged Cross-Validation Demo")
    print("=" * 70)
    
    # Test PurgedKFold
    print("\n1. Purged K-Fold:")
    pkf = PurgedKFold(n_splits=5, purge_gap=10)
    for i, (train_idx, test_idx) in enumerate(pkf.split(data)):
        print(f"  Fold {i}: train={len(train_idx)}, test={len(test_idx)}")
    
    # Test CombinatorialPurgedCV
    print("\n2. Combinatorial Purged CV:")
    cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, purge_gap=10, embargo_pct=0.02)
    print(f"  Total combinations: {cpcv.get_n_splits()}")
    
    for i, (train_idx, test_idx, meta) in enumerate(cpcv.split(data)):
        if i < 3:  # Show first 3
            print(f"  Combo {meta['combination_id']}: train={meta['train_size']}, test={meta['test_size']}, "
                  f"train_splits={meta['train_splits']}, test_splits={meta['test_splits']}")
    
    print("\n3. CPCV Summary:")
    print(f"  Method prevents data leakage via purging and embargo")
    print(f"  Generates {cpcv.get_n_splits()} independent backtest paths")
    print(f"  More robust than simple K-fold for time series")
