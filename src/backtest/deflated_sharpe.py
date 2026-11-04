"""
Deflated Sharpe Ratio (DSR) implementation.

Based on Lopez de Prado, M. (2018) "Advances in Financial Machine Learning", Chapter 9,
and the delta-method variance of the Sharpe ratio estimator (Lo, 2002; Mertens, 2002):

    Var(SR_hat) ≈ (1/n) · [1 − γ₃·SR + ((γ₄ − 1)/4)·SR²]

where SR is the *per-period* Sharpe ratio (same frequency as the observations),
γ₃ is the skewness and γ₄ is the (non-excess) kurtosis. For normal returns
(γ₃=0, γ₄=3) this reduces to the classic (1 + SR²/2)/n.

The Deflated Sharpe Ratio adjusts the standard Sharpe ratio for:
1. Multiple testing (when many strategies are tested, some will appear significant by chance)
2. Non-normality of returns (skewness and kurtosis)

This helps prevent false discoveries in strategy selection.
"""

import logging
import math

import numpy as np
from scipy import stats
from typing import Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SharpeMetrics:
    """Container for Sharpe ratio metrics."""
    sharpe_ratio: float
    deflated_sharpe: float
    p_value: float
    is_significant: bool
    skewness: float
    kurtosis: float
    n_trials: int
    n_observations: int
    annualization_factor: float


class DeflatedSharpeRatio:
    """
    Calculate the Deflated Sharpe Ratio to account for multiple testing
    and non-normality in returns.
    
    From Lopez de Prado (2018):
    "The Deflated Sharpe Ratio (DSR) corrects for two major flaws in the
    standard Sharpe ratio: non-normality and multiple testing."
    
    Example:
        >>> returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
        >>> dsr = DeflatedSharpeRatio(n_trials=100)  # 100 strategies tested
        >>> metrics = dsr.calculate(returns)
        >>> print(f"SR: {metrics.sharpe_ratio:.3f}, DSR: {metrics.deflated_sharpe:.3f}")
    """
    
    def __init__(
        self,
        n_trials: int = 1,
        annualization_factor: float = 252.0,
        significance_level: float = 0.05
    ):
        """
        Initialize DSR calculator.
        
        Args:
            n_trials: Number of strategies tested (for multiple testing correction)
            annualization_factor: Factor to annualize returns (252 for daily, 12 for monthly)
            significance_level: Threshold for statistical significance

        Raises:
            ValueError: If annualization_factor is not a positive finite number.
        """
        if not math.isfinite(annualization_factor) or annualization_factor <= 0:
            raise ValueError(
                f"annualization_factor must be a positive finite number, "
                f"got {annualization_factor!r}"
            )
        self.n_trials = max(1, n_trials)
        self.annualization_factor = annualization_factor
        self.significance_level = significance_level
    
    def calculate(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        n_trials: Optional[int] = None
    ) -> SharpeMetrics:
        """
        Calculate Sharpe ratio and Deflated Sharpe Ratio.

        Non-finite returns (NaN, ±inf) are dropped at ingress with a loud log:
        they are elements of a collection, so siblings survive (Lipschitz
        doctrine), and the observation count used in the variance formula is
        the count of *valid* observations.

        Scale convention: the variance / DSR / p-value math operates on the
        *per-period* Sharpe ratio (same frequency as ``returns``), per
        Lo (2002) / Lopez de Prado (2018). Annualization is applied only to
        the reported ``sharpe_ratio`` and ``deflated_sharpe`` fields. Mixing
        an annualized SR into the per-period variance formula (or vice versa)
        changes the z-score by orders of magnitude.

        Args:
            returns: Array of returns (daily or monthly)
            risk_free_rate: Risk-free rate (annualized)
            n_trials: Override number of trials (uses instance value if None)

        Returns:
            SharpeMetrics containing SR, DSR, and related statistics
        """
        returns = np.asarray(returns, dtype=float).flatten()
        finite_mask = np.isfinite(returns)
        n_dropped = int(np.count_nonzero(~finite_mask))
        if n_dropped:
            logger.warning(
                "Dropped %d non-finite return(s) from DSR input; %d valid "
                "observation(s) used.",
                n_dropped, int(np.count_nonzero(finite_mask)),
            )
        returns = returns[finite_mask]
        n_obs = len(returns)

        if n_obs < 2:
            raise ValueError("At least 2 observations required")

        trials = n_trials if n_trials is not None else self.n_trials

        # Basic Sharpe ratio (annualized for display)
        excess_returns = returns - risk_free_rate / self.annualization_factor
        mean_return = float(np.mean(excess_returns))
        std_return = float(np.std(excess_returns, ddof=1))

        if not math.isfinite(std_return) or not math.isfinite(mean_return) \
                or std_return < 1e-15:
            # Degenerate series (constant, or overflowed moments): no
            # estimable ratio. Guard on finiteness too — NaN comparisons are
            # always False, so a bare `< 1e-15` check lets NaN through.
            sharpe_ratio = 0.0
            sr_period = 0.0
        else:
            sr_period = mean_return / std_return
            sharpe_ratio = sr_period * math.sqrt(self.annualization_factor)

        # Calculate moments for non-normality adjustment
        if n_obs < 3 or std_return < 1e-15 or not math.isfinite(std_return):
            skewness = 0.0
            kurtosis = 3.0
        else:
            skewness = float(stats.skew(returns))
            kurtosis = float(stats.kurtosis(returns, fisher=False))  # Normal = 3.0

        # Variance / DSR / p-value operate on the per-period SR scale.
        deflated_sharpe = self._calculate_dsr(
            sr_period, skewness, kurtosis, trials, n_obs
        ) * math.sqrt(self.annualization_factor)

        p_value = self._calculate_p_value(
            sr_period, skewness, kurtosis, trials, n_obs
        )
        
        is_significant = p_value < self.significance_level
        
        return SharpeMetrics(
            sharpe_ratio=sharpe_ratio,
            deflated_sharpe=deflated_sharpe,
            p_value=p_value,
            is_significant=is_significant,
            skewness=skewness,
            kurtosis=kurtosis,
            n_trials=trials,
            n_observations=n_obs,
            annualization_factor=self.annualization_factor
        )
    
    def _calculate_dsr(
        self,
        sharpe_ratio: float,
        skewness: float,
        kurtosis: float,
        n_trials: int,
        n_obs: int
    ) -> float:
        """
        Calculate the Deflated Sharpe Ratio.
        
        The DSR adjusts the SR for multiple testing and non-normality.
        ``sharpe_ratio`` here is the PER-PERIOD SR (same frequency as the
        observations). All quantities are on that scale.

        Variance of the SR estimator (Lo, 2002; Mertens, 2002;
        Lopez de Prado, 2018, AFML ch. 9):

            Var(SR_hat) ≈ (1/n) · [1 − γ₃·SR + ((γ₄ − 1)/4)·SR²]

        where γ₄ is the NON-excess kurtosis (normal = 3). The coefficient
        (γ₄−1)/4 — not (γ₄−3)/4 — is what reduces to the classic
        (1 + SR²/2)/n for Gaussian returns.
        """
        # Variance of SR under normality
        var_sr = (1 + 0.5 * sharpe_ratio**2) / n_obs

        # Adjust variance for non-normality (Lopez de Prado formula)
        # This accounts for skewness and kurtosis
        var_sr_adjusted = var_sr * (
            1 - skewness * sharpe_ratio +
            ((kurtosis - 1.0) / 4.0) * sharpe_ratio**2
        )
        
        if var_sr_adjusted <= 0:
            var_sr_adjusted = var_sr
        
        # Standard deviation of SR
        std_sr = np.sqrt(var_sr_adjusted)
        
        # Multiple testing correction using Bonferroni approach
        # This is a simplified version; more sophisticated methods exist
        if n_trials > 1:
            # Adjust significance threshold for multiple testing
            # The probability that at least one strategy appears significant by chance
            # increases with the number of trials
            bonferroni_factor = np.log(n_trials)
        else:
            bonferroni_factor = 1.0
        
        # Deflated SR = SR - adjustment for non-normality - adjustment for multiple testing
        # This is a simplified version; see Lopez de Prado for the full formula
        adjustment = std_sr * np.sqrt(2 * np.log(n_trials)) if n_trials > 1 else 0
        
        deflated_sharpe = sharpe_ratio - adjustment
        
        return deflated_sharpe
    
    def _calculate_p_value(
        self,
        sharpe_ratio: float,
        skewness: float,
        kurtosis: float,
        n_trials: int,
        n_obs: int
    ) -> float:
        """
        Calculate the p-value for the Sharpe ratio accounting for multiple testing.

        Returns the probability that the observed SR could occur by chance.
        ``sharpe_ratio`` is the PER-PERIOD SR; the variance formula is
        scale-matched to it (see :meth:`_calculate_dsr`).
        """
        # Standard error of SR
        var_sr = (1 + 0.5 * sharpe_ratio**2) / n_obs
        var_sr_adjusted = var_sr * (
            1 - skewness * sharpe_ratio +
            ((kurtosis - 1.0) / 4.0) * sharpe_ratio**2
        )
        
        if var_sr_adjusted <= 0:
            var_sr_adjusted = var_sr
        
        std_sr = np.sqrt(var_sr_adjusted)
        
        if std_sr == 0:
            return 1.0
        
        # Standard normal test statistic
        z_score = sharpe_ratio / std_sr
        
        # Single-trial p-value
        p_single = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Multiple testing correction (Bonferroni)
        if n_trials > 1:
            # Probability that at least one of n_trials strategies
            # shows significance by chance
            p_corrected = min(1.0, p_single * n_trials)
        else:
            p_corrected = p_single
        
        return p_corrected
    
    def compare_strategies(
        self,
        strategies: List[Tuple[str, np.ndarray]],
        risk_free_rate: float = 0.0
    ) -> List[Tuple[str, SharpeMetrics]]:
        """
        Compare multiple strategies with multiple testing correction.
        
        Args:
            strategies: List of (name, returns_array) tuples
            risk_free_rate: Risk-free rate (annualized)
        
        Returns:
            List of (name, metrics) tuples sorted by DSR
        """
        n_strategies = len(strategies)
        results = []
        
        for name, returns in strategies:
            metrics = self.calculate(
                returns,
                risk_free_rate=risk_free_rate,
                n_trials=n_strategies
            )
            results.append((name, metrics))
        
        # Sort by Deflated Sharpe Ratio (descending)
        results.sort(key=lambda x: x[1].deflated_sharpe, reverse=True)
        
        return results
    
    def false_discovery_rate(
        self,
        p_values: List[float],
        method: str = "benjamini-hochberg"
    ) -> List[Tuple[float, float, bool]]:
        """
        Control False Discovery Rate (FDR) for multiple testing.
        
        Args:
            p_values: List of p-values from strategy tests
            method: FDR control method ("benjamini-hochberg" or "bonferroni")
        
        Returns:
            List of (p_value, q_value, is_significant) tuples
        """
        p_values = np.array(p_values)
        n = len(p_values)
        
        if method == "bonferroni":
            # Conservative: control family-wise error rate
            q_values = np.minimum(p_values * n, 1.0)
            significant = q_values < self.significance_level
        
        elif method == "benjamini-hochberg":
            # Less conservative: control FDR
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            q_values = np.zeros(n)
            for i, p in enumerate(sorted_p):
                q_values[sorted_indices[i]] = min(
                    1.0,
                    p * n / (i + 1)
                )
            
            # Ensure q-values are monotonic
            for i in range(n - 2, -1, -1):
                q_values[sorted_indices[i]] = min(
                    q_values[sorted_indices[i]],
                    q_values[sorted_indices[i + 1]]
                )
            
            significant = q_values < self.significance_level
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return list(zip(p_values, q_values, significant))


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0
) -> float:
    """
    Calculate the Probabilistic Sharpe Ratio (PSR).

    The PSR gives the probability that the observed Sharpe ratio
    is greater than a benchmark Sharpe ratio, accounting for
    non-normality and sample size.

    ``observed_sr`` and ``benchmark_sr`` must be on the SAME scale as the
    underlying observations (per-period), and ``n_observations`` must count
    those observations — the variance formula
    (1 − γ₃·SR + ((γ₄−1)/4)·SR²)/n is scale-matched to that scale
    (Lo, 2002; Lopez de Prado, 2018).

    Args:
        observed_sr: Observed Sharpe ratio (per-period scale)
        benchmark_sr: Benchmark Sharpe ratio to compare against (same scale)
        n_observations: Number of observations
        skewness: Return skewness
        kurtosis: Return kurtosis (normal = 3.0, non-excess)

    Returns:
        Probability that observed SR > benchmark SR

    Raises:
        ValueError: If inputs are non-finite.
    """
    inputs = {
        "observed_sr": observed_sr,
        "benchmark_sr": benchmark_sr,
        "n_observations": n_observations,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }
    for name, value in inputs.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")

    if n_observations < 2:
        return 0.5

    # Standard error of SR
    var_sr = (1 + 0.5 * observed_sr**2) / n_observations

    # Adjust for non-normality: coefficient (γ₄−1)/4 with NON-excess kurtosis,
    # reducing to (1 + SR²/2)/n for Gaussian returns.
    var_sr_adjusted = var_sr * (
        1 - skewness * observed_sr +
        ((kurtosis - 1.0) / 4.0) * observed_sr**2
    )
    
    if var_sr_adjusted <= 0:
        var_sr_adjusted = var_sr
    
    std_sr = np.sqrt(var_sr_adjusted)
    
    if std_sr == 0:
        return 1.0 if observed_sr > benchmark_sr else 0.0
    
    # Z-score for difference
    z_score = (observed_sr - benchmark_sr) / std_sr
    
    # Probability
    psr = stats.norm.cdf(z_score)
    
    return psr


def minimum_track_record_length(
    target_sharpe: float,
    confidence_level: float = 0.95,
    skewness: float = 0.0,
    kurtosis: float = 3.0
) -> int:
    """
    Calculate minimum track record length needed to achieve
    a target Sharpe ratio with given confidence.
    
    From Lopez de Prado: Helps determine if a strategy has
    been tested for sufficient time to be statistically valid.

    ``target_sharpe`` is on the per-period scale of the observations.

    Args:
        target_sharpe: Target Sharpe ratio (per-period scale, non-zero)
        confidence_level: Confidence level (e.g., 0.95)
        skewness: Expected return skewness
        kurtosis: Expected return kurtosis (normal = 3.0, non-excess)

    Returns:
        Minimum number of observations needed

    Raises:
        ValueError: If target_sharpe is zero or any input is non-finite.
            A zero target degenerates the formula (division by SR²): no
            finite track record establishes a Sharpe ratio of exactly zero.
    """
    inputs = {
        "target_sharpe": target_sharpe,
        "confidence_level": confidence_level,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }
    for name, value in inputs.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if target_sharpe == 0:
        raise ValueError("target_sharpe must be non-zero (the formula divides by SR²)")

    z_alpha = stats.norm.ppf(confidence_level)

    # Simplified formula (see Lopez de Prado for exact derivation)
    # This gives approximate minimum sample size
    # Variance adjustment factor: coefficient (γ₄−1)/4 with NON-excess
    # kurtosis, consistent with the SR variance formula elsewhere in this
    # module.
    adjustment = 1 - skewness * target_sharpe + ((kurtosis - 1.0) / 4.0) * target_sharpe**2
    if adjustment <= 0:
        adjustment = 1.0
    
    # Minimum observations
    n_min = int(np.ceil(
        (z_alpha**2 * (1 + 0.5 * target_sharpe**2) * adjustment) / target_sharpe**2
    ))
    
    return max(n_min, 2)
