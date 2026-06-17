"""
Market Regime Detection Module

Détecte les régimes de marché pour adapter la stratégie:
- Volatility regime (high/low)
- Trend regime (trending/mean-reverting)
- Correlation regime (diversification vs concentration)
"""

import pandas as pd
import numpy as np
from typing import Literal, Dict, Tuple, Any
from dataclasses import dataclass


@dataclass
class RegimeState:
    """État complet du régime de marché"""
    volatility_regime: Literal["high", "normal", "low"]
    trend_regime: Literal["trending_up", "trending_down", "mean_reverting", "neutral"]
    correlation_regime: Literal["high_correlation", "normal", "low_correlation"]
    volatility_percentile: float  # 0-100
    adx_value: float  # Average Directional Index
    avg_correlation: float  # Moyenne des corrélations inter-assets
    
    def summary(self) -> str:
        """Résumé lisible du régime"""
        return (
            f"Vol: {self.volatility_regime} ({self.volatility_percentile:.0f}th pct), "
            f"Trend: {self.trend_regime} (ADX: {self.adx_value:.1f}), "
            f"Corr: {self.correlation_regime} ({self.avg_correlation:.2f})"
        )


class RegimeDetector:
    """
    Détecteur de régime de marché basé sur:
    - Volatilité relative (percentile historique)
    - ADX pour la force de tendance
    - Matrice de corrélation pour la diversification
    """
    
    def __init__(
        self,
        vol_lookback: int = 20,
        vol_percentile_threshold_high: float = 75.0,
        vol_percentile_threshold_low: float = 25.0,
        adx_period: int = 14,
        adx_trending_threshold: float = 25.0,
        adx_mean_reverting_threshold: float = 20.0,
        correlation_lookback: int = 60,
    ):
        self.vol_lookback = vol_lookback
        self.vol_percentile_threshold_high = vol_percentile_threshold_high
        self.vol_percentile_threshold_low = vol_percentile_threshold_low
        self.adx_period = adx_period
        self.adx_trending_threshold = adx_trending_threshold
        self.adx_mean_reverting_threshold = adx_mean_reverting_threshold
        self.correlation_lookback = correlation_lookback
    
    def detect_volatility_regime(
        self,
        prices: pd.DataFrame,
        historical_window: int = 252
    ) -> Tuple[str, float]:
        """
        Détecte le régime de volatilité basé sur le percentile historique.
        
        Returns:
            (regime, percentile): "high"/"normal"/"low", percentile 0-100
        """
        # Calcul des rendements
        returns = prices.pct_change().dropna()
        
        # Volatilité actuelle (annualisée)
        current_vol = returns.iloc[-self.vol_lookback:].std() * np.sqrt(252)
        
        # Historique des volatilités rolling
        rolling_vols = (
            returns.rolling(self.vol_lookback)
            .std()
            .dropna()
            .iloc[-historical_window:]
            * np.sqrt(252)
        )
        
        # Moyenne des vols actuelles cross-asset
        avg_current_vol = current_vol.mean()
        avg_historical_vols = rolling_vols.mean(axis=1)
        
        # Percentile
        n = len(avg_historical_vols)
        percentile = (
            (avg_historical_vols < avg_current_vol).sum() / n * 100
        ) if n > 0 else 50.0
        
        if percentile >= self.vol_percentile_threshold_high:
            regime = "high"
        elif percentile <= self.vol_percentile_threshold_low:
            regime = "low"
        else:
            regime = "normal"
        
        return regime, percentile
    
    def calculate_adx(
        self,
        prices: pd.DataFrame,
        high: pd.DataFrame = None,
        low: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Calcule l'Average Directional Index (ADX) pour mesurer la force de tendance.
        
        L'ADX ne donne pas la direction, juste la force de la tendance.
        ADX > 25: tendance forte
        ADX < 20: pas de tendance (mean-reverting)
        """
        # Si pas de données high/low, approximer avec close
        if high is None:
            high = prices
        if low is None:
            low = prices
        
        adx_values = {}
        
        for ticker in prices.columns:
            close = prices[ticker]
            ticker_high = high[ticker]
            ticker_low = low[ticker]
            
            # True Range
            tr1 = ticker_high - ticker_low
            tr2 = abs(ticker_high - close.shift(1))
            tr3 = abs(ticker_low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # +DM et -DM
            plus_dm = ticker_high.diff()
            minus_dm = -ticker_low.diff()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            plus_dm[plus_dm <= minus_dm] = 0
            minus_dm[minus_dm <= plus_dm] = 0
            
            # Smoothed averages
            atr = tr.ewm(alpha=1/self.adx_period, min_periods=self.adx_period).mean()
            plus_di = 100 * plus_dm.ewm(alpha=1/self.adx_period, min_periods=self.adx_period).mean() / atr
            minus_di = 100 * minus_dm.ewm(alpha=1/self.adx_period, min_periods=self.adx_period).mean() / atr
            
            # DX et ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.ewm(alpha=1/self.adx_period, min_periods=self.adx_period).mean()
            
            adx_values[ticker] = adx
        
        return pd.DataFrame(adx_values)
    
    def detect_trend_regime(
        self,
        prices: pd.DataFrame,
        adx: pd.DataFrame = None
    ) -> Tuple[str, float]:
        """
        Détecte le régime de tendance basé sur l'ADX.
        
        Returns:
            (regime, avg_adx): régime et valeur ADX moyenne
        """
        if adx is None:
            adx = self.calculate_adx(prices)
        
        current_adx = adx.iloc[-1].mean()
        
        # Déterminer la direction avec les moyennes mobiles
        sma_20 = prices.rolling(20).mean().iloc[-1]
        sma_50 = prices.rolling(50).mean().iloc[-1]
        
        # Score de tendance par asset
        trend_scores = (sma_20 > sma_50).astype(int).mean()
        
        if current_adx >= self.adx_trending_threshold:
            if trend_scores > 0.6:
                regime = "trending_up"
            elif trend_scores < 0.4:
                regime = "trending_down"
            else:
                regime = "trending_mixed"
        elif current_adx <= self.adx_mean_reverting_threshold:
            regime = "mean_reverting"
        else:
            regime = "neutral"
        
        return regime, current_adx
    
    def detect_correlation_regime(
        self,
        prices: pd.DataFrame
    ) -> Tuple[str, float]:
        """
        Détecte le régime de corrélation entre assets.
        
        High correlation = difficult diversification
        Low correlation = good diversification opportunities
        """
        returns = prices.pct_change().dropna()
        
        # Matrice de corrélation sur lookback
        if len(returns) < self.correlation_lookback:
            # Pas assez d'historique
            return "normal", 0.5
        
        corr_matrix = returns.iloc[-self.correlation_lookback:].corr()
        
        # Moyenne des corrélations (hors diagonale)
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        corr_values = corr_matrix.values[mask]
        avg_corr = corr_values.mean() if len(corr_values) > 0 else 0.0
        
        if avg_corr > 0.7:
            regime = "high_correlation"
        elif avg_corr < 0.3:
            regime = "low_correlation"
        else:
            regime = "normal"
        
        return regime, avg_corr
    
    def analyze(
        self,
        prices: pd.DataFrame,
        high: pd.DataFrame = None,
        low: pd.DataFrame = None
    ) -> RegimeState:
        """
        Analyse complète du régime de marché.
        
        Args:
            prices: DataFrame avec les prix de clôture (index=date, columns=tickers)
            high: DataFrame avec les prix hauts (optionnel)
            low: DataFrame avec les prix bas (optionnel)
        
        Returns:
            RegimeState avec tous les régimes détectés
        """
        # Volatilité
        vol_regime, vol_pct = self.detect_volatility_regime(prices)
        
        # Tendance
        adx = self.calculate_adx(prices, high, low)
        trend_regime, adx_value = self.detect_trend_regime(prices, adx)
        
        # Corrélation
        corr_regime, avg_corr = self.detect_correlation_regime(prices)
        
        return RegimeState(
            volatility_regime=vol_regime,
            trend_regime=trend_regime,
            correlation_regime=corr_regime,
            volatility_percentile=vol_pct,
            adx_value=adx_value,
            avg_correlation=avg_corr
        )
    
    def get_strategy_recommendation(self, state: RegimeState) -> Dict[str, Any]:
        """
        Génère des recommandations de stratégie basées sur le régime.
        
        Returns:
            Dict avec recommendations pour:
            - position_sizing: "conservative", "normal", "aggressive"
            - stop_loss_tightening: bool
            - mean_reversion_opportunities: bool
            - trend_following: bool
        """
        recommendations = {
            "position_sizing": "normal",
            "stop_loss_tightening": False,
            "mean_reversion_opportunities": False,
            "trend_following": False,
            "reduce_correlated_exposure": False,
        }
        
        # Volatilité
        if state.volatility_regime == "high":
            recommendations["position_sizing"] = "conservative"
            recommendations["stop_loss_tightening"] = True
        elif state.volatility_regime == "low":
            recommendations["position_sizing"] = "aggressive"
        
        # Tendance
        if state.trend_regime in ["trending_up", "trending_down"]:
            recommendations["trend_following"] = True
        elif state.trend_regime == "mean_reverting":
            recommendations["mean_reversion_opportunities"] = True
        
        # Corrélation
        if state.correlation_regime == "high_correlation":
            recommendations["reduce_correlated_exposure"] = True
        
        return recommendations


def format_regime_for_llm(state: RegimeState, recommendations: Dict) -> str:
    """
    Formate l'état du régime pour l'inclusion dans un prompt LLM.
    """
    return f"""
## Market Regime Analysis

**Current Regime:** {state.summary()}

**Strategic Implications:**
- Position Sizing: {recommendations['position_sizing'].upper()}
- Stop-Loss Adjustment: {'TIGHTEN' if recommendations['stop_loss_tightening'] else 'Normal'}
- Mean Reversion Trades: {'ENABLED' if recommendations['mean_reversion_opportunities'] else 'Disabled'}
- Trend Following: {'ENABLED' if recommendations['trend_following'] else 'Disabled'}
- Correlation Risk: {'REDUCE exposure' if recommendations['reduce_correlated_exposure'] else 'Normal diversification'}

**Interpretation:**
{'Volatility is elevated — prioritize capital preservation and tighten risk controls.' if state.volatility_regime == 'high' else 'Volatility is compressed — opportunities for larger positions with tight stops.' if state.volatility_regime == 'low' else 'Volatility is within normal ranges.'}
{'Markets are trending strongly — momentum strategies favored over contrarian.' if 'trending' in state.trend_regime else 'Markets are range-bound — mean reversion and volatility compression trades favored.' if state.trend_regime == 'mean_reverting' else 'Trend strength is neutral.'}
{'Assets are highly correlated — diversification benefits are limited.' if state.correlation_regime == 'high_correlation' else 'Assets show low correlation — good diversification opportunities.' if state.correlation_regime == 'low_correlation' else 'Correlation levels are normal.'}
"""
