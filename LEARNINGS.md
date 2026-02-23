# LEARNINGS.md — Almost Surely Profitable

Leçons apprises du projet de trading LLM-powered.

---

## 2026-02-23 — Implémentation du CVaR

**Contexte** : Besoin de métriques de risque quantitatives pour le LLM

**Solution** : Module `risk/cvar.py` avec calcul du Conditional Value at Risk

**Implémentation** :
- CVaR 95% : perte attendue si on dépasse le VaR 95%
- VaR 95% : perte maximale avec 95% de confiance
- Tail risk metrics : skewness, kurtosis, Sortino ratio
- Intégration dans `daily_run.py` pour enrichir le contexte LLM

**Formule** : CVaR_α = E[X | X ≤ VaR_α]

**Règle** :
- CVaR > VaR (toujours — c'est la moyenne des queues)
- Si CVaR 95% = 2% → en cas de perte extrême, attendre -2%
- Utiliser le CVaR pour ajuster le cash buffer (plus CVaR élevé → plus de cash)

---

## 2026-02-20 — Partial Profit Taking sur momentum extremes

**Contexte** : MC.PA (LVMH) +4.78% intraday après 24h de détention

**Décision** : Vente de 50% à +4.78%, puis 50% restant au close

**Résultat** : +€26 realized P&L sur €544 investi (+4.8%)

**Leçon** : Quand un mouvement atteint 2+ sigma (95e percentile historique), prendre des profits partiels est prudent. Conserver de l'exposition pour la suite mais sécuriser les gains.

**Règle** : 
- Si mouvement > 4% en une séance sur position récente (< 1 semaine) → vendre 50%
- Laisser courir le reste avec stop-loss au breakeven
- Jamais regretter les profits pris trop tôt

---

## 2026-02-20 — Cash buffer comme alpha

**Observation** : Cash à 56.7% après rebalancing

**Leçon** : Un cash élevé n'est pas de l'inaction, c'est de l'optionnalité. En période de volatilité élevée, la capacité à acheter les dips est un avantage compétitif.

**Règle** : Maintenir 30-50% de cash en période d'incertitude élevée (VIX > 20, correlations cassées).

---

## 2026-02-19 — Diversification sectorielle vs concentration

**Observation** : Positions initiales trop dispersées (6 actifs)

**Leçon** : La diversification excessive dilue les gains. Mieux vaut 3-4 positions fortes avec conviction qu'une dizaine de demi-mesures.

**Règle** : Maximum 5 positions ouvertes simultanément. Concentration sur les meilleures opportunités.

---

## 2026-02-17 — System prompt et behavioral bias

**Observation** : Le LLM applique bien les principes de prospect theory

**Leçon** : Le system prompt avec loss aversion (λ = 2.25) et référence points fonctionne. Le LLM évite les pertes et sécurise les gains plus vite qu'un algorithme classique.

**Règle** : Continuer à affiner les paramètres CPT (Cumulative Prospect Theory) dans le system prompt.

---

## 2026-02-17 — Monitoring intraday

**Observation** : 4 alertes MC.PA déclenchées dans la journée

**Leçon** : Le monitoring toutes les 2h est suffisant pour capturer les mouvements majeurs sans overtrader.

**Règle** : Garder le monitoring 2h, mais ne réagir que sur des mouvements > 3% ou breakouts techniques.

---

## Patterns identifiés

### Entry signals qui fonctionnent
- RSI < 40 + Bollinger < 0.3 (mean reversion)
- Drawdown > 15% sur blue-chip (value play)
- Volatilité < 30% annualisée (stabilité)

### Exit signals qui fonctionnent  
- RSI > 70 + Bollinger > 0.9 (overbought)
- Mouvement > 4% intraday (profit taking)
- Drawdown position > 5% (stop loss)

### Ce qui ne fonctionne pas
- Chaser les breakouts après +3% de move
- Ignorer les signaux de volatilité extrême (> 100%)
- Sous-estimer les correlations en crise

---

## Métriques de suivi

| Métrique | Valeur cible | Actuelle |
|----------|--------------|----------|
| Sharpe ratio | > 1.0 | ? |
| Max drawdown | < 10% | ? |
| Win rate | > 55% | ? |
| Cash moyen | 30-50% | 45% |
| Positions max | 5 | 3 |

---

*Document mis à jour régulièrement avec les apprentissages du live trading.*
