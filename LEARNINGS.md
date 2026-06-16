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

## 2026-03-17 — Market Regime Detection

**Contexte** : Besoin d'adapter la stratégie aux conditions de marché macro

**Solution** : Module `analysis/regime_detector.py` avec 3 dimensions :
- Volatility regime (high/normal/low) via percentile historique
- Trend regime (trending/mean-reverting/neutral) via ADX
- Correlation regime via matrice de corrélation 60j

**Implémentation** :
- Détection automatique à chaque exécution de daily_run.py
- Recommandations dynamiques : position sizing, stop-loss tightening, trend vs mean-reversion
- Intégration dans le prompt LLM via `format_regime_for_llm()`

**Métriques** :
- ADX > 25 : trending | ADX < 20 : mean-reverting
- Vol percentile > 75% : high vol | < 25% : low vol
- Avg correlation > 0.7 : high correlation (diversification difficile)

**Règle** :
- High vol → conservative sizing + tight stops
- Mean-reverting → favoriser contrarian trades (RSI extremes)
- High correlation → réduire l'exposition équity totale

---

## 2026-04-23 — Timezone-aware datetime comparisons in pandas

**Contexte** : Le backtest engine retournait "No data fetched" alors que yfinance retournait bien des données

**Erreur** : yfinance retourne des DataFrames avec un DatetimeIndex timezone-aware (America/New_York). Le `BacktestEngine` utilisait `datetime.strptime()` qui produit des datetimes naive. La comparaison `df.index >= start_date` lève un `TypeError: can't compare offset-naive and offset-aware datetimes`. Le code fallback utilisait des comparaisons de strings (`strftime("%Y-%m-%d")`) qui filtraient silencieusement toutes les lignes.

**Fix** :
1. Normaliser l'index dans `fetch_historical_data()` : `hist.index.tz_convert("UTC").tz_localize(None)`
2. Utiliser `start`/`end` au lieu de `period` pour les requêtes backtest (yfinance `period` est relatif à aujourd'hui)
3. Remplacer toutes les comparaisons string par des comparaisons datetime directes

**Règle** :
- Toujours normaliser les indices de temps yfinance à naive/UTC avant toute comparaison
- Ne jamais comparer des strings pour filtrer des dates — c'est fragile et silencieux
- Vérifier la plage de dates des données fetchées avant de lancer un backtest

---

## 2026-03-26 — Vérifier les signatures avant d'écrire des tests

**Contexte** : Tentative d'ajout de tests pour decision_analyzer, backtest, evaluation

**Erreur** : Tests écrits basés sur des suppositions de signatures (class MetaLabeling, def generate_report)

**Réalité** : Les vraies signatures étaient différentes (MetaLabeler, generate_comprehensive_report)

**Leçon** : Toujours lire le fichier source avant d'écrire des tests

**Commande** : `grep -n "^def\|^class" fichier.py` pour voir les vraies signatures

**Règle** : Pas de test sans avoir vu la signature réelle de la fonction/classe

---

## 2026-05-11 — LLM API Timeout (api.kimi.com)

**Contexte** : Trois timeouts consécutifs sur l'API Kimi (21:05, 22:32, 22:37 UTC) — Read timed out après 180s

**Impact** : Pipeline daily_run tombe en fallback "hold all positions" — pas de décision de trading possible

**Analyse** :
- Le timeout est côté serveur (pas de réponse HTTP, pas d'erreur 4xx/5xx)
- Probablement temporaire (maintenance, surcharge, ou changement d'endpoint)
- Le fallback "hold" est le comportement correct — mieux vaut ne pas trader que trader sans analyse

**Règle** :
- Si timeout répété 2 jours de suite → investiguer endpoint alternatif ou fallback LLM local (llama.cpp)
- Toujours vérifier la connectivité API avant la session de trading
- Ne jamais forcer une décision sans LLM — le system prompt contient des règles de risk management critiques

---

## 2026-05-11 — NaN Close prices from yfinance (Euronext pre-market)

**Contexte** : AI.PA, SAN.PA, MC.PA retournent une ligne pour 2026-05-11 avec Close = NaN

**Cause** : yfinance retourne une ligne pour "aujourd'hui" même si le marché n'a pas encore clôturé (ou les données ne sont pas encore propagées). Euronext ferme à 17:30 CET, mais à 22:30 UTC les closing prices individuelles ne sont pas toujours disponibles via yfinance.

**Impact** : Tous les indicateurs techniques (RSI, Bollinger, drawdown, daily_return) devenaient NaN car les calculs rolling incluaient la valeur NaN

**Fix** : `calculate_all_indicators()` dans `src/data/indicators.py` — ajout de `df.dropna(subset=['Close'])` avant les calculs

**Règle** :
- Toujours nettoyer les NaN dans les données brutes avant calcul d'indicateurs
- `dropna(subset=['Close'])` est préférable à `ffill()` car une donnée manquante réelle ne doit pas être interpolée silencieusement
- Vérifier la date du dernier point valide pour s'assurer qu'on n'utilise pas des données obsolètes

---

## 2026-06-16 — Test flakiness due to hardcoded dates in `test_decision_memory.py`

**Contexte** : `TestEdgeCases::test_large_numbers` échoue avec `KeyError: 'best_trade'`

**Cause** : `make_record()` utilise une date par défaut fixe (2026-05-10). `get_decision_summary(days=30)` filtre sur les 30 derniers jours. Depuis le 2026-06-16, cette date est hors fenêtre, donc `recent_decisions` est vide et la méthode retourne un dict sans les clés `best_trade`/`worst_trade`.

**Fix** : Passer explicitement une date récente (`date="2026-06-15"`) dans les appels `make_record()` du test concerné.

**Règle** :
- Ne jamais utiliser de dates hardcodées relatives à "aujourd'hui" dans les tests sans les surcharger explicitement
- Toujours paramétriser la date de référence ou utiliser `datetime.now()` dans les helpers de test
- Si un test dépend de la date courante, le rendre explicite et idempotent

---

*Document mis à jour régulièrement avec les apprentissages du live trading.*
