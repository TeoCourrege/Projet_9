# Résultats des tests

Exécution du 27/10/2026 à 18:24 · Python 3.11.14

**92 passés · 0 échoués · 5 ignorés** en 0.7 s

| Module | Périmètre | Passés | Échoués | Ignorés | Durée (s) |
|---|---|---|---|---|---|
| `test_api` | routes, validation, erreurs, sécurité | 24 | 0 | 0 | 0.36 |
| `test_data_pipeline` | collecte, nettoyage, filtres, chunking | 21 | 0 | 0 | 0.09 |
| `test_engine` | recherche, prompt, génération | 16 | 0 | 0 | 0.10 |
| `test_evaluation` | jeu annoté, métriques | 15 | 0 | 0 | 0.02 |
| `test_indexing` | vectorisation, index FAISS, persistance | 16 | 0 | 0 | 0.15 |
| `test_real_index` | intégration : vraies données et vrai index | 0 | 0 | 5 | 0.00 |

Les tests ignorés sont les tests d'intégration : ils demandent un index construit (`scripts/build_index.py`) et s'exécutent automatiquement dès qu'il existe.
