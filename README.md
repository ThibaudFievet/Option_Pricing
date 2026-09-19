# Option Pricing & Greeks Engine — Across Vanilla & Structured Products

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![UI Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

> **Moteur d'évaluation de produits dérivés, d'analyse des sensibilités (Grecques) et de modélisation financière, progressant des contrats vanilles jusqu'aux produits structurés.**

---

## Contexte du Projet & Démarche Personnelle

### Motivation
En faisant l'inventaire de mon profil GitHub public, un constat s'est imposé : bien que j'aie développé de nombreux projets quantitatifs et d'analyse financière, la quasi-totalité d'entre eux demeuraient en dépôts privés, en grande partie parce qu'ils avaient été générés ou accélérés massivement avec l'intelligence artificielle. Aucun projet en accès libre ne mettait concrètement en valeur ma **technicité propre en finance de marché et ma compréhension des produits dérivés.

D'un point de vue strictement personnel et pratique, ce projet n'a pas vocation immédiate à alimenter une stratégie de compte propre : n'ayant pas encore eu 21 ans (l'âge légal et réglementaire requis par la plupart des courtiers pour négocier les options sur marge et dérivés complexes), je ne négociais pas ces instruments en direct sur les marchés.

La finalité de ce dépôt est donc 100 % académique, démonstrative et orientée apprentissage :
1. **Démontrer ma technicité** sur les modèles stochastiques, la théorie de l'évaluation neutre au risque et la gestion des sensibilités de portefeuille.
2. **Apprendre par la pratique** : aller bien plus loin que le cadre théorique et des sujets traités dispensés en cours magistraux à l'université en implémentant moi-même les modèles, en affrontant les cas limites numériques et en découvrant les subtilités algorithmiques.

---

## Philosophie de Conception

Ce projet applique une frontière entre la logique financière et l'interface utilisateur.

```text
Option_Pricing/
├── core/   # [100 % HUMAIN] Moteur de calcul & modèles financiers
└── app/    # [100 % IA]     Interface graphique Streamlit
```

### Le dossier `core/` :
* **Conception manuelle intégrale** : Chaque fonction mathématique, chaque classe de contrat et chaque algorithme de sensibilité a été réfléchi, architecturé et tapé par mes soins, sans génération de code par IA.
* **Documentation & Commentaires approfondis** : Le code est volontairement jalonné de commentaires détaillés expliquant l'intuition économique, les hypothèses sous-jacentes et les remarques personnelles que je me suis faites lors du développement.
* **Alignement avec mon diplôme** : Je suis parfaitement lucide sur le fait qu'une IA code plus vite et mieux. Cependant, pour les compétences fondamentales que je suis censé maîtriser au sortir de mes études, il est crucial de conserver des projets où l'effort intellectuel et la validation mathématique proviennent exclusivement de mon raisonnement.

### Le dossier `app/` :
Je n'ai jamais appris le développement web ou le frontend à l'école, je ne l'ai pas appris par moi-même, et je n'ai aucune vocation ni envie de devenir développeur web. 
L'interface graphique est donc réalisée avec l'assistance de l'IA. Elle illustre ma capacité à prompter efficacement et à piloter un agent pour concevoir une application fonctionnelle et élégante. J'ai quand même assez de notions de développement web pour choisir le framework qui me semble le plus adapté pour ce genre de projet : **Streamlit** pour éviter d'inutiles couches de complexité (architectures séparées backend/frontend, Uvicorn, FastAPI, React, etc.). L'application Streamlit agit comme un simple client qui consomme le package `core/` sans jamais recalculer de logique financière de son côté.

---

## Progression au fur et à mesure

Le projet est actuellement en cours de développement. La complexité des modèles et des sous-jacents traités est conçue pour progresser par paliers successifs.

### Étape 1 — Fondations Vanilles Européennes (Actuel)
* Formalisation orientée objet des contrats vanilles ([instruments.py](core/instruments.py)).
* Modèle de **Black-Scholes-Merton analytique** avec rendement de dividende continu `q` ([black_and_scholes.py](core/black_and_scholes.py)) :
  * Calcul des métriques intermédiaires :
    * `d1 = (ln(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))`
    * `d2 = d1 - sigma * sqrt(T)`
  * Prix exacts fermés pour Calls et Puts avec gestion des cas d'expiration (`T = 0`) et de volatilité nulle.
* Moteur de calcul des sensibilités (**Grecques**) sous deux approches complémentaires ([greeks.py](core/greeks.py)) :
  * Méthode analytique exacte : formules fermées de Black-Scholes-Merton pour Delta, Gamma, Vega, Theta et Rho.
  * Méthode numérique par différences finies centrales (bump-and-revalue) : approximation locale par décalages calibrés (`dS`, `dsigma`, `dT`, `dr`).
