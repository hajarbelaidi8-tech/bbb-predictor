# BBB Predictor — Prédiction de la perméabilité de la barrière hémato-encéphalique

## Présentation du projet

**BBB Predictor** est un projet de Machine Learning appliqué à la découverte de médicaments (*Drug Discovery*). Il vise à prédire la capacité d'une molécule à franchir la barrière hémato-encéphalique (*Blood-Brain Barrier — BBB*) à partir de sa structure chimique.

La barrière hémato-encéphalique protège le cerveau et limite le passage de nombreuses substances. L'estimation de la perméabilité des molécules peut contribuer au filtrage précoce de candidats médicaments destinés au système nerveux central.

Ce projet associe le traitement de données chimiques, l'extraction de caractéristiques moléculaires, la comparaison de modèles de classification et le développement d'une application web interactive.

## Objectifs

- Préparer et valider des structures moléculaires représentées en SMILES.
- Extraire des caractéristiques chimiques avec RDKit.
- Entraîner et comparer plusieurs modèles de classification.
- Évaluer les performances à l'aide de métriques adaptées.
- Développer une interface permettant d'analyser une molécule ou un ensemble de molécules.

## Dataset

Le projet utilise le jeu de données public **BBBP**, consacré à la perméabilité de la barrière hémato-encéphalique.

Selon le rapport du projet :
- Nombre initial de molécules : 2 050.
- Molécules conservées après validation chimique : 2 039.
- Classes prédites : perméable (1) et non perméable (0).

## Méthodologie

### 1. Exploration et préparation des données

- Analyse exploratoire des données (EDA).
- Vérification des structures moléculaires avec RDKit.
- Analyse du déséquilibre entre les classes.
- Préparation des données pour l'entraînement et l'évaluation.

### 2. Feature engineering

Deux familles de caractéristiques moléculaires sont étudiées :

- **Morgan Fingerprints (ECFP4)** : représentation binaire de motifs structuraux moléculaires.
- **Descripteurs physico-chimiques** : notamment la masse moléculaire, la lipophilie (LogP), la surface polaire topologique (TPSA) et les caractéristiques liées aux liaisons hydrogène.

### 3. Modélisation

Trois algorithmes de classification sont comparés :

- Extra Trees Classifier
- Random Forest
- Gradient Boosting

La stratégie d'évaluation comprend une séparation entraînement/test et une validation croisée à cinq plis.

### 4. Évaluation

Les métriques étudiées comprennent :
- Accuracy : proportion de prédictions correctes.
- F1-score : compromis entre précision et rappel.
- AUC-ROC : capacité de discrimination entre les classes.
- Matrice de confusion : répartition des prédictions correctes et des erreurs.

### 5. Application web

L'application BBB Predictor propose, selon le rapport du projet :
- L'analyse d'une molécule à partir de son nom ou de sa structure SMILES.
- L'affichage de caractéristiques moléculaires.
- La prédiction de la perméabilité BBB.
- Le traitement de plusieurs molécules à partir d'un fichier CSV ou Excel.

## Résultats

Les résultats rapportés dans l'étude comprennent un AUC-ROC moyen de **0,918 ± 0,026** pour Extra Trees lors de la validation croisée à cinq plis.

Le rapport présente également une comparaison entre Extra Trees, Random Forest et Gradient Boosting sur un jeu de test.

Les métriques finales doivent être interprétées à partir des résultats effectivement produits par le code d'évaluation.

## Technologies utilisées

- Python
- pandas et NumPy
- scikit-learn
- RDKit
- Streamlit
- Matplotlib et/ou Plotly, selon les visualisations effectivement utilisées

## Installation et exécution

1. Cloner le dépôt :

   `git clone https://github.com/hajarbelaidi8-tech/bbb-predictor.git`

2. Accéder au dossier :

   `cd bbb-predictor`

3. Créer un environnement virtuel :

   `python -m venv .venv`

4. Activer l'environnement sous Windows :

   `.venv\Scripts\activate`

5. Installer les dépendances :

   `pip install -r requirements.txt`

6. Lancer l'application, si elle utilise Streamlit :

   `streamlit run app.py`

Vérifier les dépendances et la commande de lancement dans le code avant utilisation.

## Limites

Ce modèle fournit une prédiction informatique de la perméabilité BBB. Il ne remplace ni les essais expérimentaux ni une validation pharmacologique. Les performances dépendent notamment du dataset, de sa représentativité et du protocole d'évaluation.

## Auteurs

**Hiba Bounaga et Hajar Benlaidi**

Encadrants : Pr. El Habib Ben Lahmar et Pr. Salma Hanoune

Année universitaire : 2025–2026

Projet académique — Machine Learning appliqué à la découverte de médicaments.
