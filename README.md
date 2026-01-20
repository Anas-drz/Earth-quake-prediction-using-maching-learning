# Earthquake Magnitude Prediction

Projet de Machine Learning pour la prédiction de la magnitude des tremblements de terre à partir de données sismiques.

## Fichiers du dépôt

- `earthquakes.csv` : dataset utilisé  
- `earthquake-linearegression.py` : modèle de régression linéaire  
- `earthquake-randomforest.py` : modèle Random Forest  
- `Rapport.pdf` : rapport détaillé du projet  
- `README.md` : description du projet  

## Objectif

Prédire la **magnitude des séismes** à partir de caractéristiques géophysiques  
(longitude, latitude, profondeur, signaux sismiques, etc.) en utilisant des modèles de régression.

## Méthodologie

- Nettoyage des données et suppression des colonnes inutiles  
- Gestion des valeurs manquantes  
- Détection et traitement des valeurs aberrantes (méthode IQR)  
- Encodage ordinal de la variable `alert`  
- Entraînement de deux modèles :
  - Régression Linéaire  
  - Random Forest Regressor  
- Évaluation avec :
  - MAE (Mean Absolute Error)  
  - MSE (Mean Squared Error)  

## Prérequis

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
## Résultats

- Le modèle **Random Forest** donne les meilleures performances après optimisation des hyperparamètres avec **GridSearchCV**.
- Comparaison des performances basée sur :
  - MAE (Mean Absolute Error)  
  - MSE (Mean Squared Error)  
- Amélioration observée entre les modèles avant et après optimisation.
