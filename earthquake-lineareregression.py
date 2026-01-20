import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Charger les données
st.title("Prédiction de la Magnitude des Séismes - Régression Linéaire")
uploaded_file = st.file_uploader("E:/ENIAD/S3/maching learning/Projet/earthquakes", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données :")
    st.write(data.head())

    # Colonnes pertinentes
    features = ['latitude', 'longitude', 'depth', 'felt', 'cdi', 'mmi', 'sig', 'nst', 'dmin', 'rms', 'gap', 'distanceKM']
    target = 'magnitude'

    # Vérification des colonnes
    if all(col in data.columns for col in features + [target]):
        # Préparation des données
        X = data[features].fillna(0)
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modèle de Régression Linéaire
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Évaluation
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("### Résultats du modèle :")
        st.write(f"- **MAE** (Mean Absolute Error) : {mae}")
        st.write(f"- **MSE** (Mean Squared Error) : {mse}")
        st.write(f"- **R² Score** : {r2}")

        # Visualisation des résultats
        st.write("### Comparaison des valeurs réelles et prédites")
        results = pd.DataFrame({"Valeur réelle": y_test.values, "Valeur prédite": y_pred})
        st.write(results.head())

        st.line_chart(results)
    else:
        st.error("Votre fichier ne contient pas toutes les colonnes nécessaires !")
