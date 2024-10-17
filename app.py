import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from stats import *
from load import load_data
from sklearn.ensemble import IsolationForest

df_aapl = load_data()

# Interface Streamlit
st.title('Analyse des Outliers dans la Volatilité Implicite')

important_cols = ['Strike','Last','Mark','Bid','Bid_size','Ask','Ask_size','Volume_option','Open_interest','Implied_volatility','Delta','Gamma','Theta','Vega','Rho']

# Appliquer Box-Cox et obtenir les outliers
method = st.selectbox("Sélectionnez le méthode pour détécter des outliers", ['Statistique (Interquartile Range)', 'Machine Learning (Isolation Forest)'])

# Sélection de la métrique
col_selected = st.selectbox("Sélectionnez la métrique pour analyser les outliers", important_cols)

if method == 'Machine Learning':
    df_transformed, _, _ = boxcox_outliers(df_aapl, col_selected)

    # Predizer anomalias
    X = df_transformed[important_cols]
    # Criar e treinar o modelo
    model = IsolationForest(contamination='auto')  # Ajuste a contaminação conforme necessário
    model.fit(X)
    df_transformed['Anomaly'] = model.predict(X)
    outliers = df_transformed[df_transformed['Anomaly'] == -1].drop(columns='Anomaly')
    
else:
    df_transformed, outliers, not_outliers = boxcox_outliers(df_aapl, col_selected)
    
if len(outliers) == 0: 
    st.write(f"Il n'y a pas des outliers pour la métrique {col_selected}")

else:
    # Afficher les outliers
    st.subheader(f'Outliers identifiés pour la métrique {col_selected}')
    st.write(outliers)

    # Sélection d'un contrat spécifique
    contract_id_selected = st.selectbox("Sélectionnez le contrat à analyser", outliers['Contractid'].unique())

    st.markdown(f"## Analyse de l'Outlier pour le Contrat `{contract_id_selected}`")

    # Sélectionner les features numériques importantes pour l'analyse
    X = df_aapl[df_aapl['Contractid'] != contract_id_selected][important_cols].drop(columns=col_selected)
    y = df_aapl[df_aapl['Contractid'] != contract_id_selected][col_selected]

    # Se concentrer sur le contrat spécifique
    contract_data = df_aapl[df_aapl['Contractid'] == contract_id_selected]

    # Prédire la volatilité implicite pour le contrat spécifique
    X_contract = contract_data[important_cols].drop(columns=col_selected)
    y_contract = contract_data[col_selected]

    # Calculer la moyenne, l'écart-type et les Z-scores
    mean_features = X.mean()
    std_features = X.std()
    z_scores = (X_contract.values[0] - mean_features) / std_features
    z_scores_df = pd.DataFrame({'Feature': X_contract.columns, 'Z-Score': z_scores, 'Mean': mean_features, 'Observed Value': X_contract.iloc[0]})
    z_scores_df['Absolute Z-Score'] = z_scores_df['Z-Score'].abs()

    # Afficher les features les plus éloignées de la moyenne
    st.subheader(f'Synthèse de la sensibilité pour le contrat')
    st.write(z_scores_df.drop(columns='Feature').sort_values(by='Absolute Z-Score', ascending=False))

    # Visualiser le Z-Score dans un graphique
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Z-Score', y='Feature', hue='Feature', data=z_scores_df.sort_values(by='Z-Score', ascending=False), palette='coolwarm', legend=False)
    plt.title(f'Métriques pour le Contrat {contract_id_selected}')
    st.pyplot(plt)

    explanation(z_scores_df, col_selected)