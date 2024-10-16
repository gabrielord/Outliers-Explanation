import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import requests
import json
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PowerTransformer
import streamlit as st

# Charger les données
def load_data():
    # Charger les données du fichier JSON (adapté à votre exemple)
    with open('./stock_data.json', 'r') as file:
        data_stock = json.load(file)

    with open('./options_data.json', 'r') as file:
        data_opt = json.load(file)

    df_stock = pd.DataFrame.from_dict(data_stock['Time Series (Daily)'], orient='index')
    df_opt = pd.DataFrame.from_dict(data_opt['data'])
    df_stock.reset_index(inplace=True)
    df_stock.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume_stock']
    df_opt.columns = df_opt.columns.map(lambda x: x.capitalize())
    df_opt.rename(columns={'Volume': 'Volume_option'}, inplace=True)

    # Fusionner les données
    df_aapl = df_stock.merge(df_opt, on='Date', how='inner')

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume_stock', 'Strike', 'Last', 'Mark', 'Bid', 'Bid_size', 'Ask', 'Ask_size', 'Volume_option', 'Open_interest', 'Implied_volatility', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
    df_aapl[numeric_cols] = df_aapl[numeric_cols].astype(float)
    
    df_aapl = df_aapl[~df_aapl.isnull().any(axis=1)]

    return df_aapl

# Fonction pour appliquer Box-Cox et calculer les outliers
def boxcox_outliers(df, col):
    # Vérifier s'il y a des valeurs négatives dans la colonne
    if (df[col] < 0).any():
        # Appliquer la transformation Yeo-Johnson (pour gérer les valeurs négatives)
        pt = PowerTransformer(method='yeo-johnson')
        df[f'boxcox_{col}'] = pt.fit_transform(df[[col]])
        print(f'Utilisation de Yeo-Johnson pour la colonne {col}')
    else:
        # Appliquer la transformation Box-Cox à la colonne spécifiée
        df[f'boxcox_{col}'], _ = stats.boxcox(df[col] + 1)  # Ajouter 1 pour éviter les zéros
        print(f'Utilisation de Box-Cox pour la colonne {col}')

    # Calculer l'IQR (Intervalle Interquartile) dans la colonne transformée
    Q1 = df[f'boxcox_{col}'].quantile(0.25)
    Q3 = df[f'boxcox_{col}'].quantile(0.75)
    IQR = Q3 - Q1

    # Identifier les outliers en dehors de l'intervalle [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    outliers_iqr = df[(df[f'boxcox_{col}'] < (Q1 - 1.5 * IQR)) | 
                      (df[f'boxcox_{col}'] > (Q3 + 1.5 * IQR))]

    # Identifier les données qui ne sont pas des outliers
    not_outliers_iqr = df[~((df[f'boxcox_{col}'] < (Q1 - 1.5 * IQR)) | 
                            (df[f'boxcox_{col}'] > (Q3 + 1.5 * IQR)))]

    # Retourner le DataFrame original avec la colonne transformée, les outliers et les non-outliers
    return df, outliers_iqr, not_outliers_iqr

def explanation(z_scores_df, col_selected):
    # Trouver la ligne avec le Z-score le plus élevé en valeur absolue
    max_zscore_row = z_scores_df.loc[z_scores_df['Absolute Z-Score'].idxmax()]

    # Afficher la feature avec le Z-score le plus élevé (en valeur absolue)
    st.subheader(f"**Explication possible de l'outlier dans la cologne {col_selected}:**")
    st.markdown(f"La plus grande différence observée concerne la variable **`{max_zscore_row['Feature']}`**.")

    # Diviser la mise en page en trois colonnes
    col1, col2, col3 = st.columns(3)

    # Afficher les valeurs dans chaque colonne
    with col1:
        st.metric(label="Valeur Observée", value=round(max_zscore_row['Observed Value'], 2))

    with col2:
        st.metric(label="Moyenne Historique", value=round(max_zscore_row['Mean'], 2))

    with col3:
        st.metric(label="Z-Score", value=round(max_zscore_row['Z-Score'], 2))

df_aapl = load_data()

# Interface Streamlit
st.title('Analyse des Outliers dans la Volatilité Implicite')

important_cols = ['Strike','Last','Mark','Bid','Bid_size','Ask','Ask_size','Volume_option','Open_interest','Implied_volatility','Delta','Gamma','Theta','Vega','Rho']

# Sélection de la métrique
col_selected = st.selectbox("Sélectionnez la métrique pour analyser les outliers", important_cols)

# Appliquer Box-Cox et obtenir les outliers
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
    sns.barplot(x='Z-Score', y='Feature', data=z_scores_df.sort_values(by='Z-Score', ascending=False), palette='coolwarm')
    plt.title(f'Métriques pour le Contrat {contract_id_selected}')
    st.pyplot(plt)

    explanation(z_scores_df, col_selected)