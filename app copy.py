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
from sklearn.ensemble import IsolationForest


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

    # Retourner le DataFrame original avec la colonne transformée, les outliers et les non-outliers
    return df

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

if __name__ == '__main__':

    df_aapl = load_data()

    # Interface Streamlit
    st.title('Analyse des Outliers dans les Sensi')

    important_cols = ['Strike','Last','Mark','Bid','Bid_size','Ask','Ask_size','Volume_option','Open_interest','Implied_volatility','Delta','Gamma','Theta','Vega','Rho']

    # Sélection de la métrique
    col_selected = st.selectbox("Sélectionnez la métrique pour analyser les outliers", important_cols)

    # Appliquer Box-Cox et obtenir les outliers
    method = st.selectbox("Sélectionnez le méthode pour détécter des outliers", ['Statistique (Interquartile Range)', 'Machine Learning (Isolation Forest)'])

    if method == 'Machine Learning (Isolation Forest)':
        df_transformed = boxcox_outliers(df_aapl, col_selected)
        
        # Use the transformed column
        X = df_transformed[[f'boxcox_{col_selected}']]
        
        # Handle missing or infinite values
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.dropna(inplace=True)
        
        # Initialize Isolation Forest with auto contamination
        model = IsolationForest(contamination='auto', random_state=42)
        model.fit(X)
        
        # Predict anomalies
        df_transformed['Anomaly'] = model.predict(X)
        outliers_custom = df_transformed[df_transformed['Anomaly'] == -1].drop(columns='Anomaly')
        print(f"Number of inliers: {len(X) - len(outliers_custom)}")
        
    else:
        # Definir o layout com sliders e inputs numéricos para percentis
        st.subheader("Ajuste des percentiles pour la détection d'outliers")

        # Colocar sliders e caixas de entrada lado a lado
        col1, col2 = st.columns(2)

        # Percentil inferior
        with col1:
            low_percentile = st.slider("Sélectionnez le percentile inférieur", 0.0, 50.0, 1.0, 0.1)
            low_percentile_input = st.number_input("ou entrez le percentile inférieur", min_value=0.0, max_value=50.0, value=low_percentile, step=0.1)
            if low_percentile_input != low_percentile:
                low_percentile = low_percentile_input  # Sincroniza o valor do slider com a entrada numérica

        # Percentil superior
        with col2:
            high_percentile = st.slider("Sélectionnez le percentile supérieur", 50.0, 100.0, 99.0, 0.1)
            high_percentile_input = st.number_input("ou entrez le percentile supérieur", min_value=50.0, max_value=100.0, value=high_percentile, step=0.1)
            if high_percentile_input != high_percentile:
                high_percentile = high_percentile_input  # Sincroniza o valor do slider com a entrada numérica

        # Aplicar a transformação
        df_transformed = boxcox_outliers(df_aapl, col_selected)

        # Calcular os limites de percentil
        lower_bound = df_transformed[col_selected].quantile(low_percentile / 100)
        upper_bound = df_transformed[col_selected].quantile(high_percentile / 100)

        # Filtrar os outliers com base nos limites calculados
        outliers_custom = df_transformed[(df_transformed[col_selected] < lower_bound) | 
                                        (df_transformed[col_selected] > upper_bound)]

        # Visualizar a curva de distribuição
        st.subheader(f'Distribution de {col_selected} avec les limites de percentile')

        plt.figure(figsize=(10, 6))
        # sns.histplot(df_transformed[col_selected], kde=True, color='skyblue')
        sns.kdeplot(df_transformed[col_selected], color='skyblue')
        plt.axvline(lower_bound, color='red', linestyle='--', label=f'{round(low_percentile,2)} Percentile')
        plt.axvline(upper_bound, color='green', linestyle='--', label=f'{round(high_percentile,2)} Percentile')
        plt.title(f'Distribution de {col_selected} avec les limites de outliers')
        plt.xlabel(col_selected)
        plt.ylabel('Density')
        plt.legend()
        # plt.xlim(0, 1)  # Ajuste o valor máximo para focar na área de maior densidade
        st.pyplot(plt)
        plt.close()

    if outliers_custom.empty:
        st.write(f"Il n'y a pas des outliers pour la métrique {col_selected} avec le percentile sélectionné.")

    else:
        # Afficher les outliers
        st.subheader(f'Outliers identifiés pour la métrique {col_selected}')
        st.write(outliers_custom)

        # Sélection d'un contrat spécifique
        contract_id_selected = st.selectbox("Sélectionnez le contrat à analyser", outliers_custom['Contractid'].unique())

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