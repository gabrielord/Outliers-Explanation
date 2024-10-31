# %%
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import requests
import json
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest


# %%
# Charge les données des actions à partir d'un fichier JSON
with open('./stock_data.json', 'r') as file:
    data_stock = json.load(file)

# Charge les données des options à partir d'un fichier JSON
with open('./options_data.json', 'r') as file:
    data_opt = json.load(file)

# Convertit les données des actions en un DataFrame pandas
df_stock = pd.DataFrame.from_dict(data_stock['Time Series (Daily)'], orient='index')

# Convertit les données des options en un DataFrame pandas
df_opt = pd.DataFrame.from_dict(data_opt['data'])

# %%
# Réinitialise l'index du DataFrame df_stock et le transforme en colonne
df_stock.reset_index(inplace=True)

# Définit les noms des colonnes de df_stock
df_stock.columns = ['Date','Open','High','Low','Close','Volume_stock']

# Convertit les noms des colonnes de df_opt pour qu'ils commencent par une majuscule
df_opt.columns = df_opt.columns.map(lambda x: x.capitalize())

# Renomme la colonne 'Volume' en 'Volume_option'
df_opt.rename(columns={'Volume':'Volume_option'}, inplace = True)

# Fusionne les DataFrames df_stock et df_opt en se basant sur la colonne 'Date'
df_aapl = df_stock.merge(df_opt, on='Date', how='inner')

# %%
# Liste des colonnes numériques à convertir en type float
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume_stock',
              'Strike', 'Last', 'Mark', 'Bid',
              'Bid_size', 'Ask', 'Ask_size', 'Volume_option', 'Open_interest',
              'Implied_volatility', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho']

# Convertit les colonnes numériques sélectionnées en type float
df_aapl[numeric_cols] = df_aapl[numeric_cols].astype(float)

def explanation(z_scores_df, contract_id):
    # Encontrar a linha com o maior Z-score em módulo
    max_zscore_row = z_scores_df.loc[z_scores_df['Absolute Z-Score'].idxmax()]

    # Exibir a explicação
    print(f"O valor outlier encontrado na volatilidade implícita para o contrato {contract_id} pode ser devido a um valor incomum encontrado para {max_zscore_row['Feature']}: \n"
          f"Valor observado: {max_zscore_row['Observed Value']} \n"
          f"Média: {max_zscore_row['Mean']} \n"
          f"Z-Score: {max_zscore_row['Z-Score']}")
    
def boxcox_outliers(df, col):
    # Aplicar a transformação Box-Cox na coluna especificada (somar 1 para evitar zeros)
    df[f'boxcox_{col}'], _ = stats.boxcox(df[col] + 1)
    
    # Calcular o IQR (Intervalo Interquartil) na coluna transformada
    Q1 = df[f'boxcox_{col}'].quantile(0.25)
    Q3 = df[f'boxcox_{col}'].quantile(0.75)
    IQR = Q3 - Q1

    # Identificar os outliers fora do intervalo [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    outliers_iqr = df[(df[f'boxcox_{col}'] < (Q1 - 1.5 * IQR)) | 
                      (df[f'boxcox_{col}'] > (Q3 + 1.5 * IQR))]

    # Identificar os dados que não são outliers
    not_outliers_iqr = df[~((df[f'boxcox_{col}'] < (Q1 - 1.5 * IQR)) | 
                            (df[f'boxcox_{col}'] > (Q3 + 1.5 * IQR)))]
    
    print(outliers_iqr.columns)

    # Retornar o DataFrame original com a coluna transformada, os outliers e os não outliers
    return df, outliers_iqr, not_outliers_iqr

# %%
# Aplicar a função no DataFrame df_aapl e na coluna 'Implied_volatility'
df_transformed, outliers, not_outliers = boxcox_outliers(df_aapl, 'Implied_volatility')

# Visualizar os outliers identificados
outliers[['Expiration', 'Strike', 'Type', 'Last', 'Mark', 'Bid',
                'Bid_size', 'Ask', 'Ask_size', 'Volume_option', 'Open_interest',
                'Implied_volatility', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho',
                'boxcox_Implied_volatility']]

# %%
important_cols = ['Strike','Last','Mark',
 'Bid',
 'Bid_size',
 'Ask',
 'Ask_size',
 'Volume_option',
 'Open_interest',
 'Implied_volatility',
 'Delta',
 'Gamma',
 'Theta',
 'Vega',
 'Rho']

corr_matrix = df_aapl[important_cols].corr()

corr_matrix_outliers = outliers[important_cols].corr()

corr_matrix_not_outliers = not_outliers[important_cols].corr()

# %%
# Prever a volatilidade implícita para o contrato específico
X = df_aapl[important_cols].drop(columns=['Implied_volatility'])
y = df_aapl['Implied_volatility']

# %%
# Filtrar o contrato específico
contract_id = 'AAPL241011C00100000'  # Substitua pelo ID do contrato que você está analisando
contract_data = df_aapl[df_aapl['Contractid'] == contract_id]

# Prever a volatilidade implícita para o contrato específico
X_contract = contract_data[important_cols].drop(columns=['Implied_volatility'])
y_contract = contract_data['Implied_volatility']

# %%
# Calcular a média e o desvio padrão históricos para cada feature
mean_features = X.mean()
std_features = X.std()

# Calcular quantos desvios padrões o valor do contrato específico está da média histórica
z_scores = (X_contract.values[0] - mean_features) / std_features

# Criar um DataFrame para mostrar as distâncias em desvios padrões
z_scores_df = pd.DataFrame({'Z-Score': z_scores, 'Mean': mean_features, 'Observed Value': X_contract.iloc[0]})
z_scores_df['Absolute Z-Score'] = z_scores_df['Z-Score'].abs()

z_scores_df = z_scores_df.sort_values(by='Absolute Z-Score', ascending=False)

# %%
explanation(z_scores_df, contract_id)


