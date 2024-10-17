import pandas as pd
import json
import os
import requests

# Charger les données
def load_data(symbol='AAPL'):
    # Charger les données du fichier JSON (adapté à votre exemple)
    try:
        with open('./stock_data.json', 'r') as file:
            data_stock = json.load(file)
        with open('./options_data.json', 'r') as file:
            data_opt = json.load(file)
    except:
        api_key = os.getenv('MY_API_KEY')
        # api_key = ''
        url_stock = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}'

        # Requête HTTP
        response_stock = requests.get(url_stock)
        data_stock = response_stock.json()

        # URL pour obtenir les données historiques des options
        url_opt = f'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={symbol}&apikey={api_key}'


        # Requête HTTP
        response_opt = requests.get(url_opt)
        data_opt = response_opt.json()

        # Création des dataframes
        df_stock = pd.DataFrame.from_dict(data_stock['data'])
        df_opt = pd.DataFrame.from_dict(data_opt['data'])

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


'''
# URL pour obtenir les données quotidiennes d'une action

'''