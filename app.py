import pandas as pd
import numpy as np
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.decomposition import PCA
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

# Function to load data
def load_data():
    # Load data from JSON files (adapt to your example)
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

    # Merge data
    df_aapl = df_stock.merge(df_opt, on='Date', how='inner')

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume_stock', 'Strike', 'Last', 'Mark', 'Bid', 'Bid_size', 'Ask', 'Ask_size', 'Volume_option', 'Open_interest', 'Implied_volatility', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
    df_aapl[numeric_cols] = df_aapl[numeric_cols].astype(float)
    
    df_aapl = df_aapl.dropna(subset=numeric_cols)

    return df_aapl

# Function to normalize and scale data
def normalize_scale(df, important_cols):
    # Step 1: Identify skewed features
    skewness = df[important_cols].skew()
    skewed_features = skewness[abs(skewness) > 0.5].index

    # Step 2: Apply transformations
    positive_features = [col for col in skewed_features if (df[col] > 0).all()]
    if positive_features:
        pt_boxcox = PowerTransformer(method='box-cox')
        df[positive_features] = pt_boxcox.fit_transform(df[positive_features])

    other_features = list(set(skewed_features) - set(positive_features))
    if other_features:
        pt_yeojohnson = PowerTransformer(method='yeo-johnson')
        df[other_features] = pt_yeojohnson.fit_transform(df[other_features])

    # Step 3: Scale features
    scaler = RobustScaler()
    df[important_cols] = scaler.fit_transform(df[important_cols])

    return df

# Function to provide explanation for a specific contract
def explain_contract(df, contract_id, important_cols):
    # Get data for the contract
    contract_data = df[df['Contractid'] == contract_id]

    # Calculate deviations from the median
    deviations = contract_data[important_cols] - df[important_cols].median()
    deviations = deviations.T  # Transpose for easier handling
    deviations.columns = ['Deviation']

    # Calculate absolute deviations
    deviations['Absolute Deviation'] = deviations['Deviation'].abs()

    # Sort features by absolute deviation
    deviations = deviations.sort_values(by='Absolute Deviation', ascending=False)

    return deviations

if __name__ == '__main__':

    df_aapl = load_data()

    # Define the features to use
    important_cols = ['Strike', 'Last', 'Mark', 'Bid', 'Bid_size', 'Ask', 'Ask_size',
                    'Volume_option', 'Open_interest', 'Implied_volatility', 'Delta',
                    'Gamma', 'Theta', 'Vega', 'Rho']

    st.title('Outlier Detection in Derivative Sensitivities')

    # User selects the method
    method = st.selectbox(
        "Choose the outlier detection method:",
        ["Statistical Method (IQR)", "Machine Learning Method (Isolation Forest)"]
    )

    df_aapl = normalize_scale(df_aapl, important_cols)

    if method == "Statistical Method (IQR)":
        # User sets IQR multiplier
        iqr_multiplier_input = st.selectbox('Tolerance to outliers',
                                            ['Low', 'Mid', 'High'])
        map_iqr_multiplier = {'Low': 0.5, 'Mid': 1.5, 'High': 5}
        iqr_multiplier = map_iqr_multiplier[iqr_multiplier_input]

        # Compute bounds for each feature
        df_aapl['Anomaly'] = 0  # Initialize as normal
        for col in important_cols:
            Q1 = df_aapl[col].quantile(0.25)
            Q3 = df_aapl[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            df_aapl.loc[(df_aapl[col] < lower_bound) | (df_aapl[col] > upper_bound), 'Anomaly'] = -1

    else:
        # Isolation Forest parameters
        contamination_input = st.selectbox('Presence of outliers',
                                           ['Unknown', 'Low', 'Mid', 'High'])
        map_contamination = {'Unknown': 'auto', 'Low': 0.01, 'Mid': 0.1, 'High': 0.2}
        contamination = map_contamination[contamination_input]

        model = IsolationForest(contamination=contamination, random_state=42)
        X = df_aapl[important_cols]
        model.fit(X)
        df_aapl['Anomaly'] = model.predict(X)

    # Display outliers
    outliers = df_aapl[df_aapl['Anomaly'] == -1]
    if not outliers.empty:
        st.subheader('Detected Outliers')
        st.write(outliers)
    else:
        st.write("No outliers detected with the current parameters.")

    # Dimensionality Reduction and Visualization
    st.subheader('Visualization of Contracts')
    # Perform PCA or t-SNE
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(df_aapl[important_cols])
    plot_df = pd.DataFrame({
        'Component 1': X_reduced[:, 0],
        'Component 2': X_reduced[:, 1],
        'Anomaly': df_aapl['Anomaly'],
        'Contractid': df_aapl['Contractid']
    })
    # Map anomaly labels to colors
    color_map = {1: 'normal', -1: 'anomaly', 0: 'normal'}
    plot_df['Color'] = plot_df['Anomaly'].map(color_map)

    fig = px.scatter(
        plot_df,
        x='Component 1',
        y='Component 2',
        color='Color',
        color_discrete_map={'normal': 'white', 'anomaly': 'blue'},
        hover_data=['Contractid'],
        title='Contracts Visualized with PCA'
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=1, color='black')))

    st.plotly_chart(fig)

    # Option to further analyze a specific feature
    st.subheader('Further Analysis')
    analyze_feature = st.checkbox('Do you want to further analyze a specific feature?')
    if analyze_feature:
        # User selects the feature
        feature_selected = st.selectbox('Select a feature to analyze:', important_cols)

        # Allow user to set thresholds using sliders
        min_val = df_aapl[feature_selected].min()
        max_val = df_aapl[feature_selected].max()
        threshold = st.slider(f'Select threshold for {feature_selected}', min_val, max_val, (min_val, max_val))
        df_filtered = df_aapl[(df_aapl[feature_selected] >= threshold[0]) & (df_aapl[feature_selected] <= threshold[1])]
        st.write(f'Number of contracts within selected range: {len(df_filtered)}')

        # Update KDE plot with shaded area
        fig, ax = plt.subplots()
        sns.kdeplot(df_aapl[feature_selected], shade=True, ax=ax)
        ax.axvspan(threshold[0], threshold[1], color='red', alpha=0.3)
        st.pyplot(fig)

    # Option to delve into a specific contract
    st.subheader('Contract Analysis')
    analyze_contract = st.checkbox('Do you want to delve into a specific contract?')
    if analyze_contract:
        # User selects the contract
        contract_ids = outliers['Contractid'].unique()
        if len(contract_ids) == 0:
            st.write("No outlier contracts to analyze.")
        else:
            contract_selected = st.selectbox('Select a contract to analyze:', contract_ids)
            deviations = explain_contract(df_aapl, contract_selected, important_cols)
            st.write(f'Analysis of Contract {contract_selected}')
            st.write(deviations)

            # Visualize deviations
            fig, ax = plt.subplots(figsize=(10, 6))
            deviations['Deviation'].plot(kind='bar', ax=ax)
            ax.set_title(f'Deviations from Median for Contract {contract_selected}')
            ax.set_ylabel('Deviation')
            st.pyplot(fig)
