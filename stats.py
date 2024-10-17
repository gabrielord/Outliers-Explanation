from scipy import stats
from sklearn.preprocessing import PowerTransformer
import streamlit as st

# Fonction pour appliquer Box-Cox et calculer les outliers
def normalize_columns(df, important_cols):
    df_norm = df.copy()
    for col in df[important_cols].columns:
        # Vérifier s'il y a des valeurs négatives dans la colonne
        if (df_norm[col] < 0).any():
            # Appliquer la transformation Yeo-Johnson (pour gérer les valeurs négatives)
            pt = PowerTransformer(method='yeo-johnson')
            df_norm[col] = pt.fit_transform(df_norm[[col]])
            print(f'Utilisation de Yeo-Johnson pour la colonne {col}')
        else:
            # Appliquer la transformation Box-Cox à la colonne spécifiée
            df_norm[col], _ = stats.boxcox(df_norm[col] + 1)  # Ajouter 1 pour éviter les zéros
            print(f'Utilisation de Box-Cox pour la colonne {col}')
    return df_norm
    
            
def boxcox_outliers(df, col):
    # Calculer l'IQR (Intervalle Interquartile) dans la colonne transformée
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Identifier les outliers en dehors de l'intervalle [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    outliers_iqr = df[(df[col] < (Q1 - 1.5 * IQR)) | 
                      (df[col] > (Q3 + 1.5 * IQR))]

    # Identifier les données qui ne sont pas des outliers
    not_outliers_iqr = df[~((df[col] < (Q1 - 1.5 * IQR)) | 
                            (df[col] > (Q3 + 1.5 * IQR)))]

    # Retourner le DataFrame original avec la colonne transformée, les outliers et les non-outliers
    return outliers_iqr, not_outliers_iqr

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