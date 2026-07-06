import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import os

def load_and_preprocess_data(parquet_path: str = "data/processed/train_features.parquet", test_size: float = 0.2, random_state: int = 42):
    """
    Carrega o dataset parquet e prepara dois conjuntos de dados:
    1. Dense (para Regressão Logística e MLP): Imputado, Escalonado e One-Hot Encoded.
    2. Tree (para XGBoost, LightGBM, CatBoost): Mantém nulos e usa Ordinal/Label Encoding para categóricos.
    """
    print(f"Lendo dados de {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Identifica identificador e variável alvo
    id_col = "SK_ID_CURR"
    target_col = "TARGET"
    
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada no dataset.")
        
    y = df[target_col].values
    X_raw = df.drop(columns=[id_col, target_col])
    
    # Identifica colunas categóricas e numéricas
    cat_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_raw.select_dtypes(include=['number']).columns.tolist()
    
    print(f"Colunas encontradas: {len(num_cols)} numéricas, {len(cat_cols)} categóricas.")
    
    # --- PIPELINE TREE (Árvores de Decisão) ---
    print("Preparando pipeline para modelos de árvore (Tree)...")
    X_tree = X_raw.copy()
    
    # Ordinal Encoder para as colunas categóricas (lidar com desconhecidos e nulos de forma segura)
    # Convertemos colunas categóricas para string para evitar erros no Encoder
    for col in cat_cols:
        X_tree[col] = X_tree[col].astype(str).fillna("MISSING")
        
    encoder_tree = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if cat_cols:
        X_tree[cat_cols] = encoder_tree.fit_transform(X_tree[cat_cols])
        
    # Garantimos que todos os dados sejam do tipo float32 para as árvores
    X_tree = X_tree.astype(np.float32)
    
    # --- PIPELINE DENSE (Regressão Logística e Redes Neurais) ---
    print("Preparando pipeline para modelos densos (LR / MLP)...")
    
    # 1. Tratamento numérico (Imputação da mediana + StandardScaler)
    imputer_num = SimpleImputer(strategy='median')
    scaler_num = StandardScaler()
    
    X_num_imputed = imputer_num.fit_transform(X_raw[num_cols])
    X_num_scaled = scaler_num.fit_transform(X_num_imputed)
    df_num_dense = pd.DataFrame(X_num_scaled, columns=num_cols)
    
    # 2. Tratamento categórico (Imputação de 'MISSING' + One-Hot Encoding)
    if cat_cols:
        X_cat_raw = X_raw[cat_cols].fillna("MISSING").astype(str)
        encoder_cat = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
        X_cat_ohe = encoder_cat.fit_transform(X_cat_raw)
        
        # Cria nomes amigáveis para as novas colunas One-Hot
        ohe_col_names = encoder_cat.get_feature_names_out(cat_cols)
        df_cat_dense = pd.DataFrame(X_cat_ohe, columns=ohe_col_names)
        
        # Concatena numéricos e categóricos OHE
        X_dense = pd.concat([df_num_dense, df_cat_dense], axis=1)
    else:
        X_dense = df_num_dense
        
    # --- DIVISÃO DOS DADOS (TRAIN / VALIDATION ESTRATIFICADO) ---
    print("Dividindo os dados de forma estratificada...")
    
    # Divisão para os modelos de árvore
    X_train_tree, X_val_tree, y_train, y_val = train_test_split(
        X_tree, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Divisão correspondente para os modelos densos usando os mesmos índices
    train_idx, val_idx = X_train_tree.index, X_val_tree.index
    X_train_dense = X_dense.iloc[train_idx].reset_index(drop=True)
    X_val_dense = X_dense.iloc[val_idx].reset_index(drop=True)
    
    # Reset indices para os conjuntos Tree também para evitar problemas com índices antigos
    X_train_tree = X_train_tree.reset_index(drop=True)
    X_val_tree = X_val_tree.reset_index(drop=True)
    
    print(f"Sucesso! Conjunto de Treino: {X_train_tree.shape[0]} amostras, Validação: {X_val_tree.shape[0]} amostras.")
    print(f"Colunas de Árvore: {X_train_tree.shape[1]} | Colunas Densa (OHE): {X_train_dense.shape[1]}")
    
    return (
        X_train_dense, X_val_dense, 
        X_train_tree, X_val_tree, 
        y_train, y_val
    )

if __name__ == "__main__":
    X_tr_d, X_val_d, X_tr_t, X_val_t, y_tr, y_val = load_and_preprocess_data()
