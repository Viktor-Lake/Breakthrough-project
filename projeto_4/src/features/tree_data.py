import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def load_tree_data(
    parquet_path: str = "data/processed/train_features.parquet",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Recarrega o dataset processado e reproduz exatamente o pipeline de árvore
    (Ordinal Encoding + split estratificado 80/20, random_state=42) usado em
    src/models/data_loader.py, sem depender do torch (não necessário para
    XGBoost/LightGBM/CatBoost/SHAP). Mantém os dois pipelines em sincronia:
    qualquer mudança no split de data_loader.py deve ser espelhada aqui.
    """
    df = pd.read_parquet(parquet_path)

    id_col = "SK_ID_CURR"
    target_col = "TARGET"

    y = df[target_col].values
    X_raw = df.drop(columns=[id_col, target_col])

    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train_raw = X_train_raw.reset_index(drop=True)
    X_val_raw = X_val_raw.reset_index(drop=True)

    X_train_tree = X_train_raw.copy()
    X_val_tree = X_val_raw.copy()

    for col in cat_cols:
        X_train_tree[col] = X_train_tree[col].astype(str).fillna("MISSING")
        X_val_tree[col] = X_val_tree[col].astype(str).fillna("MISSING")

    if cat_cols:
        encoder_tree = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_tree[cat_cols] = encoder_tree.fit_transform(X_train_tree[cat_cols])
        X_val_tree[cat_cols] = encoder_tree.transform(X_val_tree[cat_cols])

    X_train_tree = X_train_tree.astype(np.float32)
    X_val_tree = X_val_tree.astype(np.float32)

    return X_train_tree, X_val_tree, y_train, y_val, cat_cols
