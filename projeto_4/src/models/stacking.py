import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def run_stacking_ensemble(model_preds: dict, y_val: np.ndarray) -> dict:
    """
    Combina as predições de vários modelos em diferentes estratégias de ensemble:
    1. Average Blend (Média aritmética simples).
    2. Top-3 Trees Blend (Média de XGB, LGBM e CatBoost).
    3. Meta-Model Stacking (Regressão Logística treinada sobre as probabilidades).
    """
    print("\n==============================================")
    print(" INICIANDO STRATEGY OF STACKING ENSEMBLE ")
    print("==============================================")
    
    # Prepara o DataFrame de predições
    df_preds = pd.DataFrame(model_preds)
    print("Predições dos modelos de validação carregadas:")
    print(df_preds.head())
    
    results = {}
    
    # --- 1. AVERAGE BLEND ---
    avg_preds = df_preds.mean(axis=1).values
    avg_auc = roc_auc_score(y_val, avg_preds)
    print(f"\n[Ensemble] Média Simples de todos os 5 modelos | ROC AUC: {avg_auc:.5f}")
    results["Average Blend"] = {"preds": avg_preds, "auc": avg_auc}
    
    # --- 2. TOP-3 TREES BLEND ---
    tree_cols = [col for col in ["XGBoost", "LightGBM", "CatBoost"] if col in df_preds.columns]
    if len(tree_cols) >= 2:
        top_trees_preds = df_preds[tree_cols].mean(axis=1).values
        top_trees_auc = roc_auc_score(y_val, top_trees_preds)
        print(f"[Ensemble] Média dos {len(tree_cols)} modelos de Árvore | ROC AUC: {top_trees_auc:.5f}")
        results["Top Trees Blend"] = {"preds": top_trees_preds, "auc": top_trees_auc}
        
    # --- 3. META-MODEL STACKING ---
    # As predições em model_preds foram todas geradas no MESMO conjunto de
    # validação (y_val). Ajustar o meta-modelo e avaliá-lo nesse mesmo
    # conjunto vazaria informação (o meta-modelo "veria" os rótulos que está
    # tentando prever), inflando artificialmente o ROC AUC. Para evitar isso,
    # dividimos y_val em duas metades estratificadas: uma para treinar o
    # meta-modelo (meta-treino) e outra, nunca vista pelo meta-modelo durante
    # o ajuste, para avaliá-lo (meta-holdout). O ROC AUC do Meta Stacking é
    # calculado apenas nessa fatia de holdout — por isso não é diretamente
    # comparável, em tamanho de amostra, aos demais ensembles desta função,
    # que usam o y_val inteiro.
    X_meta = df_preds.values
    (
        X_meta_train, X_meta_holdout,
        y_meta_train, y_meta_holdout,
    ) = train_test_split(X_meta, y_val, test_size=0.5, stratify=y_val, random_state=42)

    # Usamos uma Regressão Logística com regularização L2 forte para evitar overfit
    meta_model = LogisticRegression(C=0.1, penalty='l2', random_state=42)
    meta_model.fit(X_meta_train, y_meta_train)

    # Predições do Meta-Modelo avaliadas SOMENTE no meta-holdout (nunca visto no fit)
    meta_preds = meta_model.predict_proba(X_meta_holdout)[:, 1]
    meta_auc = roc_auc_score(y_meta_holdout, meta_preds)

    print(f"[Ensemble] Meta-Modelo Stacking (Logistic Regression, avaliado em holdout "
          f"de {len(y_meta_holdout)} amostras) | ROC AUC: {meta_auc:.5f}")
    
    # Exibe os coeficientes aprendidos pelo Meta-Modelo (importância de cada modelo)
    coefs = meta_model.coef_[0]
    print("\nPesos atribuídos pelo Meta-Modelo a cada algoritmo original:")
    for name, coef in zip(df_preds.columns, coefs):
        print(f"  - {name}: {coef:.4f}")
    print(f"  - Intercepto: {meta_model.intercept_[0]:.4f}")
    
    results["Meta Stacking"] = {
        "preds": meta_preds,
        "auc": meta_auc,
        "model": meta_model,
        "coefficients": dict(zip(df_preds.columns, coefs))
    }
    
    return results
