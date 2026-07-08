import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Treina e avalia a Regressão Logística"""
    print("\n--- Treinando Regressão Logística ---")
    start_time = time.time()
    
    # Usamos solver='lbfgs' com max_iter elevado para convergência segura
    model = LogisticRegression(max_iter=500, random_state=42, solver='lbfgs', n_jobs=-1)
    model.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probs)
    
    print(f"Regressão Logística finalizada em {elapsed_time:.1f}s | Val ROC AUC: {auc:.5f}")
    return model, probs, auc, elapsed_time

def train_xgboost(X_train, y_train, X_val, y_val):
    """Treina e avalia o XGBoost com Early Stopping"""
    print("\n--- Treinando XGBoost ---")
    start_time = time.time()
    
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric="auc",
        tree_method="hist"  # Extremamente rápido para dados grandes
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    elapsed_time = time.time() - start_time
    probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probs)
    
    print(f"XGBoost finalizado em {elapsed_time:.1f}s | Val ROC AUC: {auc:.5f}")
    return model, probs, auc, elapsed_time

def train_lightgbm(X_train, y_train, X_val, y_val):
    """Treina e avalia o LightGBM com Early Stopping"""
    print("\n--- Treinando LightGBM ---")
    start_time = time.time()
    
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
    )
    
    elapsed_time = time.time() - start_time
    probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probs)
    
    print(f"LightGBM finalizado em {elapsed_time:.1f}s | Val ROC AUC: {auc:.5f}")
    return model, probs, auc, elapsed_time

def train_catboost(X_train, y_train, X_val, y_val):
    """Treina e avalia o CatBoost com Early Stopping"""
    print("\n--- Treinando CatBoost ---")
    start_time = time.time()
    
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.03,
        depth=6,
        random_seed=42,
        early_stopping_rounds=50,
        verbose=100,
        thread_count=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    
    elapsed_time = time.time() - start_time
    probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probs)
    
    print(f"CatBoost finalizado em {elapsed_time:.1f}s | Val ROC AUC: {auc:.5f}")
    return model, probs, auc, elapsed_time
