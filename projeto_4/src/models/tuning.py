"""
Busca de hiperparâmetros (RandomizedSearchCV) para os três modelos de
Gradient Boosting sobre árvores (XGBoost, LightGBM, CatBoost).

A busca roda em uma sub-amostra estratificada do treino (mais rápida e mais
leve em memória); os melhores parâmetros encontrados são então retreinados
no conjunto de treino completo, com early stopping contra o conjunto de
validação (mesmo split 80/20 usado no restante do projeto), para produzir um
número de ROC AUC diretamente comparável à Tabela 1 do relatório.
"""
import json
import os
import time

import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from src.features.tree_data import load_tree_data

N_ITER = 20
CV_FOLDS = 3
SEARCH_SAMPLE_SIZE = 100_000
RANDOM_STATE = 42


def _stratified_subsample(X, y, n, random_state):
    if n >= len(y):
        return X, y
    rng = np.random.RandomState(random_state)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    frac = n / len(y)
    n_pos = max(1, int(len(idx_pos) * frac))
    n_neg = n - n_pos
    sel_pos = rng.choice(idx_pos, size=n_pos, replace=False)
    sel_neg = rng.choice(idx_neg, size=n_neg, replace=False)
    sel = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(sel)
    return X.iloc[sel].reset_index(drop=True), y[sel]


def tune_xgboost(X_train, y_train, X_val, y_val):
    param_dist = {
        "n_estimators": randint(200, 600),
        "max_depth": randint(3, 9),
        "learning_rate": uniform(0.01, 0.14),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 12),
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.5, 2.0),
    }
    base = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
    )
    return _run_search("XGBoost", base, param_dist, X_train, y_train, X_val, y_val)


def tune_lightgbm(X_train, y_train, X_val, y_val):
    param_dist = {
        "n_estimators": randint(200, 600),
        "num_leaves": randint(15, 127),
        "learning_rate": uniform(0.01, 0.14),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_samples": randint(5, 100),
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.0, 2.0),
    }
    base = lgb.LGBMClassifier(
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1,
    )
    return _run_search("LightGBM", base, param_dist, X_train, y_train, X_val, y_val)


def tune_catboost(X_train, y_train, X_val, y_val):
    param_dist = {
        "iterations": randint(200, 600),
        "depth": randint(4, 10),
        "learning_rate": uniform(0.01, 0.14),
        "l2_leaf_reg": uniform(1.0, 9.0),
        "border_count": randint(32, 255),
    }
    base = CatBoostClassifier(
        random_seed=RANDOM_STATE,
        verbose=0,
        thread_count=-1,
    )
    return _run_search("CatBoost", base, param_dist, X_train, y_train, X_val, y_val)


def _run_search(name, base_model, param_dist, X_train, y_train, X_val, y_val):
    print(f"\n--- [Tuning] RandomizedSearchCV para {name} "
          f"({N_ITER} combinações x {CV_FOLDS} folds em amostra de "
          f"{min(SEARCH_SAMPLE_SIZE, len(y_train))} linhas) ---")

    X_search, y_search = _stratified_subsample(X_train, y_train, SEARCH_SAMPLE_SIZE, RANDOM_STATE)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,  # os próprios modelos já paralelizam internamente (threads)
        verbose=1,
        refit=False,
    )

    t0 = time.time()
    search.fit(X_search, y_search)
    search_time = time.time() - t0

    print(f"[{name}] Melhor ROC AUC (CV na amostra): {search.best_score_:.5f}")
    print(f"[{name}] Melhores parâmetros: {search.best_params_}")

    # Retreina com os melhores parâmetros no TREINO COMPLETO, avaliando na
    # validação oficial do projeto, para gerar um número comparável à Tabela 1.
    final_params = dict(search.best_params_)
    t0 = time.time()
    if name == "XGBoost":
        final_model = xgb.XGBClassifier(
            **final_params, random_state=RANDOM_STATE, eval_metric="auc",
            tree_method="hist", n_jobs=-1, early_stopping_rounds=50,
        )
        final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "LightGBM":
        final_model = lgb.LGBMClassifier(
            **final_params, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
        )
        final_model.fit(
            X_train, y_train, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    else:  # CatBoost
        final_model = CatBoostClassifier(
            **final_params, random_seed=RANDOM_STATE, verbose=0,
            thread_count=-1, early_stopping_rounds=50,
        )
        final_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    final_train_time = time.time() - t0

    val_probs = final_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"[{name}] ROC AUC na validação oficial (params tunados): {val_auc:.5f} "
          f"(retreino em {final_train_time:.1f}s, busca em {search_time:.1f}s)")

    return {
        "model": final_model,
        "best_params": search.best_params_,
        "cv_search_auc": search.best_score_,
        "val_auc_tuned": val_auc,
        "search_time_s": search_time,
        "retrain_time_s": final_train_time,
    }


def main():
    X_train, X_val, y_train, y_val, _ = load_tree_data()

    # Baselines medidos NESTA máquina com os hiperparâmetros padrão de
    # src/models/boosters.py (mesmo split de dados), para uma comparação justa
    # com a busca de hiperparâmetros abaixo. Não usar números de outra máquina
    # (ex.: os da Tabela 1 do relatório, gerados em outro ambiente) como
    # baseline aqui, pois diferenças de hardware/versão de biblioteca não são
    # comparáveis a um delta de tuning.
    baseline_auc = {
        "XGBoost": 0.78691,
        "LightGBM": 0.78379,
        "CatBoost": 0.78279,
    }

    results = {}
    results["XGBoost"] = tune_xgboost(X_train, y_train, X_val, y_val)
    results["LightGBM"] = tune_lightgbm(X_train, y_train, X_val, y_val)
    results["CatBoost"] = tune_catboost(X_train, y_train, X_val, y_val)

    print("\n=================================================================")
    print("           RESUMO: BASELINE (main.py) vs. TUNADO (busca)         ")
    print("=================================================================")
    summary = []
    for name, res in results.items():
        summary.append({
            "model": name,
            "baseline_val_auc": baseline_auc[name],
            "tuned_val_auc": res["val_auc_tuned"],
            "delta": res["val_auc_tuned"] - baseline_auc[name],
            "best_params": res["best_params"],
        })
        print(f"{name:10s} | baseline: {baseline_auc[name]:.5f} | "
              f"tunado: {res['val_auc_tuned']:.5f} | "
              f"delta: {res['val_auc_tuned'] - baseline_auc[name]:+.5f}")

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/hyperparameter_tuning_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResultados salvos em '{out_path}'.")


if __name__ == "__main__":
    main()
