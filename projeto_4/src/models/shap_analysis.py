"""
Explicabilidade via SHAP (SHapley Additive exPlanations) sobre o modelo de
árvore salvo pelo pipeline principal (data/processed/saved_models/best_model_xgboost.joblib).

Usa shap.TreeExplainer, que calcula os valores de Shapley de forma exata e
eficiente para modelos baseados em árvore (sem a necessidade de amostragem
aproximada exigida por modelos genéricos), explorando a estrutura interna das
árvores para propagar as contribuições de cada atributo.
"""
import os

import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap

from src.features.tree_data import load_tree_data

SAMPLE_SIZE = 3000
RANDOM_STATE = 42
OUT_DIR = "data/processed/shap"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Carregando dados e modelo salvo...")
    X_train, X_val, y_train, y_val, cat_cols = load_tree_data()
    model = joblib.load("data/processed/saved_models/best_model_xgboost.joblib")

    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(len(X_val), size=min(SAMPLE_SIZE, len(X_val)), replace=False)
    X_sample = X_val.iloc[sample_idx].reset_index(drop=True)

    print(f"Calculando valores SHAP (TreeExplainer) em {len(X_sample)} amostras de validação...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # --- Importância global (mean |SHAP|) ---
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance = (
        pd_series := __import__("pandas").Series(mean_abs_shap, index=X_sample.columns)
    ).sort_values(ascending=False)
    print("\nTop 15 atributos por importância média |SHAP|:")
    print(importance.head(15).to_string())
    importance.head(30).to_csv(os.path.join(OUT_DIR, "shap_feature_importance.csv"))

    # --- Gráfico 1: Bar plot (importância global) ---
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_bar_importance.png"), dpi=300)
    plt.close()

    # --- Gráfico 2: Beeswarm (direção + magnitude do impacto) ---
    plt.figure()
    shap.summary_plot(shap_values, X_sample, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_beeswarm.png"), dpi=300)
    plt.close()

    # --- Gráfico 3: Dependence plots para atributos discutidos no relatório ---
    for feat in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "PAYMENT_RATE"]:
        if feat not in X_sample.columns:
            continue
        plt.figure()
        shap.dependence_plot(feat, shap_values.values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"shap_dependence_{feat}.png"), dpi=300)
        plt.close()

    print(f"\nGráficos e importâncias salvos em '{OUT_DIR}/'.")


if __name__ == "__main__":
    main()
