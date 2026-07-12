import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_auc_score

# Importa módulos internos do projeto
from src.models.data_loader import load_and_preprocess_data
from src.models.autoencoder import train_dae
from src.models.neural_net import train_mlp
from src.models.boosters import (
    train_logistic_regression,
    train_xgboost,
    train_lightgbm,
    train_catboost
)
from src.models.tabtransformer import train_tabtransformer
from src.models.stacking import run_stacking_ensemble

def main():
    print("=================================================================")
    print("   INICIANDO PIPELINE DE COMPARAÇÃO & SUPERMODELO HOME CREDIT   ")
    print("=================================================================")
    
    start_pipeline_time = time.time()
    
    # 1. Carregamento e Pré-processamento dos dados
    # OBS: as opções de normalização (dados numéricos) e de codificação (dados
    # categóricos) são configuradas em src/models/data_loader.py, no bloco
    # "CONFIGURAÇÃO DE PRÉ-PROCESSAMENTO" no topo do arquivo.
    (
        X_train_d, X_val_d, X_train_t, X_val_t,
        y_train, y_val, cat_cols, num_cols,
        preprocessing_info, pipeline_preproc
    ) = load_and_preprocess_data()
    
    X_train_d_arr = X_train_d.values
    X_val_d_arr = X_val_d.values
    X_train_t_arr = X_train_t.values
    X_val_t_arr = X_val_t.values
    
    results_summary = []
    model_preds = {}
    
    # 2. Treinamento do Denoising Autoencoder (DAE) - Autossupervisão (SSL)
    print("\n--- [Fase SSL] Treinando Denoising Autoencoder ---")
    dae_start = time.time()
    # Usamos 5 epochs para manter o tempo razoável mas de alta eficiência
    weights_path = train_dae(X_train_d_arr, epochs=15, batch_size=512, lr=0.001)
    dae_time = time.time() - dae_start
    print(f"Autoencoder treinado com sucesso em {dae_time:.1f} segundos.")
    
    # 3. Treinamento da MLP com Transfer Learning (SSL)
    print("\n--- [Classificação] Treinando MLP com Pesos do Autoencoder (Transfer Learning) ---")
    mlp_ssl_start = time.time()
    mlp_ssl_model, mlp_ssl_preds = train_mlp(
        X_train_d_arr, y_train, X_val_d_arr, y_val, 
        epochs=8, batch_size=512, lr=0.001,
        pretrained_weights=weights_path
    )
    mlp_ssl_time = time.time() - mlp_ssl_start
    mlp_ssl_auc = roc_auc_score(y_val, mlp_ssl_preds)
    results_summary.append({
        "Model": "MLP (SSL Pretrained)",
        "Val ROC AUC": mlp_ssl_auc,
        "Train Time (s)": mlp_ssl_time + dae_time
    })
    model_preds["MLP (SSL)"] = mlp_ssl_preds
    
    # 4. Treinamento da MLP do Zero (Scratch) para Comparação
    print("\n--- [Classificação] Treinando MLP do Zero (Sem Pre-treino) ---")
    mlp_scratch_start = time.time()
    mlp_scratch_model, mlp_scratch_preds = train_mlp(
        X_train_d_arr, y_train, X_val_d_arr, y_val, 
        epochs=8, batch_size=512, lr=0.001,
        pretrained_weights=None
    )
    mlp_scratch_time = time.time() - mlp_scratch_start
    mlp_scratch_auc = roc_auc_score(y_val, mlp_scratch_preds)
    results_summary.append({
        "Model": "MLP (From Scratch)",
        "Val ROC AUC": mlp_scratch_auc,
        "Train Time (s)": mlp_scratch_time
    })
    model_preds["MLP (Scratch)"] = mlp_scratch_preds
    
    # 5. Treinamento da Regressão Logística
    lr_model, lr_preds, lr_auc, lr_time = train_logistic_regression(X_train_d_arr, y_train, X_val_d_arr, y_val)
    results_summary.append({
        "Model": "Regressão Logística",
        "Val ROC AUC": lr_auc,
        "Train Time (s)": lr_time
    })
    model_preds["Logistic Regression"] = lr_preds
    
    # 6. Treinamento do XGBoost
    xgb_model, xgb_preds, xgb_auc, xgb_time = train_xgboost(X_train_t_arr, y_train, X_val_t_arr, y_val)
    results_summary.append({
        "Model": "XGBoost",
        "Val ROC AUC": xgb_auc,
        "Train Time (s)": xgb_time
    })
    model_preds["XGBoost"] = xgb_preds
    
    # 7. Treinamento do LightGBM
    lgb_model, lgb_preds, lgb_auc, lgb_time = train_lightgbm(X_train_t_arr, y_train, X_val_t_arr, y_val)
    results_summary.append({
        "Model": "LightGBM",
        "Val ROC AUC": lgb_auc,
        "Train Time (s)": lgb_time
    })
    model_preds["LightGBM"] = lgb_preds
    
    # 8. Treinamento do CatBoost
    cat_model, cat_preds, cat_auc, cat_time = train_catboost(X_train_t_arr, y_train, X_val_t_arr, y_val)
    results_summary.append({
        "Model": "CatBoost",
        "Val ROC AUC": cat_auc,
        "Train Time (s)": cat_time
    })
    model_preds["CatBoost"] = cat_preds
    
    # 9. Treinamento do TabTransformer (Atenção sobre colunas categóricas + features numéricas)
    print("\n--- [Classificação] Treinando TabTransformer ---")
    tabtransformer_start = time.time()
    tabtransformer_model, tabtransformer_preds = train_tabtransformer(
        X_train_t, y_train, X_val_t, y_val,
        cat_cols=cat_cols, num_cols=num_cols,
        epochs=8, batch_size=512, lr=0.001,
        embed_dim=32, n_heads=8, n_layers=3
    )
    tabtransformer_time = time.time() - tabtransformer_start
    tabtransformer_auc = roc_auc_score(y_val, tabtransformer_preds)
    results_summary.append({
        "Model": "TabTransformer",
        "Val ROC AUC": tabtransformer_auc,
        "Train Time (s)": tabtransformer_time
    })
    model_preds["TabTransformer"] = tabtransformer_preds
    
    # 10. Executa Stacking Ensemble de Modelos
    ensemble_results = run_stacking_ensemble(model_preds, y_val)
    
    # Adiciona resultados do ensemble no sumário
    for ens_name, data in ensemble_results.items():
        results_summary.append({
            "Model": f"Ensemble: {ens_name}",
            "Val ROC AUC": data["auc"],
            "Train Time (s)": np.nan  # Combinado de todos
        })
        
    # 11. Apresenta Resultados e Gera Visualizações
    df_results = pd.DataFrame(results_summary).sort_values(by="Val ROC AUC", ascending=False)
    
    print("\n=================================================================")
    print("                   RESULTADOS COMPARATIVOS FINAIS                ")
    print("=================================================================")
    print(f"Normalização numérica utilizada : {preprocessing_info['num_normalization']}")
    print(f"Encoding categórico utilizado   : {preprocessing_info['cat_encoding']}")
    print(df_results.to_string(index=False, formatters={"Val ROC AUC": "{:.5f}".format, "Train Time (s)": "{:.1f}".format}))
    print("=================================================================")
    
    # Salva tabela de resultados
    os.makedirs("data/processed", exist_ok=True)
    df_results.to_csv("data/processed/model_results.csv", index=False)
    print("Resultados salvos em 'data/processed/model_results.csv'.")

    print("\n=================================================================")
    print("                 SALVANDO MODELOS PARA O KAGGLE                  ")
    print("=================================================================")
    os.makedirs("data/processed/modelos_salvos", exist_ok=True)
    
    # 1. Salvando o Pipeline de Pré-processamento (Obrigatório)
    joblib.dump(pipeline_preproc, "data/processed/saved_models/pipeline_preproc.joblib")
    
    # 2. Salvando o seu melhor modelo individual (XGBoost)
    joblib.dump(
        xgb_model,
        os.path.join("data/processed/saved_models/best_model_xgboost.joblib")
    )
    
    # Gera o gráfico de comparação
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Remove as linhas nulas de tempo para colorir de forma diferente ou focar apenas no ROC AUC
    barplot = sns.barplot(
        x="Val ROC AUC", 
        y="Model", 
        data=df_results, 
        palette="viridis",
        hue="Model",
        legend=False
    )
    plt.xlim(0.5, 0.85)  # Ajusta limites para destacar a diferença no ROC AUC

    # Deixa explícito no próprio gráfico qual normalização/encoding foi usada
    norm_label = preprocessing_info["num_normalization"]
    enc_label = preprocessing_info["cat_encoding"]
    subtitle = f"Normalização numérica: {norm_label}  |  Encoding categórico: {enc_label}"

    plt.suptitle("Comparação de Desempenho dos Modelos (Home Credit - ROC AUC)", fontsize=14, y=0.99)
    plt.title(subtitle, fontsize=10, style="italic", color="dimgray", pad=10)
    plt.xlabel("Validation ROC AUC Score", fontsize=12)
    plt.ylabel("Modelo", fontsize=12)
    
    # Adiciona rótulos numéricos em cada barra
    for index, row in df_results.iterrows():
        plt.text(
            row["Val ROC AUC"] + 0.005, 
            plt.gca().get_yticklabels()[df_results.index.get_loc(index)].get_position()[1],
            f"{row['Val ROC AUC']:.5f}", 
            va='center', 
            fontsize=10, 
            fontweight='bold'
        )
        
    plt.tight_layout()
    norm_tag = str(norm_label).lower().replace(" ", "_")
    enc_tag = str(enc_label).lower().replace(" ", "_")
    chart_path = f"data/processed/model_comparison_norm-{norm_tag}_enc-{enc_tag}.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Gráfico comparativo salvo em '{chart_path}'.")
    
    total_pipeline_time = time.time() - start_pipeline_time
    print(f"\nPipeline finalizado com sucesso absoluto em {total_pipeline_time/60:.2f} minutos!")

if __name__ == "__main__":
    main()
