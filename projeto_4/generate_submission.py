import os
import joblib
import pandas as pd
import numpy as np
import polars as pl

# Importa a sua função de transformação diretamente do data_loader
from src.models.data_loader import transform_test_data

def generate_submission():
    print("=================================================================")
    print("          GERANDO ARQUIVO DE SUBMISSÃO PARA O KAGGLE             ")
    print("=================================================================")
    
    # MUDANÇA AQUI: Apontando para os dados com as features já criadas!
    test_data_path = "data/processed/test_features.parquet" 
    output_path = "data/processed/submission_kaggle.csv"
    pipeline_path = "data/processed/saved_models/pipeline_preproc.joblib"
    model_path = "data/processed/saved_models/best_model_xgboost.joblib"
    
    # 1. Verificações de segurança
    if not os.path.exists(test_data_path):
        print(f"Erro: Arquivo de teste não encontrado em {test_data_path}")
        print("Certifique-se de que o data_eng.py foi executado para gerar este arquivo.")
        return
        
    if not os.path.exists(model_path) or not os.path.exists(pipeline_path):
        print("Erro: Arquivos do modelo ou pipeline não encontrados.")
        print(f"Verifique se a pasta 'data/processed/saved_models' contém os arquivos.")
        return

    # 2. Carregar o modelo e o pipeline
    print("Carregando pipeline de pré-processamento...")
    pipeline_preproc = joblib.load(pipeline_path)
    
    print("Carregando o modelo vencedor (XGBoost)...")
    best_model = joblib.load(model_path)

    # 3. Ler dados de teste (já com as features criadas)
    print(f"Lendo dados de teste de: {test_data_path}...")
    
    # Lendo direto com Pandas (facilita o processo pois o parquet retém as tipagens)
    df_test_pd = pd.read_parquet(test_data_path)
    test_ids = df_test_pd["SK_ID_CURR"].to_numpy()

    # 4. Transformar os dados e alinhar features
    print("Aplicando transformações matemáticas e alinhando colunas...")
    
    X_test_tree = transform_test_data(
        test_df=df_test_pd, 
        pipeline_preproc=pipeline_preproc, 
        metadata=pipeline_preproc
    )

    # 5. Fazer Previsões
    print("Calculando o risco de inadimplência (TARGET=1)...")
    predictions = best_model.predict_proba(X_test_tree)[:, 1]

    # 6. Salvar submissão
    print("Gerando arquivo final...")
    submission = pl.DataFrame({
        "SK_ID_CURR": test_ids,
        "TARGET": predictions
    })
    submission.write_csv(output_path)
    
    print(f"\nSUCESSO! Submissão salva em: {output_path}")
    print("Pronto para envio no Kaggle!")

if __name__ == "__main__":
    generate_submission()