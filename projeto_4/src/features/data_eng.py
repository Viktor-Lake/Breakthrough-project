import os
import polars as pl
import numpy as np

def aggregate_bureau(path: str) -> pl.LazyFrame:
    """Processa e agrega a tabela bureau.csv"""
    print("Processando bureau.csv...")
    df = pl.scan_csv(path)
    
    # Agregações por cliente
    agg_df = df.group_by("SK_ID_CURR").agg([
        pl.len().alias("BUREAU_LOAN_COUNT"),
        pl.col("AMT_CREDIT_SUM").mean().alias("BUREAU_AMT_CREDIT_SUM_MEAN"),
        pl.col("AMT_CREDIT_SUM").max().alias("BUREAU_AMT_CREDIT_SUM_MAX"),
        pl.col("AMT_CREDIT_SUM_DEBT").mean().alias("BUREAU_AMT_CREDIT_SUM_DEBT_MEAN"),
        pl.col("AMT_CREDIT_SUM_DEBT").sum().alias("BUREAU_AMT_CREDIT_SUM_DEBT_SUM"),
        pl.col("DAYS_CREDIT").mean().alias("BUREAU_DAYS_CREDIT_MEAN"),
        pl.col("DAYS_CREDIT").min().alias("BUREAU_DAYS_CREDIT_MIN")
    ])
    return agg_df

def aggregate_previous_applications(path: str) -> pl.LazyFrame:
    """Processa e agrega a tabela previous_application.csv"""
    print("Processando previous_application.csv...")
    df = pl.scan_csv(path)
    
    # Agregações por cliente
    agg_df = df.group_by("SK_ID_CURR").agg([
        pl.len().alias("PREV_LOAN_COUNT"),
        pl.col("AMT_CREDIT").mean().alias("PREV_AMT_CREDIT_MEAN"),
        pl.col("AMT_CREDIT").max().alias("PREV_AMT_CREDIT_MAX"),
        pl.col("AMT_ANNUITY").mean().alias("PREV_AMT_ANNUITY_MEAN"),
        pl.col("AMT_ANNUITY").max().alias("PREV_AMT_ANNUITY_MAX"),
        pl.col("RATE_DOWN_PAYMENT").mean().alias("PREV_RATE_DOWN_PAYMENT_MEAN"),
        pl.col("DAYS_DECISION").mean().alias("PREV_DAYS_DECISION_MEAN")
    ])
    return agg_df

def aggregate_pos_cash(path: str) -> pl.LazyFrame:
    """Processa e agrega a tabela POS_CASH_balance.csv"""
    print("Processando POS_CASH_balance.csv...")
    df = pl.scan_csv(path)
    
    # Agregações por cliente
    agg_df = df.group_by("SK_ID_CURR").agg([
        pl.len().alias("POS_LOAN_COUNT"),
        pl.col("MONTHS_BALANCE").max().alias("POS_MONTHS_BALANCE_MAX"),
        pl.col("SK_DPD").max().alias("POS_SK_DPD_MAX"),
        pl.col("SK_DPD").mean().alias("POS_SK_DPD_MEAN"),
        pl.col("CNT_INSTALMENT_FUTURE").mean().alias("POS_CNT_INSTALMENT_FUTURE_MEAN")
    ])
    return agg_df

def aggregate_installments(path: str) -> pl.LazyFrame:
    """Processa e agrega a tabela installments_payments.csv"""
    print("Processando installments_payments.csv...")
    df = pl.scan_csv(path)
    
    # Calcula atraso de pagamento
    df = df.with_columns([
        (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT")).alias("DPD")
    ]).with_columns([
        pl.when(pl.col("DPD") > 0).then(pl.col("DPD")).otherwise(0).alias("DPD_POS")
    ])
    
    # Agregações por cliente
    agg_df = df.group_by("SK_ID_CURR").agg([
        pl.len().alias("INST_PAYMENT_COUNT"),
        pl.col("DPD_POS").mean().alias("INST_DPD_MEAN"),
        pl.col("DPD_POS").max().alias("INST_DPD_MAX"),
        pl.col("AMT_PAYMENT").mean().alias("INST_AMT_PAYMENT_MEAN"),
        pl.col("AMT_PAYMENT").sum().alias("INST_AMT_PAYMENT_SUM"),
        pl.col("AMT_INSTALMENT").mean().alias("INST_AMT_INSTALMENT_MEAN")
    ])
    return agg_df

def process_application(path: str, is_train: bool = True) -> pl.LazyFrame:
    """Processa e limpa a tabela principal application_train.csv / application_test.csv"""
    print(f"Processando tabela principal: {path}...")
    df = pl.scan_csv(path)
    
    # Razões financeiras e engenharia básica de atributos
    df = df.with_columns([
        (pl.col("AMT_INCOME_TOTAL") / pl.col("AMT_CREDIT")).alias("INCOME_CREDIT_PERC"),
        (pl.col("AMT_ANNUITY") / pl.col("AMT_INCOME_TOTAL")).alias("ANNUITY_INCOME_PERC"),
        (pl.col("AMT_ANNUITY") / pl.col("AMT_CREDIT")).alias("PAYMENT_RATE"),
        (pl.col("DAYS_BIRTH") / -365.25).alias("AGE"),
        (pl.col("DAYS_EMPLOYED") / -365.25).alias("YEARS_EMPLOYED")
    ])
    
    # Corrige anomalia comum no DAYS_EMPLOYED (365243 dias é erro de preenchimento)
    df = df.with_columns([
        pl.when(pl.col("DAYS_EMPLOYED") == 365243).then(None).otherwise(pl.col("DAYS_EMPLOYED")).alias("DAYS_EMPLOYED")
    ])
    
    return df

def optimize_types(df: pl.DataFrame) -> pl.DataFrame:
    """Otimiza os tipos de dados para reduzir consumo de memória (de Float64 para Float32, Int64 para Int32)"""
    print("Otimizando tipos de dados para reduzir uso de memória...")
    new_cols = []
    for col in df.columns:
        dtype = df.schema[col]
        if dtype == pl.Float64:
            new_cols.append(pl.col(col).cast(pl.Float32))
        elif dtype == pl.Int64 and col != "SK_ID_CURR" and col != "TARGET":
            new_cols.append(pl.col(col).cast(pl.Int32))
        else:
            new_cols.append(pl.col(col))
    return df.with_columns(new_cols)

def run_feature_engineering():
    """Função principal para executar a engenharia de atributos completa"""
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Carrega tabelas secundárias como LazyFrames
    bureau_lazy = aggregate_bureau(os.path.join(raw_dir, "bureau.csv"))
    prev_lazy = aggregate_previous_applications(os.path.join(raw_dir, "previous_application.csv"))
    pos_lazy = aggregate_pos_cash(os.path.join(raw_dir, "POS_CASH_balance.csv"))
    inst_lazy = aggregate_installments(os.path.join(raw_dir, "installments_payments.csv"))
    
    # Processa datasets de treino e teste
    for phase, file_name in [("train", "application_train.csv"), ("test", "application_test.csv")]:
        file_path = os.path.join(raw_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Aviso: Arquivo {file_path} não encontrado. Pulando etapa.")
            continue
            
        app_lazy = process_application(file_path, is_train=(phase == "train"))
        
        # Junção à esquerda (Left Joins) com todas as tabelas agregadas
        print(f"Realizando joins para o conjunto de {phase}...")
        final_lazy = (
            app_lazy
            .join(bureau_lazy, on="SK_ID_CURR", how="left")
            .join(prev_lazy, on="SK_ID_CURR", how="left")
            .join(pos_lazy, on="SK_ID_CURR", how="left")
            .join(inst_lazy, on="SK_ID_CURR", how="left")
        )
        
        # Executa as transformações e traz o resultado para a memória (coleta física)
        print(f"Coletando e processando os dados de {phase} de forma paralela...")
        final_df = final_lazy.collect()
        
        # Otimiza o tipo das colunas
        final_df = optimize_types(final_df)
        
        # Salva o arquivo final no formato altamente eficiente Parquet
        output_path = os.path.join(processed_dir, f"{phase}_features.parquet")
        print(f"Salvando o dataset final em {output_path}...")
        final_df.write_parquet(output_path)
        print(f"Sucesso! Dimensões do dataset de {phase}: {final_df.shape}\n")

if __name__ == "__main__":
    run_feature_engineering()
