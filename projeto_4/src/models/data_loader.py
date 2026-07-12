import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    QuantileTransformer,
    PowerTransformer,
    TargetEncoder,
)
from sklearn.impute import SimpleImputer
import os

# =====================================================================================
# CONFIGURAÇÃO DE PRÉ-PROCESSAMENTO — MODIFIQUE AQUI
# =====================================================================================
# Este é o arquivo (src/models/data_loader.py) onde se controla como os dados
# numéricos e categóricos são tratados ANTES de virarem tensores para os modelos
# densos (Regressão Logística / MLP). As duas variáveis abaixo são o "painel de
# controle" do pipeline.
#
# ---------------------------------------------------------------------------
# 1) NORMALIZAÇÃO DOS DADOS NUMÉRICOS (NUM_NORMALIZATION)
# ---------------------------------------------------------------------------
# Escolha UMA das opções abaixo (strings) ou None para não normalizar:
#   None         -> NÃO aplica nenhuma normalização (usa os valores imputados "crus")
#   "zscore"     -> Z-Score / StandardScaler        (média 0, desvio padrão 1)
#   "minmax"     -> Min-Max Scaling                 (escala tudo para o intervalo [0, 1])
#   "robust"     -> Robust Scaler                   (usa mediana/IQR, robusto a outliers)
#   "maxabs"     -> MaxAbsScaler                    (escala pelo valor absoluto máximo, mantém esparsidade/sinal)
#   "quantile"   -> Quantile Transformer            (mapeia para distribuição uniforme/normal)
#   "power"      -> Power Transformer (Yeo-Johnson) (aproxima os dados de uma distribuição normal)
NUM_NORMALIZATION = "zscore"

# ---------------------------------------------------------------------------
# 2) CODIFICAÇÃO DOS DADOS CATEGÓRICOS (CAT_ENCODING)
# ---------------------------------------------------------------------------
# Escolha UMA das opções abaixo:
#   None          -> NÃO usa as colunas categóricas no pipeline denso (elas são descartadas)
#   "target"      -> Target Encoding (cada categoria vira a média do TARGET nessa categoria,
#                    com cross-fitting interno do sklearn para reduzir vazamento de dados)
#   "embedding"   -> Embeddings aprendidos por rede neural (Entity Embeddings), treinados de
#                    forma supervisionada para prever o TARGET e depois extraídos como
#                    features densas
CAT_ENCODING = "target"

# Dimensão de cada embedding categórico (só é usado se CAT_ENCODING == "embedding")
CAT_EMBEDDING_DIM = 8
# =====================================================================================


def _get_scaler(method: str):
    """
    Fábrica de scalers para os dados numéricos. Retorna None se nenhuma
    normalização deve ser aplicada.
    """
    if method is None:
        return None

    method = method.lower()
    if method == "zscore":
        return StandardScaler()
    if method == "minmax":
        return MinMaxScaler()
    if method == "robust":
        return RobustScaler()
    if method == "maxabs":
        return MaxAbsScaler()
    if method == "quantile":
        return QuantileTransformer(output_distribution="normal", random_state=42)
    if method == "power":
        return PowerTransformer(method="yeo-johnson")

    raise ValueError(
        f"Método de normalização '{method}' desconhecido. "
        f"Opções válidas: None, 'zscore', 'minmax', 'robust', 'maxabs', 'quantile', 'power'."
    )


class _EntityEmbeddingNet(nn.Module):
    """
    Rede neural simples usada apenas para aprender Embeddings supervisionados
    (Entity Embeddings) das colunas categóricas. Cada coluna categórica recebe sua
    própria camada nn.Embedding; todas são concatenadas e passam por uma camada
    linear que tenta prever o TARGET. Depois do treino, descartamos a camada de
    previsão e usamos apenas os vetores de embedding como features densas.
    """

    def __init__(self, cardinalities: list, embedding_dim: int):
        super().__init__()
        # +1 em cada cardinalidade para reservar o índice 0 a categorias desconhecidas
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card + 1, embedding_dim) for card in cardinalities]
        )
        self.head = nn.Linear(embedding_dim * len(cardinalities), 1)

    def forward(self, x):
        embs = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        concat = torch.cat(embs, dim=1)
        logits = self.head(concat)
        return logits, concat


def _learn_categorical_embeddings(
    X_cat_train_int: np.ndarray,
    y_train: np.ndarray,
    X_cat_val_int: np.ndarray,
    cardinalities: list,
    embedding_dim: int = 8,
    epochs: int = 5,
    batch_size: int = 1024,
    lr: float = 0.01,
):
    """
    Treina os Entity Embeddings usando SOMENTE os dados de treino (para não vazar
    informação do TARGET de validação) e depois aplica (apenas forward, sem
    treinar) a mesma rede nos dados de treino e validação para extrair os vetores
    finais de embedding.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _EntityEmbeddingNet(cardinalities, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_cat_train_int, dtype=torch.long)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_train_dev = torch.tensor(X_cat_train_int, dtype=torch.long).to(device)
        X_val_dev = torch.tensor(X_cat_val_int, dtype=torch.long).to(device)
        _, emb_train = model(X_train_dev)
        _, emb_val = model(X_val_dev)

    return emb_train.cpu().numpy(), emb_val.cpu().numpy()


def load_and_preprocess_data(
    parquet_path: str = "data/processed/train_features.parquet",
    test_size: float = 0.2,
    random_state: int = 42,
    num_normalization: str = NUM_NORMALIZATION,
    cat_encoding: str = CAT_ENCODING,
):
    """
    Carrega o dataset parquet e prepara dois conjuntos de dados:
    1. Dense (para Regressão Logística e MLP): Imputado, Normalizado (opcional) e
       com codificação categórica configurável (nenhuma / Target Encoding / Embeddings).
    2. Tree (para XGBoost, LightGBM, CatBoost): Mantém nulos e usa Ordinal/Label
       Encoding para categóricos.

    Parâmetros
    ----------
    num_normalization : ver NUM_NORMALIZATION no topo do arquivo.
    cat_encoding      : ver CAT_ENCODING no topo do arquivo.
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
    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_raw.select_dtypes(include=["number"]).columns.tolist()

    print(f"Colunas encontradas: {len(num_cols)} numéricas, {len(cat_cols)} categóricas.")
    print(f"Normalização numérica selecionada: {num_normalization!r}")
    print(f"Codificação categórica selecionada: {cat_encoding!r}")

    # --- DIVISÃO DOS DADOS (TRAIN / VALIDATION ESTRATIFICADO) ---
    # Feita ANTES de ajustar qualquer scaler/encoder, para que a normalização e a
    # codificação categórica sejam aprendidas apenas com dados de treino (evita
    # vazamento de informação da validação, especialmente crítico no Target Encoding).
    print("Dividindo os dados de forma estratificada...")
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train_raw = X_train_raw.reset_index(drop=True)
    X_val_raw = X_val_raw.reset_index(drop=True)

    # --- PIPELINE TREE (Árvores de Decisão) ---
    print("Preparando pipeline para modelos de árvore (Tree)...")
    X_train_tree = X_train_raw.copy()
    X_val_tree = X_val_raw.copy()

    for col in cat_cols:
        X_train_tree[col] = X_train_tree[col].astype(str).fillna("MISSING")
        X_val_tree[col] = X_val_tree[col].astype(str).fillna("MISSING")

    if cat_cols:
        encoder_tree = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_tree[cat_cols] = encoder_tree.fit_transform(X_train_tree[cat_cols])
        X_val_tree[cat_cols] = encoder_tree.transform(X_val_tree[cat_cols])
    else:
        encoder_tree = None

    X_train_tree = X_train_tree.astype(np.float32)
    X_val_tree = X_val_tree.astype(np.float32)

    # --- PIPELINE DENSE (Regressão Logística e Redes Neurais) ---
    print("Preparando pipeline para modelos densos (LR / MLP)...")

    # 1. Tratamento numérico: Imputação (mediana) + Normalização configurável
    imputer_num = SimpleImputer(strategy="median")
    X_train_num_imputed = imputer_num.fit_transform(X_train_raw[num_cols])
    X_val_num_imputed = imputer_num.transform(X_val_raw[num_cols])

    scaler_num = _get_scaler(num_normalization)
    if scaler_num is not None:
        X_train_num_final = scaler_num.fit_transform(X_train_num_imputed)
        X_val_num_final = scaler_num.transform(X_val_num_imputed)
    else:
        X_train_num_final = X_train_num_imputed
        X_val_num_final = X_val_num_imputed

    df_num_train = pd.DataFrame(X_train_num_final, columns=num_cols)
    df_num_val = pd.DataFrame(X_val_num_final, columns=num_cols)

    # 2. Tratamento categórico: nenhuma / Target Encoding / Embeddings
    target_encoder = None  # <--- ADICIONE ESTA LINHA AQUI

    if cat_encoding is None or not cat_cols:
        df_cat_train = None
        df_cat_val = None

    elif cat_encoding.lower() == "target":
        X_cat_train_raw = X_train_raw[cat_cols].fillna("MISSING").astype(str)
        X_cat_val_raw = X_val_raw[cat_cols].fillna("MISSING").astype(str)

        target_encoder = TargetEncoder(target_type="binary", random_state=random_state)
        X_cat_train_enc = target_encoder.fit_transform(X_cat_train_raw, y_train)
        X_cat_val_enc = target_encoder.transform(X_cat_val_raw)

        enc_col_names = [f"{col}_target_enc" for col in cat_cols]
        df_cat_train = pd.DataFrame(X_cat_train_enc, columns=enc_col_names)
        df_cat_val = pd.DataFrame(X_cat_val_enc, columns=enc_col_names)

    elif cat_encoding.lower() == "embedding":
        # Reaproveita a codificação ordinal (ajustada só no treino) já calculada
        # acima para o pipeline de árvores, deslocando +1 para reservar o 0 aos
        # valores desconhecidos (-1 -> 0) e alimentar a camada nn.Embedding.
        X_cat_train_int = (X_train_tree[cat_cols].values.astype(int) + 1)
        X_cat_val_int = (X_val_tree[cat_cols].values.astype(int) + 1)
        cardinalities = [len(cats) for cats in encoder_tree.categories_]

        emb_train, emb_val = _learn_categorical_embeddings(
            X_cat_train_int, y_train, X_cat_val_int,
            cardinalities=cardinalities,
            embedding_dim=CAT_EMBEDDING_DIM,
        )

        emb_col_names = [
            f"{col}_emb_{i}" for col in cat_cols for i in range(CAT_EMBEDDING_DIM)
        ]
        df_cat_train = pd.DataFrame(emb_train, columns=emb_col_names)
        df_cat_val = pd.DataFrame(emb_val, columns=emb_col_names)

    else:
        raise ValueError(
            f"Método de codificação categórica '{cat_encoding}' desconhecido. "
            f"Opções válidas: None, 'target', 'embedding'."
        )

    if df_cat_train is not None:
        X_train_dense = pd.concat([df_num_train, df_cat_train], axis=1)
        X_val_dense = pd.concat([df_num_val, df_cat_val], axis=1)
    else:
        X_train_dense = df_num_train
        X_val_dense = df_num_val

    print(f"Sucesso! Conjunto de Treino: {X_train_tree.shape[0]} amostras, Validação: {X_val_tree.shape[0]} amostras.")
    print(f"Colunas de Árvore: {X_train_tree.shape[1]} | Colunas Densa: {X_train_dense.shape[1]}")

    # Descrição legível usada nos gráficos/relatórios (ex.: título do gráfico de ROC AUC)
    preprocessing_info = {
        "num_normalization": num_normalization if num_normalization else "Nenhuma",
        "cat_encoding": cat_encoding if cat_encoding else "Nenhuma",
    }

    fitted_preprocessors = {
        "imputer_num": imputer_num,
        "scaler_num": scaler_num,
        "encoder_tree": encoder_tree,
        "target_encoder": target_encoder,
        "cat_cols": cat_cols,  # Salvamos o nome das colunas para facilitar no teste
        "num_cols": num_cols,
        "feature_order": list(X_train_tree.columns)
    }

    return (
        X_train_dense, X_val_dense,
        X_train_tree, X_val_tree,
        y_train, y_val,
        cat_cols, num_cols,
        preprocessing_info,
        fitted_preprocessors
    )


def transform_test_data(
    test_df: pd.DataFrame,
    pipeline_preproc: dict,
    metadata: dict,
):
    """
    Aplica ao conjunto de teste exatamente o mesmo pré-processamento
    aprendido durante o treinamento.

    Esta função NÃO realiza nenhum ajuste (fit) dos preprocessadores,
    apenas reutiliza aqueles salvos pelo main.py.
    """

    id_col = "SK_ID_CURR"

    if id_col in test_df.columns:
        test_df = test_df.drop(columns=[id_col])

    cat_cols = metadata["cat_cols"]
    num_cols = metadata["num_cols"]
    feature_order = metadata["feature_order"]

    # -----------------------------
    # Pipeline Tree (XGBoost)
    # -----------------------------
    X_tree = test_df.copy()

    for col in cat_cols:
        X_tree[col] = (
            X_tree[col]
            .astype(str)
            .fillna("MISSING")
        )

    encoder_tree = pipeline_preproc["encoder_tree"]

    if encoder_tree is not None:
        X_tree[cat_cols] = encoder_tree.transform(
            X_tree[cat_cols]
        )

    X_tree = X_tree.astype(np.float32)

    # Garante exatamente a mesma ordem das colunas
    X_tree = X_tree[feature_order]

    return X_tree

if __name__ == "__main__":
    result = load_and_preprocess_data()
    print(result[-1])
