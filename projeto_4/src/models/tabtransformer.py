import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


class TabTransformerDataset(Dataset):
    """Dataset que separa colunas categóricas (índices inteiros) das colunas numéricas contínuas."""
    def __init__(self, X_cat: np.ndarray, X_num: np.ndarray, y: np.ndarray = None):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_cat[idx], self.X_num[idx], self.y[idx]
        return self.X_cat[idx], self.X_num[idx]


class TabTransformer(nn.Module):
    """
    Implementação do TabTransformer (Huang et al., 2020).

    - Cada coluna categórica recebe um embedding próprio.
    - Os embeddings categóricos são tratados como uma sequência de "tokens" e passam
      por um Transformer Encoder (self-attention entre as colunas categóricas).
    - As features numéricas são normalizadas (LayerNorm) e concatenadas com a saída
      contextualizada do Transformer (achatada).
    - Uma MLP final (head) produz o logit de classificação binária.
    """
    def __init__(self, cat_cardinalities: list, num_continuous: int,
                 embed_dim: int = 32, n_heads: int = 8, n_layers: int = 3,
                 ff_dim: int = 128, attn_dropout: float = 0.1, ff_dropout: float = 0.1,
                 mlp_hidden_dims: tuple = (128, 64)):
        super(TabTransformer, self).__init__()

        self.n_cat = len(cat_cardinalities)
        self.embed_dim = embed_dim

        # Um embedding independente por coluna categórica
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embed_dim) for cardinality in cat_cardinalities
        ])

        # Bloco Transformer para contextualizar as colunas categóricas entre si
        if self.n_cat > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=attn_dropout,
                activation="relu",
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            self.transformer = None

        # Normalização das features numéricas contínuas
        self.num_norm = nn.LayerNorm(num_continuous) if num_continuous > 0 else None

        # Dimensão de entrada da MLP: (colunas categóricas contextualizadas achatadas) + (numéricas)
        mlp_input_dim = (self.n_cat * embed_dim) + num_continuous

        head_layers = []
        prev_dim = mlp_input_dim
        for hidden_dim in mlp_hidden_dims:
            head_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(ff_dropout)
            ])
            prev_dim = hidden_dim
        head_layers.append(nn.Linear(prev_dim, 1))  # Logit final (BCEWithLogitsLoss)
        self.mlp_head = nn.Sequential(*head_layers)

    def forward(self, x_cat, x_num):
        parts = []

        if self.n_cat > 0:
            # Gera embedding de cada coluna categórica e monta a sequência (batch, n_cat, embed_dim)
            embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
            tokens = torch.stack(embeddings, dim=1)

            # Self-attention entre as colunas categóricas
            contextualized = self.transformer(tokens)

            # Achata a sequência contextualizada para concatenar com as features numéricas
            contextualized_flat = contextualized.reshape(contextualized.size(0), -1)
            parts.append(contextualized_flat)

        if self.num_norm is not None:
            parts.append(self.num_norm(x_num))

        x = torch.cat(parts, dim=1)
        logits = self.mlp_head(x)
        return logits


def _prepare_cat_num_arrays(df: pd.DataFrame, cat_cols: list, num_cols: list,
                             num_medians: pd.Series = None, num_stats: tuple = None):
    """
    Extrai e prepara as matrizes categórica (int64) e numérica (float32, imputada e padronizada)
    a partir do dataframe "tree" (que já vem com categorias ordinal-encoded e nulos preservados
    nas colunas numéricas).
    """
    X_cat = df[cat_cols].values.astype(np.int64) if cat_cols else np.zeros((len(df), 0), dtype=np.int64)

    if num_cols:
        X_num_df = df[num_cols].copy()

        # Imputação de nulos pela mediana (calculada no treino e reaplicada na validação)
        if num_medians is None:
            num_medians = X_num_df.median()
        X_num_df = X_num_df.fillna(num_medians)

        X_num = X_num_df.values.astype(np.float32)

        # Padronização (mean/std calculados no treino e reaplicados na validação)
        if num_stats is None:
            mean = X_num.mean(axis=0)
            std = X_num.std(axis=0)
            std[std == 0] = 1.0
            num_stats = (mean, std)
        mean, std = num_stats
        X_num = (X_num - mean) / std
    else:
        X_num = np.zeros((len(df), 0), dtype=np.float32)
        num_medians = pd.Series(dtype=float)
        num_stats = (np.zeros(0), np.ones(0))

    return X_cat, X_num, num_medians, num_stats


def train_tabtransformer(X_train_tree: pd.DataFrame, y_train: np.ndarray,
                          X_val_tree: pd.DataFrame, y_val: np.ndarray,
                          cat_cols: list, num_cols: list,
                          epochs: int = 10, batch_size: int = 512, lr: float = 0.001,
                          embed_dim: int = 32, n_heads: int = 8, n_layers: int = 3) -> tuple:
    """
    Treina o TabTransformer para classificação binária e retorna o modelo treinado
    e as probabilidades de validação (melhor época por ROC AUC), no mesmo padrão
    das demais funções train_* do projeto.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando TabTransformer no dispositivo: {device}")

    # Prepara matrizes categóricas (int) e numéricas (float, imputadas/padronizadas)
    X_train_cat, X_train_num, num_medians, num_stats = _prepare_cat_num_arrays(X_train_tree, cat_cols, num_cols)
    X_val_cat, X_val_num, _, _ = _prepare_cat_num_arrays(X_val_tree, cat_cols, num_cols,
                                                          num_medians=num_medians, num_stats=num_stats)

    # Cardinalidade de cada coluna categórica (considera treino + validação, +1 de margem de segurança)
    cat_cardinalities = []
    for i in range(len(cat_cols)):
        max_code = max(int(X_train_cat[:, i].max()) if len(X_train_cat) else 0,
                        int(X_val_cat[:, i].max()) if len(X_val_cat) else 0)
        cat_cardinalities.append(max_code + 2)  # +1 para índice 0-based, +1 de margem

    model = TabTransformer(
        cat_cardinalities=cat_cardinalities,
        num_continuous=len(num_cols),
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers
    ).to(device)

    train_dataset = TabTransformerDataset(X_train_cat, X_train_num, y_train)
    val_dataset = TabTransformerDataset(X_val_cat, X_val_num, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_auc = 0.0
    best_preds = None
    best_weights_path = "data/processed/tabtransformer_best_model.pth"
    os.makedirs("data/processed", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"TabTransformer Epoch {epoch}/{epochs}")
        for x_cat_batch, x_num_batch, y_batch in progress_bar:
            x_cat_batch = x_cat_batch.to(device)
            x_num_batch = x_num_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_cat_batch, x_num_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for x_cat_batch, x_num_batch, y_batch in val_loader:
                x_cat_batch = x_cat_batch.to(device)
                x_num_batch = x_num_batch.to(device)

                logits = model(x_cat_batch, x_num_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.append(probs)
                val_targets.append(y_batch.numpy())

        val_preds = np.vstack(val_preds).flatten()
        val_targets = np.vstack(val_targets).flatten()

        epoch_auc = roc_auc_score(val_targets, val_preds)
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch} finalizada. Loss: {avg_loss:.4f} | Val ROC AUC: {epoch_auc:.5f}")

        if epoch_auc > best_auc:
            best_auc = epoch_auc
            best_preds = val_preds
            torch.save(model.state_dict(), best_weights_path)
            print(f"--> Novo melhor modelo salvo com ROC AUC: {best_auc:.5f}!")

    print(f"Treinamento concluído. Melhor ROC AUC atingido na validação: {best_auc:.5f}")

    model.load_state_dict(torch.load(best_weights_path))
    return model, best_preds


if __name__ == "__main__":
    # Teste rápido se executado diretamente
    n_samples = 1000
    df_dummy = pd.DataFrame({
        "cat_a": np.random.randint(0, 5, n_samples).astype(float),
        "cat_b": np.random.randint(0, 3, n_samples).astype(float),
        "num_a": np.random.randn(n_samples),
        "num_b": np.random.randn(n_samples),
    })
    y_dummy = np.random.randint(0, 2, n_samples)
    train_tabtransformer(df_dummy, y_dummy, df_dummy, y_dummy,
                          cat_cols=["cat_a", "cat_b"], num_cols=["num_a", "num_b"], epochs=1)
