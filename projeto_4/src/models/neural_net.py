import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

class ClassificationDataset(Dataset):
    """Dataset para treinamento supervisionado de classificação binária"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLPClassifier(nn.Module):
    """
    Classificador MLP que usa opcionalmente o encoder pré-treinado do DAE
    e adiciona camadas finais para classificação binária.
    """
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dim: int = 256):
        super(MLPClassifier, self).__init__()
        
        # O encoder deve ter exatamente a mesma estrutura do Denoising Autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        
        # Cabeça de Classificação (Classification Head)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Retorna logits para BCEWithLogitsLoss
        )
        
    def load_pretrained_encoder(self, weights_path: str):
        """Carrega os pesos do encoder pré-treinado pelo Autoencoder"""
        if os.path.exists(weights_path):
            print(f"Carregando pesos pré-treinados do Autoencoder de: {weights_path}")
            self.encoder.load_state_dict(torch.load(weights_path))
        else:
            print(f"Aviso: Arquivo de pesos '{weights_path}' não encontrado. Iniciando do zero.")
            
    def forward(self, x):
        latent = self.encoder(x)
        logits = self.classifier(latent)
        return logits

def train_mlp(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 8, batch_size: int = 512, lr: float = 0.001,
              pretrained_weights: str = None,
              save_path: str = "data/processed/mlp_best_model.pth") -> tuple:
    """
    Treina a MLP de classificação e retorna o modelo treinado e as probabilidades de validação.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando MLP de Classificação no dispositivo: {device}")
    
    input_dim = X_train.shape[1]
    model = MLPClassifier(input_dim).to(device)
    
    # Se pesos do autoencoder forem providos, carrega para Transfer Learning
    if pretrained_weights:
        model.load_pretrained_encoder(pretrained_weights)
        
    train_dataset = ClassificationDataset(X_train, y_train)
    val_dataset = ClassificationDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Usamos BCEWithLogitsLoss por estabilidade numérica (aplica Sigmoid internamente)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_auc = 0.0
    best_preds = None
    best_weights_path = save_path
    os.makedirs(os.path.dirname(best_weights_path) or ".", exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        # Modo Treino
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"MLP Epoch {epoch}/{epochs}")
        for x_batch, y_batch in progress_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Forward
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        # Modo Avaliação (Validação)
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                logits = model(x_batch)
                
                # Converte os logits em probabilidades usando sigmoid
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.append(probs)
                val_targets.append(y_batch.numpy())
                
        val_preds = np.vstack(val_preds).flatten()
        val_targets = np.vstack(val_targets).flatten()
        
        # Calcula ROC AUC de validação
        epoch_auc = roc_auc_score(val_targets, val_preds)
        avg_loss = total_loss / len(train_loader)
        
        print(f"Epoch {epoch} finalizada. Loss: {avg_loss:.4f} | Val ROC AUC: {epoch_auc:.5f}")
        
        # Salva o melhor modelo com base em ROC AUC
        if epoch_auc > best_auc:
            best_auc = epoch_auc
            best_preds = val_preds
            torch.save(model.state_dict(), best_weights_path)
            print(f"--> Novo melhor modelo salvo com ROC AUC: {best_auc:.5f}!")
            
    print(f"Treinamento concluído. Melhor ROC AUC atingido na validação: {best_auc:.5f}")
    
    # Carrega os melhores pesos antes de retornar
    model.load_state_dict(torch.load(best_weights_path))
    return model, best_preds

if __name__ == "__main__":
    # Teste rápido se executado diretamente
    X_dummy = np.random.randn(1000, 50)
    y_dummy = np.random.randint(0, 2, 1000)
    train_mlp(X_dummy, y_dummy, X_dummy, y_dummy, epochs=1)
