import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

class TabularDataset(Dataset):
    """Dataset simples para carregar dados do numpy no PyTorch"""
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx]

class DenoisingAutoencoder(nn.Module):
    """
    Arquitetura de Denoising Autoencoder (DAE) para Dados Tabulares.
    """
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dim: int = 256):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder (Compactação)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        
        # Decoder (Reconstrução)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x, noise_factor: float = 0.15):
        # Injeta ruído do tipo Masking (zerar uma % aleatória das variáveis)
        if self.training and noise_factor > 0:
            mask = torch.rand_like(x) > noise_factor
            x_corrupted = x * mask
        else:
            x_corrupted = x
            
        latent = self.encoder(x_corrupted)
        reconstruction = self.decoder(latent)
        return reconstruction

def train_dae(X_train: np.ndarray, epochs: int = 5, batch_size: int = 512, lr: float = 0.001, latent_dim: int = 128):
    """
    Função de treinamento do Denoising Autoencoder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando Autoencoder no dispositivo: {device}")
    
    input_dim = X_train.shape[1]
    model = DenoisingAutoencoder(input_dim, latent_dim=latent_dim).to(device)
    
    dataset = TabularDataset(X_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        
        for batch in progress_bar:
            batch = batch.to(device)
            
            # Forward
            reconstruction = model(batch, noise_factor=0.2)
            loss = criterion(reconstruction, batch)  # Compara reconstrução com o dado limpo original
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} finalizada. Erro médio de reconstrução (MSE): {avg_loss:.5f}")
        
    # Salva os pesos do encoder para transfer learning
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "encoder_weights.pth")
    
    # Salva apenas a parte do encoder
    torch.save(model.encoder.state_dict(), weights_path)
    print(f"Pesos do Encoder salvos em {weights_path} com sucesso!")
    return weights_path

if __name__ == "__main__":
    # Teste rápido se executado diretamente
    X_dummy = np.random.randn(1000, 100)
    train_dae(X_dummy, epochs=1)
