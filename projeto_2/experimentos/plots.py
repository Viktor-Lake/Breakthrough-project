import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class Plots:
    """
    Classe adaptada para ler os resultados em JSON gerados pelo ExperimentRunner.
    """

    def __init__(self, filename: str):
        # Define o caminho subindo um nível, já que os JSONs estão fora da pasta 'experimentos'
        self.path = os.path.join("..", filename)
        self.data = None
        self.df_vals = None
        self.df_rewards = None

    def load_data(self):
        """Carrega o JSON e prepara os DataFrames para plotagem."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Arquivo não encontrado: {self.path}")

        with open(self.path, 'r') as f:
            self.data = json.load(f)

        # Se for uma lista (resultado do Grid Search do Q-Learning)
        if isinstance(self.data, list):
            # Pegamos o primeiro resultado como exemplo para os plots básicos
            main_data = self.data[0]
            print(f"Carregando dados do Q-Learning: {main_data['params']}")
        else:
            main_data = self.data

        # Converte a Q-Table ou V-Table para DataFrame
        if "final_q_table" in main_data:
            self.df_vals = pd.DataFrame(main_data["final_q_table"])
        elif "v_table" in main_data:
            # V-Table costuma ser um vetor, convertemos para DataFrame de 1 coluna
            self.df_vals = pd.DataFrame(main_data["v_table"], columns=["Value"])

        # Converte o histórico de recompensas
        if "rewards_history" in main_data:
            self.df_rewards = pd.DataFrame(main_data["rewards_history"], columns=["recompensa"])
        elif "backtest_reward" in main_data:
            # No Bellman temos apenas um valor final ou histórico de net_worth
            self.df_rewards = pd.DataFrame(main_data.get("net_worth_history", []), columns=["recompensa"])

    def plot_heatmap(self):
        """Plota heatmap dos valores (Q-Table ou V-Table)."""
        if self.df_vals is None:
            print("Sem dados de valores para plotar.")
            return

        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df_vals, cmap="RdBu_r", annot=True, fmt=".2f")
        plt.title(f"Mapa de Valores ({self.data[0]['model'] if isinstance(self.data, list) else self.data['model']})")
        plt.xlabel("Ações")
        plt.ylabel("Estados")
        plt.show()

    def plot_learning_curve(self):
        """Plota a evolução das recompensas ou saldo final."""
        if self.df_rewards is None or self.df_rewards.empty:
            print("Sem histórico de recompensas para plotar.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.df_rewards["recompensa"], label="Performance")
        plt.title("Evolução do Desempenho")
        plt.xlabel("Episódios / Passos")
        plt.ylabel("Valor")
        plt.legend()
        plt.grid(True)
        plt.show()

    def extract_policy(self):
        """Extrai a melhor ação por estado."""
        if self.df_vals is None:
            return None
        
        # Para Q-Table, o argmax é na linha. Para V-table (1 coluna), a política já vem pronta no JSON.
        if self.df_vals.shape[1] > 1:
            return self.df_vals.idxmax(axis=1)
        else:
            # Se for Bellman, a política costuma estar na chave "policy" do JSON
            source = self.data[0] if isinstance(self.data, list) else self.data
            return source.get("policy", "Política não disponível")