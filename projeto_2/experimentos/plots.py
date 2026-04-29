import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class Plots:
    """
    Classe para análise de experimentos de RL:
        1. Heatmap do mapa de valores (Q-table ou V-table)
        2. Extração da política aprendida
        3. Curva de aprendizado
    """

    def __init__(self, path_valores: str, path_recompensa: str):
        self.path_valores = path_valores
        self.path_recompensa = path_recompensa

        self.df_vals = None
        self.df_rewards = None
        self.df_vals_norm = None

    # Carregamento de planilhas

    def load_data(self):
        """Carrega os dados dos arquivos Excel."""
        self.df_vals = pd.read_excel(self.path_valores)
        self.df_rewards = pd.read_excel(self.path_recompensa)

    # Pré-processamento

    def clean_values(self):
        """Remove colunas vazias ou inválidas."""
        self.df_vals = self.df_vals.drop(
            columns=[col for col in self.df_vals.columns if str(col).strip() == ""]
        )

    def normalize_values(self):
        """Aplica normalização Z-score nos valores."""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.df_vals)
        self.df_vals_norm = pd.DataFrame(scaled, columns=self.df_vals.columns)

    # Visualizações

    def plot_heatmap(self, normalized: bool = True):
        """Plota heatmap dos valores."""
        data = self.df_vals_norm if normalized else self.df_vals

        if data is None:
            raise ValueError("Dados não carregados ou normalizados.")

        plt.figure(figsize=(8, 15))
        sns.heatmap(
            data,
            cmap="RdBu_r",
            annot=False,
            yticklabels=True
        )
        plt.title("Heatmap do Mapa de Valores")
        plt.show()

    def plot_learning_curve(self):
        """Plota curva de aprendizado."""
        if self.df_rewards is None:
            raise ValueError("Dados de recompensa não carregados.")

        time = np.arange(len(self.df_rewards))

        plt.figure(figsize=(10, 5))
        sns.lineplot(
            x=time,
            y=self.df_rewards["recompensa"]
        )
        plt.title("Curva de Aprendizado")
        plt.xlabel("Episódios")
        plt.ylabel("Recompensa")
        plt.show()

    # Política

    def extract_policy(self):
        """
        Extrai política aprendida (argmax das ações).
        Retorna a melhor ação por estado.
        """
        if self.df_vals is None:
            raise ValueError("Dados de valores não carregados.")

        policy = self.df_vals.idxmax(axis=1)
        return policy

    def print_policy(self):
        """Imprime a política aprendida."""
        policy = self.extract_policy()
        print("Política aprendida:")
        print(policy)


"""
    def run_all(self):
        """"""Executa todo o pipeline automaticamente.""""""
        self.load_data()
        self.clean_values()
        self.normalize_values()

        self.plot_heatmap(normalized=True)
        self.plot_learning_curve()
        self.print_policy()

"""