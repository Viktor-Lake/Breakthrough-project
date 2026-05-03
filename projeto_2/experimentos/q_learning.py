# import numpy as np
# import random
# from typing import Any
#
# class QAgent:
#     """
#     Agente de Q-Learning.
#     """
#     def __init__(self, num_actions: int = 3, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 1.0):
#         self.num_actions = num_actions
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#
#         # Tabela Q inicia vazia. Estados são adicionados conforme o agente os visita.
#         self.q_table = {}
#
#     def _get_q_values(self, state: Any) -> np.ndarray:
#         """
#         Função auxiliar: Se o estado não estiver na tabela, inicializa com zeros.
#         """
#         if state not in self.q_table:
#             self.q_table[state] = np.zeros(self.num_actions)
#         return self.q_table[state]
#
#     def choose_action(self, state: Any, is_training: bool = True) -> int:
#         # Busca os valores Q para o estado atual (cria se não existir)
#         q_values = self._get_q_values(state)
#
#         if is_training and random.random() < self.epsilon:
#             return random.randint(0, self.num_actions - 1)
#
#         return int(np.argmax(q_values))
#
#     def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
#         """
#         Atualiza a tabela Q usando a regra de Bellman / Diferença Temporal.
#         """
#         # Garante que o estado atual existe na tabela
#         current_q_values = self._get_q_values(state)
#
#         if done:
#             max_future_q = 0.0
#         else:
#             # Garante que o próximo estado existe na tabela antes de pegar o máximo
#             next_q_values = self._get_q_values(next_state)
#             max_future_q = np.max(next_q_values)
#
#         # Cálculo do erro de Diferença Temporal
#         td_target = reward + self.gamma * max_future_q
#         td_error = td_target - current_q_values[action]
#
#         # Atualização da tabela
#         self.q_table[state][action] += self.alpha * td_error
#
#     def set_epsilon(self, epsilon: float):
#         self.epsilon = epsilon
#
#     def get_q_table(self) -> np.ndarray:
#         """Converte o dicionário para matriz NumPy para exportar os resultados."""
#         if not self.q_table:
#             return np.zeros((1, self.num_actions))
#
#         max_state_id = max(self.q_table.keys())
#         q_array = np.zeros((max_state_id + 1, self.num_actions))
#
#         for s, actions in self.q_table.items():
#             q_array[s] = actions
#
#         return q_array
#
import numpy as np
import random
from typing import Any

class QAgent:
    """
    Agente de Q-Learning Otimizado com Matriz NumPy.
    """
    def __init__(self, num_states: int = 6, num_actions: int = 3, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 1.0):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Tabela Q alocada previamente na memória
        self.q_table = np.zeros((self.num_states, self.num_actions))

    def choose_action(self, state: int, is_training: bool = True) -> int:
        if is_training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Se os valores Q forem todos iguais (ex: início), escolhe aleatoriamente para evitar viés
        if np.all(self.q_table[state] == self.q_table[state][0]):
            return random.randint(0, self.num_actions - 1)
            
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Atualiza a tabela Q usando a regra de Bellman / Diferença Temporal.
        """
        max_future_q = 0.0 if done else np.max(self.q_table[next_state])
        
        td_target = reward + self.gamma * max_future_q
        td_error = td_target - self.q_table[state, action]
        
        self.q_table[state, action] += self.alpha * td_error

    def set_epsilon(self, epsilon: float):
        self.epsilon = epsilon

    def get_q_table(self) -> np.ndarray:
        return self.q_table.copy()
