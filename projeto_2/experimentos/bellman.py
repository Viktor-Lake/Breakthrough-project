import numpy as np

from typing import Dict, Any

class BellmanAgent:
    def __init__(
        self, 
        env_model: Dict[int, Dict[int, list]], # {state: {action: [(prob, next_state, reward, done), ...]}}
        num_states: int, # Número total de estados no ambiente
        num_actions: int = 3, # Número total de ações (0=Comprar, 1=Vender, 2=Manter)
        gamma: float = 0.99 # Fator de desconto (0 < gamma < 1)
    ):
        if gamma <= 0 or gamma >= 1:
            raise ValueError("O fator de desconto gamma deve estar entre 0 e 1.")
        
        self.env_model = env_model
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        
        # Inicializa a tabela de valores V(s) e a política π(s)
        self.V = np.zeros(num_states)
        self.policy = np.zeros(num_states, dtype=int)
        
        self.has_converged = False
        self.iterations = 0

    def run_value_iteration(self, theta: float = 1e-5) -> int:
        iteration = 0
        
        while True:
            delta = 0.0
            iteration += 1
            
            for state in range(self.num_states):
                action_values = self._compute_action_values(state)
                
                best_action_value = np.max(action_values)
                delta = max(delta, abs(best_action_value - self.V[state]))
                self.V[state] = best_action_value
                
            if delta < theta:
                break
        
        self._extract_policy()
        self.has_converged = True
                
        print(
            f"[Value Iteration] Convergiu em {iteration} iterações "
            f"(theta={theta}, gamma={self.gamma})"
        )
        return iteration
    
    def get_v_table(self) -> np.ndarray:
        return self.V

    def get_policy(self) -> np.ndarray:
        return self.policy

    # ---------------------------------------------------------------------------
    # Métodos auxiliares
    # ---------------------------------------------------------------------------
    
    def _compute_action_values(self, state: int) -> np.ndarray:
        action_values = np.zeros(self.num_actions)
        
        transitions = self.env_model.get(state, {})
        
        for action in range(self.num_actions):
            trans_list = transitions.get(action, [])
            
            if not trans_list:
                action_values[action] = 0.0
                continue
            
            q_value = 0.0
            for (prob, next_state, reward, done) in trans_list:
                future_value = 0.0 if done else self.gamma * self.V[next_state]
                q_value += prob * (reward + future_value)
                
            action_values[action] = q_value
            
        return action_values
    
    def _extract_policy(self):
        for state in range(self.num_states):
            action_values = self._compute_action_values(state)
            self.policy[state] = int(np.argmax(action_values))
            
    def get_action(self, state: int) -> int:
        if not self.has_converged:
            raise RuntimeError("A política ainda não convergiu. Execute run_value_iteration() primeiro.")
        return int(self.policy[state])
    
    def get_summary(self) -> Dict[str, Any]:

        return {
            "gamma": self.gamma,
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "converged": self.has_converged,
            "v_table_mean": float(np.mean(self.V)),
            "v_table_max": float(np.max(self.V)),
            "v_table_min": float(np.min(self.V)),
            "action_distribution": {
                int(a): int(np.sum(self.policy == a))
                for a in range(self.num_actions)
            },
        }
        
# ---------------------------------------------------------------------------
# TESTE RÁPIDO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 3 estados (baixa, neutra, alta tendência) × 3 ações (comprar, vender, manter)
    NUM_STATES = 3
    NUM_ACTIONS = 3  # 0=comprar, 1=vender, 2=manter
 
    # Modelo de transição manual:
    # P[estado][ação] = [(probabilidade, próx_estado, recompensa, done)]
    mock_model = {
        0: {  # estado: tendência de baixa
            0: [(0.3, 0, -1.0, False), (0.4, 1, 0.0, False), (0.3, 2, 1.0, False)],  # comprar
            1: [(0.5, 0,  0.5, False), (0.3, 1, 0.0, False), (0.2, 2,-0.5, False)],  # vender
            2: [(0.6, 0, -0.2, False), (0.3, 1, 0.0, False), (0.1, 2, 0.2, False)],  # manter
        },
        1: {  # estado: tendência neutra
            0: [(0.3, 0, -0.5, False), (0.4, 1, 0.5, False), (0.3, 2, 1.0, False)],
            1: [(0.3, 0,  0.5, False), (0.4, 1, 0.5, False), (0.3, 2,-0.5, False)],
            2: [(0.3, 0,  0.0, False), (0.4, 1, 0.0, False), (0.3, 2, 0.0, False)],
        },
        2: {  # estado: tendência de alta
            0: [(0.1, 0,  0.5, False), (0.3, 1, 1.0, False), (0.6, 2, 2.0, False)],
            1: [(0.4, 0, -1.0, False), (0.3, 1, 0.0, False), (0.3, 2, 0.5, False)],
            2: [(0.2, 0,  0.2, False), (0.3, 1, 0.5, False), (0.5, 2, 1.0, False)],
        },
    }
 
    agent = BelllmanAgent(
        env_model=mock_model,
        num_states=NUM_STATES,
        num_actions=NUM_ACTIONS,
        gamma=0.99,
    )
 
    iters = agent.run_value_iteration(theta=1e-6)
 
    print("\n--- Resultados ---")
    print(f"Iterações para convergir : {iters}")
    print(f"V-table                  : {agent.get_v_table()}")
    print(f"Política ótima (ações)   : {agent.get_policy()}")
    action_names = {0: "Comprar", 1: "Vender", 2: "Manter"}
    for s, a in enumerate(agent.get_policy()):
        print(f"  Estado {s}: {action_names[a]}")
    print(f"\nResumo: {agent.get_summary()}")
   
