import numpy as np
import time
import json
from typing import Protocol, Tuple, Any, Dict, List, Callable

# ==========================================
# 1. CONTRATOS
# ==========================================

class Environment(Protocol):
    """Contrato que o arquivo environment.py deve respeitar."""
    def reset(self) -> Any:
        ...
    def step(self, action: int) -> Tuple[Any, float, bool, dict]:
        ...

class QLearningAgent(Protocol):
    """Contrato que o arquivo q_learning.py deve respeitar."""
    def choose_action(self, state: Any, is_training: bool = True) -> int:
        ...
    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
        ...
    def set_epsilon(self, epsilon: float):
        ...
    def get_q_table(self) -> np.ndarray:
        ...

class BellmanAgent(Protocol):
    """Contrato que o arquivo bellman.py deve respeitar."""
    def run_value_iteration(self, theta: float = 1e-5) -> int:
        """Deve rodar o algoritmo e retornar o número de iterações até convergir."""
        ...
    def get_v_table(self) -> np.ndarray:
        ...
    def get_policy(self) -> np.ndarray:
        ...

# ==========================================
# 2. MOTOR DE EXPERIMENTOS
# ==========================================

class ExperimentRunner:
    def __init__(self, env: Environment):
        self.env = env

    def run_q_learning_session(
        self, 
        agent: QLearningAgent, 
        num_episodes: int, 
        epsilon_start: float = 1.0, 
        epsilon_min: float = 0.05, 
        epsilon_decay_rate: float = 0.995
    ) -> Dict[str, Any]:
        """
        Roda uma sessão completa de treinamento de Q-Learning e coleta as métricas.
        """
        rewards_history = []
        steps_history = []
        epsilon_history = []
        
        epsilon = epsilon_start
        start_time = time.time()

        for ep in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            
            # Atualiza a taxa de exploração no agente
            agent.set_epsilon(epsilon)
            
            while not done:
                action = agent.choose_action(state, is_training=True)
                next_state, reward, done, info = self.env.step(action)
                
                agent.update(state, action, reward, next_state, done)
                
                total_reward += reward
                state = next_state
                steps += 1
                
            rewards_history.append(total_reward)
            steps_history.append(steps)
            epsilon_history.append(epsilon)
            
            # Decaimento do Epsilon (após cada episódio)
            epsilon = max(epsilon_min, epsilon * epsilon_decay_rate)

        execution_time = time.time() - start_time

        return {
            "model": "Q-Learning",
            "episodes_run": num_episodes,
            "execution_time_seconds": execution_time,
            "rewards_history": rewards_history,
            "epsilon_history": epsilon_history,
            "final_q_table": agent.get_q_table().tolist() # Convertido para lista para facilitar exportação
        }

    def run_bellman_session(self, agent: BellmanAgent, theta: float = 1e-5) -> Dict[str, Any]:
        """
        Executa o planejamento do agente de Bellman e coleta os resultados.
        """
        start_time = time.time()
        
        # Bellman resolve tudo de uma vez sem loop de episódios
        iterations = agent.run_value_iteration(theta=theta)
        
        execution_time = time.time() - start_time

        return {
            "model": "Value Iteration (Bellman)",
            "iterations_to_converge": iterations,
            "execution_time_seconds": execution_time,
            "v_table": agent.get_v_table().tolist(),
            "policy": agent.get_policy().tolist()
        }

    def grid_search_q_learning(
        self, 
        agent_factory: Callable[..., QLearningAgent], # Função que cria um agente novo
        alphas: List[float], 
        gammas: List[float], 
        num_episodes: int
    ) -> List[Dict[str, Any]]:
        """
        Testa automaticamente todas as combinações de Alpha e Gamma para Q-Learning.
        """
        results = []
        total_configs = len(alphas) * len(gammas)
        current = 1

        print(f"Iniciando Grid Search com {total_configs} configurações...")

        for alpha in alphas:
            for gamma in gammas:
                print(f"Testando [{current}/{total_configs}]: Alpha={alpha}, Gamma={gamma}")
                
                # O agent_factory garante que cada teste comece com um agente "limpo" (Q-table zerada)
                agent = agent_factory(alpha=alpha, gamma=gamma)
                
                # Roda a sessão
                session_metrics = self.run_q_learning_session(agent, num_episodes)
                
                # Adiciona metadados
                session_metrics["params"] = {"alpha": alpha, "gamma": gamma}
                results.append(session_metrics)
                current += 1
                
        return results

# ==========================================
# 3. EXPORTAÇÃO PARA O ARQUIVO DE PLOTS
# ==========================================

def save_results(data: Any, filename: str):
    """Salva os dicionários de resultados em JSON para a Pessoa 5 (Plots) ler."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Resultados salvos com sucesso em: {filename}")

# ==========================================
# EXEMPLO DE USO (Para você testar localmente)
# ==========================================
if __name__ == "__main__":
    # Importe as classes dos seus colegas quando estiverem prontas
    # from environment import PortfolioEnv
    # from q_learning import QAgent
    # from bellman import ValueIterationAgent

    print("Este arquivo é um módulo de experimentação.")
    print("Para usar, instancie os ambientes e agentes e chame o ExperimentRunner.")
    
    # --- Exemplo de script de execução principal ---
    # env = PortfolioEnv()
    # runner = ExperimentRunner(env)
    
    # 1. Testando Bellman
    # bellman_agent = ValueIterationAgent(env_model=env.get_model(), gamma=0.99)
    # bellman_results = runner.run_bellman_session(bellman_agent)
    # save_results(bellman_results, "bellman_results.json")
    
    # 2. Grid Search Q-Learning
    # def create_q_agent(alpha, gamma):
    #     return QAgent(num_states=env.num_states, num_actions=env.num_actions, alpha=alpha, gamma=gamma)
    #
    # alphas_to_test = [0.1, 0.5, 0.9]
    # gammas_to_test = [0.5, 0.9, 0.99]
    # 
    # q_results = runner.grid_search_q_learning(create_q_agent, alphas_to_test, gammas_to_test, num_episodes=2000)
    # save_results(q_results, "q_learning_grid_results.json")
