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
        Executa o planeamento do agente de Bellman e realiza um Backtest financeiro
        para traduzir a matemática em valor monetário.
        """
        print("\n" + "="*50)
        print("INICIANDO PLANEAMENTO: VALUE ITERATION (BELLMAN)")
        print(f"Parâmetros a usar -> Theta: {theta}")
        print("="*50)
        
        start_time = time.time()
        
        # 1. Planeamento Matemático (Offline)
        # O agente resolve as equações diferenciais/estocásticas até estabilizar
        iterations = agent.run_value_iteration(theta=theta)
        execution_time = time.time() - start_time
        
        print(f"-> Convergência matemática alcançada em {iterations} iterações ({execution_time:.4f}s).")
        
        # 2. Backtest Financeiro (Testando a política na prática)
        print("\nSimulando a aplicação da Política Ótima no mercado...")
        state = self.env.reset()
        done = False
        total_reward = 0.0
        final_net_worth = 0.0
        
        net_worth_history = []
        
        while not done:
            # O agente não explora; executa estritamente a decisão ótima do V(s)
            action = agent.get_action(state)
            next_state, reward, done, info = self.env.step(action)
            
            total_reward += reward
            state = next_state
            net_worth_history.append(info.get("net_worth", 0.0))
            
            if done:
                final_net_worth = info.get("net_worth", 0.0)
                
        print(f"-> Fim da Simulação | Recompensa Total: {total_reward:.2f} | Saldo Final do Portefólio: ${final_net_worth:.2f}")

        return {
            "model": "Value Iteration (Bellman)",
            "params": {"theta": theta}, # 1º Pedido: Parâmetros explícitos
            "iterations_to_converge": iterations,
            "execution_time_seconds": execution_time,
            "backtest_reward": total_reward,
            "net_worth_history": net_worth_history, # 3º Pedido: Evolução do dinheiro!
            "v_table": agent.get_v_table().tolist(),
            "policy": agent.get_policy().tolist()
        }
    def grid_search_q_learning(self, agent_factory: Callable, alphas: List[float], gammas: List[float], num_episodes: int = 2000) -> List[Dict]:
        results = []
        total_configs = len(alphas) * len(gammas)
        current_config = 1
        
        print(f"\n" + "="*50)
        print(f"INICIANDO GRID SEARCH DE Q-LEARNING")
        print(f"Total de combinações a testar: {total_configs}")
        print(f"Episódios por combinação: {num_episodes}")
        print("="*50)
        
        for alpha in alphas:
            for gamma in gammas:
                # 1. Parâmetros explícitos
                params = {"alpha": alpha, "gamma": gamma, "episodes": num_episodes}
                print(f"\n[{current_config}/{total_configs}] Rodando Configuração -> Alpha: {alpha} | Gamma: {gamma}")
                
                agent = agent_factory(alpha, gamma)
                start_time = time.time()
                
                rewards_history = []
                net_worth_history = []  # 3. Histórico de dinheiro no bolso
                q_table_changes = []    # 2. Histórico de interpretabilidade
                
                last_q_table = agent.get_q_table().copy()
                
                for ep in range(num_episodes):
                    state = self.env.reset()
                    done = False
                    total_reward = 0.0
                    final_net_worth = 0.0
                    
                    while not done:
                        action = agent.choose_action(state, is_training=True)
                        next_state, reward, done, info = self.env.step(action)
                        agent.update(state, action, reward, next_state, done)
                        state = next_state
                        total_reward += reward
                        
                        if done:
                            # Captura o dinheiro exato que sobrou no último dia do episódio
                            final_net_worth = info.get("net_worth", 0.0)
                    
                    # Decaimento do epsilon (opcional, ajustável ao seu projeto)
                    agent.set_epsilon(max(0.05, agent.epsilon * 0.995))
                    
                    rewards_history.append(total_reward)
                    net_worth_history.append(final_net_worth)
                    
                    # 2. Resumo de mudanças a cada "N" episódios (ex: 20% do treinamento)
                    check_interval = max(1, num_episodes // 5)
                    if (ep + 1) % check_interval == 0:
                        current_q = agent.get_q_table()
                        
                        # O Max Delta mostra o quão agressivamente a tabela ainda está mudando
                        max_delta = float(np.max(np.abs(current_q - last_q_table)))
                        avg_net_worth = float(np.mean(net_worth_history[-check_interval:]))
                        
                        q_table_changes.append({
                            "episode": ep + 1,
                            "max_delta_q": max_delta,
                            "avg_net_worth_period": avg_net_worth
                        })
                        last_q_table = current_q.copy()
                        
                        # Output clean e profissional no terminal
                        print(f"   Ep {ep+1:4d}/{num_episodes} | "
                              f"Reward: {total_reward:7.2f} | "
                              f"Saldo Final: ${final_net_worth:7.2f} | "
                              f"Delta-Q: {max_delta:.4f}")
                
                exec_time = time.time() - start_time
                
                # Salvando o pacote completo para a análise final
                results.append({
                    "model": "Q-Learning",
                    "params": params,
                    "execution_time_seconds": exec_time,
                    "rewards_history": rewards_history,
                    "net_worth_history": net_worth_history,
                    "q_table_convergence": q_table_changes,
                    "final_q_table": agent.get_q_table().tolist()
                })
                
                current_config += 1
                
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
    from environment import PortfolioEnvironment, PortfolioConfig
    from q_learning import QAgent
    from bellman import BellmanAgent 

    # print("Iniciando teste de integração do Bellman...")
    #
    # # 1. Cria o ambiente forçando o modo sintético
    cfg = PortfolioConfig(price_source="synthetic")
    env = PortfolioEnvironment(cfg)
    
    # cfg = PortfolioConfig(market_symbol="PETR4.SA", horizon=100)
    # env = PortfolioEnvironment(cfg)

    runner = ExperimentRunner(env)
    modelo_mdp = env.get_transition_model(num_samples=50000)

    # 3. Cria o agente usando a variável gerada no passo anterior
    bellman_agent = BellmanAgent(
        env_model=modelo_mdp, 
        num_states=env.state_space_size, 
        num_actions=env.action_space_size, 
        gamma=0.99
    )

    #4. Roda o planejamento e mostra o resultado
    print("Política Ótima:", bellman_agent.get_policy())    
    bellman_results = runner.run_bellman_session(bellman_agent)
    # save_results(bellman_results, "bellman_results.json")
    #
    # 2. Grid Search Q-Learning
    # def create_q_agent(alpha, gamma):
    #     return QAgent(
    #         num_states=env.state_space_size, 
    #         num_actions=env.action_space_size, 
    #         alpha=alpha, 
    #         gamma=gamma
    #     )
    # alphas_to_test = [0.1, 0.5, 0.9]
    # gammas_to_test = [0.5, 0.9, 0.99]
    #
    # q_results = runner.grid_search_q_learning(create_q_agent, alphas_to_test, gammas_to_test, num_episodes=2000)
    # save_results(q_results, "q_learning_grid_results.json")

