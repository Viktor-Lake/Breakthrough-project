# 📈 Projeto T2 - Aprendizado por Reforço: Portfólio Financeiro Simplificado

Este repositório contém a implementação do Agente de Aprendizado por Reforço para tomada de decisões sequenciais em um portfólio financeiro. 

Dado o nosso prazo final no dia 30 de Abril, adotamos uma **Arquitetura Plug and Play**. Isso significa que os módulos de Ambiente, Agentes e Experimentos estão desacoplados. A validação e a geração de gráficos não dependem da lógica interna de como vocês implementaram o modelo, **desde que as classes respeitem os Contratos de Interface definidos abaixo**.

---

## 🏗️ Divisão de Arquitetura e Contratos

Para que o módulo de testes (`experiments.py` - Pessoa 4) consiga rodar os milhares de episódios necessários e gerar os dados para os gráficos (`plots.py` - Pessoa 5) sem quebrar, as classes construídas pelas Pessoas 1, 2 e 3 precisam ter **exatamente** os métodos descritos a seguir.

Não importa como a lógica interna funciona, mas as funções precisam receber e retornar esses formatos específicos.

### 1. Contrato do Ambiente (Pessoa 1: `environment.py`)
O ambiente deve ser o árbitro do mercado. A classe do ambiente (ex: `PortfolioEnv`) precisa expor:

* `reset(self) -> Any`: Reinicia o ambiente para o episódio e retorna o `estado_inicial`.
* `step(self, action: int) -> Tuple[Any, float, bool, dict]`: Recebe uma ação discreta e retorna uma tupla exata contendo:
    1.  `next_state`: O próximo estado numérico/discreto.
    2.  `reward`: O valor float da recompensa calculada.
    3.  `done`: Booleano informando se o episódio acabou (fim do horizonte).
    4.  `info`: Um dicionário opcional para dados extras de debug (pode ser vazio `{}`).

### 2. Contrato de Bellman (Pessoa 2: `bellman.py`)
A classe do agente de Value Iteration precisa planejar a política offline e expor os resultados matriciais para o Heatmap.

* `run_value_iteration(self, theta: float = 1e-5) -> int`: Executa o loop de convergência e retorna a quantidade de iterações que levou para convergir.
* `get_v_table(self) -> np.ndarray`: Retorna a matriz de valores $V(s)$ final.
* `get_policy(self) -> np.ndarray`: Retorna a matriz indicando a melhor ação para cada estado.

### 3. Contrato de Q-Learning (Pessoa 3: `q_learning.py`)
A classe do agente Q-Learning vai aprender interagindo com o módulo da Pessoa 1. O código de experimentos vai chamar estas funções a cada passo de tempo:

* `choose_action(self, state: Any, is_training: bool = True) -> int`: Retorna a ação com base no estado (usando a regra $\epsilon$-greedy internamente se `is_training` for True).
* `update(self, state: Any, action: int, reward: float, next_state: Any, done: bool)`: Aplica a equação de atualização da tabela Q usando o erro de diferença temporal.
* `set_epsilon(self, epsilon: float)`: Setter simples para o módulo de experimentos forçar o decaimento dinâmico da exploração de fora para dentro.
* `get_q_table(self) -> np.ndarray`: Retorna a matriz $Q(s, a)$ completa ao final do treinamento.

---

## 🚀 Fluxo de Trabalho e Integração

1.  **Desenvolvimento Isolado:** Vocês podem testar as lógicas dos agentes usando *mock* ou rodando pequenos scripts locais.
2.  **Validação Final:** Quando o código de vocês estiver pronto, instanciem as classes e passem para a classe `ExperimentRunner` no arquivo `
