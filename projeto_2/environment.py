from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

@dataclass
class PortfolioConfig:
    """Configs para o environment. A maioria é alto-explicativa, mas pode perguntar do Lucca."""

    # Configurações de episódio
    initial_cash: float = 1000.0
    horizon: int = 100

    # Definição de fonte
    price_source: str = "synthetic"

    # Para o modo "synthetic"
    initial_price: float = 100.0
    drift: float = 0.0005
    volatility: float = 0.01

    # Para o modo "yfinance"
    market_symbol: str = "AAPL"
    market_start: str = "2020-01-01"
    market_end: str = "2024-01-01"
    market_interval: str = "1d"
    market_field: str = "Close"
    random_window_start: bool = True

    # Regras de trading
    transaction_cost: float = 0.0
    invalid_action_penalty: float = -1
    bankruptcy_ends_episode: bool = False
    seed: Optional[int] = None

    # Estados
    trend_window: int = 3
    allow_short: bool = False   # Pesquisar "stock shorting" no Google


class PortfolioEnvironment:
    """
    Environment para a aplicação.

    Ações:
        0 = HOLD
        1 = BUY
        2 = SELL

    Estados:
        trend ∈ {0,1,2} -> {DOWN, FLAT, UP}
        position ∈ {0,1} -> {CASH, HOLDING}
        encoded_state = trend * 2 + position
    """

    ACTION_HOLD = 0
    ACTION_BUY = 1
    ACTION_SELL = 2

    TREND_DOWN = 0
    TREND_FLAT = 1
    TREND_UP = 2

    POSITION_CASH = 0
    POSITION_HOLDING = 1

    def __init__(
        self,
        config: Optional[PortfolioConfig] = None,
        prices: Optional[Sequence[float]] = None,
    ) -> None:
        self.config = config or PortfolioConfig()
        self._rng = np.random.default_rng(self.config.seed)

        self._external_prices = (
            np.asarray(prices, dtype=float).copy() if prices is not None else None
        )

        self.prices: np.ndarray = np.array([], dtype=float)
        self.current_step: int = 0
        self.episode_horizon: int = 0

        self.cash: float = float(self.config.initial_cash)
        self.position: int = self.POSITION_CASH
        self.entry_price: float = 0.0
        self.done: bool = False

        self._last_state: int = 0

    @property
    def state_space_size(self) -> int:
        return 6

    @property
    def action_space_size(self) -> int:
        return 3

    def reset(self) -> int:
        """Reinicia o episódio e retorna o estado inicial discreto."""
        self.prices = self._prepare_prices()

        if len(self.prices) < 2:
            raise ValueError("A série de preços precisa ter pelo menos 2 valores.")

        self.episode_horizon = len(self.prices) - 1
        self.current_step = 0

        self.cash = float(self.config.initial_cash)
        self.position = self.POSITION_CASH
        self.entry_price = 0.0
        self.done = False

        self._last_state = self._get_state()
        return self._last_state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Recebe uma ação discreta e retorna:
            next_state, reward, done, info
        """
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() before step().")

        if action not in (self.ACTION_HOLD, self.ACTION_BUY, self.ACTION_SELL):
            raise ValueError("action must be one of {0: HOLD, 1: BUY, 2: SELL}.")

        current_price = float(self.prices[self.current_step])
        prev_net_worth = self._net_worth(current_price)

        invalid_action = False

        # Executa a ação no preço atual
        if action == self.ACTION_BUY:
            if self.position == self.POSITION_CASH:
                self.position = self.POSITION_HOLDING
                self.entry_price = current_price
                self.cash -= current_price * (1.0 + self.config.transaction_cost)
            else:
                invalid_action = True

        elif action == self.ACTION_SELL:
            if self.position == self.POSITION_HOLDING:
                self.position = self.POSITION_CASH
                self.cash += current_price * (1.0 - self.config.transaction_cost)
                self.entry_price = 0.0
            else:
                invalid_action = True

        # Avança o mercado sempre, mesmo em ação inválida
        if self.current_step < len(self.prices) - 1:
            self.current_step += 1

        next_price = float(self.prices[self.current_step])
        next_net_worth = self._net_worth(next_price)

        reward = float(next_net_worth - prev_net_worth)

        if invalid_action:
            reward += float(self.config.invalid_action_penalty)

        # Terminação por falência
        net_worth = next_net_worth
        if self.config.bankruptcy_ends_episode and net_worth <= 0:
            self.done = True

        # Terminação por horizonte
        if self.current_step >= self.episode_horizon:
            self.done = True

        next_state = self._get_state()
        info = {
            "step": self.current_step,
            "price": round(float(self.prices[self.current_step]), 2),
            "cash": round(float(self.cash), 2),
            "position": int(self.position),
            "entry_price": round(float(self.entry_price), 2),
            "net_worth": round(float(net_worth), 2),
            "invalid_action": bool(invalid_action),
        }

        self._last_state = next_state
        return next_state, reward, self.done, info

    def _prepare_prices(self) -> np.ndarray:
        """
        Prepara a série de preços conforme o modo escolhido:
        - synthetic: gera série artificial
        - external: usa um arquivo (não testei se funciona ainda)
        - yfinance: baixa dados online com a biblioteca yfinance
        """
        source = self.config.price_source.lower().strip()

        if source == "external":
            if self._external_prices is None:
                raise ValueError(
                    "price_source='external' exige que o parâmetro prices seja informado."
                )
            prices = np.asarray(self._external_prices, dtype=float).copy()

        elif source == "yfinance":
            prices = self._load_prices_from_yfinance()

        elif source == "synthetic":
            prices = self._generate_synthetic_prices()

        else:
            raise ValueError(
                "price_source deve ser 'synthetic', 'external' ou 'yfinance'."
            )

        prices = prices[np.isfinite(prices)]
        prices = prices[prices > 0]

        if len(prices) < 2:
            raise ValueError("A série de preços precisa conter ao menos 2 valores válidos.")

        # Para treino, usa uma janela contínua do tamanho do horizonte + 1.
        target_len = self.config.horizon + 1

        if len(prices) > target_len and self.config.random_window_start:
            start = self._rng.integers(0, len(prices) - target_len + 1)
            prices = prices[start : start + target_len]
        else:
            prices = prices[:target_len]

        if len(prices) < 2:
            raise ValueError("Janela de preços insuficiente após o recorte.")

        return prices.astype(float)

    def _generate_synthetic_prices(self) -> np.ndarray:
        """Gera uma série artificial de preços via random walk geométrico."""
        n = self.config.horizon + 1
        prices = np.empty(n, dtype=float)
        prices[0] = float(self.config.initial_price)

        drift = float(self.config.drift)
        vol = float(self.config.volatility)

        for t in range(1, n):
            shock = self._rng.normal(loc=drift, scale=vol)
            prices[t] = max(0.01, prices[t - 1] * (1.0 + shock))

        return prices

    def _load_prices_from_yfinance(self) -> np.ndarray:
        """Baixa preços reais usando yfinance."""
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "Para usar price_source='yfinance', instale a biblioteca yfinance."
            ) from exc

        df = yf.download(
            self.config.market_symbol,
            start=self.config.market_start,
            end=self.config.market_end,
            interval=self.config.market_interval,
            auto_adjust=True,
            progress=False,
        )

        if df is None or len(df) == 0:
            raise ValueError(
                f"Não foi possível obter dados para {self.config.market_symbol}."
            )

        field = self.config.market_field
        if field not in df.columns:
            if "Close" in df.columns:
                field = "Close"
            else:
                raise ValueError(
                    f"Campo '{self.config.market_field}' não encontrado. "
                    f"Colunas disponíveis: {list(df.columns)}"
                )

        prices = df[field].dropna().to_numpy(dtype=float)

        if len(prices) < 2:
            raise ValueError("A série baixada do mercado tem menos de 2 preços válidos.")

        return prices

    def _trend_label(self) -> int:
        """
        Classifica a tendência com base na média dos retornos recentes.
        DOWN / FLAT / UP.
        """
        if self.current_step <= 0:
            return self.TREND_FLAT

        window = max(1, int(self.config.trend_window))
        start_idx = max(1, self.current_step - window + 1)

        segment = self.prices[start_idx - 1 : self.current_step + 1]
        if len(segment) < 2:
            return self.TREND_FLAT

        prev_prices = segment[:-1]
        next_prices = segment[1:]
        returns = (next_prices - prev_prices) / np.maximum(prev_prices, 1e-12)
        mean_ret = float(np.mean(returns))

        threshold = 0.002
        if mean_ret > threshold:
            return self.TREND_UP
        if mean_ret < -threshold:
            return self.TREND_DOWN
        return self.TREND_FLAT

    def _get_state(self) -> int:
        trend = self._trend_label()
        return trend * 2 + int(self.position)

    def _net_worth(self, price: float) -> float:
        return float(self.cash + self.position * price)

    def decode_state(self, state: int) -> Tuple[str, str]:
        """Converte o estado codificado em rótulos legíveis."""
        if state < 0 or state >= self.state_space_size:
            raise ValueError("Invalid state index.")

        trend = state // 2
        position = state % 2

        trend_name = {0: "DOWN", 1: "FLAT", 2: "UP"}[trend]
        position_name = {0: "CASH", 1: "HOLDING"}[position]
        return trend_name, position_name

    def render(self) -> Dict[str, Any]:
        """Retorna uma visão simples do estado atual."""
        return {
            "step": self.current_step,
            "price": float(self.prices[self.current_step]) if len(self.prices) else None,
            "cash": float(self.cash),
            "position": int(self.position),
            "entry_price": float(self.entry_price),
            "state": int(self._last_state),
            "decoded_state": self.decode_state(self._last_state)
            if len(self.prices)
            else None,
        }


# Exemplo de uso

if __name__ == "__main__":

    '''
    PortfolioConfig - Define o modo de uso e os parâmetros:

        Ex:
        price_source = "synthetic",
        horizon = 10,
        seed = 15

    Se quiser ver todos os parâmetros, ir até a definição a partir da linha 9 do código
    '''
    cfg = PortfolioConfig(
        initial_cash=1000,
        price_source="yfinance",
        market_symbol="PETR4.SA",   # Tá pegando da Petrobras
        market_start="2022-01-01",
        market_end="2024-01-01",
        horizon=100,
        seed=42
    )

    env = PortfolioEnvironment(cfg)

    s = env.reset()

    print("initial_state:", s, env.decode_state(s))

    total_reward = 0.0

    # Para printar melhor a ação
    action_tags = ["HOLD", "BUY ", "SELL"]

    # Loop de ações
    done = False
    while not done:
        action = int(env._rng.integers(0, 3))   # Fazendo ações aleatórias aqui
        s, r, done, info = env.step(action)
        total_reward += r
        print(
            f"step={info['step']:3d} | "
            f"price={info['price']:7.2f} | "
            f"pos={info['position']} | "
            f"act={action_tags[action]} | "
            f"cash={info['cash']:8.2f} | "
            f"net={info['net_worth']:8.2f} | "
            f"r={r:7.2f} | "
            f"invalid={info['invalid_action']}"
        )

    print("Reward total:", total_reward)

    ''' 
    OBS:    Tem uma penalidade por ficar fazendo ação inválida, então a reward não vai ser o lucro.
            Se desligar ela nas definições (só faz ser igual a 0), a reward vai ser o lucro.  
    '''