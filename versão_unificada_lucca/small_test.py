from game import Game
from agent_alpha_beta import AgentAlphaBeta
from agent_minimax import AgentMinimax
from heuristics import (
    heuristic_material_and_advance,
    heuristic_defensive_structures
)


def main():
    # Configuração dos agentes
    agent1 = AgentAlphaBeta(
        player=1,
        heuristic_func=heuristic_defensive_structures,
        time_limit=10
    )

    agent2 = AgentMinimax(
        player=2,
        heuristic_func=heuristic_material_and_advance,
        time_limit=10
    )

    # Cria jogo
    game = Game(size=8)

    # Executa partida
    result = game.play_match(agent1, agent2, verbose=True)

    # Mostra resultado final
    print("\n=== RESULTADO FINAL ===")
    print(f"Winner: {result['winner']}")
    print(f"Turns: {result['turns']}")
    print(f"Nodes P1: {result['nodes_p1']}")
    print(f"Nodes P2: {result['nodes_p2']}")
    print(f"Max Depth P1: {result['max_depth_p1']:.2f}")
    print(f"Max Depth P2: {result['max_depth_p2']:.2f}")
    print(f"Avg Time P1: {result['avg_time_p1']:.4f} seconds")
    print(f"Avg Time P2: {result['avg_time_p2']:.4f} seconds")


if __name__ == "__main__":
    main()