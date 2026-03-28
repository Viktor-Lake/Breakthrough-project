from game import Game
from agent_alpha_beta import AgentAlphaBeta
from agent_minimax import AgentMinimax
from heuristics import (
    heuristic_material_and_advance,
    heuristic_defensive_structures
)

def create_agent(config, player, time_limit):
    search = config["search"]
    heuristic = config["heuristic"]

    if search == "alphabeta":
        return AgentAlphaBeta(player, heuristic, time_limit)
    elif search == "minimax":
        return AgentMinimax(player, heuristic, time_limit)
    else:
        raise ValueError(f"Unknown search algorithm: {search}")


def run_match(agent_a_cfg, agent_b_cfg, game_size=8, time_limit=1.0, verbose=False):
    game = Game(size=game_size)

    agent1 = create_agent(agent_a_cfg, 1, time_limit)
    agent2 = create_agent(agent_b_cfg, 2, time_limit)

    result = game.play_match(agent1, agent2, verbose=verbose)

    return {
        "agent1_name": agent_a_cfg["name"],
        "agent2_name": agent_b_cfg["name"],
        "winner": result["winner"],
        "turns": result["turns"],
        "nodes_p1": result["nodes_p1"],
        "nodes_p2": result["nodes_p2"],
        "avg_depth_p1": result["avg_depth_p1"],
        "avg_depth_p2": result["avg_depth_p2"],
        "avg_time_p1": result["avg_time_p1"],
        "avg_time_p2": result["avg_time_p2"],
    }


def run_experiments(agent_a_cfg, agent_b_cfg, num_games=10, time_limit=1.0, game_size=8):
    summary = {
        agent_a_cfg["name"]: 0,
        agent_b_cfg["name"]: 0,
        "draws": 0,
        "total_turns": 0,

        "total_nodes_a": 0,
        "total_nodes_b": 0,

        "total_depth_a": 0.0,
        "total_depth_b": 0.0,

        "total_time_a": 0.0,
        "total_time_b": 0.0,
    }

    for i in range(num_games):
        # Alterna lados
        if i % 2 == 0:
            cfg1, cfg2 = agent_a_cfg, agent_b_cfg
            agent_a_player = 1
        else:
            cfg1, cfg2 = agent_b_cfg, agent_a_cfg
            agent_a_player = 2

        result = run_match(cfg1, cfg2, game_size=game_size, time_limit=time_limit, verbose=False)

        winner = result["winner"]
        turns = result["turns"]

        if agent_a_player == 1:
            nodes_a = result["nodes_p1"]
            nodes_b = result["nodes_p2"]
            depth_a = result["avg_depth_p1"]
            depth_b = result["avg_depth_p2"]
            time_a = result["avg_time_p1"]
            time_b = result["avg_time_p2"]
        else:
            nodes_a = result["nodes_p2"]
            nodes_b = result["nodes_p1"]
            depth_a = result["avg_depth_p2"]
            depth_b = result["avg_depth_p1"]
            time_a = result["avg_time_p2"]
            time_b = result["avg_time_p1"]

        # Resultado
        if winner is None:
            summary["draws"] += 1
        elif winner == agent_a_player:
            summary[agent_a_cfg["name"]] += 1
        else:
            summary[agent_b_cfg["name"]] += 1

        # Acumular métricas
        summary["total_turns"] += turns
        summary["total_nodes_a"] += nodes_a
        summary["total_nodes_b"] += nodes_b
        summary["total_depth_a"] += depth_a
        summary["total_depth_b"] += depth_b
        summary["total_time_a"] += time_a
        summary["total_time_b"] += time_b

        print(
            f"Game {i+1}: winner={winner}, turns={turns}\n"
            f"  {agent_a_cfg['name']} -> nodes={nodes_a}, depth={depth_a:.2f}, time={time_a:.4f}s\n"
            f"  {agent_b_cfg['name']} -> nodes={nodes_b}, depth={depth_b:.2f}, time={time_b:.4f}s"
        )

    print("\n=== FINAL RESULTS ===")
    print(f"Agent A: {agent_a_cfg['name']}")
    print(f"Agent B: {agent_b_cfg['name']}")
    print(f"Games: {num_games}")

    wins_a = summary[agent_a_cfg["name"]]
    wins_b = summary[agent_b_cfg["name"]]

    print(f"{agent_a_cfg['name']} wins: {wins_a}")
    print(f"{agent_b_cfg['name']} wins: {wins_b}")
    print(f"Draws: {summary['draws']}")

    # TAXA DE VITÓRIA
    print(f"Win rate {agent_a_cfg['name']}: {wins_a / num_games:.2f}")
    print(f"Win rate {agent_b_cfg['name']}: {wins_b / num_games:.2f}")

    # MÉDIAS
    print(f"Average turns: {summary['total_turns'] / num_games:.2f}")

    print(f"Average nodes {agent_a_cfg['name']}: {summary['total_nodes_a'] / num_games:.2f}")
    print(f"Average nodes {agent_b_cfg['name']}: {summary['total_nodes_b'] / num_games:.2f}")

    print(f"Average depth {agent_a_cfg['name']}: {summary['total_depth_a'] / num_games:.2f}")
    print(f"Average depth {agent_b_cfg['name']}: {summary['total_depth_b'] / num_games:.2f}")

    print(f"Average time per move {agent_a_cfg['name']}: {summary['total_time_a'] / num_games:.4f}s")
    print(f"Average time per move {agent_b_cfg['name']}: {summary['total_time_b'] / num_games:.4f}s")


if __name__ == "__main__":

    agent_alpha_h1 = {
        "name": "AlphaBeta + H1",
        "search": "alphabeta",
        "heuristic": heuristic_material_and_advance,
    }

    agent_minimax_h1 = {
        "name": "Minimax + H1",
        "search": "minimax",
        "heuristic": heuristic_material_and_advance,
    }

    agent_alpha_h2 = {
        "name": "AlphaBeta + H2",
        "search": "alphabeta",
        "heuristic": heuristic_defensive_structures,
    }

    agent_minimax_h2 = {
        "name": "Minimax + H2",
        "search": "minimax",
        "heuristic": heuristic_defensive_structures,
    }

    # Rodar aqui o run_experiments para comparar os agentes
    run_experiments(agent_alpha_h1, agent_minimax_h1, num_games=1, time_limit=1.0)
