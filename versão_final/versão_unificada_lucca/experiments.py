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
        "moves_p1": result["moves_p1"],
        "moves_p2": result["moves_p2"],
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
        "wins_as_p1_a": 0,
        "wins_as_p2_a": 0,
        "wins_as_p1_b": 0,
        "wins_as_p2_b": 0,
        "total_turns": 0,
        "total_nodes_a": 0,
        "total_nodes_b": 0,
        "total_moves_a": 0,
        "total_moves_b": 0,
        "total_depth_a": 0.0,
        "total_depth_b": 0.0,
        "total_time_a": 0.0,
        "total_time_b": 0.0,
    }

    for i in range(num_games):
        if i % 2 == 0:
            cfg1, cfg2 = agent_a_cfg, agent_b_cfg
            agent_a_player = 1
        else:
            cfg1, cfg2 = agent_b_cfg, agent_a_cfg
            agent_a_player = 2

        result = run_match(cfg1, cfg2, game_size=game_size, time_limit=time_limit, verbose=False)

        winner_player = result["winner"]
        turns = result["turns"]

        if winner_player is None:
            winner_label = "Draw"
        elif winner_player == 1:
            winner_label = f"{cfg1['name']} (P1)"
        else:
            winner_label = f"{cfg2['name']} (P2)"

        if agent_a_player == 1:
            nodes_a, nodes_b = result["nodes_p1"], result["nodes_p2"]
            moves_a, moves_b = result["moves_p1"], result["moves_p2"]
            depth_a, depth_b = result["avg_depth_p1"], result["avg_depth_p2"]
            time_a, time_b = result["avg_time_p1"], result["avg_time_p2"]
        else:
            nodes_a, nodes_b = result["nodes_p2"], result["nodes_p1"]
            moves_a, moves_b = result["moves_p2"], result["moves_p1"]
            depth_a, depth_b = result["avg_depth_p2"], result["avg_depth_p1"]
            time_a, time_b = result["avg_time_p2"], result["avg_time_p1"]

        if winner_player is None:
            summary["draws"] += 1
        elif winner_player == agent_a_player:
            summary[agent_a_cfg["name"]] += 1
            if agent_a_player == 1:
                summary["wins_as_p1_a"] += 1
            else:
                summary["wins_as_p2_a"] += 1
        else:
            summary[agent_b_cfg["name"]] += 1
            if agent_a_player == 1:
                summary["wins_as_p2_b"] += 1
            else:
                summary["wins_as_p1_b"] += 1

        summary["total_turns"] += turns
        summary["total_nodes_a"] += nodes_a
        summary["total_nodes_b"] += nodes_b
        summary["total_moves_a"] += moves_a
        summary["total_moves_b"] += moves_b
        summary["total_depth_a"] += depth_a
        summary["total_depth_b"] += depth_b
        summary["total_time_a"] += time_a
        summary["total_time_b"] += time_b

        avg_nodes_a = nodes_a / moves_a if moves_a else 0
        avg_nodes_b = nodes_b / moves_b if moves_b else 0

        print(
            f"Game {i+1} ({'A=P1,B=P2' if agent_a_player == 1 else 'A=P2,B=P1'}): "
            f"winner={winner_label}, turns={turns}\n"
            f"  {agent_a_cfg['name']} -> avg_nodes={avg_nodes_a:.0f}, avg_depth={depth_a:.2f}, time={time_a:.4f}s\n"
            f"  {agent_b_cfg['name']} -> avg_nodes={avg_nodes_b:.0f}, avg_depth={depth_b:.2f}, time={time_b:.4f}s"
        )

    wins_a = summary[agent_a_cfg["name"]]
    wins_b = summary[agent_b_cfg["name"]]

    print("\n=== FINAL RESULTS ===")
    print(f"Agent A: {agent_a_cfg['name']}")
    print(f"Agent B: {agent_b_cfg['name']}")
    print(f"Games: {num_games}")
    print(f"\n= WIN RATE PER AGENT =")
    print(f"{agent_a_cfg['name']} wins: {wins_a} (Win Rate: {wins_a / num_games:.2f})")
    print(f"{agent_b_cfg['name']} wins: {wins_b} (Win Rate: {wins_b / num_games:.2f})")
    print(f"Draws: {summary['draws']}")
    print(f"\n= WIN RATE PER SIDE =")
    print(f"{agent_a_cfg['name']} venceu como P1: {summary['wins_as_p1_a']} | como P2: {summary['wins_as_p2_a']}")
    print(f"{agent_b_cfg['name']} venceu como P1: {summary['wins_as_p1_b']} | como P2: {summary['wins_as_p2_b']}")
    total_p1_wins = summary["wins_as_p1_a"] + summary["wins_as_p1_b"]
    total_p2_wins = summary["wins_as_p2_a"] + summary["wins_as_p2_b"]
    print(f"Total vitórias do lado P1: {total_p1_wins} | lado P2: {total_p2_wins}")
    print(f"\n= REMAINING DATA =")
    print(f"Average Turns: {summary['total_turns'] / num_games:.2f}")
    print(f"Total Nodes {agent_a_cfg['name']}: {summary['total_nodes_a']}")
    print(f"Total Nodes {agent_b_cfg['name']}: {summary['total_nodes_b']}")
    print(f"Average Nodes per Move {agent_a_cfg['name']}: {summary['total_nodes_a'] / summary['total_moves_a']:.2f}")
    print(f"Average Nodes per Move {agent_b_cfg['name']}: {summary['total_nodes_b'] / summary['total_moves_b']:.2f}")
    print(f"Average Depth {agent_a_cfg['name']}: {summary['total_depth_a'] / num_games:.2f}")
    print(f"Average Depth {agent_b_cfg['name']}: {summary['total_depth_b'] / num_games:.2f}")
    # print(f"Average time per move {agent_a_cfg['name']}: {summary['total_time_a'] / num_games:.4f}s")
    # print(f"Average time per move {agent_b_cfg['name']}: {summary['total_time_b'] / num_games:.4f}s")
    print("\n")


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

    run_experiments(agent_alpha_h2, agent_minimax_h1, num_games=4, time_limit=0.5)