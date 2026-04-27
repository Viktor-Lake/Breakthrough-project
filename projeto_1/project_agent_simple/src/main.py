from breakthrough import GameState
from og_heuristics import heuristic_material_and_advance, heuristic_defensive_structures
from agent import AIAgent

def print_board(state):
    print("\n  " + " ".join(str(i) for i in range(state.size)))
    for i, row in enumerate(state.board):
        # Usando caracteres para facilitar visualização: 1=X, 2=O, 0=.
        symbols = {0: '.', 1: 'X', 2: 'O'}
        print(f"{i} " + " ".join(symbols[cell] for cell in row))
    print()

def main():
    state = GameState(size=8)
    
    # Exemplo: Agente vs Agente com heurísticas diferentes (ótimo para a Avaliação Experimental [cite: 84])
    agent1 = AIAgent(player=1, heuristic_func=heuristic_material_and_advance, time_limit=1.0)
    agent2 = AIAgent(player=2, heuristic_func=heuristic_defensive_structures, time_limit=1.0)
    
    print("Jogo Breakthrough Inicializado!")
    print("Player 1: X | Player 2: O")
    
    while True:
        print_board(state)
        is_term, winner = state.is_terminal()
        
        if is_term:
            print(f"Fim de Jogo! Vencedor: Player {winner}")
            break
            
        print(f"Turno do Player {state.current_player}...")
        
        if state.current_player == 1:
            move, nodes, depth = agent1.get_best_move(state)
            print(f"P1 jogou {move}. Nós expandidos: {nodes} | Profundidade: {depth}")
        else:
            move, nodes, depth = agent2.get_best_move(state)
            print(f"P2 jogou {move}. Nós expandidos: {nodes} | Profundidade: {depth}")
            
        if move is None:
            print(f"Player {state.current_player} sem movimentos válidos.")
            break
            
        state = state.apply_move(move)

if __name__ == "__main__":
    main()