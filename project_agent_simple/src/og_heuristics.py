def heuristic_material_and_advance(state, player):
    """
    Heurística 1: Combina a diferença de material e o avanço médio das peças.
    """
    is_term, winner = state.is_terminal()
    if is_term:
        return 10000 if winner == player else -10000

    opponent = 2 if player == 1 else 1
    score = 0
    
    for r in range(state.size):
        for c in range(state.size):
            if state.board[r][c] == player:
                score += 10  # Peso para material
                # Peso para avanço (quanto mais perto do final, melhor)
                score += (r if player == 1 else (state.size - 1 - r))
            elif state.board[r][c] == opponent:
                score -= 10
                score -= (r if opponent == 1 else (state.size - 1 - r))
                
    return score

def heuristic_defensive_structures(state, player):
    """
    Heurística 2: Foca em estruturas de apoio e evitar deixar peças desprotegidas[cite: 78].
    """
    is_term, winner = state.is_terminal()
    if is_term:
        return 10000 if winner == player else -10000

    opponent = 2 if player == 1 else 1
    score = 0
    direction = -1 if player == 1 else 1 # Para checar quem está 'atrás' protegendo
    
    for r in range(state.size):
        for c in range(state.size):
            if state.board[r][c] == player:
                score += 10 # Material
                # Checa se tem suporte de uma peça amiga atrás
                back_r = r + direction
                if 0 <= back_r < state.size:
                    if c - 1 >= 0 and state.board[back_r][c - 1] == player:
                        score += 3 # Suporte diagonal
                    if c + 1 < state.size and state.board[back_r][c + 1] == player:
                        score += 3 # Suporte diagonal
            elif state.board[r][c] == opponent:
                score -= 10
                # Lógica inversa para o oponente
                back_r_opp = r - direction
                if 0 <= back_r_opp < state.size:
                    if c - 1 >= 0 and state.board[back_r_opp][c - 1] == opponent:
                        score -= 3
                    if c + 1 < state.size and state.board[back_r_opp][c + 1] == opponent:
                        score -= 3
    return score