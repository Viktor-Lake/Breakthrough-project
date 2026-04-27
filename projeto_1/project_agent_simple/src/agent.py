import time

class AIAgent:
    def __init__(self, player, heuristic_func, time_limit=1.0):
        self.player = player
        self.heuristic = heuristic_func
        self.time_limit = time_limit # Orçamento computacional fixo por jogada [cite: 14, 15]
        self.nodes_expanded = 0      # Instrumentação obrigatória 
        self.start_time = 0

    def order_moves(self, state, moves):
        """Move ordering: Prioriza capturas para melhorar a poda Alpha-Beta """
        def move_score(move):
            (from_r, from_c), (to_r, to_c) = move
            # Se o destino não está vazio, é uma captura (maior prioridade)
            if state.board[to_r][to_c] != 0:
                return 1 
            return 0
        
        # Ordena as jogadas (capturas primeiro)
        return sorted(moves, key=move_score, reverse=True)

    def get_best_move(self, state):
        self.nodes_expanded = 0
        self.start_time = time.time()
        best_move = None
        depth = 1
        
        # Iterative Deepening
        try:
            while True:
                # Checa o tempo antes de iniciar uma nova profundidade
                if time.time() - self.start_time >= self.time_limit:
                    break
                    
                current_best_move, _ = self.alpha_beta(state, depth, float('-inf'), float('inf'), True)
                
                if current_best_move:
                    best_move = current_best_move
                    
                depth += 1
        except TimeoutError:
            pass # Estourou o tempo, usa a melhor jogada encontrada na última profundidade completa
            
        return best_move, self.nodes_expanded, depth - 1

    def alpha_beta(self, state, depth, alpha, beta, maximizing_player):
        # Controle rígido de tempo dentro da recursão
        if time.time() - self.start_time >= self.time_limit:
            raise TimeoutError()

        is_term, _ = state.is_terminal()
        if depth == 0 or is_term:
            self.nodes_expanded += 1
            return None, self.heuristic(state, self.player)

        moves = state.get_legal_moves(state.current_player)
        moves = self.order_moves(state, moves) # Aplica move ordering para melhorar a eficiência da poda

        best_move = moves[0] if moves else None

        if maximizing_player:
            max_eval = float('-inf')
            for move in moves:
                new_state = state.apply_move(move)
                _, eval = self.alpha_beta(new_state, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break # Poda Alpha-Beta
            return best_move, max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                new_state = state.apply_move(move)
                _, eval = self.alpha_beta(new_state, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break # Poda Alpha-Beta
            return best_move, min_eval