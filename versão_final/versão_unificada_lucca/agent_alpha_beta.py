import time

class AgentAlphaBeta:
    def __init__(self, player, heuristic_func, time_limit=1.0):
        self.player = player
        self.heuristic = heuristic_func
        self.time_limit = time_limit 
        self.nodes_expanded = 0      
        self.start_time = 0

    def order_moves(self, state, moves):
        """
        Move ordering obrigatório. 
        Prioriza capturas, o que aumenta a eficiência da poda Alpha-Beta significativamente.
        """
        def move_score(move):
            from_r, from_c, capture_direction = move
            return 1 if capture_direction is not None else 0
        
        return sorted(moves, key=move_score, reverse=True)

    def get_best_move(self, state):
        self.nodes_expanded = 0
        self.start_time = time.time()

        best_move = None
        depth = 1
        completed_depth = 0
        commited_nodes = 0

        try:
            while True:
                if time.time() - self.start_time >= self.time_limit:
                    break

                self.nodes_expanded = 0
                current_best_move, _ = self.alpha_beta(
                    state, depth, float('-inf'), float('inf'), True
                )

                if current_best_move is not None:
                    best_move = current_best_move
                    completed_depth = depth
                    commited_nodes += self.nodes_expanded

                if time.time() - self.start_time >= self.time_limit:
                    break

                depth += 1

        except TimeoutError:
            pass

        self.nodes_expanded = commited_nodes
        return best_move, self.nodes_expanded, completed_depth

    def alpha_beta(self, state, depth, alpha, beta, maximizing_player):
        if time.time() - self.start_time >= self.time_limit:
            raise TimeoutError()

        # is_term, _ = state.is_terminal()
        # if depth == 0 or is_term:
        #     self.nodes_expanded += 1
        #     return None, self.heuristic(state, self.player)
        is_term, winner = state.is_terminal()

        if is_term:
            self.nodes_expanded += 1

            if winner == self.player:
                return None, 10000 + depth
            elif winner is None:
                return None, 0
            else:
                return None, -10000 - depth

        if depth == 0:
            self.nodes_expanded += 1
            return None, self.heuristic(state, self.player)

        moves = state.get_legal_moves(state.current_player)
        moves = self.order_moves(state, moves) # Aplica a ordenação das jogadas

        best_move = moves[0] if moves else None

        if maximizing_player:
            max_eval = float('-inf')
            for move in moves:
                new_state = state.apply_move(move)
                _, eval = self.alpha_beta(new_state, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                
                # Atualiza o Alpha e faz a poda 
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break # Poda! O oponente já tem uma opção melhor antes
            return best_move, max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                new_state = state.apply_move(move)
                _, eval = self.alpha_beta(new_state, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                
                # Atualiza o Beta e faz a poda 
                beta = min(beta, eval)
                if beta <= alpha:
                    break # Poda! Nós já temos uma opção melhor antes
            return best_move, min_eval