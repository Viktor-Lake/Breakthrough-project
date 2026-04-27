import time

class AgentMinimax:
    def __init__(self, player, heuristic_func, time_limit=1.0):
        self.player = player
        self.heuristic = heuristic_func
        self.time_limit = time_limit # Limite fixo por jogada 
        self.nodes_expanded = 0      # Instrumentação obrigatória
        self.start_time = 0

    def get_best_move(self, state):
        self.nodes_expanded = 0
        self.start_time = time.time()
        best_move = None
        depth = 1
        
        # Iterative Deepening
        try:
            while True:
                # Interrompe se o tempo limite estourar antes da próxima profundidade
                if time.time() - self.start_time >= self.time_limit:
                    break
                    
                current_best_move, _ = self.minimax(state, depth, True)
                
                if current_best_move:
                    best_move = current_best_move
                    
                depth += 1
        except TimeoutError:
            pass # Estourou o tempo no meio da recursão
            
        return best_move, self.nodes_expanded, depth - 1

    def minimax(self, state, depth, maximizing_player):
        # Checagem de tempo
        if time.time() - self.start_time >= self.time_limit:
            raise TimeoutError()

        is_term, _ = state.is_terminal()
        # Teste de terminal e limite de profundidade
        if depth == 0 or is_term:
            self.nodes_expanded += 1
            return None, self.heuristic(state, self.player)

        # Geração de ações
        moves = state.get_legal_moves(state.current_player)
        best_move = moves[0] if moves else None

        if maximizing_player:
            max_eval = float('-inf')
            for move in moves:
                new_state = state.apply_move(move)
                # Chamada recursiva SEM passagem de alpha ou beta
                _, eval = self.minimax(new_state, depth - 1, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
            return best_move, max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                new_state = state.apply_move(move)
                _, eval = self.minimax(new_state, depth - 1, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
            return best_move, min_eval