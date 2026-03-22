import copy

class GameState:
    def __init__(self, size=8):
        self.size = size
        self.board = self.initialize_board(size)
        self.current_player = 1 # Player 1 (Move para baixo), Player 2 (Move para cima)

    def initialize_board(self, size):
        # 0 = vazio, 1 = P1, 2 = P2
        board = [[0 for _ in range(size)] for _ in range(size)] # inciailiza o tabuleiro vazio 8x8
        for player, rows in ((1, range(0, 2)), (2, range(size - 2, size))):# Preenche as duas primeiras fileiras de cada jogador
            for row in rows:
                board[row] = [player] * size
        return board

    def get_legal_moves(self, player):
        """Retorna uma lista de tuplas com os movimentos válidos: ((from_r, from_c), (to_r, to_c))"""
        moves = []
        direction = 1 if player == 1 else -1
        opponent = 2 if player == 1 else 1

        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == player:
                    to_r = r + direction
                    
                    if 0 <= to_r < self.size:
                        # 1. Mover para frente (precisa estar vazio)
                        if self.board[to_r][c] == 0:
                            moves.append(((r, c), (to_r, c)))
                        
                        # 2. Capturar na diagonal esquerda
                        if c - 1 >= 0 and self.board[to_r][c - 1] == opponent:
                            moves.append(((r, c), (to_r, c - 1)))
                        
                        # 3. Capturar na diagonal direita
                        if c + 1 < self.size and self.board[to_r][c + 1] == opponent:
                            moves.append(((r, c), (to_r, c + 1)))
        return moves

    def apply_move(self, move):
        """Aplica um movimento e retorna um NOVO estado do jogo (imprescindível para a busca)"""
        new_state = copy.deepcopy(self)
        (from_r, from_c), (to_r, to_c) = move
        
        # Move a peça e atualiza o turno
        new_state.board[to_r][to_c] = new_state.board[from_r][from_c]
        new_state.board[from_r][from_c] = 0
        new_state.current_player = 2 if self.current_player == 1 else 1
        
        return new_state

    def is_terminal(self):
        """Verifica se o jogo acabou """
        # Verifica se alguém alcançou a última fileira adversária
        if 1 in self.board[self.size - 1]: return True, 1
        if 2 in self.board[0]: return True, 2
        
        # Verifica se alguém ficou sem peças
        p1_pieces = sum(row.count(1) for row in self.board)
        p2_pieces = sum(row.count(2) for row in self.board)
        
        if p1_pieces == 0: return True, 2
        if p2_pieces == 0: return True, 1
        
        return False, 0