import time

class GameState:
    def __init__(self, board=None, current_player=1, size=8):
        if isinstance(board, int) and size == 8:
            size = board
            board = None

        if board is None:
            board = initialize_board(size)

        self.board = [row[:] for row in board]
        self.size = len(self.board)
        self.current_player = current_player

    def is_terminal(self):
        for player in (1, 2):
            if win_condition(self.board, player):
                return True, player
        return False, None

    def get_legal_moves(self, player):
        moves = []
        board_size = len(self.board)

        for row in range(board_size):
            for col in range(len(self.board[0])):
                if self.board[row][col] != player:
                    continue

                to_row = row + 1 if player == 1 else row - 1
                if 0 <= to_row < board_size and self.board[to_row][col] == 0:
                    moves.append((row, col, None))

                for direction in [-1, 1]:
                    to_row = row + 1 if player == 1 else row - 1
                    to_col = col + direction
                    if 0 <= to_row < board_size and 0 <= to_col < len(self.board[0]):
                        opponent = 2 if player == 1 else 1
                        if self.board[to_row][to_col] == opponent:
                            moves.append((row, col, direction))

        return moves

    def apply_move(self, move):
        from_row, from_col, capture_direction = move

        new_board = [row[:] for row in self.board]

        if capture_direction is not None:
            capture_piece(new_board, self.current_player, from_row, from_col, capture_direction)
        else:
            move_piece(new_board, self.current_player, from_row, from_col)

        return GameState(new_board, 3 - self.current_player)


def initialize_board(size):
    board = [[0 for _ in range(size)] for _ in range(size)]

    for player, rows in ((1, range(0, 2)), (2, range(size - 2, size))):
        for row in rows:
            board[row] = [player] * size

    return board


def win_condition(board, player):
    opponent = 2 if player == 1 else 1
    target_row = len(board) - 1 if player == 1 else 0

    if any(board[target_row][col] == player for col in range(len(board[0]))):
        return True

    return not any(cell == opponent for row in board for cell in row)


def capture_piece(board, player, from_row, from_col, direction):
    if board[from_row][from_col] != player:
        raise ValueError("Invalid move: No piece of the player at the source position.")

    if player == 1:
        to_row = from_row + 1
        to_col = from_col + direction
    else:
        to_row = from_row - 1
        to_col = from_col + direction

    if to_col < 0 or to_col >= len(board[0]) or to_row < 0 or to_row >= len(board):
        raise ValueError("Invalid move: Destination position is out of bounds.")

    if board[to_row][to_col] == 0:
        raise ValueError("Invalid move: No piece to capture at the destination position.")

    board[to_row][to_col] = player
    board[from_row][from_col] = 0


def move_piece(board, player, from_row, from_col):
    if board[from_row][from_col] != player:
        raise ValueError("Invalid move: No piece of the player at the source position.")

    to_row = from_row + 1 if player == 1 else from_row - 1

    if to_row < 0 or to_row >= len(board):
        raise ValueError("Invalid move: Destination position is out of bounds.")

    if board[to_row][from_col] != 0:
        raise ValueError("Invalid move: Destination position is already occupied.")

    board[to_row][from_col] = player
    board[from_row][from_col] = 0


def print_board(board):
    print("\n  " + " ".join(str(i) for i in range(len(board[0]))))
    symbols = {0: ".", 1: "X", 2: "O"}

    for i, row in enumerate(board):
        print(f"{i} " + " ".join(symbols[cell] for cell in row))
    print()


def game_loop(board):
    current_player = 1

    while True:
        print_board(board)
        print(f"Player {current_player}'s turn.")

        print("Enter move ([ROW] [COL] ADD TO CAPTURE <-[-1] | [1]->): ", end="")
        move_input = input().strip().split()
        from_row, from_col = int(move_input[0]), int(move_input[1])
        capture_direction = int(move_input[2]) if len(move_input) > 2 else None

        try:
            if capture_direction is not None:
                capture_piece(board, current_player, from_row, from_col, capture_direction)
            else:
                move_piece(board, current_player, from_row, from_col)
        except ValueError as e:
            print(e)
            continue

        if win_condition(board, current_player):
            print_board(board)
            print(f"Player {current_player} wins!")
            return current_player

        current_player = 2 if current_player == 1 else 1


class Game:
    def __init__(self, size=8):
        self.size = size

    def play_match(self, agent1, agent2, verbose=True, max_turns=1000):
        state = GameState(size=self.size)
        turn = 0

        total_nodes_p1 = 0
        total_nodes_p2 = 0
        max_depth_p1 = 0
        max_depth_p2 = 0
        moves_p1 = 0
        moves_p2 = 0
        total_time_p1 = 0.0
        total_time_p2 = 0.0

        if verbose:
            print("Jogo Breakthrough Inicializado!")
            print("Player 1: X | Player 2: O")

        while turn < max_turns:
            if verbose:
                print_board(state.board)

            is_term, winner = state.is_terminal()
            if is_term:
                if verbose:
                    print(f"Fim de Jogo! Vencedor: Player {winner}")

                return {
                    "winner": winner,
                    "turns": turn,
                    "nodes_p1": total_nodes_p1,
                    "nodes_p2": total_nodes_p2,
                    "max_depth_p1": max_depth_p1,
                    "max_depth_p2": max_depth_p2,
                    "avg_time_p1": (total_time_p1 / moves_p1) if moves_p1 else 0,
                    "avg_time_p2": (total_time_p2 / moves_p2) if moves_p2 else 0,
                }

            if state.current_player == agent1.player:
                start_time = time.time()  # Start time
                move, nodes, depth = agent1.get_best_move(state)
                elapsed = time.time() - start_time  # Stop time
                total_time_p1 += elapsed
                total_nodes_p1 += nodes
                max_depth_p1 = max(max_depth_p1, depth)
                moves_p1 += 1

                if verbose:
                    print(f"P1 jogou {move}. Nós: {nodes} | Depth: {depth}")

            else:
                start_time = time.time()  # Start time
                move, nodes, depth = agent2.get_best_move(state)
                elapsed = time.time() - start_time  # Stop time
                total_time_p2 += elapsed
                total_nodes_p2 += nodes
                max_depth_p2 = max(max_depth_p2, depth)
                moves_p2 += 1

                if verbose:
                    print(f"P2 jogou {move}. Nós: {nodes} | Depth: {depth}")

            if move is None:
                return {
                    "winner": 3 - state.current_player,
                    "turns": turn,
                    "nodes_p1": total_nodes_p1,
                    "nodes_p2": total_nodes_p2,
                    "max_depth_p1": max_depth_p1,
                    "max_depth_p2": max_depth_p2,
                    "avg_time_p1": (total_time_p1 / moves_p1) if moves_p1 else 0,
                    "avg_time_p2": (total_time_p2 / moves_p2) if moves_p2 else 0,
                }

            state = state.apply_move(move)
            turn += 1

        return {
            "winner": None,
            "turns": turn,
            "nodes_p1": total_nodes_p1,
            "nodes_p2": total_nodes_p2,
            "max_depth_p1": max_depth_p1,
            "max_depth_p2": max_depth_p2,
            "avg_time_p1": (total_time_p1 / moves_p1) if moves_p1 else 0,
            "avg_time_p2": (total_time_p2 / moves_p2) if moves_p2 else 0,
        }


# def main():
#     size = 8
#     board = initialize_board(size)
#     game_loop(board)
#     return 0


# if __name__ == "__main__":
#     main()