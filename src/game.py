def win_condition(board, player):
    opponent = 2 if player == 1 else 1
    target_row = len(board) - 1 if player == 1 else 0
    
    # Check if player reached the target row
    if any(board[target_row][col] == player for col in range(len(board[0]))):
        return True
    
    # Check if opponent has any pieces left
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
    
    # Capture the piece
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
    
    # Move the piece 
    board[to_row][from_col] = player
    board[from_row][from_col] = 0

def initialize_board(size):
    board = [[0 for _ in range(size)] for _ in range(size)]
    
    for player, rows in ((1, range(0, 2)), (2, range(size - 2, size))):
        for row in rows:
            board[row] = [player] * size
            
    return board

def print_board(board):
    for row in board:
        print(" ".join(str(cell) for cell in row))
        
def game_loop(board):
    current_player = 1
    
    while True:
        print_board(board)
        print(f"Player {current_player}'s turn.")
        
        print("Enter move (from_row from_col [capture_direction]): ", end="")
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
        
def main():
    size = 8
    board = initialize_board(size)
    game_loop(board)
    return 0
    
if __name__ == "__main__":    
    main()