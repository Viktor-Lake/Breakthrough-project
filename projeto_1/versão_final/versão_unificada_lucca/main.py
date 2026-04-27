import sys
from dataclasses import dataclass
from typing import Callable

import pygame

from game import GameState
from agent_alpha_beta import AgentAlphaBeta
from agent_minimax import AgentMinimax
from heuristics import heuristic_material_and_advance, heuristic_defensive_structures


BOARD_SIZE = 8
WINDOW_WIDTH = 935
WINDOW_HEIGHT = 660
FPS = 60
MOVE_ANIM_DURATION = 0.18

BOARD_MARGIN = 32
PANEL_WIDTH = 300
TOP_MARGIN = 24
BOTTOM_MARGIN = 24

LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
BG = (24, 26, 32)
PANEL_BG = (32, 35, 42)
TEXT = (235, 238, 242)
MUTED = (170, 176, 185)
ACCENT = (72, 130, 255)
GREEN = (72, 180, 110)
RED = (220, 92, 92)
YELLOW = (235, 197, 75)
GOLD = (198, 163, 66)
BOARD_BORDER = (60, 64, 74)
HIGHLIGHT = (80, 170, 255)
MOVE_DOT = (52, 92, 160)
LAST_MOVE = (120, 120, 120)


HEURISTICS: dict[str, tuple[str, Callable[[GameState, int], float]]] = {
    "material_and_advance": ("Material + Advance", heuristic_material_and_advance),
    "defensive_structures": ("Defensive Structures", heuristic_defensive_structures),
}

AGENT_TYPES = ("alpha_beta", "minimax")
PLAYER_MODES = ("human", "ai")


@dataclass
class AnimationState:
    move: tuple[int, int, int | None]
    piece: int
    start_pos: tuple[float, float]
    end_pos: tuple[float, float]
    started_at: float
    duration: float = MOVE_ANIM_DURATION

    def progress(self) -> float:
        if self.duration <= 0:
            return 1.0
        t = (pygame.time.get_ticks() / 1000.0) - self.started_at
        return max(0.0, min(1.0, t / self.duration))

    def current_pos(self) -> tuple[float, float]:
        t = self.progress()
        eased = t * t * (3.0 - 2.0 * t)
        x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * eased
        y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * eased
        return x, y

    def finished(self) -> bool:
        return self.progress() >= 1.0


class BreakthroughUI:
    def __init__(
        self,
        size=BOARD_SIZE,
        player1_mode="human",
        player2_mode="ai",
        time_limit=1.0,
        agent_types: dict[int, str] | None = None,
        heuristics: dict[int, str] | None = None,
    ):
        pygame.init()
        pygame.display.set_caption("Breakthrough")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 22)
        self.font_small = pygame.font.SysFont("arial", 18)
        self.font_big = pygame.font.SysFont("arial", 30, bold=True)

        self.size = size
        self.state = GameState(size=size)
        self.time_limit = time_limit

        self.player_modes = {
            1: player1_mode if player1_mode in PLAYER_MODES else "human",
            2: player2_mode if player2_mode in PLAYER_MODES else "ai",
        }

        # Defaults
        self.agent_types = {
            1: "alpha_beta",
            2: "alpha_beta",
        }
        self.heuristics = {
            1: "material_and_advance",
            2: "material_and_advance",
        }

        # Apply overrides
        if agent_types:
            for p in (1, 2):
                if p in agent_types and agent_types[p] in AGENT_TYPES:
                    self.agent_types[p] = agent_types[p]

        if heuristics:
            for p in (1, 2):
                if p in heuristics and heuristics[p] in HEURISTICS:
                    self.heuristics[p] = heuristics[p]

        self.agents = self.build_agents()

        self.selected: tuple[int, int] | None = None
        self.legal_moves: list[tuple[int, int, int | None]] = []
        self.last_move: tuple[int, int, int | None] | None = None
        self.game_over = False
        self.winner: int | None = None
        self.status_message = ""
        self.ai_thinking = False
        self.move_history: list[str] = []
        self.animation: AnimationState | None = None

        self.board_size_px = min(
            WINDOW_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN,
            WINDOW_WIDTH - PANEL_WIDTH - BOARD_MARGIN * 2,
        )
        self.square_size = self.board_size_px // self.size
        self.board_px = self.square_size * self.size
        self.board_x = BOARD_MARGIN
        self.board_y = (WINDOW_HEIGHT - self.board_px) // 2
        self.panel_x = self.board_x + self.board_px + BOARD_MARGIN
        self.panel_y = BOARD_MARGIN
        self.panel_rect = pygame.Rect(self.panel_x, self.panel_y, PANEL_WIDTH, WINDOW_HEIGHT - 2 * BOARD_MARGIN)

    def build_agent(self, player: int):
        agent_type = self.agent_types[player]
        heuristic_key = self.heuristics[player]
        _, heuristic_func = HEURISTICS[heuristic_key]
        if agent_type == "minimax":
            return AgentMinimax(player=player, heuristic_func=heuristic_func, time_limit=self.time_limit)
        return AgentAlphaBeta(player=player, heuristic_func=heuristic_func, time_limit=self.time_limit)

    def build_agents(self):
        agents = {}
        for player in (1, 2):
            if self.player_modes[player] == "ai":
                agents[player] = self.build_agent(player)
            else:
                agents[player] = None
        return agents

    def rebuild_agents(self):
        self.agents = self.build_agents()

    def set_agent_type(self, player: int, agent_type: str):
        if agent_type not in AGENT_TYPES:
            return
        self.agent_types[player] = agent_type
        self.rebuild_agents()
        self.status_message = f"Player {player} agent set to {agent_type.replace('_', ' ').title()}."

    def set_heuristic(self, player: int, heuristic_key: str):
        if heuristic_key not in HEURISTICS:
            return
        self.heuristics[player] = heuristic_key
        self.rebuild_agents()
        label, _ = HEURISTICS[heuristic_key]
        self.status_message = f"Player {player} heuristic set to {label}."

    def reset_game(self):
        self.state = GameState(size=self.size)
        self.selected = None
        self.legal_moves = []
        self.last_move = None
        self.game_over = False
        self.winner = None
        self.status_message = ""
        self.ai_thinking = False
        self.move_history.clear()
        self.animation = None

    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if not self.game_over and self.animation is None and self.is_current_player_human():
                        self.handle_click(event.pos)

            self.update_animation()

            if not self.game_over and self.animation is None and self.is_current_player_ai():
                self.make_ai_move()

            self.draw()
            pygame.display.flip()

        pygame.quit()
        sys.exit()

    def is_current_player_human(self) -> bool:
        return self.player_modes[self.state.current_player] == "human"

    def is_current_player_ai(self) -> bool:
        return self.player_modes[self.state.current_player] == "ai"

    def board_to_screen(self, row: int, col: int) -> pygame.Rect:
        x = self.board_x + col * self.square_size
        y = self.board_y + row * self.square_size
        return pygame.Rect(x, y, self.square_size, self.square_size)

    def square_center(self, row: int, col: int) -> tuple[float, float]:
        rect = self.board_to_screen(row, col)
        return rect.centerx, rect.centery

    def screen_to_board(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        x, y = pos
        if not (self.board_x <= x < self.board_x + self.board_px and self.board_y <= y < self.board_y + self.board_px):
            return None
        col = (x - self.board_x) // self.square_size
        row = (y - self.board_y) // self.square_size
        return int(row), int(col)

    def legal_moves_from_selected(self) -> list[tuple[int, int, int | None]]:
        if self.selected is None:
            return []
        r, c = self.selected
        return [m for m in self.state.get_legal_moves(self.state.current_player) if m[0] == r and m[1] == c]

    def handle_click(self, pos: tuple[int, int]):
        sq = self.screen_to_board(pos)
        if sq is None:
            return

        row, col = sq
        board_piece = self.state.board[row][col]

        if self.selected is None:
            if board_piece == self.state.current_player:
                self.selected = (row, col)
                self.legal_moves = self.legal_moves_from_selected()
            return

        if self.selected == (row, col):
            self.selected = None
            self.legal_moves = []
            return

        move = self.find_move_for_target(row, col)
        if move is not None:
            self.queue_human_move(move)
            return

        if board_piece == self.state.current_player:
            self.selected = (row, col)
            self.legal_moves = self.legal_moves_from_selected()
        else:
            self.selected = None
            self.legal_moves = []

    def find_move_for_target(self, to_row: int, to_col: int):
        # CHANGE: Iterate through actual legal moves from the engine
        for move in self.legal_moves:
            from_row, from_col, capture_direction = move
            
            # Breakthrough movement logic: 
            # Row always advances; Col stays same (None) or shifts by direction (-1, 1)
            expected_row = from_row + 1 if self.state.current_player == 1 else from_row - 1
            expected_col = from_col if capture_direction is None else from_col + capture_direction
            
            if expected_row == to_row and expected_col == to_col:
                return move
        return None

    def queue_human_move(self, move):
        self.start_animation_for_move(move)

    def start_animation_for_move(self, move):
        from_row, from_col, capture_direction = move
        piece = self.state.board[from_row][from_col]
        if capture_direction is None:
            to_row = from_row + 1 if self.state.current_player == 1 else from_row - 1
            to_col = from_col
        else:
            to_row = from_row + 1 if self.state.current_player == 1 else from_row - 1
            to_col = from_col + capture_direction

        self.animation = AnimationState(
            move=move,
            piece=piece,
            start_pos=self.square_center(from_row, from_col),
            end_pos=self.square_center(to_row, to_col),
            started_at=pygame.time.get_ticks() / 1000.0,
        )
        self.selected = None
        self.legal_moves = []
        self.status_message = ""

    def commit_move(self, move):
        prev_player = self.state.current_player
        self.state = self.state.apply_move(move)
        self.last_move = move

        from_row, from_col, capture_direction = move
        if capture_direction is None:
            to_row = from_row + 1 if prev_player == 1 else from_row - 1
            to_col = from_col
        else:
            to_row = from_row + 1 if prev_player == 1 else from_row - 1
            to_col = from_col + capture_direction

        self.move_history.append(self.format_move(move, to_row, to_col))

        is_term, winner = self.state.is_terminal()
        if is_term:
            self.game_over = True
            self.winner = winner

    def update_animation(self):
        if self.animation is None:
            return

        if not self.animation.finished():
            return

        move = self.animation.move
        self.animation = None
        self.commit_move(move)
        self.ai_thinking = False

    def make_ai_move(self):
        if self.ai_thinking or self.game_over or self.animation is not None:
            return

        player = self.state.current_player
        agent = self.agents.get(player)
        if agent is None:
            return

        self.ai_thinking = True
        pygame.event.pump()
        
        # CHANGE: Pass self.state directly instead of creating a new GameState
        move, nodes, depth = agent.get_best_move(self.state)

        if move is None:
            self.game_over = True
            self.winner = 3 - player
            self.ai_thinking = False
            return

        agent_type = self.agent_types[player]
        heuristic_key = self.heuristics[player]
        label, _ = HEURISTICS[heuristic_key]

        self.status_message = f"Player {player} ({agent_type.replace('_', ' ').title()}, {label}) searched {nodes} nodes at depth {depth}."
        self.start_animation_for_move(move)

    def piece_color(self, piece: int):
        return (245, 245, 245) if piece == 1 else (35, 35, 35)

    def piece_edge_color(self, piece: int):
        return (210, 210, 210) if piece == 1 else (90, 90, 90)

    def draw(self):
        self.screen.fill(BG)
        self.draw_board()
        self.draw_panel()

    def draw_board(self):
        board_outer = pygame.Rect(self.board_x - 6, self.board_y - 6, self.board_px + 12, self.board_px + 12)
        pygame.draw.rect(self.screen, BOARD_BORDER, board_outer, border_radius=10)

        for row in range(self.size):
            for col in range(self.size):
                rect = self.board_to_screen(row, col)
                square_color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(self.screen, square_color, rect)

                if self.selected == (row, col):
                    pygame.draw.rect(self.screen, HIGHLIGHT, rect, 4)

                if self.last_move is not None:
                    fr, fc, cap = self.last_move
                    if self.is_last_move_square(row, col, fr, fc, cap):
                        overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                        overlay.fill((*LAST_MOVE, 55))
                        self.screen.blit(overlay, rect.topleft)

                if any(self.move_destination(move) == (row, col) for move in self.legal_moves):
                    center = rect.center
                    radius = max(5, self.square_size // 8)
                    pygame.draw.circle(self.screen, MOVE_DOT, center, radius)

                piece = self.state.board[row][col]
                if piece != 0 and not self.is_animating_piece_from_square(row, col):
                    self.draw_piece(rect.center, piece)

        if self.animation is not None:
            x, y = self.animation.current_pos()
            self.draw_piece((x, y), self.animation.piece, floating=True)

        self.draw_coordinates()

    def is_animating_piece_from_square(self, row: int, col: int) -> bool:
        if self.animation is None:
            return False
        from_row, from_col, _ = self.animation.move
        return (row, col) == (from_row, from_col)

    def draw_piece(self, center, piece: int, floating: bool = False):
        radius = self.square_size // 2 - 8
        if floating:
            radius = int(radius * 0.98)
        outline = (30, 30, 30)
        fill = self.piece_color(piece)
        accent = self.piece_edge_color(piece)
        pygame.draw.circle(self.screen, outline, center, radius + 2)
        pygame.draw.circle(self.screen, fill, center, radius)
        pygame.draw.circle(self.screen, accent, center, radius - 9, 3)

    def draw_coordinates(self):
        for col in range(self.size):
            txt = self.font_small.render(str(col), True, TEXT)
            x = self.board_x + col * self.square_size + self.square_size // 2 - txt.get_width() // 2
            y = self.board_y + self.board_px + 6
            self.screen.blit(txt, (x, y))

        for row in range(self.size):
            txt = self.font_small.render(str(row), True, TEXT)
            x = self.board_x - txt.get_width() - 8
            y = self.board_y + row * self.square_size + self.square_size // 2 - txt.get_height() // 2
            self.screen.blit(txt, (x, y))

    def draw_panel(self):
        pygame.draw.rect(self.screen, PANEL_BG, self.panel_rect, border_radius=16)
        pygame.draw.rect(self.screen, BOARD_BORDER, self.panel_rect, 2, border_radius=16)

        y = self.panel_y + 18
        self.blit_text("Breakthrough", self.font_big, TEXT, self.panel_x + 18, y)
        y += 42

        turn_text = f"Turn: Player {self.state.current_player}"
        turn_color = GREEN if self.state.current_player == 1 else RED
        self.blit_text(turn_text, self.font, turn_color, self.panel_x + 18, y)
        y += 30

        for player in (1, 2):
            mode = self.player_modes[player]

            if mode == "ai":
                agent = self.agent_types[player].replace("_", " ").title()
                heur_label = HEURISTICS[self.heuristics[player]][0]

                self.blit_text(f"P{player}: {agent}", self.font_small, MUTED, self.panel_x + 18, y)
                y += 22
                self.blit_text(f"   Heuristic: {heur_label}", self.font_small, MUTED, self.panel_x + 18, y)
                y += 22
            else:
                self.blit_text(f"P{player}: Human", self.font_small, MUTED, self.panel_x + 18, y)
                y += 22

        y += 4

        self.blit_text(f"P1: {self.player_modes[1].title()} | P2: {self.player_modes[2].title()}", self.font_small, MUTED, self.panel_x + 18, y)
        y += 28

        status = self.status_message if self.status_message else self.current_status_text()
        self.wrap_and_blit(status, self.font_small, TEXT, self.panel_x + 18, y, PANEL_WIDTH - 36)
        y += 72

        if self.game_over:
            winner_text = "Draw" if self.winner is None else f"Winner: Player {self.winner}"
            self.blit_text(winner_text, self.font_big, YELLOW if self.winner is None else GOLD, self.panel_x + 18, y)
            y += 38

        self.draw_section_divider(y)
        y += 18

        self.blit_text("Controls", self.font, TEXT, self.panel_x + 18, y)
        y += 28
        self.wrap_and_blit(
            "Click a piece, then click a destination. Press R to restart.",
            self.font_small,
            MUTED,
            self.panel_x + 18,
            y,
            PANEL_WIDTH - 36,
        )
        y += 88

        self.draw_section_divider(y)
        y += 18

        self.blit_text("Moves", self.font, TEXT, self.panel_x + 18, y)
        y += 30
        self.draw_move_history(y)

    def draw_move_history(self, start_y: int):
        available_height = self.panel_rect.bottom - start_y - 10
        line_height = self.font_small.get_height() + 4
        max_visible = max(1, available_height // line_height)

        visible_moves = self.move_history[-max_visible:]
        y = start_y
        if not visible_moves:
            self.blit_text("No moves yet.", self.font_small, MUTED, self.panel_x + 18, y)
            return

        total_moves = len(self.move_history)

        for i, move_text in enumerate(reversed(visible_moves)):
            move_number = total_moves - i
            self.wrap_and_blit(
                f"{move_number}. {move_text}",
                self.font_small,
                TEXT,
                self.panel_x + 18,
                y,
                PANEL_WIDTH - 36,
            )
            y += 22

    def draw_section_divider(self, y: int):
        start = (self.panel_x + 18, y)
        end = (self.panel_x + PANEL_WIDTH - 18, y)
        pygame.draw.line(self.screen, BOARD_BORDER, start, end, 1)

    def blit_text(self, text, font, color, x, y):
        surf = font.render(text, True, color)
        self.screen.blit(surf, (x, y))

    def wrap_and_blit(self, text, font, color, x, y, max_width):
        words = text.split()
        lines = []
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            if font.size(test)[0] <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)

        for i, line in enumerate(lines):
            self.blit_text(line, font, color, x, y + i * (font.get_height() + 2))

    def current_status_text(self) -> str:
        if self.game_over:
            if self.winner is None:
                return "Game over. Draw."
            return f"Game over. Player {self.winner} wins."

        if self.is_current_player_human():
            return "Your turn."
        if self.animation is not None:
            return "Animating move..."
        return f"Player {self.state.current_player} is thinking..."

    def move_destination(self, move):
        from_row, from_col, capture_direction = move
        if capture_direction is None:
            to_row = from_row + 1 if self.state.current_player == 1 else from_row - 1
            to_col = from_col
        else:
            to_row = from_row + 1 if self.state.current_player == 1 else from_row - 1
            to_col = from_col + capture_direction
        return to_row, to_col

    def is_last_move_square(self, row, col, fr, fc, cap):
        if (row, col) == (fr, fc):
            return True
        prev_player = 2 if self.state.current_player == 1 else 1
        if cap is None:
            tr = fr + 1 if prev_player == 1 else fr - 1
            tc = fc
        else:
            tr = fr + 1 if prev_player == 1 else fr - 1
            tc = fc + cap
        return (row, col) == (tr, tc)

    def format_move(self, move, to_row, to_col) -> str:
        from_row, from_col, capture_direction = move
        return f"({from_row}, {from_col}) -> ({to_row}, {to_col})"


if __name__ == "__main__":
    ui = BreakthroughUI(
        size=8,
        player1_mode="human",
        player2_mode="ai",
        time_limit=1,
        agent_types={
            1: "alpha_beta",
            2: "alpha_beta",
        },
        heuristics={
            1: "material_and_advance",
            2: "defensive_structures",
        },
    )
    ui.run()