from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

# Constantes, tipos y funciones auxiliares.

EMPTY = 0
WHITE_MAN = 1
WHITE_KING = 2
BLACK_MAN = -1
BLACK_KING = -2

WHITE = 1
BLACK = -1

DRAW = 0

BOARD_SIZE = 8
PLAYABLE_SQUARES = 32
MAX_NO_CAPTURE_PLY = 80
MAX_GAME_PLIES = 200
INF = float("inf")

# Dataclasses


@dataclass(frozen=True)
class Move:
    path: tuple[int, ...]
    captures: tuple[int, ...] = ()
    promotes: bool = False

    @property
    def start(self) -> int:
        return self.path[0]

    @property
    def end(self) -> int:
        return self.path[-1]

    @property
    def is_capture(self) -> bool:
        return bool(self.captures)

    def label(self) -> str:
        sep = "x" if self.captures else "-"
        return sep.join(str(square) for square in self.path)


@dataclass(frozen=True)
class GameState:
    board: tuple[int, ...]
    turn: int = WHITE
    ply: int = 0
    no_capture_ply: int = 0


@dataclass
class SearchMetrics:
    nodes: int = 0
    elapsed_seconds: float = 0.0
    depth_reached: int = 0
    evaluation: float = 0.0
    simulations: int = 0


def opponent(player: int) -> int:
    """Devuelve el oponente del jugador dado."""
    return BLACK if player == WHITE else WHITE


def player_name(player: int | None) -> str:
    if player == WHITE:
        return "Blancas"
    if player == BLACK:
        return "Negras"
    if player == DRAW:
        return "Empate"
    return "En juego"


def is_playable_square(row: int, col: int) -> bool:
    """True si la casilla (fila,col) es jugable (tablero 8x8 en diagonales)."""
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and (row + col) % 2 == 1


def rc_to_index(row: int, col: int) -> int | None:
    if not is_playable_square(row, col):
        return None
    return row * 4 + col // 2


def index_to_rc(index: int) -> tuple[int, int]:
    if not 0 <= index < PLAYABLE_SQUARES:
        raise ValueError(f"Indice de casilla invalido: {index}")
    row = index // 4
    offset = 1 if row % 2 == 0 else 0
    col = (index % 4) * 2 + offset
    return row, col


def create_initial_board() -> tuple[int, ...]:
    """Crea el tablero inicial con 12 piezas por bando."""
    board = [EMPTY] * PLAYABLE_SQUARES
    for index in range(PLAYABLE_SQUARES):
        row, _ = index_to_rc(index)
        if row <= 2:
            board[index] = BLACK_MAN
        elif row >= 5:
            board[index] = WHITE_MAN
    return tuple(board)


def piece_owner(piece: int) -> int | None:
    """Devuelve el propietario de una pieza (WHITE/BLACK) o None si vacía."""
    if piece > 0:
        return WHITE
    if piece < 0:
        return BLACK
    return None


def is_king(piece: int) -> bool:
    """True si la pieza es una dama/rey."""
    return abs(piece) == WHITE_KING


def promote_if_needed(piece: int, index: int) -> int:
    row, _ = index_to_rc(index)
    if piece == WHITE_MAN and row == 0:
        return WHITE_KING
    if piece == BLACK_MAN and row == BOARD_SIZE - 1:
        return BLACK_KING
    return piece


def directions_for_piece(piece: int) -> tuple[tuple[int, int], ...]:
    if is_king(piece):
        return ((-1, -1), (-1, 1), (1, -1), (1, 1))
    owner = piece_owner(piece)
    if owner == WHITE:
        return ((-1, -1), (-1, 1))
    return ((1, -1), (1, 1))


def count_pieces(board: tuple[int, ...], player: int) -> int:
    """Cuenta piezas de `player` en el tablero."""
    return sum(1 for piece in board if piece_owner(piece) == player)


class GameEngine:
    def __init__(self, state: GameState | None = None):
        self.state = state or GameState(create_initial_board(), WHITE)

    def getlegalmoves(self, state: GameState | None = None) -> list[Move]:
        state = state or self.state
        captures: list[Move] = []
        simple_moves: list[Move] = []

        for index, piece in enumerate(state.board):
            if piece_owner(piece) != state.turn:
                continue
            captures.extend(self._capture_moves_for_piece(state, index))
            simple_moves.extend(self._simple_moves_for_piece(state, index))

        legal = captures if captures else simple_moves
        return sorted(set(legal), key=lambda move: (move.start, move.path, move.captures))

    def get_legal_moves(self, state: GameState | None = None) -> list[Move]:
        return self.getlegalmoves(state)

    def apply_move(self, move: Move, state: GameState | None = None) -> GameState:
        state = state or self.state
        legal_moves = self.getlegalmoves(state)
        if move not in legal_moves:
            raise ValueError(f"Movimiento ilegal: {move}")
        new_state = self._apply_move_unchecked(state, move)
        if state == self.state:
            self.state = new_state
        return new_state

    def evaluate(self, state: GameState | None = None, player: int | None = None) -> float:
        state = state or self.state
        return evaluate_state(state, player or state.turn, self)

    def is_terminal(self, state: GameState | None = None) -> bool:
        return self.get_winner(state) is not None

    def get_winner(self, state: GameState | None = None) -> int | None:
        state = state or self.state
        white_pieces = count_pieces(state.board, WHITE)
        black_pieces = count_pieces(state.board, BLACK)
        if white_pieces == 0:
            return BLACK
        if black_pieces == 0:
            return WHITE
        if state.no_capture_ply >= MAX_NO_CAPTURE_PLY:
            return DRAW
        if not self.getlegalmoves(state):
            return opponent(state.turn)
        return None

    def alphabeta(
        self,
        depth: int,
        alpha: float = -INF,
        beta: float = INF,
        state: GameState | None = None,
    ) -> tuple[float, Move | None]:
        agent = AlphaBetaAgent(depth=depth)
        state = state or self.state
        move = agent.choose_move(self, state, time_limit=2.0)
        return agent.last_metrics.evaluation, move

    def alpha_beta(
        self,
        depth: int,
        alpha: float = -INF,
        beta: float = INF,
        state: GameState | None = None,
    ) -> tuple[float, Move | None]:
        return self.alphabeta(depth, alpha, beta, state)

    def expectimax(self, depth: int, state: GameState | None = None) -> tuple[float, Move | None]:
        # The project only requires the methods considered convenient.
        # This light version models the opponent as choosing uniformly.
        state = state or self.state
        root_player = state.turn
        best_value = -INF
        best_move = None
        for move in self.getlegalmoves(state):
            value = self._expectimax_value(
                self._apply_move_unchecked(state, move),
                depth - 1,
                root_player,
            )
            if value > best_value:
                best_value = value
                best_move = move
        return best_value, best_move

    def mcts(self, iterations: int = 500, C: float = 1.414, state: GameState | None = None) -> Move | None:
        agent = MCTSAgent(iterations=iterations, exploration=C)
        return agent.choose_move(self, state or self.state, time_limit=None)

    def _simple_moves_for_piece(self, state: GameState, index: int) -> list[Move]:
        piece = state.board[index]
        moves: list[Move] = []
        row, col = index_to_rc(index)
        for dr, dc in directions_for_piece(piece):
            end = rc_to_index(row + dr, col + dc)
            if end is not None and state.board[end] == EMPTY:
                promoted_piece = promote_if_needed(piece, end)
                moves.append(Move(path=(index, end), promotes=promoted_piece != piece))
        return moves

    def _capture_moves_for_piece(self, state: GameState, index: int) -> list[Move]:
        piece = state.board[index]
        results: list[Move] = []
        self._extend_capture(
            board=state.board,
            current=index,
            piece=piece,
            path=(index,),
            captures=(),
            promoted=False,
            results=results,
        )
        return results

    def _extend_capture(
        self,
        board: tuple[int, ...],
        current: int,
        piece: int,
        path: tuple[int, ...],
        captures: tuple[int, ...],
        promoted: bool,
        results: list[Move],
    ) -> None:
        found = False
        row, col = index_to_rc(current)
        for dr, dc in directions_for_piece(piece):
            middle = rc_to_index(row + dr, col + dc)
            landing = rc_to_index(row + 2 * dr, col + 2 * dc)
            if middle is None or landing is None:
                continue
            if middle in captures:
                continue
            if piece_owner(board[middle]) != opponent(piece_owner(piece)) or board[landing] != EMPTY:
                continue

            found = True
            new_board = list(board)
            new_board[current] = EMPTY
            new_board[middle] = EMPTY
            new_piece = promote_if_needed(piece, landing)
            new_board[landing] = new_piece
            did_promote = promoted or new_piece != piece
            new_path = path + (landing,)
            new_captures = captures + (middle,)

            # Decision del proyecto: si un peon corona al capturar, termina el turno.
            if new_piece != piece and not is_king(piece):
                results.append(Move(new_path, new_captures, promotes=True))
            else:
                self._extend_capture(
                    tuple(new_board),
                    landing,
                    new_piece,
                    new_path,
                    new_captures,
                    did_promote,
                    results,
                )

        if captures and not found:
            results.append(Move(path, captures, promotes=promoted))

    def _apply_move_unchecked(self, state: GameState, move: Move) -> GameState:
        board = list(state.board)
        piece = board[move.start]
        board[move.start] = EMPTY
        for captured in move.captures:
            board[captured] = EMPTY
        board[move.end] = promote_if_needed(piece, move.end)
        no_capture_ply = 0 if move.captures else state.no_capture_ply + 1
        return GameState(tuple(board), opponent(state.turn), state.ply + 1, no_capture_ply)

    def _expectimax_value(self, state: GameState, depth: int, root_player: int) -> float:
        winner = self.get_winner(state)
        if depth <= 0 or winner is not None:
            return evaluate_state(state, root_player, self)
        moves = self.getlegalmoves(state)
        if not moves:
            return evaluate_state(state, root_player, self)
        if state.turn == root_player:
            return max(
                self._expectimax_value(self._apply_move_unchecked(state, move), depth - 1, root_player)
                for move in moves
            )
        return sum(
            self._expectimax_value(self._apply_move_unchecked(state, move), depth - 1, root_player)
            for move in moves
        ) / len(moves)


def evaluate_state(state: GameState, player: int, engine: GameEngine | None = None) -> float:
    engine = engine or GameEngine(state)
    winner = engine.get_winner(state)
    if winner == player:
        return 100000.0
    if winner == opponent(player):
        return -100000.0
    if winner == DRAW:
        return 0.0

    score = 0.0
    for index, piece in enumerate(state.board):
        owner = piece_owner(piece)
        if owner is None:
            continue
        sign = 1 if owner == player else -1
        row, col = index_to_rc(index)
        if abs(piece) == WHITE_MAN:
            score += sign * 100
            advancement = (7 - row) if owner == WHITE else row
            score += sign * advancement * 5
        else:
            score += sign * 175
        if 2 <= row <= 5 and 2 <= col <= 5:
            score += sign * 6
        if col in (0, 7):
            score += sign * 8

    own_moves = len(engine.getlegalmoves(GameState(state.board, player, state.ply, state.no_capture_ply)))
    rival_moves = len(engine.getlegalmoves(GameState(state.board, opponent(player), state.ply, state.no_capture_ply)))
    score += (own_moves - rival_moves) * 8
    return score


class RandomAgent:
    name = "Random"

    def __init__(self) -> None:
        self.last_metrics = SearchMetrics()

    def choose_move(self, engine: GameEngine, state: GameState, time_limit: float | None = None) -> Move | None:
        start = time.perf_counter()
        moves = engine.getlegalmoves(state)
        move = random.choice(moves) if moves else None
        self.last_metrics = SearchMetrics(nodes=1, elapsed_seconds=time.perf_counter() - start)
        return move


class MinimaxAgent:
    name = "Minimax"

    def __init__(self, depth: int = 3):
        self.depth = depth
        self.last_metrics = SearchMetrics()

    def choose_move(self, engine: GameEngine, state: GameState, time_limit: float | None = None) -> Move | None:
        start = time.perf_counter()
        root_player = state.turn
        self.last_metrics = SearchMetrics(depth_reached=self.depth)
        best_value = -INF
        best_move = None
        for move in engine.getlegalmoves(state):
            value = self._value(engine, engine._apply_move_unchecked(state, move), self.depth - 1, root_player)
            if value > best_value:
                best_value = value
                best_move = move
        self.last_metrics.elapsed_seconds = time.perf_counter() - start
        self.last_metrics.evaluation = best_value if best_move else evaluate_state(state, root_player, engine)
        return best_move

    def _value(self, engine: GameEngine, state: GameState, depth: int, root_player: int) -> float:
        self.last_metrics.nodes += 1
        if depth <= 0 or engine.is_terminal(state):
            return evaluate_state(state, root_player, engine)
        moves = engine.getlegalmoves(state)
        if state.turn == root_player:
            return max(self._value(engine, engine._apply_move_unchecked(state, move), depth - 1, root_player) for move in moves)
        return min(self._value(engine, engine._apply_move_unchecked(state, move), depth - 1, root_player) for move in moves)
