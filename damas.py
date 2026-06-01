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
