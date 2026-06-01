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


class AlphaBetaAgent:
    name = "Alpha-Beta"

    def __init__(self, depth: int = 5):
        self.depth = depth
        self.last_metrics = SearchMetrics()
        self._deadline: float | None = None
        self._root_player = WHITE

    def choose_move(self, engine: GameEngine, state: GameState, time_limit: float | None = 2.0) -> Move | None:
        start = time.perf_counter()
        self._deadline = start + time_limit if time_limit else None
        self._root_player = state.turn
        self.last_metrics = SearchMetrics(depth_reached=self.depth)
        best_value = -INF
        best_move = None
        try:
            for move in self._ordered_moves(engine, state):
                value = self._value(
                    engine,
                    engine._apply_move_unchecked(state, move),
                    self.depth - 1,
                    -INF,
                    INF,
                )
                if value > best_value:
                    best_value = value
                    best_move = move
        except TimeoutError:
            pass
        self.last_metrics.elapsed_seconds = time.perf_counter() - start
        self.last_metrics.evaluation = best_value if best_move else evaluate_state(state, state.turn, engine)
        return best_move or (engine.getlegalmoves(state)[0] if engine.getlegalmoves(state) else None)

    def _value(self, engine: GameEngine, state: GameState, depth: int, alpha: float, beta: float) -> float:
        if self._deadline and time.perf_counter() >= self._deadline:
            raise TimeoutError
        self.last_metrics.nodes += 1
        if depth <= 0 or engine.is_terminal(state):
            return evaluate_state(state, self._root_player, engine)

        moves = self._ordered_moves(engine, state)
        if state.turn == self._root_player:
            value = -INF
            for move in moves:
                value = max(value, self._value(engine, engine._apply_move_unchecked(state, move), depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = INF
        for move in moves:
            value = min(value, self._value(engine, engine._apply_move_unchecked(state, move), depth - 1, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def _ordered_moves(self, engine: GameEngine, state: GameState) -> list[Move]:
        moves = engine.getlegalmoves(state)
        return sorted(
            moves,
            key=lambda move: (len(move.captures), move.promotes, evaluate_state(engine._apply_move_unchecked(state, move), state.turn, engine)),
            reverse=True,
        )


@dataclass
class MCTSNode:
    state: GameState
    parent: "MCTSNode | None" = None
    move: Move | None = None
    untried_moves: list[Move] = field(default_factory=list)
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    wins: float = 0.0


class MCTSAgent:
    name = "MCTS"

    def __init__(self, iterations: int = 1000, exploration: float = 1.414):
        self.iterations = iterations
        self.exploration = exploration
        self.last_metrics = SearchMetrics()

    def choose_move(self, engine: GameEngine, state: GameState, time_limit: float | None = 2.0) -> Move | None:
        legal = engine.getlegalmoves(state)
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]

        start = time.perf_counter()
        deadline = start + time_limit if time_limit else None
        root_player = state.turn
        root = MCTSNode(state=state, untried_moves=legal[:])
        simulations = 0

        while simulations < self.iterations and (deadline is None or time.perf_counter() < deadline):
            node = root
            while not node.untried_moves and node.children:
                node = self._best_uct_child(node)

            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                child_state = engine._apply_move_unchecked(node.state, move)
                child = MCTSNode(child_state, parent=node, move=move, untried_moves=engine.getlegalmoves(child_state))
                node.children.append(child)
                node = child

            result = self._rollout(engine, node.state, root_player)
            while node is not None:
                node.visits += 1
                node.wins += result
                node = node.parent
            simulations += 1

        self.last_metrics = SearchMetrics(
            elapsed_seconds=time.perf_counter() - start,
            simulations=simulations,
            nodes=sum(child.visits for child in root.children),
        )
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.move

    def _best_uct_child(self, node: MCTSNode) -> MCTSNode:
        log_parent = math.log(max(1, node.visits))

        def score(child: MCTSNode) -> float:
            if child.visits == 0:
                return INF
            return (child.wins / child.visits) + self.exploration * math.sqrt(log_parent / child.visits)

        return max(node.children, key=score)

    def _rollout(self, engine: GameEngine, state: GameState, root_player: int) -> float:
        current = state
        for _ in range(80):
            winner = engine.get_winner(current)
            if winner is not None:
                if winner == root_player:
                    return 1.0
                if winner == DRAW:
                    return 0.5
                return 0.0
            moves = engine.getlegalmoves(current)
            captures = [move for move in moves if move.captures]
            current = engine._apply_move_unchecked(current, random.choice(captures or moves))
        score = evaluate_state(current, root_player, engine)
        if score > 0:
            return 1.0
        if score < 0:
            return 0.0
        return 0.5


def run_game(max_plies: int = MAX_GAME_PLIES, white_agent=None, black_agent=None, time_limit: float = 2.0) -> dict[str, object]:
    engine = GameEngine()
    agents = {WHITE: white_agent, BLACK: black_agent}
    metrics: dict[int, list[SearchMetrics]] = {WHITE: [], BLACK: []}

    while not engine.is_terminal() and engine.state.ply < max_plies:
        agent = agents[engine.state.turn]
        if agent is None:
            break
        move = agent.choose_move(engine, engine.state, time_limit=time_limit)
        if move is None:
            break
        engine.apply_move(move)
        metrics[opponent(engine.state.turn)].append(agent.last_metrics)

    winner = engine.get_winner()
    if winner is None and engine.state.ply >= max_plies:
        winner = DRAW
    return {"winner": winner, "plies": engine.state.ply, "metrics": metrics}


def benchmark_branching(max_depth: int = 5) -> Path:
    engine = GameEngine()
    rows: list[dict[str, object]] = []
    for depth in range(1, max_depth + 1):
        for agent in (MinimaxAgent(depth), AlphaBetaAgent(depth)):
            move = agent.choose_move(engine, engine.state, time_limit=2.0)
            nodes = max(1, agent.last_metrics.nodes)
            rows.append(
                {
                    "algorithm": agent.name,
                    "depth": depth,
                    "nodes": nodes,
                    "elapsed_seconds": round(agent.last_metrics.elapsed_seconds, 6),
                    "effective_branching_factor": round(nodes ** (1 / depth), 4),
                    "evaluation": round(agent.last_metrics.evaluation, 3),
                    "best_move": move.label() if move else "",
                }
            )
    return write_csv("branching.csv", rows)


def run_tournament(games: int = 20, time_limit: float = 2.0, alpha_depth: int = 5) -> Path:
    rows: list[dict[str, object]] = []
    for game_number in range(1, games + 1):
        alpha = AlphaBetaAgent(depth=alpha_depth)
        mcts = MCTSAgent(iterations=100000)
        if game_number % 2 == 1:
            white_agent, black_agent = alpha, mcts
        else:
            white_agent, black_agent = mcts, alpha
        result = run_game(white_agent=white_agent, black_agent=black_agent, time_limit=time_limit)
        alpha_metrics = []
        mcts_metrics = []
        for side_metrics in result["metrics"].values():
            for metric in side_metrics:
                if metric.simulations:
                    mcts_metrics.append(metric)
                else:
                    alpha_metrics.append(metric)
        rows.append(
            {
                "game": game_number,
                "white_agent": white_agent.name,
                "black_agent": black_agent.name,
                "winner": player_name(result["winner"]),
                "plies": result["plies"],
                "alpha_beta_avg_time": average([m.elapsed_seconds for m in alpha_metrics]),
                "mcts_avg_time": average([m.elapsed_seconds for m in mcts_metrics]),
                "alpha_beta_total_nodes": sum(m.nodes for m in alpha_metrics),
                "mcts_total_simulations": sum(m.simulations for m in mcts_metrics),
            }
        )
        print(f"Partida {game_number}: ganador={rows[-1]['winner']} plies={result['plies']}")
    return write_csv("tournament.csv", rows)


def average(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def write_csv(filename: str, rows: list[dict[str, object]]) -> Path:
    output_dir = Path("resultados")
    output_dir.mkdir(exist_ok=True)
    path = output_dir / filename
    if not rows:
        return path
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Archivo generado: {path}")
    return path


def make_plots() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        make_svg_plots()
        return

    output_dir = Path("resultados")
    branching_path = output_dir / "branching.csv"
    tournament_path = output_dir / "tournament.csv"

    if branching_path.exists():
        by_algorithm: dict[str, list[tuple[int, int, float]]] = {}
        with branching_path.open(encoding="utf-8") as file:
            for row in csv.DictReader(file):
                by_algorithm.setdefault(row["algorithm"], []).append(
                    (int(row["depth"]), int(row["nodes"]), float(row["effective_branching_factor"]))
                )
        for algorithm, values in by_algorithm.items():
            plt.plot([v[0] for v in values], [v[1] for v in values], marker="o", label=algorithm)
        plt.xlabel("Profundidad")
        plt.ylabel("Nodos visitados")
        plt.title("Minimax vs Alpha-Beta")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "branching.png")
        plt.close()

    if tournament_path.exists():
        counts: dict[str, int] = {}
        with tournament_path.open(encoding="utf-8") as file:
            for row in csv.DictReader(file):
                counts[row["winner"]] = counts.get(row["winner"], 0) + 1
        plt.bar(counts.keys(), counts.values())
        plt.ylabel("Partidas")
        plt.title("Resultados del torneo")
        plt.tight_layout()
        plt.savefig(output_dir / "tournament.png")
        plt.close()
    print(f"Graficas generadas en: {output_dir}")


def make_svg_plots() -> None:
    output_dir = Path("resultados")
    branching_path = output_dir / "branching.csv"
    tournament_path = output_dir / "tournament.csv"

    if branching_path.exists():
        rows = []
        with branching_path.open(encoding="utf-8") as file:
            for row in csv.DictReader(file):
                rows.append((row["algorithm"], int(row["depth"]), int(row["nodes"])))
        write_line_svg(output_dir / "branching.svg", rows, "Nodos visitados")

    if tournament_path.exists():
        counts: dict[str, int] = {}
        with tournament_path.open(encoding="utf-8") as file:
            for row in csv.DictReader(file):
                counts[row["winner"]] = counts.get(row["winner"], 0) + 1
        write_bar_svg(output_dir / "tournament.svg", counts, "Resultados del torneo")

    print(f"Graficas SVG generadas en: {output_dir}")


def write_line_svg(path: Path, rows: list[tuple[str, int, int]], title: str) -> None:
    width, height = 760, 420
    margin = 60
    max_depth = max((depth for _, depth, _ in rows), default=1)
    max_nodes = max((nodes for _, _, nodes in rows), default=1)
    colors = {"Minimax": "#d97706", "Alpha-Beta": "#2563eb"}
    grouped: dict[str, list[tuple[int, int]]] = {}
    for algorithm, depth, nodes in rows:
        grouped.setdefault(algorithm, []).append((depth, nodes))

    def x_for(depth: int) -> float:
        return margin + (depth - 1) * ((width - 2 * margin) / max(1, max_depth - 1))

    def y_for(nodes: int) -> float:
        return height - margin - nodes * ((height - 2 * margin) / max_nodes)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial" font-size="22">{title}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#111"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#111"/>',
    ]
    for algorithm, points in grouped.items():
        points = sorted(points)
        polyline = " ".join(f"{x_for(depth):.1f},{y_for(nodes):.1f}" for depth, nodes in points)
        svg.append(f'<polyline points="{polyline}" fill="none" stroke="{colors.get(algorithm, "#111")}" stroke-width="3"/>')
        for depth, nodes in points:
            svg.append(f'<circle cx="{x_for(depth):.1f}" cy="{y_for(nodes):.1f}" r="5" fill="{colors.get(algorithm, "#111")}"/>')
        svg.append(f'<text x="{width - margin - 140}" y="{70 + 24 * len(svg) % 80}" font-family="Arial" font-size="14" fill="{colors.get(algorithm, "#111")}">{algorithm}</text>')
    path.write_text("\n".join(svg + ["</svg>"]), encoding="utf-8")


def write_bar_svg(path: Path, counts: dict[str, int], title: str) -> None:
    width, height = 760, 420
    margin = 60
    max_value = max(counts.values(), default=1)
    bar_width = 110
    gap = 45
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial" font-size="22">{title}</text>',
    ]
    for i, (label, value) in enumerate(counts.items()):
        x = margin + i * (bar_width + gap)
        bar_height = value * ((height - 2 * margin) / max_value)
        y = height - margin - bar_height
        svg.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="#2563eb"/>')
        svg.append(f'<text x="{x + bar_width / 2}" y="{height - 30}" text-anchor="middle" font-family="Arial" font-size="14">{label}</text>')
        svg.append(f'<text x="{x + bar_width / 2}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial" font-size="14">{value}</text>')
    path.write_text("\n".join(svg + ["</svg>"]), encoding="utf-8")


def run_self_tests() -> None:
    assert len(create_initial_board()) == 32
    assert count_pieces(create_initial_board(), WHITE) == 12
    assert count_pieces(create_initial_board(), BLACK) == 12
    assert rc_to_index(7, 7) is None
    assert index_to_rc(rc_to_index(5, 0)) == (5, 0)

    engine = GameEngine()
    moves = engine.getlegalmoves()
    assert moves
    assert all(engine.state.board[move.start] > 0 for move in moves)

    board = [EMPTY] * 32
    board[21] = WHITE_MAN
    board[17] = BLACK_MAN
    capture_state = GameState(tuple(board), WHITE)
    capture_moves = engine.getlegalmoves(capture_state)
    assert capture_moves and all(move.captures for move in capture_moves)

    board = [EMPTY] * 32
    board[5] = WHITE_MAN
    state = GameState(tuple(board), WHITE)
    assert engine.get_winner(state) == WHITE

    alpha = AlphaBetaAgent(depth=2)
    assert alpha.choose_move(engine, engine.state, time_limit=0.5) in engine.getlegalmoves(engine.state)
    mcts = MCTSAgent(iterations=20)
    assert mcts.choose_move(engine, engine.state, time_limit=0.2) in engine.getlegalmoves(engine.state)
    print("Pruebas internas OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proyecto Damas IA")
    parser.add_argument("--test", action="store_true", help="Ejecuta pruebas internas rapidas")
    parser.add_argument("--benchmark", action="store_true", help="Genera resultados Minimax vs Alpha-Beta")
    parser.add_argument("--tournament", action="store_true", help="Ejecuta torneo Alpha-Beta vs MCTS")
    parser.add_argument("--plots", action="store_true", help="Genera graficas desde CSV")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--time-limit", type=float, default=2.0)
    parser.add_argument("--alpha-depth", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.test:
        run_self_tests()
    elif args.benchmark:
        benchmark_branching(args.max_depth)
    elif args.tournament:
        run_tournament(args.games, args.time_limit, args.alpha_depth)
    elif args.plots:
        make_plots()
    else:
        from ui import run_gui

        run_gui()


if __name__ == "__main__":
    main()
