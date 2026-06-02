from __future__ import annotations

import math
import threading
import time

from damas import (
    BLACK,
    DRAW,
    EMPTY,
    WHITE,
    AlphaBetaAgent,
    GameEngine,
    GameState,
    MCTSAgent,
    Move,
    SearchMetrics,
    count_pieces,
    create_initial_board,
    is_king,
    is_playable_square,
    piece_owner,
    player_name,
    rc_to_index,
)

# Diseño
CELL     = 80
BOARD_PX = CELL * 8        # 640
PANEL_W  = 340
SCREEN_W = BOARD_PX + PANEL_W  # 980
SCREEN_H = BOARD_PX            # 640

BTN_H     = 40
BTN_GAP   = 8
BTN_START = SCREEN_H - (4 * BTN_H + 3 * BTN_GAP + 30)  # 424

# Paleta
C: dict = {
    "sq_light":  (240, 217, 181),
    "sq_dark":   (101, 143, 107),
    "last_sq":   (190, 190,  70),
    "bg":        ( 22,  21,  18),
    "panel":     ( 28,  27,  25),
    "card":      ( 38,  37,  34),
    "divider":   ( 55,  52,  48),
    "text":      (220, 215, 205),
    "muted":     (130, 125, 115),
    "accent":    (197, 164, 107),
    "green":     (100, 180, 100),
    "wp_shadow": (110, 100,  82),
    "wp_rim":    (255, 250, 238),
    "wp_fill":   (238, 224, 197),
    "wp_mid":    (210, 195, 168),
    "bp_shadow": (  8,   7,   6),
    "bp_rim":    ( 75,  71,  66),
    "bp_fill":   ( 48,  46,  42),
    "bp_mid":    ( 30,  28,  25),
    "crown":     (210, 170,  60),
    "crown_hi":  (245, 210,  90),
    "crown_gem": (200,  55,  55),
    "hint_mv":   ( 80, 200, 120),
    "hint_cap":  (220,  90,  60),
    "btn":       ( 50,  48,  45),
    "btn_hover": ( 68,  65,  60),
    "btn_on":    ( 72, 118,  68),
    "btn_text":  (215, 210, 200),
    "badge_w":   ( 65,  58,  46),
    "badge_b":   ( 40,  38,  36),
}


# Worker IA

class AIWorker:
    """Ejecuta choose_move() en hilo para no bloquear la UI."""

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._thread: threading.Thread | None = None
        self._move:   Move | None    = None
        self._metrics: SearchMetrics | None = None
        self._ready   = False

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready

    def collect(self) -> tuple[Move | None, SearchMetrics]:
        with self._lock:
            m, met = self._move, self._metrics or SearchMetrics()
            self._move    = None
            self._metrics = None
            self._ready   = False
            self._thread  = None
            return m, met

    def start(self, agent, state: GameState) -> None:
        if self.busy:
            return
        self._ready = False
        self._thread = threading.Thread(
            target=self._run, args=(agent, state), daemon=True
        )
        self._thread.start()

    def _run(self, agent, state: GameState) -> None:
        eng  = GameEngine(state)
        move = agent.choose_move(eng, state, time_limit=2.0)
        with self._lock:
            self._move    = move
            self._metrics = agent.last_metrics
            self._ready   = True


# Dibujo de piezas

def draw_piece(surf, cx: int, cy: int, owner: int, king: bool) -> None:
    import pygame
    R = 30
    if owner == WHITE:
        shadow, rim, fill, mid = C["wp_shadow"], C["wp_rim"], C["wp_fill"], C["wp_mid"]
        shine = (255, 255, 255)
    else:
        shadow, rim, fill, mid = C["bp_shadow"], C["bp_rim"], C["bp_fill"], C["bp_mid"]
        shine = (105, 100, 92)
    pygame.draw.circle(surf, shadow, (cx + 3, cy + 5), R)
    pygame.draw.circle(surf, rim,    (cx,     cy    ), R)
    pygame.draw.circle(surf, fill,   (cx,     cy    ), R - 3)
    pygame.draw.circle(surf, mid,    (cx + 3, cy + 3), R - 5)
    pygame.draw.circle(surf, fill,   (cx,     cy    ), R - 9)
    pygame.draw.circle(surf, shine,  (cx - 10, cy - 11), 8)
    pygame.draw.circle(surf, shine,  (cx -  7, cy -  8), 5)
    if king:
        _draw_crown(surf, cx, cy)


def _draw_crown(surf, cx: int, cy: int) -> None:
    import pygame
    pts = [
        (cx - 13, cy + 7), (cx - 13, cy - 4), (cx -  7, cy + 1),
        (cx -  3, cy - 11), (cx,      cy - 3), (cx +  3, cy - 11),
        (cx +  7, cy + 1),  (cx + 13, cy - 4), (cx + 13, cy + 7),
    ]
    pygame.draw.polygon(surf, C["crown"],    pts)
    pygame.draw.polygon(surf, C["crown_hi"], pts, 1)
    for gx, gy in [(cx - 3, cy - 11), (cx, cy - 3), (cx + 3, cy - 11)]:
        pygame.draw.circle(surf, C["crown_gem"],  (gx, gy), 3)
        pygame.draw.circle(surf, (255, 100, 100), (gx, gy), 2)


def mini_piece(surf, cx: int, cy: int, owner: int, r: int = 8) -> None:
    import pygame
    fill = C["wp_fill"] if owner == WHITE else C["bp_fill"]
    rim  = C["wp_rim"]  if owner == WHITE else C["bp_rim"]
    pygame.draw.circle(surf, (8, 7, 6), (cx + 1, cy + 2), r)
    pygame.draw.circle(surf, rim,       (cx, cy),          r)
    pygame.draw.circle(surf, fill,      (cx, cy),          r - 2)
    pygame.draw.circle(surf, (255, 255, 255), (cx - 3, cy - 3), 2)


def alpha_surf(w: int, h: int):
    import pygame
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill((0, 0, 0, 0))
    return s


def blend(a, b, f: float):
    return tuple(int(a[i] + (b[i] - a[i]) * f) for i in range(3))


# GUI principal

def run_gui() -> None:
    try:
        import pygame
    except ImportError:
        print("Pygame no esta instalado. Ejecuta: python -m pip install pygame")
        return

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Damas IA — Proyecto Final 2026")
    clock = pygame.time.Clock()

    def mkfont(size: int, bold: bool = False):
        for name in ("Segoe UI", "Arial"):
            try:
                return pygame.font.SysFont(name, size, bold=bold)
            except Exception:
                pass
        return pygame.font.Font(None, size)

    fnt_title   = mkfont(26, bold=True)
    fnt_section = mkfont(12, bold=True)
    fnt_body    = mkfont(15)
    fnt_val     = mkfont(15, bold=True)
    fnt_big     = mkfont(52, bold=True)
    fnt_hint    = mkfont(20)

    # Estado mutable
    gs = {
        "mode":      "Humano vs IA",
        "diff":      "Media",
        "autoplay":  False,
        "selected":  None,
        "sel_moves": [],
        "last_move": None,
        "metrics":   SearchMetrics(),
        "thinking":  False,   # True si IA piensa
    }

    MODES = ["Humano vs Humano", "Humano vs IA", "IA vs IA"]
    DIFFS = ["Facil", "Media", "Dificil"]
    DEPTH = {"Facil": 2, "Media": 4, "Dificil": 5}

    engine = GameEngine()
    worker = AIWorker()

    # Auxiliares

    def is_human_turn() -> bool:
        if gs["mode"] == "Humano vs Humano":
            return True
        if gs["mode"] == "Humano vs IA":
            return engine.state.turn == WHITE
        return False

    def pick_agent():
        """Devuelve agente según el turno."""
        if engine.state.turn == BLACK:
            return AlphaBetaAgent(depth=DEPTH[gs["diff"]])
        # White in IA vs IA uses MCTS
        return MCTSAgent(iterations=100_000)

    def request_ai_move() -> None:
        """Pide al worker un movimiento (no bloquea)."""
        if worker.busy or engine.is_terminal():
            return
        gs["thinking"] = True
        worker.start(pick_agent(), engine.state)

    def apply_worker_result() -> None:
        """Aplica resultado del worker si ya terminó."""
        if not worker.ready:
            return
        move, metrics = worker.collect()
        gs["thinking"] = False
        if move and not engine.is_terminal():
            gs["last_move"] = move
            engine.apply_move(move)
            gs["metrics"] = metrics
        gs["selected"]  = None
        gs["sel_moves"] = []

    def restart() -> None:
        engine.state    = GameState(create_initial_board(), WHITE)
        gs["selected"]  = None
        gs["sel_moves"] = []
        gs["last_move"] = None
        gs["metrics"]   = SearchMetrics()
        gs["thinking"]  = False
        # hilo daemon: se ignora resultado
