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
