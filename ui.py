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

    # Tablero

    def draw_board(t: float) -> None:
        hints   = {m.end: m for m in gs["sel_moves"]}
        last_sq: set[int] = set()
        if gs["last_move"]:
            last_sq.add(gs["last_move"].start)
            last_sq.add(gs["last_move"].end)

        for row in range(8):
            for col in range(8):
                rect = pygame.Rect(col * CELL, row * CELL, CELL, CELL)
                dark = is_playable_square(row, col)
                idx  = rc_to_index(row, col)
                base = C["sq_dark"] if dark else C["sq_light"]
                if dark and idx in last_sq:
                    base = blend(base, C["last_sq"], 0.45)
                pygame.draw.rect(screen, base, rect)
                if idx is None:
                    continue
                if idx == gs["selected"]:
                    pulse = int(180 + 75 * math.sin(t * 5))
                    pygame.draw.rect(screen, (255, 215, pulse), rect, 4)
                if idx in hints:
                    hc = C["hint_cap"] if hints[idx].captures else C["hint_mv"]
                    hs = alpha_surf(CELL, CELL)
                    pygame.draw.circle(hs, (*hc, 150), (CELL // 2, CELL // 2), 20)
                    pygame.draw.circle(hs, (*hc, 220), (CELL // 2, CELL // 2), 20, 3)
                    screen.blit(hs, rect.topleft)

        for row in range(8):
            for col in range(8):
                idx = rc_to_index(row, col)
                if idx is None:
                    continue
                piece = engine.state.board[idx]
                if piece == EMPTY:
                    continue
                cx = col * CELL + CELL // 2
                cy = row * CELL + CELL // 2
                draw_piece(screen, cx, cy, piece_owner(piece), is_king(piece))

    # Panel

    def draw_card(x, y, w, h, title=None) -> None:
        pygame.draw.rect(screen, C["card"], (x, y, w, h), border_radius=8)
        if title:
            ts = fnt_section.render(title, True, C["muted"])
            screen.blit(ts, (x + 10, y + 8))

    def draw_panel(t: float, mouse_pos) -> dict:
        px  = BOARD_PX
        cw  = PANEL_W - 28
        cx_ = px + 14
        mid = px + PANEL_W // 2

        pygame.draw.rect(screen, C["panel"], (px, 0, PANEL_W, SCREEN_H))
        pygame.draw.line(screen, C["divider"], (px, 0), (px, SCREEN_H), 2)

        y = 16

        # Título
        ts = fnt_title.render("DAMAS  IA", True, C["accent"])
        screen.blit(ts, ts.get_rect(centerx=mid, top=y));  y += 34
        sub = fnt_section.render("PROYECTO FINAL — IA 2026", True, C["muted"])
        screen.blit(sub, sub.get_rect(centerx=mid, top=y)); y += 22
        pygame.draw.line(screen, C["divider"], (cx_, y), (cx_ + cw, y)); y += 10

        # Carta 1 — Turno
        draw_card(cx_, y, cw, 62, "ESTADO DEL JUEGO")
        turn = engine.state.turn
        br   = pygame.Rect(cx_ + 10, y + 26, cw - 20, 28)
        pygame.draw.rect(screen, C["badge_w"] if turn == WHITE else C["badge_b"], br, border_radius=6)
        mini_piece(screen, br.left + 16, br.centery, turn, r=8)
        tn = fnt_val.render(f"Turno: {player_name(turn)}", True, C["text"])
        screen.blit(tn, tn.get_rect(left=br.left + 30, centery=br.centery))
        y += 72

        # Carta 2 — Piezas
        wc = count_pieces(engine.state.board, WHITE)
        bc = count_pieces(engine.state.board, BLACK)
        draw_card(cx_, y, cw, 82, "PIEZAS EN JUEGO")
        _dot_row(cx_ + 10, y + 34, wc, 12, WHITE)
        _dot_row(cx_ + 10, y + 62, bc, 12, BLACK)
        y += 92

        # Carta 3 — Métricas IA
        met = gs["metrics"]
        draw_card(cx_, y, cw, 108, "METRICAS IA")
        rows = [
            ("Nodos explorados", f"{met.nodes:,}"),
            ("Simulaciones",     f"{met.simulations:,}"),
            ("Tiempo jugada",    f"{met.elapsed_seconds:.3f} s"),
            ("Valoracion",       f"{met.evaluation:+.1f}"),
        ]
        my = y + 26
        for lbl, val in rows:
            screen.blit(fnt_body.render(lbl, True, C["muted"]), (cx_ + 10, my))
            vs = fnt_val.render(val, True, C["accent"])
            screen.blit(vs, vs.get_rect(right=cx_ + cw - 10, top=my))
            my += 20

        # Indicador: IA pensando
        if gs["thinking"]:
            dots = "." * (int(t * 3) % 4)
            tsurf = fnt_body.render(f"IA pensando{dots}", True, C["green"])
            screen.blit(tsurf, tsurf.get_rect(centerx=mid, top=my + 2))
        y += 118

        # Ganador
        winner = engine.get_winner()
        if winner is not None:
            wcolor = (110, 220, 110) if winner == WHITE else \
                     (220, 110, 110) if winner == BLACK else C["accent"]
            draw_card(cx_, y, cw, 40, None)
            msg = "EMPATE" if winner == DRAW else f"GANA: {player_name(winner).upper()}"
            screen.blit(fnt_val.render(msg, True, wcolor),
                        fnt_val.render(msg, True, wcolor).get_rect(centerx=mid, centery=y + 20))

        # Botones
        btn_defs = [
            ("Modo",       gs["mode"],  False),
            ("Dificultad", gs["diff"],  False),
            ("Reiniciar",  "",          False),
        ]
        rects: dict[str, pygame.Rect] = {}
        by = BTN_START
        for key, val, active in btn_defs:
            r = pygame.Rect(cx_, by, cw, BTN_H)
            rects[key] = r
            hovered = r.collidepoint(mouse_pos)
            bg = C["btn_on"] if active else (C["btn_hover"] if hovered else C["btn"])
            pygame.draw.rect(screen, bg, r, border_radius=7)
            label = f"{key}:  {val}" if val else key
            lt = fnt_body.render(label, True, C["btn_text"])
            screen.blit(lt, lt.get_rect(center=r.center))
            by += BTN_H + BTN_GAP

        # Pie
        footer = fnt_section.render(
            f"Jugadas: {engine.state.ply}    Sin captura: {engine.state.no_capture_ply}",
            True, C["muted"],
        )
        screen.blit(footer, footer.get_rect(centerx=mid, bottom=SCREEN_H - 6))
        return rects

    def _dot_row(x: int, y: int, count: int, total: int, owner: int) -> None:
        fill  = C["wp_fill"] if owner == WHITE else C["bp_fill"]
        rim   = C["wp_rim"]  if owner == WHITE else C["bp_rim"]
        lc    = C["wp_rim"]  if owner == WHITE else C["muted"]
        name  = "Blancas" if owner == WHITE else "Negras "
        screen.blit(fnt_body.render(f"{name}: {count}", True, lc), (x, y - 14))
        for i in range(total):
            dx = x + i * 13 + 5
            alive = i < count
            pygame.draw.circle(screen, fill if alive else C["divider"], (dx, y), 5)
            pygame.draw.circle(screen, rim  if alive else C["muted"],   (dx, y), 5, 1)

    # Overlay fin de partida

    def draw_gameover(winner: int) -> None:
        ov = alpha_surf(BOARD_PX, SCREEN_H)
        ov.fill((0, 0, 0, 170))
        screen.blit(ov, (0, 0))
        if winner == DRAW:
            msg, col = "EMPATE", C["accent"]
        elif winner == WHITE:
            msg, col = "BLANCAS GANAN", (120, 230, 120)
        else:
            msg, col = "NEGRAS GANAN",  (230, 120, 120)
        bx = BOARD_PX // 2
        ts = fnt_big.render(msg, True, col)
        hs = fnt_hint.render("Pulsa  Reiniciar  para una nueva partida", True, C["muted"])
        screen.blit(ts, ts.get_rect(centerx=bx, centery=SCREEN_H // 2 - 36))
        screen.blit(hs, hs.get_rect(centerx=bx, centery=SCREEN_H // 2 + 28))

    # Bucle principal
    btn_rects: dict[str, "pygame.Rect"] = {}
    last_ai_tick = 0.0
    running = True

    while running:
        t = time.perf_counter()
        mouse = pygame.mouse.get_pos()

        # Recoger resultado IA si listo
        apply_worker_result()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                for label, rect in btn_rects.items():
                    if rect.collidepoint(mx, my):
                        if label == "Modo":
                            gs["mode"] = MODES[(MODES.index(gs["mode"]) + 1) % len(MODES)]
                        elif label == "Dificultad":
                            gs["diff"] = DIFFS[(DIFFS.index(gs["diff"]) + 1) % len(DIFFS)]
                        elif label == "Reiniciar":
                            restart()

                # Clicks en tablero
                if mx < BOARD_PX and is_human_turn() and not engine.is_terminal() and not worker.busy:
                    row, col = my // CELL, mx // CELL
                    clicked  = rc_to_index(row, col)
                    if clicked is not None:
                        chosen = next((m for m in gs["sel_moves"] if m.end == clicked), None)
                        if chosen:
                            gs["last_move"] = chosen
                            engine.apply_move(chosen)
                            gs["selected"]  = None
                            gs["sel_moves"] = []
                        elif piece_owner(engine.state.board[clicked]) == engine.state.turn:
                            gs["selected"]  = clicked
                            gs["sel_moves"] = [m for m in engine.getlegalmoves()
                                               if m.start == clicked]
                        else:
                            gs["selected"]  = None
                            gs["sel_moves"] = []

        # AI triggers

        # Humano vs IA — AI plays automatically when it's Black's turn
        if (gs["mode"] == "Humano vs IA"
                and not is_human_turn()
                and not engine.is_terminal()
                and not worker.busy):
            request_ai_move()

        # IA vs IA — both sides move automatically
        if (gs["mode"] == "IA vs IA"
                and not engine.is_terminal()
                and not worker.busy
                and t - last_ai_tick > 0.3):
            request_ai_move()
            last_ai_tick = t

        # Render
        screen.fill(C["bg"])
        draw_board(t)

        winner = engine.get_winner()
        if winner is not None:
            draw_gameover(winner)

        btn_rects = draw_panel(t, mouse)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
