from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
import socket
import time

import pygame

from controller_state import BUTTON_NAMES, ControllerState
from web_server import ServerConfig, start_http_server
from websocket_server import ControllerWebSocketServer


WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540
FPS = 60

MOVE_SPEED = 260.0
DASH_MULTIPLIER = 1.65
GROUND_FRICTION = 1400.0
AIR_CONTROL = 0.72
GRAVITY = 1900.0
JUMP_SPEED = -660.0
MAX_FALL_SPEED = 1100.0
FAST_FALL_ACCEL = 900.0
COYOTE_TIME = 0.12

BG_TOP = (203, 224, 164)
BG_BOTTOM = (138, 164, 92)
GROUND_COLOR = (59, 78, 42)
GROUND_TOP = (165, 188, 99)
PLATFORM_COLOR = (72, 94, 53)
PLATFORM_TOP = (190, 208, 118)
FACE_COLOR = (242, 242, 216)
TEXT_COLOR = (33, 44, 23)
TEXT_DIM = (76, 95, 57)
STAR_COLOR = (236, 244, 190)

PLAYER_PALETTES = (
    {"body": (52, 63, 44), "accent": (126, 149, 79), "crouch": (181, 162, 88), "chip": (88, 111, 58)},
    {"body": (93, 64, 126), "accent": (164, 125, 210), "crouch": (200, 162, 224), "chip": (126, 92, 164)},
    {"body": (66, 103, 140), "accent": (123, 175, 215), "crouch": (162, 201, 231), "chip": (82, 126, 170)},
    {"body": (140, 78, 70), "accent": (212, 126, 110), "crouch": (228, 168, 142), "chip": (171, 103, 82)},
    {"body": (86, 108, 54), "accent": (166, 197, 106), "crouch": (199, 218, 141), "chip": (112, 138, 70)},
    {"body": (125, 92, 54), "accent": (206, 156, 94), "crouch": (226, 191, 136), "chip": (164, 122, 73)},
)

SPAWN_X_POSITIONS = (90, 178, 266, 354, 442, 530, 618, 706, 794)


KEY_BINDINGS = {
    "select": (pygame.K_TAB,),
}


@dataclass
class PlayerAvatar:
    player_id: str
    label: str
    spawn_slot: int
    x: float
    y: float
    spawn_x: float
    spawn_y: float
    palette: dict[str, tuple[int, int, int]]
    width: int = 34
    height: int = 52
    vx: float = 0.0
    vy: float = 0.0
    on_ground: bool = False
    coyote_timer: float = 0.0
    crouching: bool = False
    facing: int = 1

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(round(self.x), round(self.y), self.width, self.height)

    def reset(self) -> None:
        self.x = self.spawn_x
        self.y = self.spawn_y
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False
        self.coyote_timer = 0.0
        self.crouching = False


def detect_public_host() -> str:
    probe = None
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        host = probe.getsockname()[0]
        if host and not host.startswith("127."):
            return host
    except OSError:
        pass
    finally:
        if probe is not None:
            probe.close()

    try:
        host_name = socket.gethostname()
        for address in socket.gethostbyname_ex(host_name)[2]:
            if not address.startswith("127."):
                return address
    except OSError:
        pass

    return "127.0.0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multiplayer Pygame demo controlled by phones over the local network."
    )
    parser.add_argument("--bind-host", default="0.0.0.0", help="Address to bind the local servers to.")
    parser.add_argument(
        "--public-host",
        default=None,
        help="LAN IP or hostname shown in the QR code. Auto-detected by default.",
    )
    parser.add_argument("--http-port", type=int, default=8000, help="HTTP port for the setup and controller pages.")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket port for controller input.")
    parser.add_argument("--headless", action="store_true", help="Run with SDL's dummy video driver.")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional max runtime in seconds, useful for smoke tests.",
    )
    return parser.parse_args()


def build_platforms() -> list[pygame.Rect]:
    ground = pygame.Rect(0, WINDOW_HEIGHT - 100, WINDOW_WIDTH, 100)
    return [
        ground,
        pygame.Rect(150, 360, 190, 22),
        pygame.Rect(450, 300, 210, 22),
        pygame.Rect(720, 230, 160, 22),
    ]


def palette_for_index(index: int) -> dict[str, tuple[int, int, int]]:
    return PLAYER_PALETTES[index % len(PLAYER_PALETTES)]


def next_spawn_slot(players: list[PlayerAvatar]) -> int:
    used = {player.spawn_slot for player in players}
    for slot in range(len(SPAWN_X_POSITIONS)):
        if slot not in used:
            return slot
    return len(players) % len(SPAWN_X_POSITIONS)


def build_player(player_id: str, label: str, color_index: int, spawn_slot: int, spawn_y: float) -> PlayerAvatar:
    spawn_x = float(SPAWN_X_POSITIONS[spawn_slot % len(SPAWN_X_POSITIONS)])
    return PlayerAvatar(
        player_id=player_id,
        label=label,
        spawn_slot=spawn_slot,
        x=spawn_x,
        y=spawn_y,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
        palette=palette_for_index(color_index),
    )


def gather_keyboard_state(previous: dict[str, bool]) -> tuple[dict[str, bool], dict[str, bool]]:
    keys = pygame.key.get_pressed()
    held = {
        button: any(keys[key_code] for key_code in codes)
        for button, codes in KEY_BINDINGS.items()
    }
    pressed = {
        button: held[button] and not previous[button]
        for button in KEY_BINDINGS
    }
    return held, pressed


def build_pressed_state(events: list[object]) -> dict[str, bool]:
    pressed = {button: False for button in BUTTON_NAMES}
    for event in events:
        if event.pressed:
            pressed[event.button] = True
    return pressed


def clamp_player(player: PlayerAvatar) -> None:
    max_x = WINDOW_WIDTH - player.width
    if player.x < 0.0:
        player.x = 0.0
        player.vx = 0.0
    elif player.x > max_x:
        player.x = float(max_x)
        player.vx = 0.0


def update_player(
    player: PlayerAvatar,
    dt: float,
    held: dict[str, bool],
    pressed: dict[str, bool],
    platforms: list[pygame.Rect],
) -> None:
    if pressed["start"]:
        player.reset()
        return

    move_axis = 0
    if held["left"]:
        move_axis -= 1
    if held["right"]:
        move_axis += 1

    if move_axis:
        player.facing = move_axis

    speed = MOVE_SPEED * (DASH_MULTIPLIER if held["b"] else 1.0)
    if held["down"]:
        speed *= 0.48

    if player.on_ground:
        player.coyote_timer = COYOTE_TIME
    else:
        player.coyote_timer = max(0.0, player.coyote_timer - dt)

    if pressed["a"] or pressed["up"]:
        if player.on_ground or player.coyote_timer > 0.0:
            player.vy = JUMP_SPEED
            player.on_ground = False
            player.coyote_timer = 0.0

    if move_axis:
        control = 1.0 if player.on_ground else AIR_CONTROL
        player.vx = move_axis * speed * control
    elif player.on_ground:
        drag = GROUND_FRICTION * dt
        if abs(player.vx) <= drag:
            player.vx = 0.0
        else:
            player.vx -= math.copysign(drag, player.vx)

    if held["down"] and not player.on_ground:
        player.vy += FAST_FALL_ACCEL * dt

    player.vy = min(player.vy + GRAVITY * dt, MAX_FALL_SPEED)
    player.crouching = held["down"] and player.on_ground

    player.x += player.vx * dt
    clamp_player(player)
    rect = player.rect
    for platform in platforms:
        if not rect.colliderect(platform):
            continue
        if player.vx > 0.0:
            rect.right = platform.left
        elif player.vx < 0.0:
            rect.left = platform.right
        player.x = float(rect.x)
        player.vx = 0.0

    player.on_ground = False
    player.y += player.vy * dt
    rect = player.rect
    for platform in platforms:
        if not rect.colliderect(platform):
            continue
        if player.vy >= 0.0:
            rect.bottom = platform.top
            player.on_ground = True
            player.coyote_timer = COYOTE_TIME
        else:
            rect.top = platform.bottom
        player.y = float(rect.y)
        player.vy = 0.0

    if player.y > WINDOW_HEIGHT + 220:
        player.reset()


def resolve_player_collisions(players: list[PlayerAvatar]) -> None:
    for _ in range(3):
        collided = False
        for index, first in enumerate(players):
            for second in players[index + 1:]:
                first_rect = first.rect
                second_rect = second.rect
                if not first_rect.colliderect(second_rect):
                    continue

                overlap_x = min(first_rect.right, second_rect.right) - max(first_rect.left, second_rect.left)
                overlap_y = min(first_rect.bottom, second_rect.bottom) - max(first_rect.top, second_rect.top)
                if overlap_x <= 0 or overlap_y <= 0:
                    continue

                collided = True
                if overlap_y <= overlap_x:
                    if first_rect.centery <= second_rect.centery:
                        first.y -= overlap_y
                        if first.vy > 0.0:
                            first.vy = 0.0
                        first.on_ground = True
                        first.coyote_timer = COYOTE_TIME
                    else:
                        second.y -= overlap_y
                        if second.vy > 0.0:
                            second.vy = 0.0
                        second.on_ground = True
                        second.coyote_timer = COYOTE_TIME
                else:
                    shift = overlap_x / 2.0
                    if first_rect.centerx <= second_rect.centerx:
                        first.x -= shift
                        second.x += shift
                    else:
                        first.x += shift
                        second.x -= shift
                    first.vx = 0.0
                    second.vx = 0.0
                    clamp_player(first)
                    clamp_player(second)

        if not collided:
            break


def draw_background(surface: pygame.Surface, tick: int) -> None:
    for row in range(WINDOW_HEIGHT):
        blend = row / WINDOW_HEIGHT
        color = (
            round(BG_TOP[0] * (1.0 - blend) + BG_BOTTOM[0] * blend),
            round(BG_TOP[1] * (1.0 - blend) + BG_BOTTOM[1] * blend),
            round(BG_TOP[2] * (1.0 - blend) + BG_BOTTOM[2] * blend),
        )
        pygame.draw.line(surface, color, (0, row), (WINDOW_WIDTH, row))

    for index in range(6):
        x = 90 + index * 150 + int(math.sin((tick / 40.0) + index) * 6)
        y = 70 + (index % 2) * 26
        pygame.draw.circle(surface, STAR_COLOR, (x, y), 3)
        pygame.draw.circle(surface, STAR_COLOR, (x + 22, y + 6), 2)
        pygame.draw.circle(surface, STAR_COLOR, (x + 38, y - 3), 2)


def draw_platform(surface: pygame.Surface, rect: pygame.Rect) -> None:
    pygame.draw.rect(surface, PLATFORM_COLOR, rect, border_radius=8)
    top = pygame.Rect(rect.x, rect.y, rect.width, 8)
    pygame.draw.rect(surface, PLATFORM_TOP, top, border_top_left_radius=8, border_top_right_radius=8)


def draw_player(surface: pygame.Surface, font: pygame.font.Font, player: PlayerAvatar) -> None:
    body_rect = player.rect
    body_color = player.palette["crouch"] if player.crouching else player.palette["body"]
    pygame.draw.rect(surface, body_color, body_rect, border_radius=7)
    face_rect = pygame.Rect(body_rect.x + 6, body_rect.y + 6, body_rect.width - 12, 14)
    pygame.draw.rect(surface, FACE_COLOR, face_rect, border_radius=5)

    eye_x = face_rect.x + (4 if player.facing < 0 else face_rect.width - 8)
    pygame.draw.circle(surface, TEXT_COLOR, (eye_x, face_rect.y + 7), 2)

    pack = pygame.Rect(
        body_rect.x + (body_rect.width - 8 if player.facing < 0 else 0),
        body_rect.y + 14,
        8,
        20,
    )
    pygame.draw.rect(surface, player.palette["accent"], pack, border_radius=3)

    feet_y = body_rect.bottom
    pygame.draw.line(surface, TEXT_COLOR, (body_rect.x + 7, feet_y), (body_rect.x + 11, feet_y + 5), 2)
    pygame.draw.line(surface, TEXT_COLOR, (body_rect.right - 7, feet_y), (body_rect.right - 11, feet_y + 5), 2)

    chip_text = font.render(player.label, True, FACE_COLOR)
    chip = chip_text.get_rect(midbottom=(body_rect.centerx, body_rect.y - 6))
    chip.inflate_ip(12, 6)
    pygame.draw.rect(surface, player.palette["chip"], chip, border_radius=10)
    pygame.draw.rect(surface, TEXT_COLOR, chip, width=2, border_radius=10)
    surface.blit(chip_text, chip_text.get_rect(center=chip.center))


def draw_roster(
    surface: pygame.Surface,
    font: pygame.font.Font,
    players: list[PlayerAvatar],
    y: int,
) -> None:
    x = 24
    for player in players:
        label = font.render(player.label, True, FACE_COLOR)
        chip = label.get_rect(topleft=(x, y))
        chip.inflate_ip(18, 10)
        pygame.draw.rect(surface, player.palette["chip"], chip, border_radius=12)
        pygame.draw.rect(surface, TEXT_COLOR, chip, width=2, border_radius=12)
        surface.blit(label, label.get_rect(center=chip.center))
        x = chip.right + 10


def draw_scene(
    surface: pygame.Surface,
    fonts: tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font],
    remote_players: list[PlayerAvatar],
    platforms: list[pygame.Rect],
    connected_clients: int,
    last_event: str,
    setup_url: str,
    show_debug: bool,
    tick: int,
) -> None:
    title_font, body_font, small_font = fonts
    draw_background(surface, tick)

    for platform in platforms:
        if platform.bottom >= WINDOW_HEIGHT:
            pygame.draw.rect(surface, GROUND_COLOR, platform)
            top = pygame.Rect(platform.x, platform.y, platform.width, 12)
            pygame.draw.rect(surface, GROUND_TOP, top)
        else:
            draw_platform(surface, platform)

    for player in remote_players:
        draw_player(surface, small_font, player)

    panel = pygame.Rect(18, 16, 560, 144)
    pygame.draw.rect(surface, (238, 244, 206), panel, border_radius=16)
    pygame.draw.rect(surface, (110, 129, 71), panel, width=3, border_radius=16)

    joined_labels = ", ".join(player.label for player in remote_players) or "none"
    title = title_font.render("PHONE WEBPAD ARENA", True, TEXT_COLOR)
    subtitle = body_font.render("Every phone connection spawns a player sprite.", True, TEXT_DIM)
    status = body_font.render(
        f"Phones joined: {connected_clients}   Last event: {last_event}",
        True,
        TEXT_COLOR,
    )
    joined = small_font.render(f"Remote roster: {joined_labels}", True, TEXT_DIM)
    url = small_font.render(f"Open on PC: {setup_url}", True, TEXT_DIM)

    surface.blit(title, (34, 28))
    surface.blit(subtitle, (34, 64))
    surface.blit(status, (34, 90))
    surface.blit(joined, (34, 114))
    surface.blit(url, (34, 136))

    if remote_players:
        draw_roster(surface, small_font, remote_players, WINDOW_HEIGHT - 42)

    if show_debug:
        debug_box = pygame.Rect(WINDOW_WIDTH - 244, 18, 220, 170)
        pygame.draw.rect(surface, (238, 244, 206), debug_box, border_radius=14)
        pygame.draw.rect(surface, (110, 129, 71), debug_box, width=3, border_radius=14)
        roster_text = joined_labels if len(joined_labels) <= 18 else f"{joined_labels[:15]}..."
        lines = [
            f"remote sprites: {len(remote_players)}",
            f"labels: {roster_text}",
            f"setup: {setup_url.split('//', 1)[-1]}",
            "tab toggles this panel",
        ]
        for index, line in enumerate(lines):
            rendered = small_font.render(line, True, TEXT_COLOR)
            surface.blit(rendered, (WINDOW_WIDTH - 224, 32 + index * 20))


def sync_remote_players(
    remote_players: dict[str, PlayerAvatar],
    player_snapshots: dict[str, dict[str, object]],
    spawn_y: float,
) -> None:
    active_ids = set(player_snapshots)
    stale_ids = [player_id for player_id in remote_players if player_id not in active_ids]
    for player_id in stale_ids:
        remote_players.pop(player_id, None)

    for player_id, snapshot in player_snapshots.items():
        if player_id in remote_players:
            remote_players[player_id].label = str(snapshot["label"])
            continue

        spawn_slot = next_spawn_slot(list(remote_players.values()))
        remote_players[player_id] = build_player(
            player_id=player_id,
            label=str(snapshot["label"]),
            color_index=int(snapshot["color_index"]),
            spawn_slot=spawn_slot,
            spawn_y=spawn_y,
        )


def main() -> int:
    args = parse_args()

    if args.headless and "SDL_VIDEODRIVER" not in os.environ:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    public_host = args.public_host or detect_public_host()
    controller_state = ControllerState()
    config = ServerConfig(
        bind_host=args.bind_host,
        public_host=public_host,
        http_port=args.http_port,
        ws_port=args.ws_port,
    )

    http_server, http_thread = start_http_server(controller_state, config)
    ws_server = ControllerWebSocketServer(controller_state, args.bind_host, args.ws_port)
    ws_server.start()

    pygame.init()
    pygame.display.set_caption("Mobile WebPad Arena")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    fonts = (
        pygame.font.SysFont("consolas", 28, bold=True),
        pygame.font.SysFont("consolas", 18),
        pygame.font.SysFont("consolas", 15),
    )

    platforms = build_platforms()
    spawn_y = float(platforms[0].top - 52.0)
    remote_players: dict[str, PlayerAvatar] = {}

    previous_keyboard = {button: False for button in KEY_BINDINGS}
    show_debug = True
    started_at = time.monotonic()
    tick = 0

    print(f"Setup page:      {config.setup_url}")
    print(f"Controller page: {config.controller_url}")
    print(f"WebSocket port:  ws://{config.public_host}:{config.ws_port}")
    print("Only active phone connections spawn sprites in the scene.")
    if public_host == "127.0.0.1":
        print("Warning: LAN IP auto-detection fell back to 127.0.0.1. Use --public-host if your phone cannot connect.")

    try:
        running = True
        while running:
            dt = clock.tick(FPS) / 1000.0
            tick += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keyboard_held, keyboard_pressed = gather_keyboard_state(previous_keyboard)
            previous_keyboard = keyboard_held

            remote_snapshot = controller_state.drain()
            player_snapshots = {
                str(snapshot["client_id"]): snapshot
                for snapshot in remote_snapshot["players"]
            }
            sync_remote_players(remote_players, player_snapshots, spawn_y)

            if keyboard_pressed["select"]:
                show_debug = not show_debug

            for player_id in sorted(player_snapshots):
                snapshot = player_snapshots[player_id]
                held = {
                    button: bool(snapshot["buttons"][button])
                    for button in BUTTON_NAMES
                }
                pressed = build_pressed_state(snapshot["events"])
                update_player(remote_players[player_id], dt, held, pressed, platforms)

            resolved_players = list(remote_players.values())
            resolve_player_collisions(resolved_players)

            draw_scene(
                surface=screen,
                fonts=fonts,
                remote_players=sorted(remote_players.values(), key=lambda player: player.label),
                platforms=platforms,
                connected_clients=int(remote_snapshot["connected_clients"]),
                last_event=str(remote_snapshot["last_event"]),
                setup_url=config.setup_url,
                show_debug=show_debug,
                tick=tick,
            )
            pygame.display.flip()

            if args.duration is not None and (time.monotonic() - started_at) >= args.duration:
                running = False
    finally:
        pygame.quit()
        ws_server.stop()
        http_server.shutdown()
        http_server.server_close()
        http_thread.join(timeout=3.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
