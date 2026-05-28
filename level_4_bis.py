from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pygame

from level_4 import (
    BUTTON_NAMES,
    CALIBRATION_CACHE,
    COUNTDOWN_SECONDS,
    FPS,
    GRID_H,
    GRID_W,
    LIFT_GLOW_COLOR,
    PARACHUTE_CANOPY_HEIGHT,
    PARACHUTE_CANOPY_WIDTH,
    PARACHUTE_GRAVITY_SCALE,
    PARACHUTE_LIFT_ACCEL,
    PARACHUTE_LIFT_THRESHOLD,
    PARACHUTE_MAX_FALL_SPEED,
    PARACHUTE_RISE_SPEED_MAX,
    PARACHUTE_RISE_SPEED_MIN,
    PLATFORM_HEIGHT,
    PLAYER_COYOTE_TIME,
    PLAYER_GRAVITY,
    PLAYER_HEAD_COLOR,
    PLAYER_JUMP_SPEED,
    PLAYER_MAX_FALL_SPEED,
    PLAYER_MOVE_SPEED,
    PLAYER_OUTLINE_COLOR,
    PLAYER_SIZE,
    SHOW_DEBUG_WINDOWS,
    TARGET_DISPLAY_INDEX,
    ControllerState,
    ControllerWebSocketServer,
    PLAYER_COLLISION_PASSES,
    ServerConfig,
    VideoStream,
    apply_heat_tint,
    build_pressed_state,
    calibrate_system,
    clamp,
    compute_heat_distortion,
    create_fullscreen_display,
    detect_public_host,
    draw_platforms,
    gain_offset_calibration,
    get_camera_capture,
    lighten_color,
    load_calibration,
    next_spawn_slot,
    objects_to_mask,
    palette_for_index,
    run_countdown,
    sample_heat_region,
    save_calibration,
    screen_to_camera_space,
    shadow_object_detection_calibrated,
    start_http_server,
    surface_to_camera_gray_masked,
    update_heat_field,
)


STONE_THROW_SPEED = 420.0
STONE_THROW_LIFT = -150.0
STONE_GRAVITY = 760.0
STONE_MAX_FALL_SPEED = 520.0
STONE_RADIUS = 9
STONE_LIFETIME = 3.2
STONE_COOLDOWN = 0.55
ACCOLADE_BASE_COLOR = (214, 162, 52)
ACCOLADE_TIP_COLOR = (255, 232, 168)
ACCOLADE_MAX_VISIBLE = 5


def wrap_scalar(position: float, size: int, limit: int) -> float:
    span = float(limit + size)
    while position >= limit:
        position -= span
    while position + size <= 0.0:
        position += span
    return position


def wrap_player_position(player: "RemotePlayer", screen_width: int, screen_height: int) -> None:
    player.x = wrap_scalar(player.x, player.width, screen_width)
    player.y = wrap_scalar(player.y, player.height, screen_height)


def build_level_4_bis_platforms(width: int, height: int) -> list[pygame.Rect]:
    ground_top = int(height * 0.83)
    ground_segments = [
        pygame.Rect(0, ground_top, int(width * 0.18), PLATFORM_HEIGHT),
        pygame.Rect(int(width * 0.27), ground_top, int(width * 0.20), PLATFORM_HEIGHT),
        pygame.Rect(int(width * 0.56), ground_top, int(width * 0.18), PLATFORM_HEIGHT),
        pygame.Rect(int(width * 0.83), ground_top, width - int(width * 0.83), PLATFORM_HEIGHT),
    ]

    side_width = int(width * 0.16)
    floating_platforms = [
        pygame.Rect(
            int(width * 0.14),
            int(height * 0.50),
            side_width,
            PLATFORM_HEIGHT,
        ),
        pygame.Rect(
            int(width * 0.40),
            int(height * 0.34),
            int(width * 0.20),
            PLATFORM_HEIGHT,
        ),
        pygame.Rect(
            width - int(width * 0.14) - side_width,
            int(height * 0.43),
            side_width,
            PLATFORM_HEIGHT,
        ),
    ]
    return ground_segments + floating_platforms


def build_spawn_points(platforms: list[pygame.Rect]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for platform in platforms[:4]:
        spawn_y = float(platform.top - PLAYER_SIZE[1])
        for ratio in (0.28, 0.72):
            x = platform.left + platform.width * ratio - PLAYER_SIZE[0] / 2.0
            points.append((float(x), spawn_y))
    return points


@dataclass
class StoneProjectile:
    owner_id: str
    x: float
    y: float
    vx: float
    vy: float
    radius: int = STONE_RADIUS
    lifetime: float = STONE_LIFETIME

    @property
    def rect(self) -> pygame.Rect:
        diameter = self.radius * 2
        return pygame.Rect(
            round(self.x - self.radius),
            round(self.y - self.radius),
            diameter,
            diameter,
        )

    def update(
        self,
        dt_sec: float,
        platforms: list[pygame.Rect],
        screen_width: int,
        screen_height: int,
    ) -> bool:
        self.lifetime -= dt_sec
        if self.lifetime <= 0.0:
            return False

        self.vy = min(self.vy + STONE_GRAVITY * dt_sec, STONE_MAX_FALL_SPEED)
        self.x += self.vx * dt_sec
        self.y += self.vy * dt_sec

        left = wrap_scalar(self.x - self.radius, self.radius * 2, screen_width)
        top = wrap_scalar(self.y - self.radius, self.radius * 2, screen_height)
        self.x = left + self.radius
        self.y = top + self.radius

        stone_rect = self.rect
        for platform in platforms:
            if stone_rect.colliderect(platform):
                return False
        return True


@dataclass
class RemotePlayer:
    player_id: str
    label: str
    color_index: int
    spawn_slot: int
    spawn_x: float
    spawn_y: float
    palette: dict[str, tuple[int, int, int]]
    x: float
    y: float
    width: int = PLAYER_SIZE[0]
    height: int = PLAYER_SIZE[1]
    vx: float = 0.0
    vy: float = 0.0
    on_ground: bool = True
    coyote_timer: float = PLAYER_COYOTE_TIME
    parachute_open: bool = False
    heat_value: float = 0.0
    heat_lift: float = 0.0
    facing: int = 1
    alive: bool = True
    stone_cooldown: float = 0.0
    kill_count: int = 0
    accolade_flash: float = 0.0

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(round(self.x), round(self.y), self.width, self.height)

    def reset(self) -> None:
        self.x = self.spawn_x
        self.y = self.spawn_y
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = True
        self.coyote_timer = PLAYER_COYOTE_TIME
        self.parachute_open = False
        self.heat_value = 0.0
        self.heat_lift = 0.0
        self.alive = True
        self.stone_cooldown = 0.0
        self.accolade_flash = 0.0

    def kill(self) -> None:
        self.alive = False
        self.vx = 0.0
        self.vy = 0.0
        self.parachute_open = False
        self.heat_value = 0.0
        self.heat_lift = 0.0

    def register_kill(self) -> None:
        self.kill_count += 1
        self.accolade_flash = 1.0

    def jump(self) -> None:
        if not self.parachute_open and (self.on_ground or self.coyote_timer > 0.0):
            self.vy = PLAYER_JUMP_SPEED
            self.on_ground = False
            self.coyote_timer = 0.0

    def toggle_parachute(self) -> None:
        self.parachute_open = not self.parachute_open

    def try_throw_stone(self) -> StoneProjectile | None:
        if self.stone_cooldown > 0.0:
            return None
        self.stone_cooldown = STONE_COOLDOWN
        spawn_x = self.x + self.width / 2.0 + self.facing * (self.width * 0.52)
        spawn_y = self.y + self.height * 0.34
        return StoneProjectile(
            owner_id=self.player_id,
            x=spawn_x,
            y=spawn_y,
            vx=self.facing * STONE_THROW_SPEED,
            vy=STONE_THROW_LIFT,
        )

    def get_canopy_rect(self) -> pygame.Rect:
        left = int(round(self.x + self.width / 2.0 - PARACHUTE_CANOPY_WIDTH / 2.0))
        top = int(round(self.y - PARACHUTE_CANOPY_HEIGHT - 10))
        return pygame.Rect(left, top, PARACHUTE_CANOPY_WIDTH, PARACHUTE_CANOPY_HEIGHT)

    def get_heat_probe_rect(self) -> pygame.Rect:
        canopy_rect = self.get_canopy_rect()
        probe_left = canopy_rect.left + 8
        probe_width = max(12, canopy_rect.width - 16)
        probe_height = max(16, self.rect.bottom - canopy_rect.top)
        return pygame.Rect(probe_left, canopy_rect.top, probe_width, probe_height)

    def get_exclusion_rect(self) -> pygame.Rect:
        if not self.alive:
            return pygame.Rect(0, 0, 0, 0)
        if not self.parachute_open:
            return self.rect.inflate(10, 10)
        return self.rect.union(self.get_canopy_rect()).inflate(14, 14)

    def update(
        self,
        dt_sec: float,
        held: dict[str, bool],
        pressed: dict[str, bool],
        platforms: list[pygame.Rect],
        heat_field: np.ndarray,
        screen_width: int,
        screen_height: int,
    ) -> StoneProjectile | None:
        if pressed["start"]:
            self.reset()
            return None

        self.stone_cooldown = max(0.0, self.stone_cooldown - dt_sec)
        self.accolade_flash = max(0.0, self.accolade_flash - dt_sec * 2.4)

        if not self.alive:
            return None

        if pressed["down"]:
            self.toggle_parachute()

        if pressed["a"] or pressed["up"]:
            self.jump()

        thrown_stone = None
        if pressed["b"]:
            thrown_stone = self.try_throw_stone()

        move_dir = 0
        if held["left"]:
            move_dir -= 1
        if held["right"]:
            move_dir += 1

        if move_dir:
            self.facing = move_dir

        control_scale = 0.78 if self.parachute_open else 1.0
        self.vx = move_dir * PLAYER_MOVE_SPEED * control_scale

        gravity = PLAYER_GRAVITY
        max_fall_speed = PLAYER_MAX_FALL_SPEED
        self.heat_value = 0.0
        self.heat_lift = 0.0

        if self.parachute_open:
            gravity *= PARACHUTE_GRAVITY_SCALE
            max_fall_speed = PARACHUTE_MAX_FALL_SPEED
            self.heat_value = sample_heat_region(
                heat_field,
                self.get_heat_probe_rect(),
                screen_width,
                screen_height,
            )
            self.heat_lift = max(0.0, self.heat_value - PARACHUTE_LIFT_THRESHOLD)

        self.vy = min(self.vy + gravity * dt_sec, max_fall_speed)

        if self.parachute_open and self.heat_lift > 0.0:
            self.vy -= self.heat_lift * PARACHUTE_LIFT_ACCEL * dt_sec
            lift_strength = clamp(
                self.heat_lift / max(1e-6, 1.0 - PARACHUTE_LIFT_THRESHOLD),
                0.0,
                1.0,
            )
            target_rise_speed = -(
                PARACHUTE_RISE_SPEED_MIN
                + lift_strength * (PARACHUTE_RISE_SPEED_MAX - PARACHUTE_RISE_SPEED_MIN)
            )
            if self.vy > target_rise_speed:
                self.vy = target_rise_speed

        self.x += self.vx * dt_sec
        self.x = wrap_scalar(self.x, self.width, screen_width)
        rect = self.rect
        for platform in platforms:
            if not rect.colliderect(platform):
                continue
            if self.vx > 0.0:
                rect.right = platform.left
            elif self.vx < 0.0:
                rect.left = platform.right
            self.x = float(rect.x)
            self.vx = 0.0

        self.on_ground = False
        self.y += self.vy * dt_sec
        self.y = wrap_scalar(self.y, self.height, screen_height)
        rect = self.rect
        for platform in platforms:
            if not rect.colliderect(platform):
                continue
            if self.vy >= 0.0:
                rect.bottom = platform.top
                self.on_ground = True
                self.coyote_timer = PLAYER_COYOTE_TIME
            else:
                rect.top = platform.bottom
            self.y = float(rect.y)
            self.vy = 0.0

        if self.on_ground:
            self.coyote_timer = PLAYER_COYOTE_TIME
        else:
            self.coyote_timer = max(0.0, self.coyote_timer - dt_sec)

        return thrown_stone


def build_remote_player(
    snapshot: dict[str, object],
    spawn_slot: int,
    spawn_points: list[tuple[float, float]],
) -> RemotePlayer:
    spawn_x, spawn_y = spawn_points[spawn_slot]
    color_index = int(snapshot["color_index"])
    return RemotePlayer(
        player_id=str(snapshot["client_id"]),
        label=str(snapshot["label"]),
        color_index=color_index,
        spawn_slot=spawn_slot,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
        palette=palette_for_index(color_index),
        x=spawn_x,
        y=spawn_y,
    )


def sync_remote_players(
    remote_players: dict[str, RemotePlayer],
    player_snapshots: dict[str, dict[str, object]],
    spawn_points: list[tuple[float, float]],
) -> None:
    active_ids = set(player_snapshots)
    stale_ids = [player_id for player_id in remote_players if player_id not in active_ids]
    for player_id in stale_ids:
        remote_players.pop(player_id, None)

    for player_id, snapshot in player_snapshots.items():
        if player_id in remote_players:
            remote_players[player_id].label = str(snapshot["label"])
            continue

        spawn_slot = next_spawn_slot(list(remote_players.values()), len(spawn_points))
        remote_players[player_id] = build_remote_player(snapshot, spawn_slot, spawn_points)


def resolve_player_collisions(
    players: list[RemotePlayer],
    screen_width: int,
    screen_height: int,
) -> None:
    alive_players = [player for player in players if player.alive]
    for _ in range(PLAYER_COLLISION_PASSES):
        collided = False
        for index, first in enumerate(alive_players):
            for second in alive_players[index + 1:]:
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
                        first.coyote_timer = PLAYER_COYOTE_TIME
                    else:
                        second.y -= overlap_y
                        if second.vy > 0.0:
                            second.vy = 0.0
                        second.on_ground = True
                        second.coyote_timer = PLAYER_COYOTE_TIME
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
                    wrap_player_position(first, screen_width, screen_height)
                    wrap_player_position(second, screen_width, screen_height)

        if not collided:
            break


def update_stones(
    stones: list[StoneProjectile],
    remote_players: dict[str, RemotePlayer],
    platforms: list[pygame.Rect],
    dt_sec: float,
    screen_width: int,
    screen_height: int,
) -> list[StoneProjectile]:
    active_stones: list[StoneProjectile] = []
    for stone in stones:
        if not stone.update(dt_sec, platforms, screen_width, screen_height):
            continue

        hit_player = None
        stone_rect = stone.rect
        for player in remote_players.values():
            if not player.alive or player.player_id == stone.owner_id:
                continue
            if stone_rect.colliderect(player.rect):
                hit_player = player
                break

        if hit_player is not None:
            hit_player.kill()
            killer = remote_players.get(stone.owner_id)
            if killer is not None:
                killer.register_kill()
            continue

        active_stones.append(stone)
    return active_stones


def draw_accolade(surface: pygame.Surface, player: RemotePlayer) -> None:
    visible_count = min(player.kill_count, ACCOLADE_MAX_VISIBLE)
    if visible_count <= 0:
        return

    flash_mix = clamp(player.accolade_flash, 0.0, 1.0)
    plume_color = lighten_color(ACCOLADE_BASE_COLOR, 0.18 + flash_mix * 0.24)
    plume_highlight = lighten_color(ACCOLADE_TIP_COLOR, flash_mix * 0.20)
    body_rect = player.rect
    direction = -1 if player.facing >= 0 else 1
    base_x = body_rect.left + 7 if direction == -1 else body_rect.right - 7
    base_y = body_rect.top + 20

    for index in range(visible_count):
        feather_base = (base_x + direction * index * 6, base_y - index * 5)
        feather_tip = (
            feather_base[0] + direction * (9 + index),
            feather_base[1] - (20 + index * 3),
        )
        feather_inner = (feather_base[0] - direction * 2, feather_base[1] - 6)
        feather_outer = (feather_base[0] + direction * 5, feather_base[1] - 11)
        pygame.draw.polygon(
            surface,
            plume_color,
            [feather_base, feather_outer, feather_tip, feather_inner],
        )
        pygame.draw.polygon(
            surface,
            plume_highlight,
            [feather_base, feather_outer, feather_tip, feather_inner],
            1,
        )


def draw_stone(surface: pygame.Surface, stone: StoneProjectile) -> None:
    center = (round(stone.x), round(stone.y))
    pygame.draw.circle(surface, (90, 84, 80), center, stone.radius)
    pygame.draw.circle(surface, (150, 142, 136), center, max(2, stone.radius - 4))
    highlight_pos = (center[0] - 3, center[1] - 3)
    pygame.draw.circle(surface, (208, 204, 196), highlight_pos, 2)


def draw_remote_player(
    surface: pygame.Surface,
    player: RemotePlayer,
    label_font: pygame.font.Font,
) -> None:
    if not player.alive:
        return

    palette = player.palette
    outline_color = palette["chip"]
    canopy_color = palette["accent"]
    canopy_highlight = lighten_color(canopy_color, 0.4)

    if player.parachute_open:
        canopy_rect = player.get_canopy_rect()
        if player.heat_lift > 0.0:
            glow_strength = clamp(player.heat_lift * 2.8, 0.0, 1.0)
            glow_rect = canopy_rect.union(player.rect).inflate(36, 44)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            alpha = int(90 * glow_strength)
            glow_color = lighten_color(LIFT_GLOW_COLOR, 0.1)
            pygame.draw.ellipse(
                glow_surface,
                (*glow_color, alpha),
                glow_surface.get_rect(),
            )
            surface.blit(glow_surface, glow_rect.topleft)

        canopy_points = [
            (canopy_rect.left, canopy_rect.bottom - 2),
            (canopy_rect.left + 12, canopy_rect.top + 6),
            (canopy_rect.centerx, canopy_rect.top),
            (canopy_rect.right - 12, canopy_rect.top + 6),
            (canopy_rect.right, canopy_rect.bottom - 2),
        ]
        pygame.draw.polygon(surface, canopy_color, canopy_points)
        pygame.draw.polygon(surface, canopy_highlight, canopy_points, 2)

        harness_top = player.rect.top + 6
        line_targets = [
            (player.rect.left + 6, harness_top),
            (player.rect.centerx, harness_top + 2),
            (player.rect.right - 6, harness_top),
        ]
        line_sources = [
            canopy_points[0],
            (canopy_rect.centerx, canopy_rect.bottom - 3),
            canopy_points[-1],
        ]
        for source, target in zip(line_sources, line_targets):
            pygame.draw.line(surface, canopy_highlight, source, target, 2)

    draw_accolade(surface, player)

    body_rect = player.rect
    pygame.draw.rect(surface, palette["body"], body_rect, border_radius=12)
    pygame.draw.rect(surface, outline_color, body_rect, 2, border_radius=12)

    head_center = (body_rect.centerx, body_rect.top + 12)
    pygame.draw.circle(surface, PLAYER_HEAD_COLOR, head_center, 11)
    pygame.draw.circle(surface, outline_color, head_center, 11, 2)

    torso_top = body_rect.top + 18
    torso_bottom = body_rect.bottom - 8
    pygame.draw.line(
        surface,
        PLAYER_OUTLINE_COLOR,
        (body_rect.centerx, torso_top),
        (body_rect.centerx, torso_bottom),
        3,
    )
    arm_reach = 11 if player.stone_cooldown > STONE_COOLDOWN - 0.12 else 7
    arm_height = body_rect.top + 30
    pygame.draw.line(
        surface,
        PLAYER_OUTLINE_COLOR,
        (body_rect.centerx, body_rect.top + 26),
        (body_rect.left + arm_reach, arm_height),
        3,
    )
    pygame.draw.line(
        surface,
        PLAYER_OUTLINE_COLOR,
        (body_rect.centerx, body_rect.top + 26),
        (body_rect.right - arm_reach, arm_height),
        3,
    )
    pygame.draw.line(
        surface,
        PLAYER_OUTLINE_COLOR,
        (body_rect.centerx, torso_bottom),
        (body_rect.left + 8, body_rect.bottom - 2),
        3,
    )
    pygame.draw.line(
        surface,
        PLAYER_OUTLINE_COLOR,
        (body_rect.centerx, torso_bottom),
        (body_rect.right - 8, body_rect.bottom - 2),
        3,
    )

    label_surf = label_font.render(player.label, True, PLAYER_HEAD_COLOR)
    chip_rect = label_surf.get_rect(center=(body_rect.centerx, body_rect.centery + 6))
    chip_rect.inflate_ip(10, 4)
    pygame.draw.rect(surface, palette["chip"], chip_rect, border_radius=8)
    pygame.draw.rect(surface, canopy_highlight, chip_rect, 2, border_radius=8)
    surface.blit(label_surf, label_surf.get_rect(center=chip_rect.center))


def draw_world(
    screen: pygame.Surface,
    frame_surface: pygame.Surface,
    grid_image: np.ndarray,
    platforms: list[pygame.Rect],
    players: list[RemotePlayer],
    stones: list[StoneProjectile],
    label_font: pygame.font.Font,
) -> None:
    pygame.surfarray.blit_array(frame_surface, grid_image.swapaxes(0, 1))
    pygame.transform.scale(frame_surface, screen.get_size(), screen)
    draw_platforms(screen, platforms)
    for stone in stones:
        draw_stone(screen, stone)
    for player in sorted((item for item in players if item.alive), key=lambda item: item.rect.bottom):
        draw_remote_player(screen, player, label_font)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Level 4 Bis - wraparound gliders with holes and stones"
    )
    parser.add_argument(
        "--cached-calibration",
        action="store_true",
        help="Use cached calibration from calibration_cache.npz.",
    )
    parser.add_argument(
        "--bind-host",
        default="0.0.0.0",
        help="Address to bind the local controller servers to.",
    )
    parser.add_argument(
        "--public-host",
        default=None,
        help="LAN IP or hostname shown in the QR code. Auto-detected by default.",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8000,
        help="HTTP port for the setup page and phone controller page.",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket port for mobile input.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=1,
        help="OpenCV camera index to use. This setup detected the secondary webcam at index 1.",
    )
    return parser.parse_args()


def main(
    cached_calibration: bool = False,
    bind_host: str = "0.0.0.0",
    public_host: str | None = None,
    http_port: int = 8000,
    ws_port: int = 8765,
    camera_index: int = 1,
) -> None:
    cap_stream = None
    http_server = None
    http_thread = None
    ws_server = None

    try:
        pygame.init()
        screen, width, height, display_index, monitor, monitor_count = create_fullscreen_display(
            TARGET_DISPLAY_INDEX
        )
        pygame.display.set_caption("Level 4 Bis - Heat Gliders Arena")
        clock = pygame.time.Clock()
        ready_font = pygame.font.SysFont("Arial", 64, bold=True)
        countdown_font = pygame.font.SysFont("Arial", 120, bold=True)
        label_font = pygame.font.SysFont("Consolas", 13, bold=True)

        if display_index != TARGET_DISPLAY_INDEX:
            print(
                f"Display {TARGET_DISPLAY_INDEX} not found; using display {display_index}."
            )
        print(
            "Display:",
            f"{monitor_count} total,",
            f"using index {display_index} at {monitor['width']}x{monitor['height']}",
            f"pos {monitor['left']},{monitor['top']}",
        )

        calibration = None
        if cached_calibration:
            calibration = load_calibration(CALIBRATION_CACHE)
            if calibration is None:
                print(f"No calibration cache found at {CALIBRATION_CACHE}.")
                pygame.quit()
                return
            H, mask, a_map, b_map = calibration
            H_inv = np.linalg.inv(H)
        else:
            cap = get_camera_capture(camera_index)
            H, mask, _, a_map, b_map = calibrate_system(
                cap=cap, screen=screen, WIDTH=width, HEIGHT=height
            )
            H_inv = np.linalg.inv(H)
            save_calibration(CALIBRATION_CACHE, H, mask, a_map, b_map)
            cap.release()

        cap_stream = VideoStream(camera_index)

        public_host = public_host or detect_public_host()
        controller_state = ControllerState()
        config = ServerConfig(
            bind_host=bind_host,
            public_host=public_host,
            http_port=http_port,
            ws_port=ws_port,
            app_title="Level 4 Bis Heat Gliders",
            helper_note_html="A / UP JUMP<br>DOWN PARACHUTE<br>B THROW STONE",
        )
        http_server, http_thread = start_http_server(controller_state, config)
        ws_server = ControllerWebSocketServer(controller_state, bind_host, ws_port)
        ws_server.start()

        print(f"Setup page:      {config.setup_url}")
        print(f"Controller page: {config.controller_url}")
        print(f"WebSocket port:  ws://{config.public_host}:{config.ws_port}")
        print(
            "Phone controls: LEFT/RIGHT move, A or UP jump, DOWN toggle parachute, "
            "B throw stone, START reset."
        )
        print("Arena: floor holes plus wraparound on left/right and top/bottom.")
        print("Local controls: ESC quit, V toggle shadow overlay, C clear heat.")
        if public_host == "127.0.0.1":
            print(
                "Warning: LAN IP auto-detection fell back to 127.0.0.1. "
                "Use --public-host if phones cannot connect."
            )

        bg_img = np.full((GRID_H, GRID_W, 3), 128, dtype=np.uint8)
        grid_x, grid_y = np.meshgrid(
            np.arange(GRID_W, dtype=np.float32),
            np.arange(GRID_H, dtype=np.float32),
        )
        heat_field = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        frame_surface = pygame.Surface((GRID_W, GRID_H))
        last_stage_img = bg_img.copy()
        platforms = build_level_4_bis_platforms(width, height)
        spawn_points = build_spawn_points(platforms)
        remote_players: dict[str, RemotePlayer] = {}
        stones: list[StoneProjectile] = []
        prev_player_cam_mask = None
        show_overlay = False
        frame_count = 0
        t_prev = time.time()
        debug_object_count = 0
        last_remote_event = "waiting"
        connected_clients = 0

        draw_world(
            screen,
            frame_surface,
            last_stage_img,
            platforms,
            list(remote_players.values()),
            stones,
            label_font,
        )
        pygame.display.flip()

        def draw_heat_background(dt_sec: float) -> bool:
            nonlocal heat_field, last_stage_img, prev_player_cam_mask, debug_object_count

            ret, cam_frame = cap_stream.read()
            if not ret or cam_frame is None:
                return True

            cam_gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
            cam_gray_masked = cv2.bitwise_and(cam_gray, cam_gray, mask=mask)

            h_cam, w_cam = cam_gray.shape[:2]
            warped_proj_gray_masked = screen_to_camera_space(screen, H, mask, h_cam, w_cam)
            expected = gain_offset_calibration(warped_proj_gray_masked, a_map, b_map)

            player_mask_surface = pygame.Surface(screen.get_size())
            player_mask_surface.fill((0, 0, 0))
            for player in remote_players.values():
                if not player.alive:
                    continue
                pygame.draw.rect(
                    player_mask_surface,
                    (255, 255, 255),
                    player.get_exclusion_rect(),
                    border_radius=18,
                )

            player_cam_mask = surface_to_camera_gray_masked(
                player_mask_surface,
                H,
                mask,
                h_cam,
                w_cam,
            ) > 0

            mask_diff = player_cam_mask
            if prev_player_cam_mask is not None:
                mask_diff = np.logical_or(mask_diff, prev_player_cam_mask)

            objects, contours_vis, shadow_mask_u8 = shadow_object_detection_calibrated(
                cam_gray_masked, expected, b_map, mask_diff, H_inv
            )

            shadow_screen = objects_to_mask(objects, height, width)
            shadow_small = cv2.resize(
                shadow_screen, (GRID_W, GRID_H), interpolation=cv2.INTER_NEAREST
            )
            shadow_mask = shadow_small > 0

            heat_field = update_heat_field(heat_field, shadow_mask, dt_sec)
            map_x, map_y = compute_heat_distortion(
                grid_x, grid_y, heat_field, time.perf_counter()
            )

            warped_img = cv2.remap(
                bg_img,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            apply_heat_tint(warped_img, heat_field, shadow_mask)

            if show_overlay:
                warped_img[shadow_mask] = (255, 110, 20)

            last_stage_img = warped_img
            pygame.surfarray.blit_array(frame_surface, warped_img.swapaxes(0, 1))
            pygame.transform.scale(frame_surface, screen.get_size(), screen)
            prev_player_cam_mask = player_cam_mask
            debug_object_count = len(objects)

            if SHOW_DEBUG_WINDOWS:
                cv2.imshow("Camera View", cam_frame)
                cv2.imshow("Expected Camera Image", expected)
                cv2.imshow("Contours", contours_vis)
                cv2.imshow("Shadow Mask", shadow_mask_u8)
                cv2.waitKey(1)

            return True

        if not run_countdown(
            screen,
            clock,
            COUNTDOWN_SECONDS,
            "HEAT GLIDERS BIS",
            ready_font,
            countdown_font,
            draw_frame=draw_heat_background,
        ):
            return

        draw_world(
            screen,
            frame_surface,
            last_stage_img,
            platforms,
            list(remote_players.values()),
            stones,
            label_font,
        )
        pygame.display.flip()

        running = True
        while running:
            dt_sec = min(clock.tick(FPS) / 1000.0, 0.1)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_v:
                        show_overlay = not show_overlay
                    elif event.key == pygame.K_c:
                        heat_field[:] = 0.0

            remote_snapshot = controller_state.drain()
            connected_clients = int(remote_snapshot["connected_clients"])
            player_snapshots = {
                str(snapshot["client_id"]): snapshot
                for snapshot in remote_snapshot["players"]
            }
            sync_remote_players(remote_players, player_snapshots, spawn_points)

            if str(remote_snapshot["last_event"]) != last_remote_event:
                last_remote_event = str(remote_snapshot["last_event"])
                if last_remote_event != "waiting":
                    print(f"Controllers: {last_remote_event}")

            if not draw_heat_background(dt_sec):
                running = False
                continue

            for player_id in sorted(player_snapshots):
                snapshot = player_snapshots[player_id]
                held = {
                    button: bool(snapshot["buttons"].get(button, False))
                    for button in BUTTON_NAMES
                }
                pressed = build_pressed_state(snapshot["events"])
                thrown_stone = remote_players[player_id].update(
                    dt_sec,
                    held,
                    pressed,
                    platforms,
                    heat_field,
                    width,
                    height,
                )
                if thrown_stone is not None:
                    stones.append(thrown_stone)

            resolve_player_collisions(list(remote_players.values()), width, height)
            stones = update_stones(stones, remote_players, platforms, dt_sec, width, height)

            draw_world(
                screen,
                frame_surface,
                last_stage_img,
                platforms,
                list(remote_players.values()),
                stones,
                label_font,
            )
            pygame.display.flip()

            if frame_count % 60 == 0:
                heat_peak = float(np.max(heat_field))
                fps = 1.0 / max(1e-6, time.time() - t_prev)
                alive_players = sum(1 for player in remote_players.values() if player.alive)
                print(
                    f"FPS: {fps:.1f} | Phones: {connected_clients} | Players: {len(remote_players)} "
                    f"| Alive: {alive_players} | Stones: {len(stones)} | Shadows: {debug_object_count} "
                    f"| Heat peak: {heat_peak:.2f}"
                )
                t_prev = time.time()
            frame_count += 1
    finally:
        if cap_stream is not None:
            cap_stream.release()
        pygame.quit()
        cv2.destroyAllWindows()
        if ws_server is not None:
            ws_server.stop()
        if http_server is not None:
            http_server.shutdown()
            http_server.server_close()
        if http_thread is not None:
            http_thread.join(timeout=3.0)


if __name__ == "__main__":
    args = parse_args()
    main(
        cached_calibration=args.cached_calibration,
        bind_host=args.bind_host,
        public_host=args.public_host,
        http_port=args.http_port,
        ws_port=args.ws_port,
        camera_index=args.camera_index,
    )
