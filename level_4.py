from __future__ import annotations

import argparse
from pathlib import Path
import socket
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pygame

MOBILE_WEBPAD_DIR = Path(__file__).resolve().parent / "mobile_webpad_demo"
if str(MOBILE_WEBPAD_DIR) not in sys.path:
    sys.path.insert(0, str(MOBILE_WEBPAD_DIR))

from controller_state import BUTTON_NAMES, ControllerState
from web_server import ServerConfig, start_http_server
from websocket_server import ControllerWebSocketServer

from modules.calibration import calibrate_system
from modules.camera_magic import gain_offset_calibration, screen_to_camera_space
from modules.videostream import VideoStream, get_camera_capture
from level_3 import (
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
    PLAYER_RESPAWN_Y_MARGIN,
    PLAYER_SIZE,
    SHOW_DEBUG_WINDOWS,
    TARGET_DISPLAY_INDEX,
    apply_heat_tint,
    build_platforms as build_level_3_platforms,
    clamp,
    compute_heat_distortion,
    create_fullscreen_display,
    draw_platforms,
    load_calibration,
    objects_to_mask,
    run_countdown,
    sample_heat_region,
    save_calibration,
    shadow_object_detection_calibrated,
    surface_to_camera_gray_masked,
    update_heat_field,
)


FAST_FALL_ACCEL = 940.0
PLAYER_COLLISION_PASSES = 3

PLAYER_PALETTES = (
    {"body": (50, 90, 138), "accent": (110, 182, 236), "chip": (35, 55, 82)},
    {"body": (130, 82, 42), "accent": (236, 166, 82), "chip": (72, 44, 18)},
    {"body": (86, 118, 58), "accent": (164, 220, 112), "chip": (44, 70, 28)},
    {"body": (124, 66, 112), "accent": (226, 128, 208), "chip": (68, 34, 64)},
    {"body": (88, 82, 154), "accent": (148, 156, 255), "chip": (48, 42, 88)},
    {"body": (134, 58, 70), "accent": (248, 120, 132), "chip": (78, 28, 36)},
)


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


def lighten_color(color, amount):
    return tuple(
        int(round(channel + (255 - channel) * amount))
        for channel in color
    )


def palette_for_index(index):
    return PLAYER_PALETTES[index % len(PLAYER_PALETTES)]


def build_level_4_platforms(width, height):
    base_platforms, _ = build_level_3_platforms(width, height)
    ground_rect = base_platforms[0]
    center_rect = base_platforms[1]

    side_width = int(width * 0.16)
    left_rect = pygame.Rect(
        int(width * 0.16),
        int(height * 0.48),
        side_width,
        PLATFORM_HEIGHT,
    )
    right_rect = pygame.Rect(
        width - int(width * 0.16) - side_width,
        int(height * 0.39),
        side_width,
        PLATFORM_HEIGHT,
    )

    return [ground_rect, left_rect, center_rect, right_rect]


def build_spawn_points(width, ground_rect):
    ratios = (0.09, 0.21, 0.33, 0.45, 0.57, 0.69, 0.81, 0.91)
    spawn_y = float(ground_rect.top - PLAYER_SIZE[1])
    points = []
    for ratio in ratios:
        x = width * ratio - PLAYER_SIZE[0] / 2.0
        x = clamp(x, 0.0, width - PLAYER_SIZE[0])
        points.append((float(x), spawn_y))
    return points


def build_pressed_state(events):
    pressed = {button: False for button in BUTTON_NAMES}
    for event in events:
        if event.pressed:
            pressed[event.button] = True
    return pressed


def next_spawn_slot(players, slot_count):
    used = {player.spawn_slot for player in players}
    for slot in range(slot_count):
        if slot not in used:
            return slot
    return len(players) % max(1, slot_count)


def clamp_remote_player(player, screen_width):
    max_x = screen_width - player.width
    if player.x < 0.0:
        player.x = 0.0
        player.vx = 0.0
    elif player.x > max_x:
        player.x = float(max_x)
        player.vx = 0.0


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

    @property
    def rect(self):
        return pygame.Rect(round(self.x), round(self.y), self.width, self.height)

    def reset(self):
        self.x = self.spawn_x
        self.y = self.spawn_y
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = True
        self.coyote_timer = PLAYER_COYOTE_TIME
        self.parachute_open = False
        self.heat_value = 0.0
        self.heat_lift = 0.0

    def jump(self):
        if not self.parachute_open and (self.on_ground or self.coyote_timer > 0.0):
            self.vy = PLAYER_JUMP_SPEED
            self.on_ground = False
            self.coyote_timer = 0.0

    def toggle_parachute(self):
        self.parachute_open = not self.parachute_open

    def get_canopy_rect(self):
        left = int(round(self.x + self.width / 2.0 - PARACHUTE_CANOPY_WIDTH / 2.0))
        top = int(round(self.y - PARACHUTE_CANOPY_HEIGHT - 10))
        return pygame.Rect(left, top, PARACHUTE_CANOPY_WIDTH, PARACHUTE_CANOPY_HEIGHT)

    def get_heat_probe_rect(self):
        canopy_rect = self.get_canopy_rect()
        probe_left = canopy_rect.left + 8
        probe_width = max(12, canopy_rect.width - 16)
        probe_height = max(16, self.rect.bottom - canopy_rect.top)
        return pygame.Rect(probe_left, canopy_rect.top, probe_width, probe_height)

    def get_exclusion_rect(self):
        if not self.parachute_open:
            return self.rect.inflate(10, 10)
        return self.rect.union(self.get_canopy_rect()).inflate(14, 14)

    def update(self, dt_sec, held, pressed, platforms, heat_field, screen_width, screen_height):
        if pressed["start"]:
            self.reset()
            return

        if pressed["b"]:
            self.toggle_parachute()

        if pressed["a"] or pressed["up"]:
            self.jump()

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
        elif held["down"] and not self.on_ground:
            gravity += FAST_FALL_ACCEL

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
        clamp_remote_player(self, screen_width)
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

        if self.rect.top < 0:
            self.y = 0.0
            self.vy = max(0.0, self.vy)

        if self.rect.top > screen_height + PLAYER_RESPAWN_Y_MARGIN:
            self.reset()


def build_remote_player(snapshot, spawn_slot, spawn_points):
    spawn_x, spawn_y = spawn_points[spawn_slot]
    return RemotePlayer(
        player_id=str(snapshot["client_id"]),
        label=str(snapshot["label"]),
        color_index=int(snapshot["color_index"]),
        spawn_slot=spawn_slot,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
        palette=palette_for_index(int(snapshot["color_index"])),
        x=spawn_x,
        y=spawn_y,
    )


def sync_remote_players(remote_players, player_snapshots, spawn_points):
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


def resolve_player_collisions(players, screen_width):
    for _ in range(PLAYER_COLLISION_PASSES):
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
                    clamp_remote_player(first, screen_width)
                    clamp_remote_player(second, screen_width)

        if not collided:
            break


def draw_remote_player(surface, player, label_font):
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
    pygame.draw.line(
        surface,
        PLAYER_OUTLINE_COLOR,
        (body_rect.centerx, body_rect.top + 26),
        (body_rect.left + 7, body_rect.top + 34),
        3,
    )
    pygame.draw.line(
        surface,
        PLAYER_OUTLINE_COLOR,
        (body_rect.centerx, body_rect.top + 26),
        (body_rect.right - 7, body_rect.top + 34),
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


def draw_world(screen, frame_surface, grid_image, platforms, players, label_font):
    pygame.surfarray.blit_array(frame_surface, grid_image.swapaxes(0, 1))
    pygame.transform.scale(frame_surface, screen.get_size(), screen)
    draw_platforms(screen, platforms)
    for player in sorted(players, key=lambda item: item.rect.bottom):
        draw_remote_player(screen, player, label_font)


def parse_args():
    parser = argparse.ArgumentParser(description="Level 4 - Heat Gliders with QR phone controllers")
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
    return parser.parse_args()


def main(cached_calibration=False, bind_host="0.0.0.0", public_host=None, http_port=8000, ws_port=8765):
    cap_stream = None
    http_server = None
    http_thread = None
    ws_server = None

    try:
        pygame.init()
        screen, width, height, display_index, monitor, monitor_count = create_fullscreen_display(
            TARGET_DISPLAY_INDEX
        )
        pygame.display.set_caption("Level 4 - Heat Gliders")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 52, bold=True)
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
            cap = get_camera_capture()
            H, mask, _, a_map, b_map = calibrate_system(
                cap=cap, screen=screen, WIDTH=width, HEIGHT=height
            )
            H_inv = np.linalg.inv(H)
            save_calibration(CALIBRATION_CACHE, H, mask, a_map, b_map)
            cap.release()

        cap_stream = VideoStream(0)

        public_host = public_host or detect_public_host()
        controller_state = ControllerState()
        config = ServerConfig(
            bind_host=bind_host,
            public_host=public_host,
            http_port=http_port,
            ws_port=ws_port,
            app_title="Level 4 Heat Gliders",
            helper_note_html="A / UP JUMP<br>B PARACHUTE",
        )
        http_server, http_thread = start_http_server(controller_state, config)
        ws_server = ControllerWebSocketServer(controller_state, bind_host, ws_port)
        ws_server.start()

        print(f"Setup page:      {config.setup_url}")
        print(f"Controller page: {config.controller_url}")
        print(f"WebSocket port:  ws://{config.public_host}:{config.ws_port}")
        print("Phone controls: LEFT/RIGHT move, A or UP jump, B toggle parachute, START reset.")
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
        platforms = build_level_4_platforms(width, height)
        spawn_points = build_spawn_points(width, platforms[0])
        remote_players = {}
        prev_player_cam_mask = None
        show_overlay = False
        frame_count = 0
        t_prev = time.time()
        debug_object_count = 0
        last_remote_event = "waiting"
        connected_clients = 0

        draw_world(screen, frame_surface, last_stage_img, platforms, remote_players.values(), label_font)
        pygame.display.flip()

        def draw_heat_background(dt_sec):
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
            "HEAT GLIDERS",
            ready_font,
            countdown_font,
            draw_frame=draw_heat_background,
        ):
            return

        draw_world(screen, frame_surface, last_stage_img, platforms, remote_players.values(), label_font)
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
                remote_players[player_id].update(
                    dt_sec,
                    held,
                    pressed,
                    platforms,
                    heat_field,
                    width,
                    height,
                )

            resolve_player_collisions(list(remote_players.values()), width)
            draw_world(
                screen,
                frame_surface,
                last_stage_img,
                platforms,
                remote_players.values(),
                label_font,
            )
            pygame.display.flip()

            if frame_count % 60 == 0:
                heat_peak = float(np.max(heat_field))
                fps = 1.0 / max(1e-6, time.time() - t_prev)
                print(
                    f"FPS: {fps:.1f} | Phones: {connected_clients} | Players: {len(remote_players)} "
                    f"| Shadows: {debug_object_count} | Heat peak: {heat_peak:.2f}"
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
    )
