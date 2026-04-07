import argparse
import math
import os
import time

import cv2
import numpy as np
import pygame

from modules.calibration import calibrate_system
from modules.camera_magic import gain_offset_calibration, screen_to_camera_space
from modules.settings import HEIGHT, WIDTH
from modules.videostream import VideoStream, get_camera_capture


GRID_H = 180
GRID_W = 240
FPS = 60

TARGET_DISPLAY_INDEX = 1
USE_BORDERLESS_FULLSCREEN = True

CALIBRATION_CACHE = "calibration_cache.npz"

COUNTDOWN_SECONDS = 5

SHADOW_RATIO = 0.6
SHADOW_MIN_DELTA = 15.0
SHADOW_MIN_AREA = 200

HEAT_RISE_CELLS_PER_SEC = 26.0
HEAT_DECAY_PER_SEC = 1.1
HEAT_INJECT_PER_SEC = 2.6
HEAT_DIFFUSE_SIGMA = 1.15
HEAT_SHADOW_BLUR = 1.8

HEAT_WARP_GAIN = 1.55
HEAT_WARP_X = 3.2
HEAT_WARP_Y = 1.7
HEAT_UPDRAFT = 2.25
HEAT_GRAD_WARP = 1.5
HEAT_WARP_CLAMP = 8.0

HEAT_TINT_GAIN = 1.3

SHOW_DEBUG_WINDOWS = True

PLAYER_SIZE = (36, 50)
PLAYER_MOVE_SPEED = 420.0
PLAYER_JUMP_SPEED = -760.0
PLAYER_GRAVITY = 1850.0
PLAYER_MAX_FALL_SPEED = 960.0
PLAYER_RESPAWN_Y_MARGIN = 100.0
PLAYER_COYOTE_TIME = 0.14

GROUND_HEIGHT_RATIO = 0.11
GROUND_MIN_HEIGHT = 56
PLATFORM_HEIGHT = 24
FLOATING_PLATFORM_WIDTH_RATIO = 0.22
FLOATING_PLATFORM_HEIGHT_RATIO = 0.62

PARACHUTE_TOGGLE_KEY = pygame.K_p
PARACHUTE_CANOPY_WIDTH = 92
PARACHUTE_CANOPY_HEIGHT = 24
PARACHUTE_GRAVITY_SCALE = 0.32
PARACHUTE_MAX_FALL_SPEED = 230.0
PARACHUTE_LIFT_ACCEL = 2300.0
PARACHUTE_LIFT_THRESHOLD = 0.08
PARACHUTE_RISE_SPEED_MIN = 26.0
PARACHUTE_RISE_SPEED_MAX = 180.0

PLATFORM_COLOR = (66, 84, 104)
PLATFORM_TOP_COLOR = (216, 178, 104)
PLATFORM_OUTLINE = (26, 34, 44)
PLAYER_BODY_COLOR = (238, 242, 255)
PLAYER_OUTLINE_COLOR = (46, 54, 68)
PLAYER_HEAD_COLOR = (246, 214, 178)
PARACHUTE_COLOR = (244, 126, 62)
PARACHUTE_HIGHLIGHT = (255, 214, 172)
LIFT_GLOW_COLOR = (255, 164, 78)


def clamp(value, low, high):
    return max(low, min(value, high))


def _get_windows_monitors():
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", wintypes.DWORD),
        ]

    MONITORINFOF_PRIMARY = 1
    monitors = []

    def _callback(hmonitor, hdc, lprect, lparam):
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        if user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)) == 0:
            return 1
        rect = info.rcMonitor
        monitors.append({
            "left": rect.left,
            "top": rect.top,
            "right": rect.right,
            "bottom": rect.bottom,
            "width": rect.right - rect.left,
            "height": rect.bottom - rect.top,
            "primary": bool(info.dwFlags & MONITORINFOF_PRIMARY),
        })
        return 1

    enum_proc = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(RECT),
        wintypes.LPARAM,
    )(_callback)
    user32.EnumDisplayMonitors(0, 0, enum_proc, 0)
    return monitors


def _get_pygame_monitors():
    monitors = []
    if hasattr(pygame.display, "get_desktop_sizes"):
        x_offset = 0
        for idx, (w, h) in enumerate(pygame.display.get_desktop_sizes()):
            monitors.append({
                "left": x_offset,
                "top": 0,
                "right": x_offset + w,
                "bottom": h,
                "width": w,
                "height": h,
                "primary": idx == 0,
            })
            x_offset += w
        return monitors

    info = pygame.display.Info()
    monitors.append({
        "left": 0,
        "top": 0,
        "right": info.current_w,
        "bottom": info.current_h,
        "width": info.current_w,
        "height": info.current_h,
        "primary": True,
    })
    return monitors


def get_target_monitor(target_index):
    monitors = []
    if os.name == "nt":
        try:
            monitors = _get_windows_monitors()
        except Exception:
            monitors = []

    if not monitors:
        monitors = _get_pygame_monitors()

    if not monitors:
        monitors = [{
            "left": 0,
            "top": 0,
            "right": WIDTH,
            "bottom": HEIGHT,
            "width": WIDTH,
            "height": HEIGHT,
            "primary": True,
        }]

    if target_index < 0 or target_index >= len(monitors):
        target_index = 0

    return target_index, monitors[target_index], len(monitors)


def create_fullscreen_display(target_index):
    display_index, monitor, monitor_count = get_target_monitor(target_index)
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{monitor['left']},{monitor['top']}"
    os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] = str(display_index)

    size = (monitor["width"], monitor["height"])
    flags = pygame.NOFRAME if USE_BORDERLESS_FULLSCREEN else pygame.FULLSCREEN
    screen = None

    try:
        screen = pygame.display.set_mode(size, flags, display=display_index)
    except TypeError:
        try:
            screen = pygame.display.set_mode(size, flags)
        except pygame.error:
            screen = None
    except pygame.error:
        display_index, monitor, monitor_count = get_target_monitor(0)
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{monitor['left']},{monitor['top']}"
        os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] = "0"
        size = (monitor["width"], monitor["height"])
        try:
            screen = pygame.display.set_mode(size, flags)
        except pygame.error:
            screen = None

    if screen is None:
        screen = pygame.display.set_mode(size, pygame.NOFRAME)

    width, height = screen.get_size()
    return screen, width, height, display_index, monitor, monitor_count


def save_calibration(path, H, mask, a_map, b_map):
    np.savez_compressed(path, H=H, mask=mask, a_map=a_map, b_map=b_map)


def load_calibration(path):
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        return data["H"], data["mask"], data["a_map"], data["b_map"]


def run_countdown(screen, clock, seconds, label, label_font, count_font, draw_frame=None):
    width, height = screen.get_size()
    label_surf = label_font.render(label, True, (255, 255, 255))
    label_rect = label_surf.get_rect(center=(width // 2, int(height * 0.42)))
    end_time = time.time() + seconds

    while True:
        dt_sec = clock.tick(FPS) / 1000.0
        remaining = int(math.ceil(end_time - time.time()))
        if remaining <= 0:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        count_surf = count_font.render(str(remaining), True, (255, 255, 255))
        count_rect = count_surf.get_rect(center=(width // 2, int(height * 0.62)))

        if draw_frame is None:
            screen.fill((0, 0, 0))
        else:
            if not draw_frame(dt_sec):
                return False
        screen.blit(label_surf, label_rect)
        screen.blit(count_surf, count_rect)
        pygame.display.flip()

    return True


def load_background_image(path, size):
    if not path:
        return None
    if not os.path.exists(path):
        return None
    try:
        image = pygame.image.load(path).convert()
    except pygame.error:
        return None
    image = pygame.transform.smoothscale(image, size)
    arr = pygame.surfarray.array3d(image)
    return np.swapaxes(arr, 0, 1).copy()


def build_stage_texture(grid_w, grid_h):
    x = np.linspace(0.0, 1.0, grid_w, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, grid_h, dtype=np.float32)[:, None]

    band_a = 0.5 + 0.5 * np.sin((x * 17.0) + (y * 7.0))
    band_b = 0.5 + 0.5 * np.sin((x * 6.5) - (y * 20.0))
    soft_grad = 0.2 + 0.8 * y

    red = 28.0 + soft_grad * 34.0 + band_a * 10.0
    green = 36.0 + soft_grad * 44.0 + band_b * 7.5
    blue = 58.0 + soft_grad * 34.0

    texture = np.stack([red, green, blue], axis=-1)
    return np.clip(texture, 0, 255).astype(np.uint8)


def objects_to_mask(objects, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for obj in objects:
        pts = np.array(obj["polygon_pygame"], dtype=np.int32)
        if pts.shape[0] < 3:
            continue
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
    return mask


def shadow_object_detection_calibrated(
    cam_gray_masked,
    expected,
    b_map,
    mask_diff,
    H_inv,
    ratio=SHADOW_RATIO,
    min_delta=SHADOW_MIN_DELTA,
    min_area=SHADOW_MIN_AREA,
):
    cam_f = cam_gray_masked.astype(np.float32)
    expected_f = expected.astype(np.float32)
    shadow_thresh = b_map + (expected_f - b_map) * ratio
    shadow_mask = (cam_f < shadow_thresh) & ((expected_f - cam_f) > min_delta)
    if mask_diff is not None:
        shadow_mask[mask_diff] = False

    shadow_mask_u8 = shadow_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        shadow_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    objects = []
    contours_vis = cv2.cvtColor(shadow_mask_u8, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        cv2.drawContours(contours_vis, [cnt], -1, (0, 255, 0), 2)
        pts_cam = cnt.astype(np.float32)
        pts_screen = cv2.perspectiveTransform(pts_cam, H_inv)
        pygame_polygon = []
        for p in pts_screen:
            x_cv, y_cv = p[0]
            pygame_polygon.append((int(x_cv), int(y_cv)))
        objects.append({
            "contour_cam": cnt,
            "polygon_pygame": pygame_polygon,
        })

    return objects, contours_vis, shadow_mask_u8


def _shift_up(field, cells):
    if cells <= 1e-6:
        return field.copy()

    whole = int(cells)
    frac = cells - whole
    out = field.copy()

    if whole > 0:
        out = np.roll(out, -whole, axis=0)
        out[-whole:, :] = 0.0

    if frac > 1e-6:
        up = np.roll(out, -1, axis=0)
        up[-1, :] = 0.0
        out = out * (1.0 - frac) + up * frac

    return out


def update_heat_field(heat_field, shadow_mask, dt_sec):
    if dt_sec <= 0.0:
        return heat_field

    shadow_f = shadow_mask.astype(np.float32)
    shadow_soft = cv2.GaussianBlur(
        shadow_f, (0, 0), sigmaX=HEAT_SHADOW_BLUR, sigmaY=HEAT_SHADOW_BLUR
    )

    rise = _shift_up(heat_field, HEAT_RISE_CELLS_PER_SEC * dt_sec)
    decay = math.exp(-HEAT_DECAY_PER_SEC * dt_sec)
    injected = shadow_soft * (HEAT_INJECT_PER_SEC * dt_sec)

    updated = rise * decay + injected
    updated = cv2.GaussianBlur(
        updated, (0, 0), sigmaX=HEAT_DIFFUSE_SIGMA, sigmaY=HEAT_DIFFUSE_SIGMA
    )
    return np.clip(updated, 0.0, 1.0)


def compute_heat_distortion(grid_x, grid_y, heat_field, t_now):
    if not np.any(heat_field > 1e-5):
        return grid_x, grid_y

    phase_a = t_now * 3.6 + grid_y * 0.42 + grid_x * 0.12
    phase_b = t_now * 5.1 - grid_y * 0.27 + grid_x * 0.39
    phase_c = t_now * 7.7 + grid_y * 0.21 - grid_x * 0.58

    shimmer = np.sin(phase_a) + 0.62 * np.sin(phase_b) + 0.36 * np.cos(phase_c)
    swirl = 0.5 * np.cos(phase_b * 1.3) + 0.5 * np.sin(phase_a * 1.2)

    grad_x = cv2.Sobel(heat_field, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(heat_field, cv2.CV_32F, 0, 1, ksize=3)

    gain = np.clip(heat_field * HEAT_WARP_GAIN, 0.0, 2.0)

    disp_x = (shimmer * HEAT_WARP_X + grad_x * HEAT_GRAD_WARP) * gain
    disp_y = (-HEAT_UPDRAFT + swirl * HEAT_WARP_Y + grad_y * HEAT_GRAD_WARP * 0.5) * gain

    disp_x = np.clip(disp_x, -HEAT_WARP_CLAMP, HEAT_WARP_CLAMP)
    disp_y = np.clip(disp_y, -HEAT_WARP_CLAMP, HEAT_WARP_CLAMP)

    map_x = np.clip(grid_x + disp_x, 0.0, GRID_W - 1.0).astype(np.float32)
    map_y = np.clip(grid_y + disp_y, 0.0, GRID_H - 1.0).astype(np.float32)
    return map_x, map_y


def apply_heat_tint(image_rgb, heat_field, shadow_mask):
    heat_soft = np.clip(heat_field * HEAT_TINT_GAIN, 0.0, 1.0)
    shadow_core = shadow_mask.astype(np.float32) * 0.35
    warm = np.clip(heat_soft + shadow_core, 0.0, 1.0)

    img_f = image_rgb.astype(np.float32)
    img_f[..., 0] += 95.0 * warm
    img_f[..., 1] += 55.0 * warm
    img_f[..., 2] -= 40.0 * warm
    np.clip(img_f, 0.0, 255.0, out=img_f)
    image_rgb[:] = img_f.astype(np.uint8)


def surface_to_camera_gray_masked(surface, H, mask, h_cam, w_cam):
    surface_rgb = pygame.surfarray.array3d(surface)
    surface_rgb_cv = np.swapaxes(surface_rgb, 0, 1)
    surface_gray = cv2.cvtColor(surface_rgb_cv, cv2.COLOR_RGB2GRAY)
    warped_gray = cv2.warpPerspective(surface_gray, H, (w_cam, h_cam))
    return cv2.bitwise_and(warped_gray, warped_gray, mask=mask)


def sample_heat_region(heat_field, rect, screen_width, screen_height):
    if rect.width <= 0 or rect.height <= 0:
        return 0.0

    left = clamp(rect.left, 0, screen_width - 1)
    top = clamp(rect.top, 0, screen_height - 1)
    right = clamp(rect.right, left + 1, screen_width)
    bottom = clamp(rect.bottom, top + 1, screen_height)

    x0 = int((left / screen_width) * GRID_W)
    x1 = int(math.ceil((right / screen_width) * GRID_W))
    y0 = int((top / screen_height) * GRID_H)
    y1 = int(math.ceil((bottom / screen_height) * GRID_H))

    x0 = clamp(x0, 0, GRID_W - 1)
    x1 = clamp(x1, x0 + 1, GRID_W)
    y0 = clamp(y0, 0, GRID_H - 1)
    y1 = clamp(y1, y0 + 1, GRID_H)

    region = heat_field[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    return float(0.55 * np.mean(region) + 0.45 * np.max(region))


def build_platforms(width, height):
    ground_height = max(GROUND_MIN_HEIGHT, int(height * GROUND_HEIGHT_RATIO))
    ground_rect = pygame.Rect(0, height - ground_height, width, ground_height)

    floating_width = int(width * FLOATING_PLATFORM_WIDTH_RATIO)
    floating_rect = pygame.Rect(
        (width - floating_width) // 2,
        int(height * FLOATING_PLATFORM_HEIGHT_RATIO),
        floating_width,
        PLATFORM_HEIGHT,
    )

    spawn_x = floating_rect.centerx - PLAYER_SIZE[0] / 2.0
    spawn_y = floating_rect.top - PLAYER_SIZE[1]
    return [ground_rect, floating_rect], (spawn_x, spawn_y)


def draw_platforms(screen, platforms):
    for rect in platforms:
        radius = min(10, rect.height // 2)
        pygame.draw.rect(screen, PLATFORM_COLOR, rect, border_radius=radius)
        top_band = pygame.Rect(rect.left, rect.top, rect.width, max(6, rect.height // 3))
        pygame.draw.rect(screen, PLATFORM_TOP_COLOR, top_band, border_radius=radius)
        pygame.draw.rect(screen, PLATFORM_OUTLINE, rect, 2, border_radius=radius)


class Player:
    def __init__(self, spawn_pos):
        self.spawn_pos = np.array(spawn_pos, dtype=np.float32)
        self.rect = pygame.Rect(
            int(round(spawn_pos[0])),
            int(round(spawn_pos[1])),
            PLAYER_SIZE[0],
            PLAYER_SIZE[1],
        )
        self.pos = self.spawn_pos.copy()
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False
        self.parachute_open = False
        self.heat_value = 0.0
        self.heat_lift = 0.0
        self.coyote_timer = 0.0

    def reset(self):
        self.pos[:] = self.spawn_pos
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = True
        self.parachute_open = False
        self.heat_value = 0.0
        self.heat_lift = 0.0
        self.coyote_timer = PLAYER_COYOTE_TIME
        self._sync_rect()

    def _sync_rect(self):
        self.rect.topleft = (int(round(self.pos[0])), int(round(self.pos[1])))

    def jump(self):
        if not self.parachute_open and (self.on_ground or self.coyote_timer > 0.0):
            self.vy = PLAYER_JUMP_SPEED
            self.on_ground = False
            self.coyote_timer = 0.0

    def toggle_parachute(self):
        self.parachute_open = not self.parachute_open

    def get_canopy_rect(self):
        left = int(round(self.pos[0] + self.rect.width / 2.0 - PARACHUTE_CANOPY_WIDTH / 2.0))
        top = int(round(self.pos[1] - PARACHUTE_CANOPY_HEIGHT - 10))
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

    def _resolve_collisions(self, platforms, axis):
        for platform in platforms:
            if not self.rect.colliderect(platform):
                continue

            if axis == "x":
                if self.vx > 0.0:
                    self.rect.right = platform.left
                elif self.vx < 0.0:
                    self.rect.left = platform.right
                self.pos[0] = float(self.rect.x)
                continue

            if self.vy > 0.0:
                self.rect.bottom = platform.top
                self.on_ground = True
            elif self.vy < 0.0:
                self.rect.top = platform.bottom
            self.pos[1] = float(self.rect.y)
            self.vy = 0.0

    def update(self, dt_sec, keys, platforms, heat_field, screen_width, screen_height):
        move_dir = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            move_dir -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            move_dir += 1

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

        self.pos[0] += self.vx * dt_sec
        self._sync_rect()
        self._resolve_collisions(platforms, "x")
        self.pos[0] = clamp(self.pos[0], 0.0, screen_width - self.rect.width)
        self._sync_rect()

        self.pos[1] += self.vy * dt_sec
        self._sync_rect()
        self.on_ground = False
        self._resolve_collisions(platforms, "y")

        if self.on_ground:
            self.coyote_timer = PLAYER_COYOTE_TIME
        else:
            self.coyote_timer = max(0.0, self.coyote_timer - dt_sec)

        if self.rect.top < 0:
            self.rect.top = 0
            self.pos[1] = float(self.rect.y)
            self.vy = max(0.0, self.vy)

        if self.rect.top > screen_height + PLAYER_RESPAWN_Y_MARGIN:
            self.reset()


def draw_player(screen, player):
    if player.parachute_open:
        canopy_rect = player.get_canopy_rect()
        if player.heat_lift > 0.0:
            glow_strength = clamp(player.heat_lift * 2.8, 0.0, 1.0)
            glow_rect = canopy_rect.union(player.rect).inflate(36, 44)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            alpha = int(90 * glow_strength)
            pygame.draw.ellipse(
                glow_surface,
                (*LIFT_GLOW_COLOR, alpha),
                glow_surface.get_rect(),
            )
            screen.blit(glow_surface, glow_rect.topleft)

        canopy_points = [
            (canopy_rect.left, canopy_rect.bottom - 2),
            (canopy_rect.left + 12, canopy_rect.top + 6),
            (canopy_rect.centerx, canopy_rect.top),
            (canopy_rect.right - 12, canopy_rect.top + 6),
            (canopy_rect.right, canopy_rect.bottom - 2),
        ]
        pygame.draw.polygon(screen, PARACHUTE_COLOR, canopy_points)
        pygame.draw.polygon(screen, PARACHUTE_HIGHLIGHT, canopy_points, 2)

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
            pygame.draw.line(screen, PARACHUTE_HIGHLIGHT, source, target, 2)

    pygame.draw.rect(screen, PLAYER_BODY_COLOR, player.rect, border_radius=12)
    pygame.draw.rect(screen, PLAYER_OUTLINE_COLOR, player.rect, 2, border_radius=12)

    head_center = (player.rect.centerx, player.rect.top + 12)
    pygame.draw.circle(screen, PLAYER_HEAD_COLOR, head_center, 11)
    pygame.draw.circle(screen, PLAYER_OUTLINE_COLOR, head_center, 11, 2)

    torso_top = player.rect.top + 18
    torso_bottom = player.rect.bottom - 8
    pygame.draw.line(
        screen,
        PLAYER_OUTLINE_COLOR,
        (player.rect.centerx, torso_top),
        (player.rect.centerx, torso_bottom),
        3,
    )
    pygame.draw.line(
        screen,
        PLAYER_OUTLINE_COLOR,
        (player.rect.centerx, player.rect.top + 26),
        (player.rect.left + 7, player.rect.top + 34),
        3,
    )
    pygame.draw.line(
        screen,
        PLAYER_OUTLINE_COLOR,
        (player.rect.centerx, player.rect.top + 26),
        (player.rect.right - 7, player.rect.top + 34),
        3,
    )
    pygame.draw.line(
        screen,
        PLAYER_OUTLINE_COLOR,
        (player.rect.centerx, torso_bottom),
        (player.rect.left + 8, player.rect.bottom - 2),
        3,
    )
    pygame.draw.line(
        screen,
        PLAYER_OUTLINE_COLOR,
        (player.rect.centerx, torso_bottom),
        (player.rect.right - 8, player.rect.bottom - 2),
        3,
    )


def draw_world(screen, frame_surface, grid_image, platforms, player):
    pygame.surfarray.blit_array(frame_surface, grid_image.swapaxes(0, 1))
    pygame.transform.scale(frame_surface, screen.get_size(), screen)
    draw_platforms(screen, platforms)
    draw_player(screen, player)


def main(image_path=None, cached_calibration=False):
    pygame.init()
    screen, width, height, display_index, monitor, monitor_count = create_fullscreen_display(
        TARGET_DISPLAY_INDEX
    )
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = width, height
    pygame.display.set_caption("Level 3 - Heat Glider")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 52, bold=True)
    ready_font = pygame.font.SysFont("Arial", 64, bold=True)
    countdown_font = pygame.font.SysFont("Arial", 120, bold=True)

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
            cap=cap, screen=screen, WIDTH=WIDTH, HEIGHT=HEIGHT
        )
        H_inv = np.linalg.inv(H)
        save_calibration(CALIBRATION_CACHE, H, mask, a_map, b_map)
        cap.release()

    cap_stream = VideoStream(0)

    bg_img = np.full((GRID_H, GRID_W, 3), 128, dtype=np.uint8)

    grid_x, grid_y = np.meshgrid(
        np.arange(GRID_W, dtype=np.float32),
        np.arange(GRID_H, dtype=np.float32),
    )
    heat_field = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    frame_surface = pygame.Surface((GRID_W, GRID_H))
    last_stage_img = bg_img.copy()
    platforms, spawn_pos = build_platforms(width, height)
    player = Player(spawn_pos)
    player.on_ground = True
    player.coyote_timer = PLAYER_COYOTE_TIME

    print("Background: level_1-style neutral (mid-gray)")
    print("Controls: A/D or arrows=move, Space/Up/W=jump, P=toggle parachute")
    print("Extra controls: V=toggle shadow overlay, C=clear heat, ESC=quit")

    running = True
    show_overlay = False
    frame_count = 0
    t_prev = time.time()
    debug_object_count = 0
    prev_player_cam_mask = None

    def draw_heat_background(dt_sec):
        nonlocal heat_field, debug_object_count, prev_player_cam_mask, last_stage_img

        ret, cam_frame = cap_stream.read()
        if not ret:
            return True

        cam_gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
        cam_gray_masked = cv2.bitwise_and(cam_gray, cam_gray, mask=mask)

        h_cam, w_cam = cam_gray.shape[:2]
        warped_proj_gray_masked = screen_to_camera_space(screen, H, mask, h_cam, w_cam)
        expected = gain_offset_calibration(warped_proj_gray_masked, a_map, b_map)

        player_mask_surface = pygame.Surface(screen.get_size())
        player_mask_surface.fill((0, 0, 0))
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

        shadow_screen = objects_to_mask(objects, HEIGHT, WIDTH)
        shadow_small = cv2.resize(
            shadow_screen, (GRID_W, GRID_H), interpolation=cv2.INTER_NEAREST
        )
        shadow_mask = shadow_small > 0

        heat_field = update_heat_field(heat_field, shadow_mask, dt_sec)
        map_x, map_y = compute_heat_distortion(grid_x, grid_y, heat_field, time.perf_counter())

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
        "HEAT GLIDER",
        ready_font,
        countdown_font,
        draw_frame=draw_heat_background,
    ):
        cap_stream.release()
        pygame.quit()
        cv2.destroyAllWindows()
        return

    draw_world(screen, frame_surface, last_stage_img, platforms, player)
    pygame.display.flip()

    while running:
        dt_sec = min(clock.tick(FPS) / 1000.0, 0.1)
        jump_requested = False

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
                elif event.key == PARACHUTE_TOGGLE_KEY:
                    player.toggle_parachute()
                elif event.key in (pygame.K_SPACE, pygame.K_UP, pygame.K_w):
                    jump_requested = True

        if not draw_heat_background(dt_sec):
            running = False
            continue

        if jump_requested:
            player.jump()
        player.update(
            dt_sec,
            pygame.key.get_pressed(),
            platforms,
            heat_field,
            width,
            height,
        )
        draw_world(screen, frame_surface, last_stage_img, platforms, player)
        pygame.display.flip()

        if frame_count % 60 == 0:
            fps = 1.0 / max(1e-6, time.time() - t_prev)
            heat_peak = float(np.max(heat_field))
            print(
                f"FPS: {fps:.1f} | Shadows: {debug_object_count} | Heat peak: {heat_peak:.2f} | Player heat: {player.heat_value:.2f} | Parachute: {'open' if player.parachute_open else 'closed'}"
            )
            t_prev = time.time()
        frame_count += 1

    cap_stream.release()
    pygame.quit()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Level 3 - Heat Glider")
    parser.add_argument(
        "--image",
        default=os.path.join("images", "galaxy-1489300667bhk.jpg"),
        help="Ignored in Level 3; stage background is level_1-style neutral.",
    )
    parser.add_argument(
        "--cached-calibration",
        action="store_true",
        help="Use cached calibration from calibration_cache.npz.",
    )
    args = parser.parse_args()
    main(args.image, cached_calibration=args.cached_calibration)
