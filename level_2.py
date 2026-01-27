import argparse
import math
import os
import random
import time

import cv2
import numpy as np
import pygame

from modules.calibration import calibrate_system
from modules.camera_magic import gain_offset_calibration
from modules.settings import HEIGHT, WIDTH
from modules.videostream import VideoStream, get_camera_capture


GRID_H = 180
GRID_W = 240
FPS = 60

TARGET_DISPLAY_INDEX = 1
USE_BORDERLESS_FULLSCREEN = True

CALIBRATION_CACHE = "calibration_cache.npz"

COUNTDOWN_SECONDS = 5

LENS_RADIUS_PX = 220.0
LENS_STRENGTH = 6.0
WARP_MAX = 18.0
WARP_FLOW_GAIN = 2.0

DIFF_THRESH = 40
EXPECTED_MIN = 45
BASELINE_MIN = 35
SHADOW_MIN_AREA = 200

OBJECT_SPAWN_MIN = 0.6
OBJECT_SPAWN_MAX = 1.8
OBJECT_SPEED_MIN = 120.0
OBJECT_SPEED_MAX = 240.0
OBJECT_RADIUS_MIN = 6
OBJECT_RADIUS_MAX = 12
OBJECT_OFFSCREEN_MARGIN = 80.0

STAR_ENERGY_START = 0.5
STAR_ENERGY_DECAY = 0.015
STAR_ENERGY_GAIN = 0.18
STAR_PULSE_GAIN = 0.6
STAR_PULSE_DECAY = 1.8

STAR_BASE_RADIUS = 18.0
STAR_RADIUS_GAIN = 14.0
STAR_GLOW_BASE = 65.0
STAR_GLOW_GAIN = 90.0

STARVE_SECONDS = 50.0

FAIL_HOLD_SECONDS = 2.0
SUCCESS_HOLD_SECONDS = 10.0

SHOW_DEBUG_WINDOWS = True


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


def load_background_image(path, size):
    if not os.path.exists(path):
        return None
    image = pygame.image.load(path).convert()
    image = pygame.transform.smoothscale(image, size)
    arr = pygame.surfarray.array3d(image)
    return np.swapaxes(arr, 0, 1).copy()


def load_background_surface(path, size):
    if not os.path.exists(path):
        return None
    image = pygame.image.load(path).convert()
    return pygame.transform.smoothscale(image, size)


def capture_background_baseline(cap_stream, screen, bg_surface, mask, warmup=8):
    screen.blit(bg_surface, (0, 0))
    pygame.display.flip()

    frame = None
    for _ in range(warmup):
        ret, cam_frame = cap_stream.read()
        if ret:
            frame = cam_frame
        pygame.time.delay(30)

    if frame is None:
        return None

    cam_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cam_gray_masked = cv2.bitwise_and(cam_gray, cam_gray, mask=mask)
    return cam_gray_masked


def build_camera_grid_maps(H_inv, screen_w, screen_h, grid_w, grid_h, cam_w, cam_h):
    xs, ys = np.meshgrid(
        np.arange(cam_w, dtype=np.float32),
        np.arange(cam_h, dtype=np.float32),
    )
    ones = np.ones_like(xs)
    pts = np.stack([xs, ys, ones], axis=-1).reshape(-1, 3).T
    screen_pts = H_inv @ pts
    screen_pts /= np.maximum(screen_pts[2], 1e-6)
    sx = screen_pts[0].reshape(cam_h, cam_w)
    sy = screen_pts[1].reshape(cam_h, cam_w)

    cell_w = screen_w / grid_w
    cell_h = screen_h / grid_h
    gx = sx / cell_w
    gy = sy / cell_h
    return gx.astype(np.float32), gy.astype(np.float32)


def project_grid_to_camera_gray(grid_rgb, cam_map_x, cam_map_y, mask):
    grid_gray = cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2GRAY)
    cam_gray = cv2.remap(
        grid_gray,
        cam_map_x,
        cam_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return cv2.bitwise_and(cam_gray, cam_gray, mask=mask)


def objects_to_mask(objects, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for obj in objects:
        pts = np.array(obj["polygon_pygame"], dtype=np.int32)
        if pts.shape[0] < 3:
            continue
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
    return mask


def shadow_object_detection_diff(
    cam_gray_masked,
    expected,
    mask_diff,
    H_inv,
    diff_thresh=DIFF_THRESH,
    expected_min=EXPECTED_MIN,
    valid_mask=None,
    min_area=SHADOW_MIN_AREA,
):
    diff = expected.astype(np.int16) - cam_gray_masked.astype(np.int16)
    shadow_mask = diff > diff_thresh
    if expected_min is not None:
        shadow_mask &= expected > expected_min
    if valid_mask is not None:
        shadow_mask &= valid_mask
    if mask_diff is not None:
        shadow_mask[mask_diff] = False

    shadow_mask_u8 = (shadow_mask.astype(np.uint8) * 255)
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
            "polygon_pygame": pygame_polygon
        })

    return objects, contours_vis, shadow_mask_u8


def compute_shadow_warp_map(grid_x, grid_y, shadow_mask, radius, strength, max_warp):
    radius = max(radius, 1e-3)
    if not shadow_mask.any():
        return grid_x.astype(np.float32), grid_y.astype(np.float32)

    inv_mask = (~shadow_mask).astype(np.uint8)
    dist_out = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
    grad_x = cv2.Sobel(dist_out, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(dist_out, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y) + 1e-6
    grad_x /= grad_mag
    grad_y /= grad_mag

    falloff = np.clip(1.0 - (dist_out / radius), 0.0, 1.0)
    disp_x = -grad_x * strength * falloff
    disp_y = -grad_y * strength * falloff

    disp_x = cv2.GaussianBlur(disp_x, (0, 0), sigmaX=1.2, sigmaY=1.2)
    disp_y = cv2.GaussianBlur(disp_y, (0, 0), sigmaX=1.2, sigmaY=1.2)

    disp_x = np.clip(disp_x, -max_warp, max_warp)
    disp_y = np.clip(disp_y, -max_warp, max_warp)
    map_x = (grid_x + disp_x).astype(np.float32)
    map_y = (grid_y + disp_y).astype(np.float32)
    return map_x, map_y


def sample_bilinear(field, x, y):
    h, w = field.shape
    x = max(0.0, min(w - 1.0, x))
    y = max(0.0, min(h - 1.0, y))
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    tx = x - x0
    ty = y - y0
    a = (1.0 - tx) * field[y0, x0] + tx * field[y0, x1]
    b = (1.0 - tx) * field[y1, x0] + tx * field[y1, x1]
    return (1.0 - ty) * a + ty * b


def spawn_object(width, height, rng):
    edge = rng.choice(["top", "bottom", "left", "right"])
    margin = OBJECT_OFFSCREEN_MARGIN
    if edge == "top":
        pos = np.array([rng.uniform(0, width), -margin], dtype=np.float32)
        angle = rng.uniform(math.radians(20), math.radians(160))
    elif edge == "bottom":
        pos = np.array([rng.uniform(0, width), height + margin], dtype=np.float32)
        angle = rng.uniform(math.radians(-160), math.radians(-20))
    elif edge == "left":
        pos = np.array([-margin, rng.uniform(0, height)], dtype=np.float32)
        angle = rng.uniform(math.radians(-70), math.radians(70))
    else:
        pos = np.array([width + margin, rng.uniform(0, height)], dtype=np.float32)
        angle = rng.uniform(math.radians(110), math.radians(250))

    speed = rng.uniform(OBJECT_SPEED_MIN, OBJECT_SPEED_MAX)
    vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
    radius = rng.uniform(OBJECT_RADIUS_MIN, OBJECT_RADIUS_MAX)
    shade = rng.randint(180, 240)
    color = (shade, shade, 255)
    return {
        "pos": pos,
        "vel": vel,
        "radius": radius,
        "color": color,
    }


def draw_star(screen, pos, energy, pulse):
    glow_radius = STAR_GLOW_BASE + STAR_GLOW_GAIN * energy + pulse * 20.0
    glow_radius = max(10.0, glow_radius)
    glow_size = int(glow_radius * 2)
    glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
    center = (glow_size // 2, glow_size // 2)

    glow_alpha = int(40 + 140 * energy)
    for i in range(3):
        radius = int(glow_radius * (1.0 - 0.25 * i))
        alpha = max(10, int(glow_alpha / (i + 1)))
        color = (255, 220, 150, alpha)
        pygame.draw.circle(glow_surface, color, center, radius)

    screen.blit(
        glow_surface,
        (int(pos[0] - glow_radius), int(pos[1] - glow_radius)),
    )

    core_radius = STAR_BASE_RADIUS + STAR_RADIUS_GAIN * energy
    core_color = (
        min(255, int(190 + 60 * energy)),
        min(255, int(150 + 80 * energy)),
        min(255, int(80 + 90 * energy)),
    )
    pygame.draw.circle(screen, core_color, (int(pos[0]), int(pos[1])), int(core_radius))


def draw_black_hole(screen, pos, radius):
    pygame.draw.circle(screen, (0, 0, 0), (int(pos[0]), int(pos[1])), int(radius))
    ring_color = (30, 30, 40)
    pygame.draw.circle(screen, ring_color, (int(pos[0]), int(pos[1])), int(radius), 2)


def show_message(screen, clock, text, duration, font, color):
    width, height = screen.get_size()
    text_surf = font.render(text, True, color)
    text_rect = text_surf.get_rect(center=(width // 2, height // 2))
    end_time = time.time() + duration

    while time.time() < end_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        screen.fill((0, 0, 0))
        screen.blit(text_surf, text_rect)
        pygame.display.flip()
        clock.tick(FPS)

    return True


def run_countdown(screen, clock, seconds, label_font, count_font):
    width, height = screen.get_size()
    label_surf = label_font.render("GET READY", True, (255, 255, 255))
    label_rect = label_surf.get_rect(center=(width // 2, int(height * 0.42)))
    end_time = time.time() + seconds

    while True:
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

        screen.fill((0, 0, 0))
        screen.blit(label_surf, label_rect)
        screen.blit(count_surf, count_rect)
        pygame.display.flip()
        clock.tick(FPS)

    return True


def main(image_path, cached_calibration=False):
    pygame.init()
    screen, width, height, display_index, monitor, monitor_count = create_fullscreen_display(
        TARGET_DISPLAY_INDEX
    )
    pygame.display.set_caption("Level 2 - Gravity Well")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 52, bold=True)
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

    cam_h, cam_w = mask.shape[:2]
    cam_map_x, cam_map_y = build_camera_grid_maps(
        H_inv, width, height, GRID_W, GRID_H, cam_w, cam_h
    )

    cap_stream = VideoStream(0)

    bg_img = load_background_image(image_path, (GRID_W, GRID_H))
    bg_surface = load_background_surface(image_path, (width, height))
    if bg_img is None or bg_surface is None:
        print(f"Image not found: {image_path}")
        cap_stream.release()
        pygame.quit()
        return

    print("Capturing background reference...")
    baseline_cam = capture_background_baseline(
        cap_stream, screen, bg_surface, mask
    )
    if baseline_cam is None:
        print("Failed to capture background reference.")
        cap_stream.release()
        pygame.quit()
        return

    warp_img = np.zeros_like(bg_img)
    warp_img[:] = bg_img
    frame = pygame.Surface((GRID_W, GRID_H))
    grid_x, grid_y = np.meshgrid(
        np.arange(GRID_W, dtype=np.float32),
        np.arange(GRID_H, dtype=np.float32)
    )

    cell_w = width / GRID_W
    cell_h = height / GRID_H
    lens_radius = LENS_RADIUS_PX / max(1e-6, min(cell_w, cell_h))
    baseline_mask = baseline_cam > BASELINE_MIN

    rng = random.Random()
    star_pos = np.array([width * 0.5, height * 0.5], dtype=np.float32)

    running = True
    show_overlay = False
    print("Controls: V=toggle shadow overlay, ESC=quit")

    def reset_state():
        return {
            "objects": [],
            "next_spawn_time": time.time() + rng.uniform(OBJECT_SPAWN_MIN, OBJECT_SPAWN_MAX),
            "star_energy": STAR_ENERGY_START,
            "star_pulse": 0.0,
            "last_hit_time": time.time(),
            "frame_count": 0,
            "t_prev": time.time(),
        }

    state = reset_state()
    success_mode = False
    success_start = None
    success_text = font.render("SUCCESS", True, (255, 255, 255))

    if not run_countdown(screen, clock, COUNTDOWN_SECONDS, font, countdown_font):
        cap_stream.release()
        pygame.quit()
        cv2.destroyAllWindows()
        return

    while running:
        dt_sec = min(clock.tick(FPS) / 1000.0, 0.1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                show_overlay = not show_overlay

        ret, cam_frame = cap_stream.read()
        if not ret:
            continue

        cam_gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
        cam_gray_masked = cv2.bitwise_and(cam_gray, cam_gray, mask=mask)

        proj_gray_masked = project_grid_to_camera_gray(warp_img, cam_map_x, cam_map_y, mask)
        expected = gain_offset_calibration(proj_gray_masked, a_map, b_map)

        mask_diff = np.zeros_like(cam_gray_masked, dtype=bool)
        objects_cam, contours_vis, shadow_mask_u8 = shadow_object_detection_diff(
            cam_gray_masked,
            expected,
            mask_diff,
            H_inv,
            expected_min=None,
            valid_mask=baseline_mask,
        )

        shadow_screen = objects_to_mask(objects_cam, height, width)
        shadow_small = cv2.resize(
            shadow_screen, (GRID_W, GRID_H), interpolation=cv2.INTER_NEAREST
        )
        shadow_mask = shadow_small > 0

        warp_mask = shadow_mask
        if success_mode:
            star_radius_px = STAR_BASE_RADIUS + STAR_RADIUS_GAIN * 1.0
            star_radius_grid = star_radius_px / max(1e-6, min(cell_w, cell_h))
            star_gx = star_pos[0] / cell_w
            star_gy = star_pos[1] / cell_h
            dist2 = (grid_x - star_gx) ** 2 + (grid_y - star_gy) ** 2
            black_hole_mask = dist2 <= (star_radius_grid ** 2)
            warp_mask = shadow_mask | black_hole_mask

        map_x, map_y = compute_shadow_warp_map(
            grid_x,
            grid_y,
            warp_mask,
            lens_radius,
            LENS_STRENGTH,
            WARP_MAX,
        )
        disp_x = map_x - grid_x
        disp_y = map_y - grid_y

        cv2.remap(
            bg_img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
            dst=warp_img,
        )

        now = time.time()
        hit_count = 0
        if not success_mode:
            if now >= state["next_spawn_time"]:
                state["objects"].append(spawn_object(width, height, rng))
                state["next_spawn_time"] = now + rng.uniform(OBJECT_SPAWN_MIN, OBJECT_SPAWN_MAX)

            if state["star_energy"] > 0.0:
                state["star_energy"] = max(0.0, state["star_energy"] - STAR_ENERGY_DECAY * dt_sec)
                state["star_pulse"] = max(0.0, state["star_pulse"] - STAR_PULSE_DECAY * dt_sec)

            alive = []
            for obj in state["objects"]:
                gx = obj["pos"][0] / cell_w
                gy = obj["pos"][1] / cell_h
                flow_x = sample_bilinear(disp_x, gx, gy) * cell_w
                flow_y = sample_bilinear(disp_y, gx, gy) * cell_h
                flow = np.array([flow_x, flow_y], dtype=np.float32)
                obj["vel"] += flow * WARP_FLOW_GAIN * dt_sec
                obj["pos"] += obj["vel"] * dt_sec

                dx = obj["pos"][0] - star_pos[0]
                dy = obj["pos"][1] - star_pos[1]
                star_radius = STAR_BASE_RADIUS + STAR_RADIUS_GAIN * state["star_energy"]
                if dx * dx + dy * dy <= (star_radius + obj["radius"]) ** 2:
                    hit_count += 1
                    continue

                if (
                    -OBJECT_OFFSCREEN_MARGIN <= obj["pos"][0] <= width + OBJECT_OFFSCREEN_MARGIN
                    and -OBJECT_OFFSCREEN_MARGIN <= obj["pos"][1] <= height + OBJECT_OFFSCREEN_MARGIN
                ):
                    alive.append(obj)

            state["objects"] = alive

            if hit_count:
                state["last_hit_time"] = now
                state["star_energy"] = min(1.0, state["star_energy"] + hit_count * STAR_ENERGY_GAIN)
                state["star_pulse"] = min(1.0, state["star_pulse"] + hit_count * STAR_PULSE_GAIN)

            if state["star_energy"] >= 1.0:
                success_mode = True
                success_start = now
                state["star_energy"] = 1.0
                state["star_pulse"] = 0.0
                state["objects"] = []
                print("SUCCESS")

            if state["star_energy"] <= 0.0 or (now - state["last_hit_time"]) > STARVE_SECONDS:
                running = show_message(
                    screen, clock, "STAR DIED", FAIL_HOLD_SECONDS, font, (220, 60, 60)
                )
                if not running:
                    break
                warp_img[:] = bg_img
                state = reset_state()
                success_mode = False
                success_start = None
                if not run_countdown(screen, clock, COUNTDOWN_SECONDS, font, countdown_font):
                    running = False
                continue
        else:
            if success_start is not None and (now - success_start) >= SUCCESS_HOLD_SECONDS:
                break

        if show_overlay:
            warp_img[shadow_mask] = (20, 20, 20)

        pygame.surfarray.blit_array(frame, warp_img.swapaxes(0, 1))
        pygame.transform.scale(frame, screen.get_size(), screen)

        for obj in state["objects"]:
            pos = obj["pos"]
            pygame.draw.circle(
                screen,
                obj["color"],
                (int(pos[0]), int(pos[1])),
                int(obj["radius"]),
            )

        if success_mode:
            star_radius = STAR_BASE_RADIUS + STAR_RADIUS_GAIN * 1.0
            draw_black_hole(screen, star_pos, star_radius)
            text_rect = success_text.get_rect(center=(width // 2, height // 5))
            screen.blit(success_text, text_rect)
        else:
            draw_star(screen, star_pos, state["star_energy"], state["star_pulse"])

        pygame.display.flip()

        if state["frame_count"] % 60 == 0:
            fps = 1.0 / max(1e-6, time.time() - state["t_prev"])
            print(
                f"FPS: {fps:.1f} | Objects: {len(state['objects'])} | Shadows: {len(objects_cam)}"
            )
            state["t_prev"] = time.time()
        state["frame_count"] += 1

        if SHOW_DEBUG_WINDOWS:
            cv2.imshow("Camera View", cam_frame)
            cv2.imshow("Expected Camera Image", expected)
            cv2.imshow("Contours", contours_vis)
            cv2.imshow("Shadow Diff Mask", shadow_mask_u8)
            cv2.waitKey(1)

    cap_stream.release()
    pygame.quit()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Level 2 - Gravity Well")
    parser.add_argument(
        "--image",
        default=os.path.join("images", "galaxy-1489300667bhk.jpg"),
        help="Path to the background image.",
    )
    parser.add_argument(
        "--cached-calibration",
        action="store_true",
        help="Use cached calibration from calibration_cache.npz.",
    )
    args = parser.parse_args()
    main(args.image, cached_calibration=args.cached_calibration)
