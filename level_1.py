import argparse
import math
import os
import time

import cv2
import numpy as np
import pygame

from modules.calibration import calibrate_system
from modules.camera_magic import (
    gain_offset_calibration,
    screen_to_camera_space,
)
from modules.settings import HEIGHT, WIDTH
from modules.videostream import VideoStream, get_camera_capture


GRID_H = 180
GRID_W = 240
FPS = 60

TARGET_DISPLAY_INDEX = 1
USE_BORDERLESS_FULLSCREEN = True

CALIBRATION_CACHE = "calibration_cache.npz"

COUNTDOWN_SECONDS = 5
FAIL_HOLD_SECONDS = 1.5

DT = 1.0
WAVE_SPEED = 0.45
DAMPING = 0.005
C2 = (WAVE_SPEED * DT) ** 2

SOURCE_X = 8
SOURCE_FREQ = 5.0
SOURCE_STRENGTH = 1.2
WAVE_BRUSH_RADIUS = 6
SHADOW_RATIO = 0.6
SHADOW_MIN_DELTA = 15.0
SHADOW_MIN_AREA = 200

BOAT_RESPONSE = 5.0
BOAT_DRAG = 0.98
BOAT_TURN_RATE = 0.7
BOAT_MAX_SPEED_FACTOR = 2.0

WAVE_STRENGTH_GAIN = 20.0
WAVE_VELOCITY_GAIN = 1.6
WAVE_SPEED_CLAMP = 1.5
WAVE_MIN_GRAD = 1e-4
WAVE_MIN_DU = 1e-4


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


def make_kernel(radius):
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    d2 = xx * xx + yy * yy
    kernel = np.exp(-d2 / (radius * radius * 0.5))
    kernel /= kernel.max()
    return kernel


def add_kernel(field, cy, cx, kernel, scale):
    r = kernel.shape[0] // 2
    y0 = max(0, cy - r)
    x0 = max(0, cx - r)
    y1 = min(field.shape[0], cy + r + 1)
    x1 = min(field.shape[1], cx + r + 1)

    ky0 = y0 - (cy - r)
    kx0 = x0 - (cx - r)
    ky1 = ky0 + (y1 - y0)
    kx1 = kx0 + (x1 - x0)

    field[y0:y1, x0:x1] += kernel[ky0:ky1, kx0:kx1] * scale


def get_drive(t):
    a = 0.55 + 0.35 * math.sin(t * 1.7) + 0.25 * math.sin(t * 4.9 + 1.2)
    return max(0.0, min(1.0, a))


def step(u, u_prev, u_next, solid, c2, damping):
    lap = (
        u[1:-1, 1:-1] * -4.0
        + u[1:-1, 2:]
        + u[1:-1, :-2]
        + u[2:, 1:-1]
        + u[:-2, 1:-1]
    )
    u_next[1:-1, 1:-1] = (
        (2.0 - damping) * u[1:-1, 1:-1]
        - (1.0 - damping) * u_prev[1:-1, 1:-1]
        + c2 * lap
    )
    u_next[0, :] = 0.0
    u_next[-1, :] = 0.0
    u_next[:, 0] = 0.0
    u_next[:, -1] = 0.0
    u_next[solid] = 0.0


def render(u, img, color_scale, overlay=None):
    c = np.clip(u * color_scale, -127, 127)
    img[..., 0] = (128 + c).astype(np.uint8)
    img[..., 1] = 128
    img[..., 2] = (128 - c).astype(np.uint8)
    if overlay is not None:
        img[overlay] = (30, 30, 30)


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
            "polygon_pygame": pygame_polygon
        })

    return objects, contours_vis



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


def sample_gradient(field, x, y):
    dx = 0.5 * (sample_bilinear(field, x + 1.0, y) - sample_bilinear(field, x - 1.0, y))
    dy = 0.5 * (sample_bilinear(field, x, y + 1.0) - sample_bilinear(field, x, y - 1.0))
    return dx, dy


def angle_step(current, target, max_step):
    diff = (target - current + math.pi) % (2.0 * math.pi) - math.pi
    if abs(diff) <= max_step:
        return target
    return current + max_step * (1.0 if diff > 0.0 else -1.0)


def make_boat_sprite(size):
    w, h = size
    surf = pygame.Surface((w, h), pygame.SRCALPHA)

    hull_rect = pygame.Rect(2, 3, w - 4, h - 6)
    pygame.draw.ellipse(surf, (120, 70, 30), hull_rect)
    pygame.draw.ellipse(surf, (80, 45, 18), hull_rect, 2)

    bow = [(w - 4, h // 2), (w - 14, 4), (w - 14, h - 4)]
    pygame.draw.polygon(surf, (120, 70, 30), bow)
    pygame.draw.polygon(surf, (80, 45, 18), bow, 2)

    cabin = pygame.Rect(int(w * 0.35), int(h * 0.38), int(w * 0.22), int(h * 0.24))
    pygame.draw.rect(surf, (200, 190, 170), cabin)
    pygame.draw.rect(surf, (120, 110, 90), cabin, 2)
    return surf


def build_dock_data(width, height, grid_w, grid_h):
    thickness = max(16, int(min(width, height) * 0.035))
    dock_width = int(width * 0.26)
    dock_height = int(height * 0.18)
    dock_left = max(0, width // 2 - dock_width // 2)
    dock_right = min(width, dock_left + dock_width)
    dock_bottom = height
    dock_top = max(0, dock_bottom - dock_height)

    dock_spine = pygame.Rect(dock_right - thickness, dock_top, thickness, dock_bottom - dock_top)
    dock_top_bar = pygame.Rect(dock_left, dock_top, dock_right - dock_left, thickness)
    dock_bottom_bar = pygame.Rect(
        dock_left, dock_bottom - thickness, dock_right - dock_left, thickness
    )

    dock_mask = np.zeros((height, width), dtype=np.uint8)
    dock_mask[dock_spine.top:dock_spine.bottom, dock_spine.left:dock_spine.right] = 255
    dock_mask[dock_top_bar.top:dock_top_bar.bottom, dock_top_bar.left:dock_top_bar.right] = 255
    dock_mask[
        dock_bottom_bar.top:dock_bottom_bar.bottom, dock_bottom_bar.left:dock_bottom_bar.right
    ] = 255

    margin = max(6, thickness // 2)
    goal_left = dock_left + margin
    goal_top = dock_top + thickness + margin
    goal_right = dock_right - thickness - margin
    goal_bottom = dock_bottom - thickness - margin
    if goal_right <= goal_left:
        goal_left = dock_left + thickness + 4
        goal_right = dock_right - 4
    if goal_bottom <= goal_top:
        goal_top = dock_top + thickness + 4
        goal_bottom = dock_bottom - thickness - 4

    goal_rect = pygame.Rect(
        goal_left,
        goal_top,
        max(10, goal_right - goal_left),
        max(10, goal_bottom - goal_top),
    )

    dock_surface = pygame.Surface((dock_right - dock_left, dock_bottom - dock_top), pygame.SRCALPHA)
    dock_surface.fill((0, 0, 0, 0))
    dock_surface.fill((50, 50, 50), dock_spine.move(-dock_left, -dock_top))
    dock_surface.fill((50, 50, 50), dock_top_bar.move(-dock_left, -dock_top))
    dock_surface.fill((50, 50, 50), dock_bottom_bar.move(-dock_left, -dock_top))
    pygame.draw.rect(dock_surface, (90, 90, 90), dock_spine.move(-dock_left, -dock_top), 2)
    pygame.draw.rect(dock_surface, (90, 90, 90), dock_top_bar.move(-dock_left, -dock_top), 2)
    pygame.draw.rect(dock_surface, (90, 90, 90), dock_bottom_bar.move(-dock_left, -dock_top), 2)

    dock_mask_grid = cv2.resize(dock_mask, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST) > 0
    dock_pos = (dock_left, dock_top)
    dock_rects = [dock_spine, dock_top_bar, dock_bottom_bar]
    return dock_mask_grid, goal_rect, dock_surface, dock_pos, dock_rects


def resolve_circle_rect(pos, radius, rect):
    closest_x = max(rect.left, min(pos[0], rect.right))
    closest_y = max(rect.top, min(pos[1], rect.bottom))
    dx = pos[0] - closest_x
    dy = pos[1] - closest_y
    dist2 = dx * dx + dy * dy
    if dist2 > 0.0 and dist2 < radius * radius:
        dist = math.sqrt(dist2)
        nx = dx / dist
        ny = dy / dist
        push = radius - dist
        pos[0] += nx * push
        pos[1] += ny * push
        return (nx, ny)
    if dist2 == 0.0 and rect.collidepoint(float(pos[0]), float(pos[1])):
        left_pen = pos[0] - rect.left
        right_pen = rect.right - pos[0]
        top_pen = pos[1] - rect.top
        bottom_pen = rect.bottom - pos[1]
        pen = min(left_pen, right_pen, top_pen, bottom_pen)
        if pen == left_pen:
            pos[0] = rect.left - radius
            return (-1.0, 0.0)
        if pen == right_pen:
            pos[0] = rect.right + radius
            return (1.0, 0.0)
        if pen == top_pen:
            pos[1] = rect.top - radius
            return (0.0, -1.0)
        pos[1] = rect.bottom + radius
        return (0.0, 1.0)
    return None


def main(cached_calibration=False, return_on_success=False, camera_index=1):
    pygame.init()
    screen, width, height, display_index, monitor, monitor_count = create_fullscreen_display(
        TARGET_DISPLAY_INDEX
    )
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = width, height
    pygame.display.set_caption("Level 1 - Dock")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 52, bold=True)
    ready_font = pygame.font.SysFont("Arial", 64, bold=True)
    countdown_font = pygame.font.SysFont("Arial", 120, bold=True)
    fail_font = pygame.font.SysFont("Arial", 72, bold=True)
    success_surf = font.render("SUCCESS", True, (255, 255, 255))

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
            return False
        H, mask, a_map, b_map = calibration
        H_inv = np.linalg.inv(H)
    else:
        cap = get_camera_capture(camera_index)
        H, mask, _, a_map, b_map = calibrate_system(
            cap=cap, screen=screen, WIDTH=WIDTH, HEIGHT=HEIGHT
        )
        H_inv = np.linalg.inv(H)
        save_calibration(CALIBRATION_CACHE, H, mask, a_map, b_map)
        cap.release()

    cap_stream = VideoStream(camera_index)

    cell_w = WIDTH / GRID_W
    cell_h = HEIGHT / GRID_H
    wave_px_speed = WAVE_SPEED * min(cell_w, cell_h) * FPS
    base_max_speed = wave_px_speed * BOAT_MAX_SPEED_FACTOR

    u = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    u_prev = np.zeros_like(u)
    u_next = np.zeros_like(u)
    solid = np.zeros((GRID_H, GRID_W), dtype=bool)

    source_mask = np.zeros_like(solid)
    source_mask[:, SOURCE_X] = True
    wave_kernel = make_kernel(WAVE_BRUSH_RADIUS)

    dock_mask_grid, goal_rect, dock_surface, dock_pos, dock_rects = build_dock_data(
        WIDTH, HEIGHT, GRID_W, GRID_H
    )

    img = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
    frame = pygame.Surface((GRID_W, GRID_H))

    boat_sprite = make_boat_sprite((52, 24))
    boat_radius = 12.0
    boat_pos = np.array([WIDTH * 0.6, HEIGHT * 0.55], dtype=np.float32)
    boat_vel = np.zeros(2, dtype=np.float32)
    boat_angle = -0.3

    running = True
    paused = False
    drive_on = True
    show_overlay = False
    success = False
    completed_success = False
    t = 0.0
    frame_count = 0
    t_prev = time.time()

    def draw_wave_background(dt_sec):
        nonlocal t, u, u_prev, u_next
        t += dt_sec

        ret, cam_frame = cap_stream.read()
        if not ret:
            return True

        cam_gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
        cam_gray_masked = cv2.bitwise_and(cam_gray, cam_gray, mask=mask)

        h_cam, w_cam = cam_gray.shape[:2]
        warped_proj_gray_masked = screen_to_camera_space(screen, H, mask, h_cam, w_cam)
        expected = gain_offset_calibration(warped_proj_gray_masked, a_map, b_map)

        mask_diff = np.zeros_like(cam_gray_masked, dtype=bool)
        objects, _ = shadow_object_detection_calibrated(
            cam_gray_masked, expected, b_map, mask_diff, H_inv
        )

        shadow_screen = objects_to_mask(objects, HEIGHT, WIDTH)
        shadow_small = cv2.resize(
            shadow_screen, (GRID_W, GRID_H), interpolation=cv2.INTER_NEAREST
        )
        shadow_mask = shadow_small > 0
        u[shadow_mask] = 0.0
        u_prev[shadow_mask] = 0.0

        mx, my = pygame.mouse.get_pos()
        gx = max(0, min(GRID_W - 1, int(mx / cell_w)))
        gy = max(0, min(GRID_H - 1, int(my / cell_h)))
        buttons = pygame.mouse.get_pressed(3)
        if buttons[0]:
            add_kernel(u, gy, gx, wave_kernel, 2.0)

        if not paused:
            step(u, u_prev, u_next, shadow_mask, C2, DAMPING)
            u_prev, u, u_next = u, u_next, u_prev

            if drive_on:
                drive = get_drive(t)
                amp = math.sin(t * SOURCE_FREQ) * drive * SOURCE_STRENGTH
                mask_src = source_mask & ~shadow_mask
                u[mask_src] = amp

        overlay_mask = shadow_mask if show_overlay else None
        render(u, img, color_scale=220.0, overlay=overlay_mask)
        pygame.surfarray.blit_array(frame, img.swapaxes(0, 1))
        pygame.transform.scale(frame, screen.get_size(), screen)
        return True

    print("Controls: left click=wave, right click=move boat,")
    print("C=clear, M=toggle music drive, P=pause,")
    print("R=reset boat, V=toggle shadow overlay, ESC=quit")

    if not run_countdown(
        screen, clock, COUNTDOWN_SECONDS, "GET READY", ready_font, countdown_font, draw_wave_background
    ):
        cap_stream.release()
        pygame.quit()
        cv2.destroyAllWindows()
        return False

    while running:
        dt_sec = clock.tick(FPS) / 1000.0
        t += dt_sec

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_c:
                    u[:] = 0.0
                    u_prev[:] = 0.0
                elif event.key == pygame.K_m:
                    drive_on = not drive_on
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_r:
                    boat_pos[:] = (WIDTH * 0.6, HEIGHT * 0.55)
                    boat_vel[:] = 0.0
                    success = False
                elif event.key == pygame.K_v:
                    show_overlay = not show_overlay

        ret, cam_frame = cap_stream.read()
        if not ret:
            continue

        cam_gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
        cam_gray_masked = cv2.bitwise_and(cam_gray, cam_gray, mask=mask)

        h_cam, w_cam = cam_gray.shape[:2]
        warped_proj_gray_masked = screen_to_camera_space(screen, H, mask, h_cam, w_cam)
        expected = gain_offset_calibration(warped_proj_gray_masked, a_map, b_map)

        mask_diff = np.zeros_like(cam_gray_masked, dtype=bool)
        objects, contours_vis = shadow_object_detection_calibrated(
            cam_gray_masked, expected, b_map, mask_diff, H_inv
        )

        shadow_screen = objects_to_mask(objects, HEIGHT, WIDTH)
        shadow_small = cv2.resize(
            shadow_screen, (GRID_W, GRID_H), interpolation=cv2.INTER_NEAREST
        )
        shadow_mask = shadow_small > 0
        solid_wave = shadow_mask | dock_mask_grid
        u[solid_wave] = 0.0
        u_prev[solid_wave] = 0.0

        if not success:
            mx, my = pygame.mouse.get_pos()
            gx = max(0, min(GRID_W - 1, int(mx / cell_w)))
            gy = max(0, min(GRID_H - 1, int(my / cell_h)))
            buttons = pygame.mouse.get_pressed(3)
            if buttons[0]:
                add_kernel(u, gy, gx, wave_kernel, 2.0)
            if buttons[2]:
                boat_pos[:] = (mx, my)
                boat_vel[:] = 0.0

            if not paused:
                step(u, u_prev, u_next, solid_wave, C2, DAMPING)
                u_prev, u, u_next = u, u_next, u_prev

                if drive_on:
                    drive = get_drive(t)
                    amp = math.sin(t * SOURCE_FREQ) * drive * SOURCE_STRENGTH
                    mask_src = source_mask & ~solid_wave
                    u[mask_src] = amp

            boat_gx = boat_pos[0] / cell_w
            boat_gy = boat_pos[1] / cell_h
            height = sample_bilinear(u, boat_gx, boat_gy)
            prev_height = sample_bilinear(u_prev, boat_gx, boat_gy)
            du = height - prev_height
            grad_x, grad_y = sample_gradient(u, boat_gx, boat_gy)
            grad_mag = math.hypot(grad_x, grad_y)

            target_vel = np.zeros(2, dtype=np.float32)
            if grad_mag > WAVE_MIN_GRAD and abs(du) > WAVE_MIN_DU:
                dir_sign = -1.0 if du > 0.0 else 1.0
                dir_x = dir_sign * grad_x / grad_mag
                dir_y = dir_sign * grad_y / grad_mag
                wave_speed_grid = abs(du) / (grad_mag + 1e-6)
                wave_speed_grid = min(wave_speed_grid, WAVE_SPEED_CLAMP)
                wave_speed_px = wave_speed_grid * min(cell_w, cell_h) * FPS
                strength = min(1.0, abs(du) * WAVE_STRENGTH_GAIN)
                target_vel = np.array([dir_x, dir_y], dtype=np.float32)
                target_vel *= wave_speed_px * strength * WAVE_VELOCITY_GAIN

            boat_vel += (target_vel - boat_vel) * BOAT_RESPONSE * dt_sec
            boat_vel *= BOAT_DRAG

            speed = float(np.hypot(boat_vel[0], boat_vel[1]))
            if speed > base_max_speed and speed > 1e-6:
                boat_vel *= base_max_speed / speed

            boat_pos += boat_vel * dt_sec
            boat_pos[0] = max(10.0, min(WIDTH - 10.0, boat_pos[0]))
            boat_pos[1] = max(10.0, min(HEIGHT - 10.0, boat_pos[1]))

            for dock_rect in dock_rects:
                normal = resolve_circle_rect(boat_pos, boat_radius, dock_rect)
                if normal is None:
                    continue
                nx, ny = normal
                vn = boat_vel[0] * nx + boat_vel[1] * ny
                if vn < 0.0:
                    boat_vel[0] -= vn * nx
                    boat_vel[1] -= vn * ny

            if (
                boat_pos[0] <= 10.0
                or boat_pos[0] >= WIDTH - 10.0
                or boat_pos[1] <= 10.0
                or boat_pos[1] >= HEIGHT - 10.0
            ):
                if not show_message(
                    screen, clock, "FAIL", FAIL_HOLD_SECONDS, fail_font, (220, 60, 60)
                ):
                    running = False
                    continue
                u[:] = 0.0
                u_prev[:] = 0.0
                u_next[:] = 0.0
                boat_pos[:] = (WIDTH * 0.6, HEIGHT * 0.55)
                boat_vel[:] = 0.0
                boat_angle = -0.3
                success = False
                paused = False
                t = 0.0
                if not run_countdown(
                    screen,
                    clock,
                    COUNTDOWN_SECONDS,
                    "GET READY",
                    ready_font,
                    countdown_font,
                    draw_wave_background,
                ):
                    running = False
                continue

            if speed > 10.0:
                target_angle = math.atan2(boat_vel[1], boat_vel[0])
                boat_angle = angle_step(boat_angle, target_angle, BOAT_TURN_RATE * dt_sec)
        else:
            boat_vel[:] = 0.0

        overlay_mask = shadow_mask if show_overlay else None
        render(u, img, color_scale=220.0, overlay=overlay_mask)
        pygame.surfarray.blit_array(frame, img.swapaxes(0, 1))
        pygame.transform.scale(frame, screen.get_size(), screen)
        pygame.draw.rect(screen, (40, 200, 80), goal_rect, 2)

        screen.blit(dock_surface, dock_pos)

        rotated = pygame.transform.rotate(boat_sprite, -math.degrees(boat_angle))
        rect = rotated.get_rect(center=(boat_pos[0], boat_pos[1]))
        screen.blit(rotated, rect)

        if goal_rect.colliderect(rect):
            if not success:
                success = True
                if return_on_success:
                    if not show_message(
                        screen, clock, "SUCCESS", 2.0, font, (255, 255, 255)
                    ):
                        running = False
                    else:
                        completed_success = True
                    running = False

        if success:
            text_rect = success_surf.get_rect(center=(WIDTH // 2, HEIGHT // 5))
            screen.blit(success_surf, text_rect)

        pygame.display.flip()

        if frame_count % 60 == 0:
            fps = 1.0 / max(1e-6, time.time() - t_prev)
            print(f"FPS: {fps:.1f} | Objects: {len(objects)}")
            t_prev = time.time()
        frame_count += 1

        cv2.imshow("Camera View", cam_frame)
        cv2.imshow("Expected Camera Image", expected)
        cv2.imshow("Contours", contours_vis)
        cv2.waitKey(1)

    cap_stream.release()
    pygame.quit()
    cv2.destroyAllWindows()
    return completed_success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Level 1 - Dock")
    parser.add_argument(
        "--cached-calibration",
        action="store_true",
        help="Use cached calibration from calibration_cache.npz.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=1,
        help="OpenCV camera index to use. This setup detected the secondary webcam at index 1.",
    )
    args = parser.parse_args()
    main(
        cached_calibration=args.cached_calibration,
        return_on_success=False,
        camera_index=args.camera_index,
    )
