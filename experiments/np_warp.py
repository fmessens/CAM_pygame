import argparse
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

SHADOW_RATIO = 0.6
SHADOW_MIN_DELTA = 15.0
SHADOW_MIN_AREA = 200

WARP_MAX = 18.0
WARP_STRENGTH = 220.0
WARP_SOFTEN = 6.0

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


def compute_shadow_centers(objects, min_area):
    centers = []
    for obj in objects:
        pts = np.array(obj["polygon_pygame"], dtype=np.float32)
        if pts.shape[0] < 3:
            continue
        area = abs(cv2.contourArea(pts.astype(np.int32)))
        if area < min_area:
            continue
        center = pts.mean(axis=0)
        centers.append((center[0], center[1], area))
    return centers


def compute_warp_map(grid_x, grid_y, shadow_centers):
    disp_x = np.zeros_like(grid_x, dtype=np.float32)
    disp_y = np.zeros_like(grid_y, dtype=np.float32)

    for sx, sy, mass in shadow_centers:
        dx = sx - grid_x
        dy = sy - grid_y
        dist2 = dx * dx + dy * dy + WARP_SOFTEN
        strength = WARP_STRENGTH * mass
        disp_x += dx * (strength / dist2)
        disp_y += dy * (strength / dist2)

    disp_x = np.clip(disp_x, -WARP_MAX, WARP_MAX)
    disp_y = np.clip(disp_y, -WARP_MAX, WARP_MAX)
    map_x = (grid_x + disp_x).astype(np.float32)
    map_y = (grid_y + disp_y).astype(np.float32)
    return map_x, map_y


def main(image_path, use_calibration=False, calibration_path=CALIBRATION_CACHE):
    pygame.init()
    screen, width, height, display_index, monitor, monitor_count = create_fullscreen_display(
        TARGET_DISPLAY_INDEX
    )
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = width, height
    pygame.display.set_caption("NumPy Shadow Warp")
    clock = pygame.time.Clock()

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

    if use_calibration:
        cap = get_camera_capture()
        H, mask, _, a_map, b_map = calibrate_system(
            cap=cap, screen=screen, WIDTH=WIDTH, HEIGHT=HEIGHT
        )
        cap.release()
        save_calibration(calibration_path, H, mask, a_map, b_map)
    else:
        cached = load_calibration(calibration_path)
        if cached is None:
            print("No calibration cache found. Run with --calibration first.")
            pygame.quit()
            return
        H, mask, a_map, b_map = cached

    H_inv = np.linalg.inv(H)
    cap_stream = VideoStream(0)

    bg_img = load_background_image(image_path, (GRID_W, GRID_H))
    if bg_img is None:
        print(f"Image not found: {image_path}")
        cap_stream.release()
        pygame.quit()
        return

    warp_img = np.zeros_like(bg_img)
    frame = pygame.Surface((GRID_W, GRID_H))
    grid_x, grid_y = np.meshgrid(
        np.arange(GRID_W, dtype=np.float32),
        np.arange(GRID_H, dtype=np.float32)
    )
    cell_w = WIDTH / GRID_W
    cell_h = HEIGHT / GRID_H
    shadow_mass_area = WIDTH * HEIGHT * 0.008

    running = True
    show_overlay = False
    frame_count = 0
    t_prev = time.time()

    print("Controls: V=toggle shadow overlay, ESC=quit")

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

        shadow_centers = compute_shadow_centers(objects, SHADOW_MIN_AREA)
        shadow_centers_grid = []
        for sx, sy, area in shadow_centers:
            mass = min(1.0, area / shadow_mass_area)
            shadow_centers_grid.append((sx / cell_w, sy / cell_h, mass))

        map_x, map_y = compute_warp_map(grid_x, grid_y, shadow_centers_grid)
        cv2.remap(
            bg_img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
            dst=warp_img,
        )

        if show_overlay:
            warp_img[shadow_mask] = (20, 20, 20)

        pygame.surfarray.blit_array(frame, warp_img.swapaxes(0, 1))
        pygame.transform.scale(frame, screen.get_size(), screen)
        pygame.display.flip()

        if frame_count % 60 == 0:
            fps = 1.0 / max(1e-6, time.time() - t_prev)
            print(f"FPS: {fps:.1f} | Shadows: {len(objects)}")
            t_prev = time.time()
        frame_count += 1

        if SHOW_DEBUG_WINDOWS:
            cv2.imshow("Camera View", cam_frame)
            cv2.imshow("Expected Camera Image", expected)
            cv2.imshow("Contours", contours_vis)
            cv2.waitKey(1)

    cap_stream.release()
    pygame.quit()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NumPy Shadow Warp")
    parser.add_argument("--image", required=True, help="Path to the background image.")
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Run calibration and cache the result for later reuse.",
    )
    args = parser.parse_args()
    main(args.image, use_calibration=args.calibration)
