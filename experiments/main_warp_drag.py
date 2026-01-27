import argparse
import os
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

LENS_RADIUS_PX = 220.0
LENS_STRENGTH = 6.0
WARP_MAX = 18.0

DIFF_THRESH = 40
EXPECTED_MIN = 45
BASELINE_MIN = 35
SHADOW_MIN_AREA = 200

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


def main(image_path):
    pygame.init()
    screen, width, height, display_index, monitor, monitor_count = create_fullscreen_display(
        TARGET_DISPLAY_INDEX
    )
    pygame.display.set_caption("Warp Shadow Lens")
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

    cap = get_camera_capture()
    H, mask, _, a_map, b_map = calibrate_system(
        cap=cap, screen=screen, WIDTH=width, HEIGHT=height
    )
    H_inv = np.linalg.inv(H)
    cam_h, cam_w = mask.shape[:2]
    cam_map_x, cam_map_y = build_camera_grid_maps(
        H_inv, width, height, GRID_W, GRID_H, cam_w, cam_h
    )
    cap.release()

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
    baseline_mask = (baseline_cam > BASELINE_MIN) if baseline_cam is not None else None
    running = True
    show_overlay = False
    frame_count = 0
    t_prev = time.time()
    print("Controls: V=toggle shadow overlay, ESC=quit")

    while running:
        clock.tick(FPS)

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
        objects, contours_vis, shadow_mask_u8 = shadow_object_detection_diff(
            cam_gray_masked,
            expected,
            mask_diff,
            H_inv,
            expected_min=None if baseline_mask is not None else EXPECTED_MIN,
            valid_mask=baseline_mask,
        )

        shadow_screen = objects_to_mask(objects, height, width)
        shadow_small = cv2.resize(
            shadow_screen, (GRID_W, GRID_H), interpolation=cv2.INTER_NEAREST
        )
        shadow_mask = shadow_small > 0

        map_x, map_y = compute_shadow_warp_map(
            grid_x,
            grid_y,
            shadow_mask,
            lens_radius,
            LENS_STRENGTH,
            WARP_MAX,
        )
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
            cv2.imshow("Shadow Diff Mask", shadow_mask_u8)
            cv2.waitKey(1)

    cap_stream.release()
    pygame.quit()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warp an image around detected shadows.")
    parser.add_argument(
        "--image",
        default=os.path.join("images", "galaxy-1489300667bhk.jpg"),
        help="Path to the background image.",
    )
    args = parser.parse_args()
    main(args.image)
