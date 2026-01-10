import cv2
import numpy as np
import time
from collections import deque

# ============================================================
# SETTINGS
# ============================================================

WIDTH, HEIGHT = 800, 600          # projector resolution (overridden if projector detected)
CHECKER_COLS, CHECKER_ROWS = 8, 6
WATER_SOURCE_X = WIDTH // 2
WATER_SOURCE_WIDTH = 4
LAGBUFFER_WATER = 3
CAMERA_WIN_POS = (40, 40)
DIFF_WIN_POS = (40, 360)
PROJECTOR_ORIGIN = (0, 0)
PRIMARY_ORIGIN = (0, 0)
SHOW_DIFF_WINDOW = True      # Show DIFF OpenCV window (debug only)
DRAW_SOLID_ON_PROJECTOR = False  # If True, render detected solids onto projector image

CAM_INDEX = 0


def detect_projector_geometry():
    """
    Detect monitors and return projector (non-primary if available) and primary info.
    """
    try:
        import ctypes

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        class MONITORINFOEX(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_ulong),
                ("rcMonitor", RECT),
                ("rcWork", RECT),
                ("dwFlags", ctypes.c_ulong),
                ("szDevice", ctypes.c_wchar * 32),
            ]

        monitors = []

        def _cb(hMonitor, hdcMonitor, lprcMonitor, lParam):
            info = MONITORINFOEX()
            info.cbSize = ctypes.sizeof(MONITORINFOEX)
            if ctypes.windll.user32.GetMonitorInfoW(hMonitor, ctypes.byref(info)):
                r = info.rcMonitor
                monitors.append({
                    "origin": (r.left, r.top),
                    "width": r.right - r.left,
                    "height": r.bottom - r.top,
                    "is_primary": bool(info.dwFlags & 1),
                })
            return 1

        monitor_enum_proc = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.POINTER(RECT), ctypes.c_double)(_cb)
        if not ctypes.windll.user32.EnumDisplayMonitors(0, 0, monitor_enum_proc, 0):
            return None

        if not monitors:
            return None

        primary = next((m for m in monitors if m["is_primary"]), monitors[0])
        projector = next((m for m in monitors if not m["is_primary"]), None)
        if projector is None:
            projector = max(monitors, key=lambda m: m["origin"][0] + m["width"])

        return {"projector": projector, "primary": primary}
    except Exception as e:
        print(f"Monitor detection failed, using defaults: {e}")
        return None


_proj = detect_projector_geometry()
if _proj:
    projector = _proj["projector"]
    primary = _proj["primary"]
    WIDTH = int(projector["width"])
    HEIGHT = int(projector["height"])
    WATER_SOURCE_X = WIDTH // 2
    PROJECTOR_ORIGIN = projector["origin"]
    PRIMARY_ORIGIN = primary["origin"]

# ============================================================
# DISPLAY (PROJECTOR)
# ============================================================

cv2.namedWindow("PROJECTOR", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("PROJECTOR", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.namedWindow("CAMERA", cv2.WINDOW_NORMAL)
cv2.namedWindow("DIFF", cv2.WINDOW_NORMAL)


def place_windows_for_dual_monitor():
    """
    Place windows for a dual-monitor layout with monitor 1 on the left (primary)
    and monitor 2 on the right (projector).
    """
    try:
        import ctypes
        user32 = ctypes.windll.user32
        primary_w = user32.GetSystemMetrics(0)
    except Exception as e:
        print(f"Window placement fallback, could not query monitor size: {e}")
        primary_w = 0

    proj_x, proj_y = PROJECTOR_ORIGIN
    prim_x, prim_y = PRIMARY_ORIGIN

    if proj_x == 0 and primary_w:
        proj_x = primary_w

    cv2.moveWindow("PROJECTOR", proj_x, proj_y)
    cv2.moveWindow("CAMERA", prim_x + CAMERA_WIN_POS[0], prim_y + CAMERA_WIN_POS[1])
    cv2.moveWindow("DIFF", prim_x + DIFF_WIN_POS[0], prim_y + DIFF_WIN_POS[1])


place_windows_for_dual_monitor()

# ============================================================
# CHECKERBOARD (NUMPY)
# ============================================================

def make_checkerboard(cols, rows, width, height):
    board = np.zeros((height, width), np.uint8)
    cw, ch = width // cols, height // rows
    for y in range(rows):
        for x in range(cols):
            if (x + y) % 2 == 0:
                board[y*ch:(y+1)*ch, x*cw:(x+1)*cw] = 255
    return board


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_system(cap):
    checker = make_checkerboard(CHECKER_COLS, CHECKER_ROWS, WIDTH, HEIGHT)
    cv2.imshow("PROJECTOR", checker)
    cv2.waitKey(500)

    print("Calibration: press SPACE when checkerboard is visible")

    pattern_inner = (CHECKER_COLS - 1, CHECKER_ROWS - 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("CAMERA", frame)

        key = cv2.waitKey(1)
        if key != 32:
            continue

        found, corners = cv2.findChessboardCornersSB(
            gray,
            pattern_inner,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
        )

        if not found:
            print("Checkerboard not detected, try again")
            continue

        # projector-space points
        cell_w = WIDTH / CHECKER_COLS
        cell_h = HEIGHT / CHECKER_ROWS
        xv, yv = np.meshgrid(
            np.arange(1, CHECKER_COLS) * cell_w,
            np.arange(1, CHECKER_ROWS) * cell_h
        )
        proj_pts = np.column_stack([xv.ravel(), yv.ravel()]).astype(np.float32)
        cam_pts = corners.reshape(-1, 2).astype(np.float32)

        H, _ = cv2.findHomography(proj_pts, cam_pts)
        H_inv = np.linalg.inv(H)

        # projection polygon in camera space
        screen_corners = np.float32([
            [0, 0],
            [WIDTH, 0],
            [WIDTH, HEIGHT],
            [0, HEIGHT]
        ]).reshape(1, 4, 2)

        cam_poly = cv2.perspectiveTransform(screen_corners, H)[0].astype(np.int32)

        mask = np.zeros_like(gray, np.uint8)
        cv2.fillConvexPoly(mask, cam_poly, 255)

        print("Homography calibrated")

        # photometric calibration
        a_map, b_map = photometric_calibration(cap, mask)
        return H, H_inv, mask, a_map, b_map


def photometric_calibration(cap, mask):
    def capture(color, label):
        proj = np.full((HEIGHT, WIDTH), color, np.uint8)
        cv2.imshow("PROJECTOR", proj)
        cv2.waitKey(500)
        print(f"Press any key to capture {label}")
        cv2.waitKey(0)

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.bitwise_and(gray, gray, mask=mask).astype(np.float32)

    cam_black = capture(0, "BLACK")
    cam_white = capture(255, "WHITE")

    b = cam_black
    a = (cam_white - b) / 255.0
    a = np.clip(a, 0.001, None)

    print("Gain/offset calibrated")
    return a, b


# ============================================================
# WATER SIMULATION (PURE NUMPY)
# ============================================================

def add_water(water, x, width=4):
    water[0:2, max(0, x-width):min(water.shape[1], x+width)] = 1


def update_water(water, solid):
    occ = (water | solid).astype(bool)
    new = water.copy()

    down = np.zeros_like(new)
    down[:-1] = new[:-1] & (~occ[1:])
    new[:-1] -= down[:-1]
    new[1:] += down[:-1]

    occ = (new | solid).astype(bool)

    dl = np.zeros_like(new)
    dl[:-1, 1:] = new[:-1, 1:] & (~occ[1:, :-1])
    new[:-1, 1:] -= dl[:-1, 1:]
    new[1:, :-1] += dl[:-1, 1:]

    occ = (new | solid).astype(bool)

    dr = np.zeros_like(new)
    dr[:-1, :-1] = new[:-1, :-1] & (~occ[1:, 1:])
    new[:-1, :-1] -= dr[:-1, :-1]
    new[1:, 1:] += dr[:-1, :-1]

    return (new > 0).astype(np.uint8)


# ============================================================
# SHADOW DETECTION (SIMPLE)
# ============================================================

def detect_shadows(cam_gray, expected, mask_diff):
    diff = cv2.absdiff(cam_gray, expected.astype(np.uint8))
    diff[mask_diff] = 0

    _, th = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    th = cv2.medianBlur(th, 5)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for c in contours:
        if cv2.contourArea(c) < 300:
            continue
        objects.append(c.reshape(-1, 2))

    return objects, th


# ============================================================
# MAIN
# ============================================================

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    assert cap.isOpened()

    H, H_inv, cam_mask, a_map, b_map = calibrate_system(cap)

    water = np.zeros((HEIGHT, WIDTH), np.uint8)
    solid = np.zeros_like(water)

    water_history = deque(maxlen=LAGBUFFER_WATER)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cam_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cam_gray = cv2.bitwise_and(cam_gray, cam_gray, mask=cam_mask)

        # projector image
        proj_gray = np.full((HEIGHT, WIDTH), 255, np.uint8)
        proj_gray[water > 0] = 80

        warped = cv2.warpPerspective(proj_gray, H, cam_gray.shape[::-1])
        warped = cv2.bitwise_and(warped, warped, mask=cam_mask)
        expected = a_map * warped + b_map

        water_cam = cv2.warpPerspective((water * 255).astype(np.uint8), H, cam_gray.shape[::-1])
        water_history.append(water_cam)

        mask_diff = water_history[-1] > 0

        objects, diff_vis = detect_shadows(cam_gray, expected, mask_diff)

        solid.fill(0)
        for poly in objects:
            cv2.fillPoly(solid, [poly.astype(np.int32)], 1)

        add_water(water, WATER_SOURCE_X, WATER_SOURCE_WIDTH)
        water = update_water(water, solid)

        # display
        proj_rgb = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)
        proj_rgb[water > 0] = (60, 150, 255)
        if DRAW_SOLID_ON_PROJECTOR:
            proj_rgb[solid > 0] = (40, 40, 40)

        cv2.imshow("PROJECTOR", proj_rgb)
        cv2.imshow("CAMERA", frame)
        if SHOW_DIFF_WINDOW:
            cv2.imshow("DIFF", diff_vis)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
