import pygame
import cv2
import numpy as np
import time

# ============================================================
#  FULLSCREEN CHECKERBOARD GENERATOR
# ============================================================
def make_checkerboard_fullscreen(cols=8, rows=6, width=800, height=600):
    square_w = width // cols
    square_h = height // rows
    board = np.zeros((height, width, 3), np.uint8)

    for y in range(rows):
        for x in range(cols):
            if (x + y) % 2 == 0:
                x0, y0 = x * square_w, y * square_h
                x1, y1 = (x + 1) * square_w, (y + 1) * square_h
                board[y0:y1, x0:x1] = 255
    return board


# ============================================================
#  CALIBRATION: HOMOGRAPHY + GAIN/OFFSET
# ============================================================
def calibrate_system(cap, screen, cols=8, rows=6, WIDTH=800, HEIGHT=600):
    project_checkerboard(cols, rows, screen)

    print("Calibration: Position checkerboard so the camera sees it. Press SPACE.")

    H, mask, cam_corners = wait_for_checkerboard_capture(
        cols, rows, WIDTH, HEIGHT, cap
    )

    print("✔️ Homography calibration complete.")
    print("Proceeding to photometric (gain/offset) calibration...")

    a, b = photometric_calibration(cap, screen, mask)

    print("✔️ Gain/Offset calibration complete.")
    return H, mask, cam_corners, a, b


def project_checkerboard(cols, rows, screen):
    checker = make_checkerboard_fullscreen(cols, rows)
    checker_surf = pygame.surfarray.make_surface(checker.swapaxes(0, 1))

    pygame.display.flip()
    pygame.event.pump()
    time.sleep(0.05)

    screen.blit(checker_surf, (0, 0))
    pygame.display.flip()
    time.sleep(1)


def wait_for_checkerboard_capture(cols, rows, WIDTH, HEIGHT, cap):
    pattern_inner = (cols - 1, rows - 1)

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Camera View", frame)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                raise SystemExit

            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                ret, frame = cap.read()
                if not ret:
                    continue

                found, corners = detect_checkerboard(frame, pattern_inner)
                if not found:
                    print("❌ Checkerboard not detected.")
                    continue

                objp = compute_screen_object_points(cols, rows, WIDTH, HEIGHT)
                imgp = corners.reshape(-1, 2).astype(np.float32)

                H = compute_homography(objp, imgp)
                cam_corners = compute_projection_polygon(H, WIDTH, HEIGHT)

                preview_homography(frame, pattern_inner, corners, cam_corners)

                mask = build_projection_mask(frame, cam_corners)
                return H, mask, cam_corners
            

def detect_checkerboard(frame, pattern_inner):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return cv2.findChessboardCornersSB(
        gray,
        pattern_inner,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    )


def compute_screen_object_points(cols, rows, WIDTH, HEIGHT):
    cell_w = WIDTH / cols
    cell_h = HEIGHT / rows

    xv, yv = np.meshgrid(
        np.arange(1, cols) * cell_w,
        np.arange(1, rows) * cell_h
    )

    return np.column_stack([xv.ravel(), yv.ravel()]).astype(np.float32)


def compute_homography(objp, imgp):
    H, _ = cv2.findHomography(objp, imgp)
    return H


def compute_projection_polygon(H, WIDTH, HEIGHT):
    screen_corners = np.float32([
        [0, 0],
        [WIDTH, 0],
        [WIDTH, HEIGHT],
        [0, HEIGHT]
    ]).reshape(1, 4, 2)

    return cv2.perspectiveTransform(screen_corners, H)[0].astype(np.int32)


def preview_homography(frame, pattern_inner, corners, cam_corners):
    preview = frame.copy()
    cv2.polylines(preview, [cam_corners], True, (0, 0, 255), 2)
    cv2.drawChessboardCorners(preview, pattern_inner, corners, True)

    cv2.imshow("Calibration preview (red = full screen)", preview)
    cv2.waitKey(0)
    cv2.destroyWindow("Calibration preview (red = full screen)")


def build_projection_mask(frame, cam_corners):
    cam_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(cam_gray, np.uint8)
    cv2.fillConvexPoly(mask, cam_corners, 255)
    return mask


def photometric_calibration(cap, screen, mask):
    cam_black = capture_projected_frame(cap, screen, (0, 0, 0), mask, "BLACK")
    cam_white = capture_projected_frame(cap, screen, (255, 255, 255), mask, "WHITE")

    a, b = compute_gain_offset(cam_black, cam_white)
    visualize_gain_offset(a, b)

    return a, b


def capture_projected_frame(cap, screen, color, mask, label):
    screen.fill(color)
    pygame.display.flip()

    print(f"\n=== {label} FRAME PROJECTED ===")
    print("Press any key to capture from camera.")
    cv2.waitKey(0)

    ret, frame = cap.read()
    cam_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cam_masked = cv2.bitwise_and(cam_gray, cam_gray, mask=mask)

    cv2.imshow(f"CALIB CHECK — cam_{label.lower()}", cam_masked)
    return cam_masked.astype(np.float32)


def compute_gain_offset(cam_black, cam_white):
    b = cam_black
    a = (cam_white - b) / 255.0
    a = np.clip(a, 0.001, None)
    return a, b


def visualize_gain_offset(a, b):
    a_vis = (a / a.max() * 255).astype(np.uint8)
    b_vis = (b / b.max() * 255).astype(np.uint8)

    cv2.imshow("Photometric gain a(x,y)", a_vis)
    cv2.imshow("Photometric offset b(x,y)", b_vis)

    print("\nInspect gain/offset maps. Press any key to finish calibration.")
    cv2.waitKey(0)