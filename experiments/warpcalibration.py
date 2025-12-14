import pygame
import cv2
import numpy as np
import time
import os
import threading
from collections import deque

#===========================================================
#  VIDEO STREAM CLASS (BACKGROUND THREAD)
#===========================================================

class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Still keep this low
        
        # Read the first frame to avoid errors on startup
        self.ret, self.frame = self.cap.read()
        
        self.stopped = False
        self.lock = threading.Lock()
        
        # Start the background thread
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True # Thread dies when main program dies
        self.t.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            # We only care about the latest frame. 
            # We use a lock to ensure we don't read while writing.
            with self.lock:
                self.ret, self.frame = ret, frame
            
            # Tiny sleep to not hog 100% of a CPU core if camera waits
            time.sleep(0.001)

    def read(self):
        # Return the most recent frame available
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.stopped = True
        self.t.join()
        self.cap.release()

os.environ['SDL_VIDEO_WINDOW_POS'] = "1280,0"

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

ball_history = deque(maxlen=10)

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
def calibrate_system(cols=8, rows=6):

    # Step 1 — PROJECT FULLSCREEN CHECKERBOARD (for homography)
    checker = make_checkerboard_fullscreen()
    checker_surf = pygame.surfarray.make_surface(checker.swapaxes(0,1))

    pygame.display.flip()
    pygame.event.pump()
    time.sleep(0.05)

    screen.blit(checker_surf, (0,0))
    pygame.display.flip()

    time.sleep(1)

    print("Calibration: Position checkerboard so the camera sees it. Press SPACE.")

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

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)

                # ---- Detect inner corners ----
                found, corners = cv2.findChessboardCornersSB(
                    gray, pattern_inner,
                    flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
                )

                if not found:
                    print("❌ Checkerboard not detected.")
                    continue

                # ---- Build objp (true inner-corner coordinates) ----
                cell_w = WIDTH / cols
                cell_h = HEIGHT / rows
                xv, yv = np.meshgrid(np.arange(1, cols)*cell_w,
                                     np.arange(1, rows)*cell_h)

                objp = np.column_stack([xv.ravel(), yv.ravel()]).astype(np.float32)
                imgp = corners.reshape(-1,2).astype(np.float32)

                # ---- Compute homography (screen → camera) ----
                H, _ = cv2.findHomography(objp, imgp)

                # ---- Compute projection polygon in camera space ----
                screen_corners = np.float32([[0,0],[WIDTH,0],[WIDTH,HEIGHT],[0,HEIGHT]]).reshape(1,4,2)
                cam_corners = cv2.perspectiveTransform(screen_corners, H)[0].astype(np.int32)

                # ---- Draw preview ----
                preview = frame.copy()
                cv2.polylines(preview, [cam_corners], True, (0,0,255), 2)
                cv2.drawChessboardCorners(preview, pattern_inner, corners, True)
                cv2.imshow("Calibration preview (red = full screen)", preview)
                cv2.waitKey(0)
                cv2.destroyWindow("Calibration preview (red = full screen)")

                # ---- Build region mask ----
                cam_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mask = np.zeros_like(cam_gray, np.uint8)
                cv2.fillConvexPoly(mask, cam_corners, 255)

                print("✔️ Homography calibration complete.")
                print("Proceeding to photometric (gain/offset) calibration...")

                # =====================================================
                #  STEP 2: PHOTOMETRIC CALIBRATION — WITH PAUSES
                # =====================================================

                # -----------------------------------------------------
                # 2a — Project BLACK
                # -----------------------------------------------------
                screen.fill((0,0,0))
                pygame.display.flip()

                print("\n=== BLACK FRAME PROJECTED ===")
                print("Look at the projector. It should be COMPLETELY BLACK.")
                print("Press any key here to CAPTURE black-frame from camera.")

                cv2.waitKey(0)

                ret, frame_black = cap.read()
                cam_black = cv2.cvtColor(frame_black, cv2.COLOR_BGR2GRAY)
                cam_black_masked = cv2.bitwise_and(cam_black, cam_black, mask=mask)

                cv2.imshow("CALIB CHECK — cam_black (should be dark)", cam_black_masked)


                # -----------------------------------------------------
                # 2b — Project WHITE
                # -----------------------------------------------------
                screen.fill((255,255,255))
                pygame.display.flip()

                print("\n=== WHITE FRAME PROJECTED ===")
                print("Look at the projector. It should be FULL WHITE.")
                print("Press any key here to CAPTURE white-frame from camera.")

                cv2.waitKey(0)

                ret, frame_white = cap.read()
                cam_white = cv2.cvtColor(frame_white, cv2.COLOR_BGR2GRAY)
                cam_white_masked = cv2.bitwise_and(cam_white, cam_white, mask=mask)

                cv2.imshow("CALIB CHECK — cam_white (should be bright)", cam_white_masked)


                # -----------------------------------------------------
                # Compute gain/offset maps
                # -----------------------------------------------------
                b = cam_black_masked.astype(np.float32)            # offset
                a = (cam_white_masked.astype(np.float32) - b) / 255.0
                a = np.clip(a, 0.001, None)                        # avoid zero

                # Debug visualization
                a_vis = (a / a.max() * 255).astype(np.uint8)
                b_vis = (b / b.max() * 255).astype(np.uint8)

                cv2.imshow("Photometric offset b(x,y)", b_vis)
                cv2.imshow("Photometric gain a(x,y)", a_vis)
                print("\nInspect gain/offset maps. Press any key to finish calibration.")
                cv2.waitKey(0)

                print("✔️ Gain/Offset calibration complete.")

                return H, mask, cam_corners, a, b


# ============================================================
#  RUN CALIBRATION
# ============================================================
H, mask, cam_corners, a_map, b_map = calibrate_system()
H_inv = np.linalg.inv(H)

# ============================================================
#  GAME LOOP
# ============================================================
ball = pygame.Rect(WIDTH//2, HEIGHT//2, 40, 40)
vel = [4, 3]

frame_count = 0
t_prev = time.time()

cap_stream = VideoStream(0)

running = True
while running:

    # ============================
    # HANDLE EVENTS
    # ============================
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    # =====================================================================
    # CAMERA FRAME
    # =====================================================================
    t_start = time.time()
    ret, cam_frame = cap_stream.read()
    t_camera = time.time()
    if not ret:
        continue

    cam_gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
    cam_gray_masked = cv2.bitwise_and(cam_gray, cam_gray, mask=mask)

    # =====================================================================
    # PROJECTED FRAME → CAMERA SPACE
    # =====================================================================
    proj_surface = pygame.surfarray.array3d(screen)     # (W,H,3)
    proj_bgr = cv2.cvtColor(proj_surface, cv2.COLOR_RGB2BGR)
    proj_gray = cv2.cvtColor(proj_bgr, cv2.COLOR_BGR2GRAY)

    h_cam, w_cam = cam_gray.shape[:2]

    proj_bgr_cv = np.swapaxes(proj_bgr, 0, 1)      # (HEIGHT, WIDTH, 3)
    warped_proj = cv2.warpPerspective(proj_bgr_cv, H, (w_cam, h_cam))
    warped_proj_gray = cv2.cvtColor(warped_proj, cv2.COLOR_BGR2GRAY)
    warped_proj_gray_masked = cv2.bitwise_and(warped_proj_gray, warped_proj_gray, mask=mask)

    # =====================================================================
    # EXPECTED CAMERA IMAGE (after gain + offset calibration)
    # =====================================================================
    expected = a_map * warped_proj_gray_masked + b_map
    expected = np.clip(expected, 0, 255).astype(np.uint8)
    
    # ============================================================
    # PROJECT BALL → CAMERA SPACE (current frame)
    # ============================================================

    # Ball rect in screen space
    ball_rect = np.float32([
        [ball.left,  ball.top],
        [ball.right, ball.top],
        [ball.right, ball.bottom],
        [ball.left,  ball.bottom]
    ]).reshape(1,4,2)

    # Project to camera
    ball_poly_cam = cv2.perspectiveTransform(ball_rect, H)[0].astype(np.int32)

    # Create mask in camera space
    ball_mask = np.zeros_like(cam_gray_masked, np.uint8)
    cv2.fillConvexPoly(ball_mask, ball_poly_cam, 255)

    # ============================================================
    # BUILD BALL HISTORY (past N frames)
    # ============================================================
    ball_history.append(ball_mask)

    # Union of all past ball masks
    ball_history_mask = np.max(ball_history, axis=0)


    # =====================================================================
    # SHADOW / OBJECT DETECTION
    # =====================================================================
    diff = cv2.absdiff(cam_gray_masked, expected)
    diff[ball_history_mask > 0] = 0
    _, shadow_mask = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY)


    # ==== FIND OBJECTS ====
    contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []

    contours_vis = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR)

    for cnt in contours:

        if cv2.contourArea(cnt) < 200:
            continue

        # Debug draw contour on OpenCV window
        cv2.drawContours(contours_vis, [cnt], -1, (0,255,0), 2)

        # ============================
        # Convert every contour point
        # ============================
        pts_cam = cnt.astype(np.float32)    # shape (N,1,2)

        # Camera → Screen coordinates via inverse homography
        pts_screen = cv2.perspectiveTransform(pts_cam, H_inv)   # (N,1,2)

        pygame_polygon = []

        for p in pts_screen:
            x_cv, y_cv = p[0]

            # Screen → pygame (flip X,Y 180°)
            sx = int(x_cv)
            sy = int(y_cv)


            pygame_polygon.append((sx, sy))

        # Save list of polygon points
        objects.append({
            "contour_cam": cnt,
            "polygon_pygame": pygame_polygon
        })

    t_cv = time.time()
    # =====================================================================
    # PHYSICS BEFORE DRAWING
    # =====================================================================
    ball.x += vel[0]
    ball.y += vel[1]

    if ball.left < 0 or ball.right > WIDTH:
        vel[0] *= -1
    if ball.top < 0 or ball.bottom > HEIGHT:
        vel[1] *= -1

    for obj in objects:
        for px, py in obj["polygon_pygame"]:
            if ball.collidepoint(px, py):
                vel[0] *= -1
                vel[1] *= -1
                break

    t_physics = time.time()
    # Print diagnostics every 60 frames
    if frame_count % 60 == 0:
        print(f"Cam: {(t_camera - t_start)*1000:.1f}ms | "
            f"CV: {(t_cv - t_camera)*1000:.1f}ms | "
            f"Phys: {(t_physics - t_cv)*1000:.1f}ms | "
            f"FPS: {1.0/(time.time() - t_prev):.1f}")
    t_prev = time.time()
    frame_count += 1
    # =====================================================================
    # DRAW GAME (after physics & detection)
    # =====================================================================
    screen.fill((255,255,255))
    pygame.draw.ellipse(screen, (255,0,0), ball)

    # Draw detected object polygons in Pygame
    # for obj in objects:
    #     poly = obj["polygon_pygame"]
    #     if len(poly) >= 3:
    #         pygame.draw.polygon(screen, (0,255,0), poly, width=0)   # filled green polygon
    

    pygame.display.flip()

    # =====================================================================
    # DEBUG WINDOWS
    # =====================================================================
    cv2.imshow("Camera View", cam_frame)
    cv2.imshow("Warped Projection", warped_proj)
    cv2.imshow("Expected Camera Image", expected)
    cv2.imshow("diff", diff)
    cv2.imshow("Shadow Mask", shadow_mask)
    cv2.imshow("Contours", contours_vis)
    # ============================================================
    # BALL HISTORY VISUALIZATION
    # ============================================================

    # 1. Simple combined mask
    cv2.imshow("Ball History (combined)", ball_history_mask)

    # 2. Fading history mask
    ball_history_fade = np.zeros_like(ball_history_mask, np.uint8)
    for i, hmask in enumerate(ball_history):
        fade_value = int(255 * (i+1) / len(ball_history))
        ball_history_fade[hmask > 0] = fade_value
    cv2.imshow("Ball History (fading trail)", ball_history_fade)

    # 3. Overlay on camera feed
    overlay = np.zeros_like(cam_frame, np.uint8)
    overlay[:, :, 2] = ball_history_mask  # red channel for history
    alpha = 0.5
    overlay_img = cv2.addWeighted(cam_frame, 1.0, overlay, alpha, 0)
    cv2.imshow("Ball History Overlay", overlay_img)


    debug = cam_frame.copy()
    cv2.polylines(debug, [cam_corners], True, (0,0,255), 2)
    cv2.imshow("Projection region", debug)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    clock.tick(30)


pygame.quit()
cap.release()
cv2.destroyAllWindows()
