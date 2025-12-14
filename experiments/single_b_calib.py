import pygame
import cv2
import numpy as np
import time
import os
import threading

# =============================================
#  SIMPLE VIDEO STREAM CAPTURE CLASS
# =============================================
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

def grab_latest_frame(cap, flush_count=5):
    for _ in range(flush_count):
        cap.grab()
    ret, frame = cap.read()
    return ret, frame

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
                # ============================================================
                #  STEP 3: COLOR CALIBRATION FOR BALL/SPRITE REMOVAL
                # ============================================================
                print("\n=== BALL COLOR CALIBRATION ===")
                print("Projecting the sprite/ball color. Position camera so it sees ONLY the projection area.")
                print("Press any key to capture.")

                # ---- CHANGE THIS TO YOUR BALL COLOR (projected) ----
                BALL_COLOR = (255, 0, 0)   # red in pygame (RGB)
                screen.fill(BALL_COLOR)
                pygame.display.flip()

                cv2.waitKey(0)

                # Capture what camera sees
                ret, frame_color = cap.read()
                if not ret:
                    raise RuntimeError("Camera failed during color calibration")

                # Mask to projection region only
                frame_color_masked = cv2.bitwise_and(frame_color, frame_color, mask=mask)

                # Convert to HSV
                hsv_color = cv2.cvtColor(frame_color_masked, cv2.COLOR_BGR2HSV)

                # Extract HSV pixels only inside projection region
                ys, xs = np.where(mask > 0)
                sample_pixels = hsv_color[ys, xs]

                # Compute median & MAD (robust against noise)
                h_med = np.median(sample_pixels[:,0])
                s_med = np.median(sample_pixels[:,1])
                v_med = np.median(sample_pixels[:,2])

                h_range = np.median(np.abs(sample_pixels[:,0] - h_med)) * 2.5 + 5
                s_range = np.median(np.abs(sample_pixels[:,1] - s_med)) * 3  + 20
                v_range = np.median(np.abs(sample_pixels[:,2] - v_med)) * 3  + 20

                # Construct HSV bounds
                lower_ball = np.array([
                    max(0, h_med - h_range),
                    max(0, s_med - s_range),
                    max(0, v_med - v_range)
                ], dtype=np.uint8)

                upper_ball = np.array([
                    min(179, h_med + h_range),
                    min(255, s_med + s_range),
                    min(255, v_med + v_range)
                ], dtype=np.uint8)

                print("✔️ Ball color calibrated.")
                print("Lower HSV:", lower_ball)
                print("Upper HSV:", upper_ball)

                # Show mask for verification
                ball_mask_preview = cv2.inRange(hsv_color, lower_ball, upper_ball)
                cv2.imshow("Ball Color Mask Preview", ball_mask_preview)
                cv2.waitKey(0)

                return H, mask, cam_corners, a, b, lower_ball, upper_ball


# ============================================================
#  RUN CALIBRATION
# ============================================================
H, mask, cam_corners, a_map, b_map, lower_ball, upper_ball = calibrate_system()
H_inv = np.linalg.inv(H)

# ---------------------------------------------
# Compute expected ONCE because background is static
# ---------------------------------------------
background_surface = pygame.surfarray.array3d(screen)
background_bgr = cv2.cvtColor(background_surface, cv2.COLOR_RGB2BGR)
expected_proj = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2GRAY)

# Warp to camera
h_cam, w_cam = mask.shape
expected_warp = cv2.warpPerspective(expected_proj, H, (w_cam, h_cam))
expected_warp_masked = cv2.bitwise_and(expected_warp, expected_warp, mask=mask)

# Final expected camera image
expected_once = (a_map * expected_warp_masked + b_map).astype(np.uint8)



# =============================================
#  GAME LOOP
# =============================================
ball = pygame.Rect(WIDTH//2, HEIGHT//2, 60, 60)
vel = [5, 4]

cap_stream = VideoStream(0)
running = True

while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    # Camera
    ret, frame = cap_stream.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cam_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cam_gray_masked = cv2.bitwise_and(cam_gray, cam_gray, mask=mask)

    # Sprite removal
    sprite_mask = cv2.inRange(hsv, lower_ball, upper_ball)
    sprite_mask = cv2.bitwise_and(sprite_mask, sprite_mask, mask=mask)

    # Diff vs static expected
    diff = cv2.absdiff(cam_gray_masked, expected_once)
    _, shadow = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)

    # Remove sprite from shadow detection
    shadow = cv2.bitwise_and(shadow, cv2.bitwise_not(sprite_mask))

    # Find obstacles
    contours, _ = cv2.findContours(shadow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # GAME UPDATE
    ball.x += vel[0]
    ball.y += vel[1]
    if ball.left < 0 or ball.right > WIDTH: vel[0] *= -1
    if ball.top < 0 or ball.bottom > HEIGHT: vel[1] *= -1

    # DRAW
    screen.fill((255,255,255))
    pygame.draw.ellipse(screen, (255,0,0), ball)

    # Visualization overlay: show obstacles as green polygons
    for cnt in contours:
        if cv2.contourArea(cnt) < 200: continue

        pts = cnt.astype(np.float32)
        pts_screen = cv2.perspectiveTransform(pts, H_inv)

        poly = []
        for p in pts_screen:
            x, y = p[0]
            sx = int(WIDTH - x)
            sy = int(HEIGHT - y)
            poly.append((sx, sy))

        if len(poly) >= 3:
            pygame.draw.polygon(screen, (0,255,0), poly, width=0)

    pygame.display.flip()

    # Debug windows
    cv2.imshow("Shadow", shadow)
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == 27: break

pygame.quit()
cv2.destroyAllWindows()
