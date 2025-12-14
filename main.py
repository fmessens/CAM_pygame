import pygame
import cv2
import numpy as np
import time

from modules.videostream import VideoStream, get_camera_capture
from modules.calibration import calibrate_system
from modules.camera_magic import (ball_history, screen_to_camera_space,
                                   gain_offset_calibration, ball_to_camera_space,
                                   shadow_object_detection)
from modules.settings import WIDTH, HEIGHT, start_velocity
from modules.physics import ball_physics

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
cap = get_camera_capture()
clock = pygame.time.Clock()



# ============================================================
#  RUN CALIBRATION
# ============================================================
H, mask, cam_corners, a_map, b_map = calibrate_system(cap=cap, screen=screen, WIDTH=WIDTH, HEIGHT=HEIGHT)
H_inv = np.linalg.inv(H)

# ============================================================
#  GAME LOOP
# ============================================================
ball = pygame.Rect(WIDTH//2, HEIGHT//2, 40, 40)
vel = start_velocity

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
    h_cam, w_cam = cam_gray.shape[:2]
    warped_proj_gray_masked = screen_to_camera_space(screen, H, mask, h_cam, w_cam)

    # =====================================================================
    # EXPECTED CAMERA IMAGE (after gain + offset calibration)
    # =====================================================================
    expected = gain_offset_calibration(warped_proj_gray_masked, a_map, b_map)
    
    # ============================================================
    # PROJECT BALL → CAMERA SPACE (current frame)
    # ============================================================

    ball_mask = ball_to_camera_space(ball, H, cam_gray_masked)

    # ============================================================
    # BUILD BALL HISTORY (past N frames)
    # ============================================================
    ball_history.append(ball_mask)

    # Union of all past ball masks
    ball_history_mask = np.max(ball_history, axis=0)
    mask_diff = ball_history_mask > 0

    # =====================================================================
    # SHADOW / OBJECT DETECTION
    # =====================================================================
    objects, contours_vis = shadow_object_detection(cam_gray_masked, expected, mask_diff, H_inv)

    t_cv = time.time()
    # =====================================================================
    # PHYSICS BEFORE DRAWING
    # =====================================================================
    ball, vel = ball_physics(ball, vel, objects, WIDTH, HEIGHT)

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
    

    pygame.display.flip()

    # =====================================================================
    # DEBUG WINDOWS
    # =====================================================================
    cv2.imshow("Camera View", cam_frame)
    cv2.imshow("Warped Projection", warped_proj_gray_masked)
    cv2.imshow("Expected Camera Image", expected)
    cv2.imshow("Contours", contours_vis)
    cv2.imshow("Ball History (combined)", ball_history_mask)

    clock.tick(30)


pygame.quit()
cap.release()
cv2.destroyAllWindows()
