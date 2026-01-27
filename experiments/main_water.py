import pygame
import cv2
import numpy as np
import time

from modules.videostream import VideoStream, get_camera_capture
from modules.calibration import calibrate_system
from modules.camera_magic import (screen_to_camera_space,
                                   gain_offset_calibration, 
                                   water_to_camera_space,
                                   shadow_object_detection)
from modules.settings import WIDTH, HEIGHT, lagbuffer_water
from modules.physics.waterstream import WaterStream

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
water = WaterStream(origin=(WIDTH//2, 30))

frame_count = 0
t_prev = time.time()

cap_stream = VideoStream(0)

water_history = []

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
    # PROJECT WATER → CAMERA SPACE (current frame)
    # ============================================================

    water_mask = water_to_camera_space(water.particles, H, 
                                       cam_gray_masked, radius_px=3)


    # ============================================================
    # BUILD WATER HISTORY (past N frames)
    # ============================================================
    water_history.append(water_mask)

    if len(water_history) > lagbuffer_water:
        water_history = water_history[1:]

    # Union of all past water masks
    water_history_mask = np.max(water_history[:2], axis=0)
    mask_diff = water_history_mask > 0
    # =====================================================================
    # SHADOW / OBJECT DETECTION
    # =====================================================================
    objects, contours_vis = shadow_object_detection(cam_gray_masked, expected, mask_diff, H_inv)

    t_cv = time.time()
    # =====================================================================
    # PHYSICS BEFORE DRAWING
    # =====================================================================
    water.spawn(rate=10)
    water.update(objects, WIDTH, HEIGHT)

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
    water.draw(screen)
    

    pygame.display.flip()

    # =====================================================================
    # DEBUG WINDOWS
    # =====================================================================
    cv2.imshow("Camera View", cam_frame)
    cv2.imshow("Warped Projection", warped_proj_gray_masked)
    cv2.imshow("Expected Camera Image", expected)
    cv2.imshow("Contours", contours_vis)
    cv2.imshow("Water History (combined)", water_history_mask)

    clock.tick(30)


pygame.quit()
cap.release()
cv2.destroyAllWindows()
