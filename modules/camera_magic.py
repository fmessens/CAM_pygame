import pygame
import cv2
import numpy as np

def screen_to_camera_space(screen, H, mask, h_cam, w_cam):
    proj_surface = pygame.surfarray.array3d(screen)     # (W,H,3)
    proj_bgr = cv2.cvtColor(proj_surface, cv2.COLOR_RGB2BGR)

    proj_bgr_cv = np.swapaxes(proj_bgr, 0, 1)      # (HEIGHT, WIDTH, 3)
    warped_proj = cv2.warpPerspective(proj_bgr_cv, H, (w_cam, h_cam))
    warped_proj_gray = cv2.cvtColor(warped_proj, cv2.COLOR_BGR2GRAY)
    warped_proj_gray_masked = cv2.bitwise_and(warped_proj_gray, warped_proj_gray, mask=mask)
    return warped_proj_gray_masked

def gain_offset_calibration(warped_proj_gray_masked, a_map, b_map):
    expected = a_map * warped_proj_gray_masked + b_map
    expected = np.clip(expected, 0, 255).astype(np.uint8)
    return expected


def ball_to_camera_space(ball, H, cam_gray_masked):
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
    return ball_mask


def shadow_object_detection(cam_gray_masked, expected, mask_diff, H_inv):
    diff = cv2.absdiff(cam_gray_masked, expected)
    diff[mask_diff] = 0
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

    return objects, contours_vis


def water_to_camera_space(water_particles, H, cam_gray_masked, radius_px=3):
    """
    Project water particles (screen space) into camera space
    and return a binary mask covering their footprint.
    """

    water_mask = np.zeros_like(cam_gray_masked, np.uint8)

    if len(water_particles) == 0:
        return water_mask

    # Collect particle centers
    pts = np.float32([
        [p.pos.x, p.pos.y] for p in water_particles
    ]).reshape(-1, 1, 2)

    # Project into camera space
    pts_cam = cv2.perspectiveTransform(pts, H).astype(np.int32)

    h, w = cam_gray_masked.shape[:2]

    # Draw each particle as a small filled circle
    for (x, y) in pts_cam.reshape(-1, 2):
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(water_mask, (x, y), radius_px, 255, -1)

    return water_mask
