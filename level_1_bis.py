import math
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

DT = 1.0
WAVE_SPEED = 0.45
DAMPING = 0.005
C2 = (WAVE_SPEED * DT) ** 2

SOURCE_X = 8
SOURCE_FREQ = 5.0
SOURCE_STRENGTH = 1.2
WAVE_BRUSH_RADIUS = 6
SHADOW_SOURCE_FREQ = 4.0
SHADOW_SOURCE_STRENGTH = 0.1
SHADOW_SOURCE_SIGMA = 1.2
SHADOW_RATIO = 0.6
SHADOW_MIN_DELTA = 15.0
SHADOW_MIN_AREA = 200
EXPECTED_DARK_RATIO = 0.35

BOAT_RESPONSE = 5.0
BOAT_DRAG = 0.98
BOAT_TURN_RATE = 0.7
BOAT_MAX_SPEED_FACTOR = 2.0

WAVE_STRENGTH_GAIN = 20.0
WAVE_VELOCITY_GAIN = 1.6
WAVE_SPEED_CLAMP = 1.5
WAVE_MIN_GRAD = 1e-4
WAVE_MIN_DU = 1e-4


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
    dock_surface.fill((120, 80, 40), dock_spine.move(-dock_left, -dock_top))
    dock_surface.fill((120, 80, 40), dock_top_bar.move(-dock_left, -dock_top))
    dock_surface.fill((120, 80, 40), dock_bottom_bar.move(-dock_left, -dock_top))
    pygame.draw.rect(dock_surface, (70, 45, 20), dock_spine.move(-dock_left, -dock_top), 2)
    pygame.draw.rect(dock_surface, (70, 45, 20), dock_top_bar.move(-dock_left, -dock_top), 2)
    pygame.draw.rect(dock_surface, (70, 45, 20), dock_bottom_bar.move(-dock_left, -dock_top), 2)

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


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Level 1 bis - Shadow Source")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 52, bold=True)
    success_surf = font.render("SUCCESS", True, (255, 255, 255))

    cap = get_camera_capture()
    H, mask, _, a_map, b_map = calibrate_system(cap=cap, screen=screen, WIDTH=WIDTH, HEIGHT=HEIGHT)
    H_inv = np.linalg.inv(H)
    cap.release()

    cap_stream = VideoStream(0)

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
    boat_stopped = False
    t = 0.0
    frame_count = 0
    t_prev = time.time()

    print("Controls: left click=wave, right click=move boat,")
    print("C=clear, M=toggle music drive, P=pause,")
    print("R=reset boat, V=toggle shadow overlay, ESC=quit")
    print("Shadows emit waves (no reflections).")

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
                    boat_stopped = False
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

        expected_f = expected.astype(np.float32)
        dark_threshold = b_map + (255.0 - b_map) * EXPECTED_DARK_RATIO
        mask_diff = expected_f < dark_threshold
        objects, contours_vis = shadow_object_detection_calibrated(
            cam_gray_masked, expected, b_map, mask_diff, H_inv
        )

        shadow_screen = objects_to_mask(objects, HEIGHT, WIDTH)
        shadow_small = cv2.resize(
            shadow_screen, (GRID_W, GRID_H), interpolation=cv2.INTER_NEAREST
        )
        shadow_mask = shadow_small > 0
        solid_wave = dock_mask_grid
        u[solid_wave] = 0.0
        u_prev[solid_wave] = 0.0

        shadow_float = shadow_mask.astype(np.float32)
        if SHADOW_SOURCE_SIGMA > 0.0:
            shadow_force = cv2.GaussianBlur(shadow_float, (0, 0), SHADOW_SOURCE_SIGMA)
        else:
            shadow_force = shadow_float
        shadow_force[solid_wave] = 0.0
        shadow_drive = math.sin(t * SHADOW_SOURCE_FREQ) * SHADOW_SOURCE_STRENGTH
        u += shadow_force * shadow_drive

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

        dock_contact = False
        if not boat_stopped:
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
                dock_contact = True
                nx, ny = normal
                vn = boat_vel[0] * nx + boat_vel[1] * ny
                if vn < 0.0:
                    boat_vel[0] -= vn * nx
                    boat_vel[1] -= vn * ny

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
            success = True
        if success and dock_contact:
            boat_stopped = True
            boat_vel[:] = 0.0

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
        cv2.imshow("Shadow Mask (Screen)", shadow_screen)
        cv2.waitKey(1)

    cap_stream.release()
    pygame.quit()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
