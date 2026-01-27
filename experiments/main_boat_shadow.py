import math
import time

import cv2
import numpy as np
import pygame

from modules.calibration import calibrate_system
from modules.camera_magic import (
    gain_offset_calibration,
    screen_to_camera_space,
    shadow_object_detection,
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


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Wave Boat + Shadows")
    clock = pygame.time.Clock()

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

    img = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
    frame = pygame.Surface((GRID_W, GRID_H))

    boat_sprite = make_boat_sprite((52, 24))
    boat_pos = np.array([WIDTH * 0.6, HEIGHT * 0.55], dtype=np.float32)
    boat_vel = np.zeros(2, dtype=np.float32)
    boat_angle = -0.3

    running = True
    paused = False
    drive_on = True
    show_overlay = False
    t = 0.0
    frame_count = 0
    t_prev = time.time()

    print("Controls: left click=wave, right click=move boat,")
    print("C=clear, M=toggle music drive, P=pause,")
    print("R=reset boat, V=toggle shadow overlay, ESC=quit")

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
        objects, contours_vis = shadow_object_detection(
            cam_gray_masked, expected, mask_diff, H_inv
        )

        shadow_screen = objects_to_mask(objects, HEIGHT, WIDTH)
        shadow_small = cv2.resize(
            shadow_screen, (GRID_W, GRID_H), interpolation=cv2.INTER_NEAREST
        )
        solid = shadow_small > 0
        u[solid] = 0.0
        u_prev[solid] = 0.0

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
            step(u, u_prev, u_next, solid, C2, DAMPING)
            u_prev, u, u_next = u, u_next, u_prev

            if drive_on:
                drive = get_drive(t)
                amp = math.sin(t * SOURCE_FREQ) * drive * SOURCE_STRENGTH
                mask_src = source_mask & ~solid
                u[mask_src] = amp

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

        if speed > 10.0:
            target_angle = math.atan2(boat_vel[1], boat_vel[0])
            boat_angle = angle_step(boat_angle, target_angle, BOAT_TURN_RATE * dt_sec)

        render(u, img, color_scale=220.0, overlay=solid if show_overlay else None)
        pygame.surfarray.blit_array(frame, img.swapaxes(0, 1))
        pygame.transform.scale(frame, screen.get_size(), screen)

        rotated = pygame.transform.rotate(boat_sprite, -math.degrees(boat_angle))
        rect = rotated.get_rect(center=(boat_pos[0], boat_pos[1]))
        screen.blit(rotated, rect)

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
