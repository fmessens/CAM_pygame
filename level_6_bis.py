"""Level 6 bis: pressure-stream response with lightweight turbulent wakes."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import pygame

import level_1 as base
from modules.calibration import calibrate_system
from modules.camera_magic import gain_offset_calibration, screen_to_camera_space
from modules.videostream import VideoStream, get_camera_capture


GRID_H = 72
GRID_W = 128
FPS = 60
PRESSURE_ITERATIONS = 35
BASE_FLOW = 3.5
FLOW_RELAXATION = 0.55
BUBBLE_COUNT = 2400
VECTOR_SPACING = 4
WAKE_STRENGTH = 6.5
WAKE_BACKFLOW = 3.2
WAKE_DECAY = 0.86
WAKE_NOISE = 1.2

TARGET_DISPLAY_INDEX = 1
CALIBRATION_CACHE = "calibration_cache.npz"


def sample_grid(field, x, y):
    """Vectorized bilinear sampling for scalar or vector grid fields."""
    h, w = field.shape[:2]
    x = np.clip(x, 0.0, w - 1.001)
    y = np.clip(y, 0.0, h - 1.001)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    if field.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    top = field[y0, x0] * (1.0 - fx) + field[y0, x1] * fx
    bottom = field[y1, x0] * (1.0 - fx) + field[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy


class PressureFlow:
    """Coarse incompressible velocity field with a changing solid mask."""

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.velocity = np.zeros((height, width, 2), dtype=np.float32)
        self.pressure = np.zeros((height, width), dtype=np.float32)
        self.divergence = np.zeros((height, width), dtype=np.float32)
        self.wake = np.zeros((height, width), dtype=np.float32)
        self.rng = np.random.default_rng()
        gy, gx = np.mgrid[0:height, 0:width]
        self.gx = gx.astype(np.float32)
        self.gy = gy.astype(np.float32)
        self.reset()

    def reset(self):
        self.velocity.fill(0.0)
        self.velocity[..., 0] = BASE_FLOW
        self.pressure.fill(0.0)
        self.divergence.fill(0.0)
        self.wake.fill(0.0)

    def apply_boundaries(self, solid, drive_on):
        self.velocity[solid] = 0.0
        inflow = BASE_FLOW if drive_on else 0.0
        self.velocity[:, 0, 0] = inflow
        self.velocity[:, 0, 1] = 0.0
        self.velocity[:, -1] = self.velocity[:, -2]
        self.velocity[0, :, 1] = 0.0
        self.velocity[-1, :, 1] = 0.0
        self.velocity[solid] = 0.0

    def advect(self, dt):
        old = self.velocity.copy()
        source_x = self.gx - dt * old[..., 0]
        source_y = self.gy - dt * old[..., 1]
        self.velocity[:] = sample_grid(old, source_x, source_y)

    def project(self, solid):
        fluid = ~solid
        u = self.velocity[..., 0]
        v = self.velocity[..., 1]
        self.divergence.fill(0.0)
        self.divergence[1:-1, 1:-1] = 0.5 * (
            u[1:-1, 2:]
            - u[1:-1, :-2]
            + v[2:, 1:-1]
            - v[:-2, 1:-1]
        )
        self.divergence[solid] = 0.0
        self.pressure.fill(0.0)

        total = np.empty_like(self.pressure)
        count = np.full_like(self.pressure, 4.0)
        count[0, :] -= 1.0
        count[-1, :] -= 1.0
        count[:, 0] -= 1.0
        count[:, -1] -= 1.0

        for _ in range(PRESSURE_ITERATIONS):
            p = self.pressure
            total.fill(0.0)
            total[:, 1:] += np.where(fluid[:, :-1], p[:, :-1], p[:, 1:])
            total[:, :-1] += np.where(fluid[:, 1:], p[:, 1:], p[:, :-1])
            total[1:, :] += np.where(fluid[:-1, :], p[:-1, :], p[1:, :])
            total[:-1, :] += np.where(fluid[1:, :], p[1:, :], p[:-1, :])
            self.pressure[:] = np.where(
                fluid, (total - self.divergence) / count, 0.0
            )
            self.pressure[:, 0] = 0.0
            self.pressure[:, -1] = 0.0

        p = self.pressure
        grad_x = np.zeros_like(p)
        grad_y = np.zeros_like(p)
        center = p[1:-1, 1:-1]
        p_left = np.where(solid[1:-1, :-2], center, p[1:-1, :-2])
        p_right = np.where(solid[1:-1, 2:], center, p[1:-1, 2:])
        p_up = np.where(solid[:-2, 1:-1], center, p[:-2, 1:-1])
        p_down = np.where(solid[2:, 1:-1], center, p[2:, 1:-1])
        grad_x[1:-1, 1:-1] = 0.5 * (p_right - p_left)
        grad_y[1:-1, 1:-1] = 0.5 * (p_down - p_up)
        u[fluid] -= grad_x[fluid]
        v[fluid] -= grad_y[fluid]

    def add_turbulent_wake(self, solid, dt, drive_on):
        if not drive_on or not solid.any():
            self.wake *= WAKE_DECAY
            return

        downstream = np.zeros_like(solid)
        downstream[:, 1:] |= solid[:, :-1]
        downstream[:, 2:] |= solid[:, :-2]
        downstream[:, 3:] |= solid[:, :-3]
        downstream[:, 4:] |= solid[:, :-4]
        wake_zone = downstream & ~solid

        above = np.zeros_like(solid)
        below = np.zeros_like(solid)
        above[:-1, :] = solid[1:, :]
        below[1:, :] = solid[:-1, :]
        edge_spin = (below.astype(np.float32) - above.astype(np.float32))

        self.wake *= WAKE_DECAY
        self.wake[wake_zone] += WAKE_STRENGTH * dt
        self.wake = np.clip(self.wake, 0.0, 1.0)

        noise = self.rng.uniform(-1.0, 1.0, self.wake.shape).astype(np.float32)
        wave = np.sin(self.gx * 0.47 + self.gy * 0.31 + time.perf_counter() * 7.0)
        swirl = (edge_spin + 0.65 * wave + WAKE_NOISE * noise) * self.wake

        u = self.velocity[..., 0]
        v = self.velocity[..., 1]
        u[wake_zone] -= WAKE_BACKFLOW * self.wake[wake_zone] * dt
        v[wake_zone] += swirl[wake_zone] * dt
        self.velocity[solid] = 0.0

    def step(self, dt, solid, drive_on):
        target = BASE_FLOW if drive_on else 0.0
        self.velocity[..., 0] += (
            target - self.velocity[..., 0]
        ) * FLOW_RELAXATION * dt
        self.advect(dt)
        self.add_turbulent_wake(solid, dt, drive_on)
        self.apply_boundaries(solid, drive_on)
        self.project(solid)
        self.add_turbulent_wake(solid, dt * 0.45, drive_on)
        self.apply_boundaries(solid, drive_on)


class FlowBubbles:
    """Screen-resolution tracers carried by the coarse velocity field."""

    def __init__(self, count, width, height):
        self.count = count
        self.width = width
        self.height = height
        self.rng = np.random.default_rng()
        self.position = np.empty((count, 2), dtype=np.float32)
        self.age = np.empty(count, dtype=np.float32)
        self.radius = self.rng.choice(
            np.array([1, 1, 1, 2, 2, 3], dtype=np.int8), size=count
        )
        self.reset()

    def reset(self):
        self.position[:, 0] = self.rng.uniform(0.0, self.width, self.count)
        self.position[:, 1] = self.rng.uniform(0.0, self.height, self.count)
        self.age[:] = self.rng.uniform(3.0, 14.0, self.count)

    def respawn(self, dead):
        count = int(dead.sum())
        if count == 0:
            return
        self.position[dead, 0] = self.rng.uniform(0.0, self.width, count)
        self.position[dead, 1] = self.rng.uniform(2.0, self.height - 2.0, count)
        self.age[dead] = self.rng.uniform(7.0, 14.0, count)

    def step(self, flow, solid, dt, cell_w, cell_h):
        gx = self.position[:, 0] / cell_w
        gy = self.position[:, 1] / cell_h
        velocity = sample_grid(flow.velocity, gx, gy)
        self.position[:, 0] += velocity[:, 0] * cell_w * dt
        self.position[:, 1] += velocity[:, 1] * cell_h * dt
        self.age -= dt

        ix = np.clip((self.position[:, 0] / cell_w).astype(np.int32), 0, GRID_W - 1)
        iy = np.clip((self.position[:, 1] / cell_h).astype(np.int32), 0, GRID_H - 1)
        dead = (
            (self.position[:, 0] < 0.0)
            | (self.position[:, 0] >= self.width)
            | (self.position[:, 1] < 0.0)
            | (self.position[:, 1] >= self.height)
            | (self.age <= 0.0)
            | solid[iy, ix]
        )
        self.respawn(dead)

    def draw(self, screen):
        points = self.position.astype(np.int32)
        for radius in (1, 2, 3):
            color = (172, 220, 238) if radius == 1 else (210, 239, 247)
            for x, y in points[self.radius == radius]:
                pygame.draw.circle(screen, color, (int(x), int(y)), radius)


def render_flow(flow, img, shadow_mask):
    # Keep the projection visually neutral: shadows affect only the simulated
    # velocity, which remains visible through bubbles and vector arrows.
    img[:] = (86, 154, 190)


def draw_vectors(screen, flow, solid, cell_w, cell_h):
    """Draw sparse arrows, including a dot for near-zero vectors."""
    for y in range(VECTOR_SPACING // 2, GRID_H, VECTOR_SPACING):
        for x in range(VECTOR_SPACING // 2, GRID_W, VECTOR_SPACING):
            if solid[y, x]:
                continue
            vx, vy = flow.velocity[y, x]
            start = pygame.Vector2((x + 0.5) * cell_w, (y + 0.5) * cell_h)
            delta = pygame.Vector2(vx * cell_w * 0.32, vy * cell_h * 0.32)
            if delta.length_squared() < 0.5:
                pygame.draw.circle(screen, (56, 113, 148), start, 1)
                continue
            end = start + delta
            pygame.draw.line(screen, (48, 103, 140), start, end, 1)
            direction = delta.normalize()
            side = pygame.Vector2(-direction.y, direction.x)
            head = min(5.0, max(2.0, delta.length() * 0.3))
            pygame.draw.line(screen, (48, 103, 140), end, end - direction * head + side * head * 0.5, 1)
            pygame.draw.line(screen, (48, 103, 140), end, end - direction * head - side * head * 0.5, 1)


def main(cached_calibration=False, camera_index=1):
    pygame.init()
    screen, width, height, display_index, monitor, monitor_count = (
        base.create_fullscreen_display(TARGET_DISPLAY_INDEX)
    )
    base.WIDTH, base.HEIGHT = width, height
    pygame.display.set_caption("Level 6 bis - Shadow Wake Stream")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)

    print(
        "Display:",
        f"{monitor_count} total,",
        f"using index {display_index} at {monitor['width']}x{monitor['height']}",
        f"pos {monitor['left']},{monitor['top']}",
    )

    if cached_calibration:
        calibration = base.load_calibration(CALIBRATION_CACHE)
        if calibration is None:
            print(f"No calibration cache found at {CALIBRATION_CACHE}.")
            pygame.quit()
            return
        homography, camera_mask, a_map, b_map = calibration
    else:
        cap = get_camera_capture(camera_index)
        homography, camera_mask, _, a_map, b_map = calibrate_system(
            cap=cap, screen=screen, WIDTH=width, HEIGHT=height
        )
        base.save_calibration(
            CALIBRATION_CACHE, homography, camera_mask, a_map, b_map
        )
        cap.release()

    homography_inv = np.linalg.inv(homography)
    cap_stream = VideoStream(camera_index)
    cell_w = width / GRID_W
    cell_h = height / GRID_H
    flow = PressureFlow(GRID_H, GRID_W)
    bubbles = FlowBubbles(BUBBLE_COUNT, width, height)
    image = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
    grid_surface = pygame.Surface((GRID_W, GRID_H))

    running = True
    paused = False
    drive_on = True
    show_vectors = True
    frame_count = 0
    report_time = time.perf_counter()

    print("Controls: C=reset, M=toggle current, P=pause, V=vectors, ESC=quit")

    while running:
        dt = min(clock.tick(FPS) / 1000.0, 1.0 / 30.0)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_c:
                    flow.reset()
                    bubbles.reset()
                elif event.key == pygame.K_m:
                    drive_on = not drive_on
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_v:
                    show_vectors = not show_vectors

        ret, camera_frame = cap_stream.read()
        if not ret:
            continue

        camera_gray = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY)
        camera_gray_masked = cv2.bitwise_and(
            camera_gray, camera_gray, mask=camera_mask
        )
        camera_h, camera_w = camera_gray.shape[:2]
        projected_gray = screen_to_camera_space(
            screen, homography, camera_mask, camera_h, camera_w
        )
        expected = gain_offset_calibration(projected_gray, a_map, b_map)
        mask_diff = np.zeros_like(camera_gray_masked, dtype=bool)
        objects, contours_vis = base.shadow_object_detection_calibrated(
            camera_gray_masked,
            expected,
            b_map,
            mask_diff,
            homography_inv,
        )
        shadow_screen = base.objects_to_mask(objects, height, width)
        shadow_mask = cv2.resize(
            shadow_screen, (GRID_W, GRID_H), interpolation=cv2.INTER_NEAREST
        ) > 0

        if not paused:
            flow.step(dt, shadow_mask, drive_on)
            bubbles.step(flow, shadow_mask, dt, cell_w, cell_h)

        render_flow(flow, image, shadow_mask)
        pygame.surfarray.blit_array(grid_surface, image.swapaxes(0, 1))
        pygame.transform.smoothscale(grid_surface, screen.get_size(), screen)
        bubbles.draw(screen)
        if show_vectors:
            draw_vectors(screen, flow, shadow_mask, cell_w, cell_h)

        status = (
            f"{clock.get_fps():4.0f} FPS  |  "
            f"{PRESSURE_ITERATIONS} pressure iterations + wake turbulence  |  "
            f"{'paused' if paused else 'flowing'}"
        )
        label = font.render(status, True, (235, 242, 245))
        screen.blit(label, (14, 12))
        pygame.display.flip()

        frame_count += 1
        now = time.perf_counter()
        if now - report_time >= 1.0:
            print(f"FPS: {clock.get_fps():.1f} | Shadow objects: {len(objects)}")
            report_time = now

        cv2.imshow("Camera View", camera_frame)
        cv2.imshow("Expected Camera Image", expected)
        cv2.imshow("Contours", contours_vis)
        cv2.waitKey(1)

    cap_stream.release()
    pygame.quit()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Level 6 bis - shadow wake turbulence experiment"
    )
    parser.add_argument(
        "--cached-calibration",
        action="store_true",
        help="Use cached calibration from calibration_cache.npz.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=1,
        help="OpenCV camera index.",
    )
    args = parser.parse_args()
    main(
        cached_calibration=args.cached_calibration,
        camera_index=args.camera_index,
    )
