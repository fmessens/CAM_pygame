"""Interactive 2-D incompressible-flow experiment.

Run from the project root:
    python experiments/pressure_stream.py

Arrow keys move the circular obstruction.  The simulation uses a coarse
velocity grid, semi-Lagrangian advection, and a Jacobi pressure projection.
It is intended as an understandable experiment, not an engineering CFD tool.
"""

from __future__ import annotations

import math

import numpy as np
import pygame


WINDOW_W, WINDOW_H = 1024, 576
GRID_W, GRID_H = 128, 72
CELL_W, CELL_H = WINDOW_W / GRID_W, WINDOW_H / GRID_H

SIM_HZ = 60
PRESSURE_ITERATIONS = 35
BASE_FLOW = 7.0  # grid cells per second
FLOW_RELAXATION = 0.7
OBSTACLE_RADIUS_PX = 46
OBSTACLE_SPEED_PX = 190
PARTICLE_COUNT = 2600


def sample_bilinear(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample a scalar or vector grid at floating-point grid coordinates."""
    x = np.clip(x, 0.0, GRID_W - 1.001)
    y = np.clip(y, 0.0, GRID_H - 1.001)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    x1 = np.minimum(x0 + 1, GRID_W - 1)
    y1 = np.minimum(y0 + 1, GRID_H - 1)
    fx = x - x0
    fy = y - y0

    if field.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]

    top = field[y0, x0] * (1.0 - fx) + field[y0, x1] * fx
    bottom = field[y1, x0] * (1.0 - fx) + field[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy


class Fluid:
    def __init__(self) -> None:
        self.velocity = np.zeros((GRID_H, GRID_W, 2), dtype=np.float32)
        self.velocity[..., 0] = BASE_FLOW
        self.pressure = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        self.divergence = np.zeros_like(self.pressure)
        self.solid = np.zeros((GRID_H, GRID_W), dtype=bool)

        gy, gx = np.mgrid[0:GRID_H, 0:GRID_W]
        self.gx = gx.astype(np.float32)
        self.gy = gy.astype(np.float32)

    def set_obstacle(
        self, position_px: pygame.Vector2, velocity_px: pygame.Vector2
    ) -> None:
        cx = position_px.x / CELL_W
        cy = position_px.y / CELL_H
        radius = OBSTACLE_RADIUS_PX / CELL_W
        self.solid[:] = (self.gx - cx) ** 2 + (self.gy - cy) ** 2 <= radius**2
        obstacle_velocity = np.array(
            [velocity_px.x / CELL_W, velocity_px.y / CELL_H], dtype=np.float32
        )
        self.velocity[self.solid] = obstacle_velocity

    def _apply_boundaries(self) -> None:
        # Open inflow/outflow, with slip along the horizontal walls.
        self.velocity[:, 0, 0] = BASE_FLOW
        self.velocity[:, 0, 1] = 0.0
        self.velocity[:, -1] = self.velocity[:, -2]
        self.velocity[0, :, 1] = 0.0
        self.velocity[-1, :, 1] = 0.0

    def _advect(self, dt: float) -> None:
        old = self.velocity.copy()
        source_x = self.gx - dt * old[..., 0]
        source_y = self.gy - dt * old[..., 1]
        self.velocity[:] = sample_bilinear(old, source_x, source_y)

    def _project(self) -> None:
        u = self.velocity[..., 0]
        v = self.velocity[..., 1]

        # Central differences. Solid velocities participate here, so a moving
        # obstruction injects its motion into the adjacent fluid.
        self.divergence.fill(0.0)
        self.divergence[1:-1, 1:-1] = 0.5 * (
            u[1:-1, 2:]
            - u[1:-1, :-2]
            + v[2:, 1:-1]
            - v[:-2, 1:-1]
        )
        self.divergence[self.solid] = 0.0

        fluid = ~self.solid
        self.pressure.fill(0.0)
        for _ in range(PRESSURE_ITERATIONS):
            p = self.pressure
            total = np.zeros_like(p)
            count = np.zeros_like(p)

            # A solid neighbour uses the current cell pressure (zero normal
            # pressure gradient); fluid neighbours contribute normally.
            neighbour = fluid[:, :-1]
            total[:, 1:] += np.where(neighbour, p[:, :-1], p[:, 1:])
            count[:, 1:] += 1
            neighbour = fluid[:, 1:]
            total[:, :-1] += np.where(neighbour, p[:, 1:], p[:, :-1])
            count[:, :-1] += 1
            neighbour = fluid[:-1, :]
            total[1:, :] += np.where(neighbour, p[:-1, :], p[1:, :])
            count[1:, :] += 1
            neighbour = fluid[1:, :]
            total[:-1, :] += np.where(neighbour, p[1:, :], p[:-1, :])
            count[:-1, :] += 1

            self.pressure[:] = np.where(
                fluid, (total - self.divergence) / np.maximum(count, 1), 0.0
            )
            # Anchor the pressure gauge and permit open horizontal boundaries.
            self.pressure[:, 0] = 0.0
            self.pressure[:, -1] = 0.0

        p = self.pressure
        grad_x = np.zeros_like(p)
        grad_y = np.zeros_like(p)
        grad_x[:, 1:-1] = 0.5 * (p[:, 2:] - p[:, :-2])
        grad_y[1:-1, :] = 0.5 * (p[2:, :] - p[:-2, :])
        u[fluid] -= grad_x[fluid]
        v[fluid] -= grad_y[fluid]

    def step(
        self,
        dt: float,
        obstacle_position: pygame.Vector2,
        obstacle_velocity: pygame.Vector2,
    ) -> None:
        # A gentle body force maintains the current without erasing the wake.
        self.velocity[..., 0] += (
            BASE_FLOW - self.velocity[..., 0]
        ) * FLOW_RELAXATION * dt
        self._advect(dt)
        self.set_obstacle(obstacle_position, obstacle_velocity)
        self._apply_boundaries()
        self._project()
        self.set_obstacle(obstacle_position, obstacle_velocity)
        self._apply_boundaries()


class Particles:
    def __init__(self) -> None:
        self.position = np.empty((PARTICLE_COUNT, 2), dtype=np.float32)
        self.age = np.empty(PARTICLE_COUNT, dtype=np.float32)
        self.rng = np.random.default_rng()
        self.respawn(np.ones(PARTICLE_COUNT, dtype=bool), anywhere=True)

    def respawn(self, mask: np.ndarray, anywhere: bool = False) -> None:
        count = int(mask.sum())
        if not count:
            return
        if anywhere:
            self.position[mask, 0] = self.rng.uniform(0, GRID_W, count)
        else:
            self.position[mask, 0] = self.rng.uniform(0.0, 1.5, count)
        self.position[mask, 1] = self.rng.uniform(1.0, GRID_H - 2.0, count)
        self.age[mask] = self.rng.uniform(4.0, 12.0, count)

    def step(self, fluid: Fluid, dt: float) -> None:
        velocity = sample_bilinear(
            fluid.velocity, self.position[:, 0], self.position[:, 1]
        )
        self.position += velocity * dt
        self.age -= dt

        ix = np.clip(self.position[:, 0].astype(np.int32), 0, GRID_W - 1)
        iy = np.clip(self.position[:, 1].astype(np.int32), 0, GRID_H - 1)
        dead = (
            (self.position[:, 0] < 0)
            | (self.position[:, 0] >= GRID_W - 1)
            | (self.position[:, 1] < 1)
            | (self.position[:, 1] >= GRID_H - 1)
            | (self.age <= 0)
            | fluid.solid[iy, ix]
        )
        self.respawn(dead)


def field_surface(fluid: Fluid) -> pygame.Surface:
    speed = np.linalg.norm(fluid.velocity, axis=2)
    pressure_scale = np.tanh(fluid.pressure * 0.22)

    rgb = np.empty((GRID_H, GRID_W, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(15 + 75 * np.maximum(pressure_scale, 0), 0, 255)
    rgb[..., 1] = np.clip(34 + speed * 8, 0, 150)
    rgb[..., 2] = np.clip(65 + speed * 15 + 65 * np.maximum(-pressure_scale, 0), 0, 255)
    rgb[fluid.solid] = (38, 45, 50)

    surface = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
    return pygame.transform.smoothscale(surface, (WINDOW_W, WINDOW_H))


def draw_vectors(screen: pygame.Surface, fluid: Fluid) -> None:
    for y in range(3, GRID_H, 6):
        for x in range(3, GRID_W, 6):
            if fluid.solid[y, x]:
                continue
            vx, vy = fluid.velocity[y, x]
            start = (int((x + 0.5) * CELL_W), int((y + 0.5) * CELL_H))
            end = (int(start[0] + vx * 1.8), int(start[1] + vy * 1.8))
            pygame.draw.line(screen, (145, 196, 220), start, end, 1)


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Pressure stream — arrow keys move the obstruction")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    fluid = Fluid()
    particles = Particles()
    obstacle = pygame.Vector2(WINDOW_W * 0.48, WINDOW_H * 0.5)
    show_vectors = True
    paused = False
    accumulator = 0.0
    fixed_dt = 1.0 / SIM_HZ
    running = True

    while running:
        frame_dt = min(clock.tick(60) / 1000.0, 0.05)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_v:
                    show_vectors = not show_vectors
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    fluid = Fluid()
                    particles = Particles()

        keys = pygame.key.get_pressed()
        direction = pygame.Vector2(
            int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT]),
            int(keys[pygame.K_DOWN]) - int(keys[pygame.K_UP]),
        )
        if direction.length_squared() > 0:
            direction = direction.normalize()
        obstacle_velocity = direction * OBSTACLE_SPEED_PX

        margin = OBSTACLE_RADIUS_PX + 8
        old_obstacle = obstacle.copy()
        obstacle += obstacle_velocity * frame_dt
        obstacle.x = max(margin, min(WINDOW_W - margin, obstacle.x))
        obstacle.y = max(margin, min(WINDOW_H - margin, obstacle.y))
        if frame_dt:
            obstacle_velocity = (obstacle - old_obstacle) / frame_dt

        if not paused:
            accumulator = min(accumulator + frame_dt, fixed_dt * 3)
            while accumulator >= fixed_dt:
                fluid.step(fixed_dt, obstacle, obstacle_velocity)
                particles.step(fluid, fixed_dt)
                accumulator -= fixed_dt

        screen.blit(field_surface(fluid), (0, 0))

        particle_xy = particles.position.copy()
        particle_xy[:, 0] *= CELL_W
        particle_xy[:, 1] *= CELL_H
        for x, y in particle_xy.astype(np.int32):
            pygame.draw.circle(screen, (177, 226, 242), (x, y), 1)

        if show_vectors:
            draw_vectors(screen, fluid)
        pygame.draw.circle(
            screen,
            (210, 218, 220),
            (round(obstacle.x), round(obstacle.y)),
            OBSTACLE_RADIUS_PX,
        )
        pygame.draw.circle(
            screen,
            (25, 30, 34),
            (round(obstacle.x), round(obstacle.y)),
            OBSTACLE_RADIUS_PX,
            3,
        )

        label = "Arrows: move   V: vectors   Space: pause   R: reset   Esc: quit"
        if paused:
            label += "   [PAUSED]"
        text = font.render(label, True, (235, 241, 244))
        screen.blit(text, (14, 12))
        fps = font.render(f"{clock.get_fps():4.0f} FPS", True, (235, 241, 244))
        screen.blit(fps, (WINDOW_W - fps.get_width() - 14, 12))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
