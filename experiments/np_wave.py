import math
import numpy as np
import pygame


H, W = 180, 240
SCALE = 3
FPS = 60

DT = 1.0
WAVE_SPEED = 0.45
DAMPING = 0.005
C2 = (WAVE_SPEED * DT) ** 2

SOURCE_X = 8
SOURCE_FREQ = 5.0
SOURCE_STRENGTH = 1.2

WAVE_BRUSH_RADIUS = 6
SOLID_BRUSH_RADIUS = 6
OBSTACLE_RADIUS = 18
OBSTACLE_START_X = W // 2 + 20
OBSTACLE_START_Y = H // 2


def make_kernel(radius):
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    d2 = xx * xx + yy * yy
    kernel = np.exp(-d2 / (radius * radius * 0.5))
    kernel /= kernel.max()
    return kernel


def make_brush(radius):
    ax = np.arange(-radius, radius + 1, dtype=np.int32)
    xx, yy = np.meshgrid(ax, ax)
    return (xx * xx + yy * yy) <= radius * radius


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


def paint_solid(mask, cy, cx, brush, value=True):
    r = brush.shape[0] // 2
    y0 = max(0, cy - r)
    x0 = max(0, cx - r)
    y1 = min(mask.shape[0], cy + r + 1)
    x1 = min(mask.shape[1], cx + r + 1)

    by0 = y0 - (cy - r)
    bx0 = x0 - (cx - r)
    by1 = by0 + (y1 - y0)
    bx1 = bx0 + (x1 - x0)

    sub = mask[y0:y1, x0:x1]
    sub_mask = brush[by0:by1, bx0:bx1]
    if value:
        sub[sub_mask] = True
    else:
        sub[sub_mask] = False


def update_obstacle_mask(mask, cy, cx, brush):
    mask[:] = False
    paint_solid(mask, cy, cx, brush, value=True)


def update_solid_mask(solid, base_solid, obstacle_mask, u, u_prev):
    np.logical_or(base_solid, obstacle_mask, out=solid)
    u[solid] = 0.0
    u_prev[solid] = 0.0


def get_drive(t):
    # Replace with audio RMS in [0, 1] for music-reactive waves.
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


def render(u, solid, img, color_scale):
    c = np.clip(u * color_scale, -127, 127)
    img[..., 0] = (128 + c).astype(np.uint8)
    img[..., 1] = 128
    img[..., 2] = (128 - c).astype(np.uint8)
    img[solid] = (30, 30, 30)


def main():
    pygame.init()
    screen = pygame.display.set_mode((W * SCALE, H * SCALE))
    pygame.display.set_caption("NumPy Wavefronts (click to disturb, right click to draw solids)")
    clock = pygame.time.Clock()

    u = np.zeros((H, W), dtype=np.float32)
    u_prev = np.zeros_like(u)
    u_next = np.zeros_like(u)
    base_solid = np.zeros((H, W), dtype=bool)
    obstacle_mask = np.zeros_like(base_solid)
    solid = np.zeros_like(base_solid)

    source_mask = np.zeros_like(solid)
    source_mask[:, SOURCE_X] = True

    wave_kernel = make_kernel(WAVE_BRUSH_RADIUS)
    solid_brush = make_brush(SOLID_BRUSH_RADIUS)
    obstacle_brush = make_brush(OBSTACLE_RADIUS)

    obstacle_x = OBSTACLE_START_X
    obstacle_y = OBSTACLE_START_Y
    update_obstacle_mask(obstacle_mask, obstacle_y, obstacle_x, obstacle_brush)
    update_solid_mask(solid, base_solid, obstacle_mask, u, u_prev)

    img = np.zeros((H, W, 3), dtype=np.uint8)
    frame = pygame.Surface((W, H))

    running = True
    paused = False
    drive_on = True
    move_obstacle = False
    t = 0.0

    print("Controls: left click=wave, right click=solid, shift+right=erase,")
    print("Middle click=move obstacle, O=toggle obstacle follow,")
    print("C=clear, M=toggle music drive, P=pause, ESC=quit")

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
                elif event.key == pygame.K_o:
                    move_obstacle = not move_obstacle
                elif event.key == pygame.K_SPACE:
                    add_kernel(u, H // 2, SOURCE_X + 2, wave_kernel, 2.5)

        mx, my = pygame.mouse.get_pos()
        gx = max(0, min(W - 1, int(mx / SCALE)))
        gy = max(0, min(H - 1, int(my / SCALE)))
        buttons = pygame.mouse.get_pressed(3)
        if buttons[0]:
            add_kernel(u, gy, gx, wave_kernel, 2.0)
        if buttons[2]:
            erase = pygame.key.get_mods() & pygame.KMOD_SHIFT
            paint_solid(base_solid, gy, gx, solid_brush, value=not erase)
            update_solid_mask(solid, base_solid, obstacle_mask, u, u_prev)
        if buttons[1] or move_obstacle:
            obstacle_x = gx
            obstacle_y = gy
            update_obstacle_mask(obstacle_mask, obstacle_y, obstacle_x, obstacle_brush)
            update_solid_mask(solid, base_solid, obstacle_mask, u, u_prev)

        if not paused:
            step(u, u_prev, u_next, solid, C2, DAMPING)
            u_prev, u, u_next = u, u_next, u_prev

            if drive_on:
                drive = get_drive(t)
                amp = math.sin(t * SOURCE_FREQ) * drive * SOURCE_STRENGTH
                mask = source_mask & ~solid
                u[mask] = amp

        render(u, solid, img, color_scale=220.0)
        pygame.surfarray.blit_array(frame, img.swapaxes(0, 1))
        pygame.transform.scale(frame, screen.get_size(), screen)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
