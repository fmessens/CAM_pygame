"""
Project a high‑contrast pattern on a projector (beamer).
Use fullscreen on the projector display for best results.

Controls:
  Q            Quit
  1            Checkerboard pattern
  2            Dot grid pattern
  I            Invert colors
  F            Toggle fullscreen

Tips:
- On multi‑monitor setups, set which display to use with the env var
  SDL_VIDEO_FULLSCREEN_DISPLAY, e.g. on Windows PowerShell:
    $env:SDL_VIDEO_FULLSCREEN_DISPLAY = 1
  or run with:  python projector_pattern.py --display 1 --fullscreen
"""
import os, argparse, pygame

DEF_W, DEF_H = 1920, 1080


def draw_checker(surface, cell=80, invert=False):
    w, h = surface.get_size()
    c1 = (255, 255, 255) if not invert else (0, 0, 0)
    c2 = (0, 0, 0) if not invert else (255, 255, 255)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            rect = pygame.Rect(x, y, cell, cell)
            color = c1 if ((x // cell + y // cell) % 2 == 0) else c2
            surface.fill(color, rect)


def draw_dots(surface, spacing=80, radius=6, invert=False):
    w, h = surface.get_size()
    bg = (0, 0, 0) if not invert else (255, 255, 255)
    fg = (255, 255, 255) if not invert else (0, 0, 0)
    surface.fill(bg)
    for y in range(spacing // 2, h, spacing):
        for x in range(spacing // 2, w, spacing):
            pygame.draw.circle(surface, fg, (x, y), radius)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--display', type=int, default=None, help='fullscreen display index')
    ap.add_argument('--fullscreen', action='store_true', help='start fullscreen')
    ap.add_argument('--width', type=int, default=DEF_W)
    ap.add_argument('--height', type=int, default=DEF_H)
    ap.add_argument('--pattern', choices=['checker', 'dots'], default='checker')
    ap.add_argument('--cell', type=int, default=80)
    ap.add_argument('--spacing', type=int, default=80)
    ap.add_argument('--radius', type=int, default=6)
    ap.add_argument('--invert', action='store_true')
    args = ap.parse_args()

    if args.display is not None:
        os.environ['SDL_VIDEO_FULLSCREEN_DISPLAY'] = str(args.display)

    pygame.init()
    flags = pygame.NOFRAME
    size = (args.width, args.height)
    if args.fullscreen:
        flags |= pygame.FULLSCREEN
        size = (0, 0)
    screen = pygame.display.set_mode(size, flags)
    pygame.display.set_caption('Projector Pattern')
    clock = pygame.time.Clock()

    pattern = args.pattern
    invert = args.invert

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_q:
                    running = False
                elif e.key == pygame.K_1:
                    pattern = 'checker'
                elif e.key == pygame.K_2:
                    pattern = 'dots'
                elif e.key == pygame.K_i:
                    invert = not invert
                elif e.key == pygame.K_f:
                    flags ^= pygame.FULLSCREEN
                    pygame.display.quit(); pygame.display.init()
                    screen = pygame.display.set_mode(size, flags)

        if pattern == 'checker':
            draw_checker(screen, cell=args.cell, invert=invert)
        else:
            draw_dots(screen, spacing=args.spacing, radius=args.radius, invert=invert)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    main()