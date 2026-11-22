"""Technical notes and the menu’s fullscreen audience display."""

import pygame
from pygame._sdl2.video import Renderer, Texture, Window


TECHNOLOGY = (
    ("Projector and camera", "#83d4f2", (
        "A projector draws the playfield and a camera watches it.",
        "OpenCV fits a homography from checkerboard corners.",
        "A red dot identifies the top edge. Black and white",
        "captures estimate each camera pixel’s gain and offset.",
        "The calibration is saved and shared across levels.",
    )),
    ("Shadow detection", "#a4df9b", (
        "The game image is warped into camera coordinates.",
        "Gain and offset predict how bright it should look.",
        "Pixels darker than expected form shadow contours.",
        "The inverse homography maps them back onto the",
        "playfield for obstacles, platforms, or heat sources.",
    )),
    ("Shadow Harbor waves", "#83d4f2", (
        "NumPy stores the current and previous wave heights.",
        "A five-point finite-difference stencil advances a",
        "damped wave equation. Shadows, the dock, and edges",
        "pin the height to zero. Clicks and a driven source",
        "make waves. Surface slopes help steer the boat.",
    )),
    ("Water Wakes fluid solver", "#72ded9", (
        "A 128 × 72 grid stores a two-dimensional velocity field.",
        "Semi-Lagrangian advection moves the flow. A Jacobi",
        "pressure solve reduces divergence. The model adds",
        "a wake with slowing and sideways disturbances.",
        "Bubbles trace the resulting velocity field.",
    )),
    ("Counting vortices", "#a4df9b", (
        "Velocity gradients identify local swirling motion.",
        "Nearby rotating cells with the same spin form one",
        "detected region. Its bubbles turn soft green.",
        "The score is the live count of those regions.",
    )),
    ("Game and phone controls", "#f5cb83", (
        "Pygame handles rendering, input, and audio playback.",
        "Phones load a local controller page over HTTP.",
        "A WebSocket streams buttons and joystick input.",
        "Shadow Percussion synthesizes its sounds in NumPy.",
    )),
)


class InfoDisplay:
    def __init__(self, monitor):
        self.window = Window(
            "Shadow Arcade · Info screen",
            size=(monitor["width"], monitor["height"]),
            position=(monitor["left"], monitor["top"]),
            borderless=True,
        )
        self.renderer = Renderer(self.window)
        self.renderer.draw_color = (16, 23, 29, 255)
        self.viewport = pygame.Rect(0, 0, *self.window.size)
        self.view = None

    def draw(self, view):
        self.view = view
        width, height = self.window.size
        self.viewport = view.surface.get_rect().fit((0, 0, width, height))
        texture = Texture.from_surface(self.renderer, view.surface)
        self.renderer.clear()
        self.renderer.blit(texture, self.viewport)
        self.renderer.present()

    def handle_pointer(self, event):
        if self.view is None or not self.viewport.collidepoint(event.pos):
            return
        x, y = event.pos
        width, height = self.view.surface.get_size()
        point = ((x - self.viewport.x) * width // self.viewport.width,
                 (y - self.viewport.y) * height // self.viewport.height)
        if event.type == pygame.MOUSEMOTION:
            self.view.mouse = point
        elif event.button == 1:
            self.view.click(point)

    def close(self):
        del self.renderer
        self.window.destroy()
