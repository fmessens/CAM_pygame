"""Launch balls into shadow reflections to play corner instruments."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
import pygame
from pygame._sdl2.video import Renderer, Texture, Window

from level_1 import (
    CALIBRATION_CACHE,
    TARGET_DISPLAY_INDEX,
    create_fullscreen_display,
    get_target_monitor,
    load_calibration,
    save_calibration,
)
from level_3 import shadow_object_detection_calibrated
from modules.calibration import calibrate_system
from modules.camera_magic import gain_offset_calibration, screen_to_camera_space
from modules.videostream import VideoStream, get_camera_capture


TITLE = "Shadow Percussion"
MAX_BALLS = 24
SEAM_GUARD_RADII = 3


@dataclass
class Instrument:
    name: str
    color: tuple[int, int, int]
    center: pygame.Vector2
    radius: int
    sound: pygame.mixer.Sound | None = None
    flash: float = 0.0

    def play(self):
        if self.sound is not None:
            self.sound.play()
        self.flash = 0.24


@dataclass
class Ball:
    position: pygame.Vector2
    velocity: pygame.Vector2
    touching: set[int] = field(default_factory=set)

    def update(self, dt, size, radius, shadows, instruments):
        steps = max(1, math.ceil(self.velocity.length() * dt / (radius * 0.5)))
        width, height = size
        seam_guard = SEAM_GUARD_RADII * radius
        for _ in range(steps):
            self.position += self.velocity * (dt / steps)
            self.position.x %= width
            self.position.y %= height
            # Keep projection-edge artifacts from acting as walls at the wrap seams.
            if (seam_guard <= self.position.x < width - seam_guard
                    and seam_guard <= self.position.y < height - seam_guard):
                for contour, bounds in shadows:
                    if bounds.collidepoint(self.position):
                        self.bounce(contour, radius)
                self.position.x %= width
                self.position.y %= height

            touching = {
                index for index, instrument in enumerate(instruments)
                if self.position.distance_squared_to(instrument.center) <= (instrument.radius + radius) ** 2
            }
            for index in sorted(touching - self.touching):
                instruments[index].play()
            self.touching = touching

    def bounce(self, contour, radius):
        x, y = self.position
        distance = cv2.pointPolygonTest(contour, (x, y), True)
        if distance <= -radius:
            return

        # Signed distance increases toward the interior of the shadow.
        normal = pygame.Vector2(
            cv2.pointPolygonTest(contour, (x - 1, y), True)
            - cv2.pointPolygonTest(contour, (x + 1, y), True),
            cv2.pointPolygonTest(contour, (x, y - 1), True)
            - cv2.pointPolygonTest(contour, (x, y + 1), True),
        )
        if normal.length_squared() < 1e-8:
            normal = -self.velocity
        normal.normalize_ip()
        self.position += normal * (radius + distance + 0.25)
        if self.velocity.dot(normal) < 0:
            self.velocity.reflect_ip(normal)


def make_instruments(size):
    width, height = size
    radius = max(12, round(min(width, height) * 0.075))
    margin = round(radius * 1.9)
    corners = [
        Instrument(name, color, pygame.Vector2(x, y), radius)
        for name, color, x, y in (
            ("Kick", (57, 81, 112), margin, margin),
            ("Cymbal", (119, 82, 41), width - margin, margin),
            ("Snare", (112, 53, 64), margin, height - margin),
            ("Guitar", (48, 98, 76), width - margin, height - margin),
        )
    ]
    return corners + [
        Instrument("Big Beat", (86, 65, 112), pygame.Vector2(width / 2, height / 2),
                   round(radius * 1.65)),
    ]


def make_sounds(instruments):
    config = pygame.mixer.get_init()
    if config is None:
        print("Audio output is unavailable.")
        return
    sample_rate, _, channels = config
    pygame.mixer.set_num_channels(24)
    t = np.arange(int(sample_rate * 0.85)) / sample_rate
    noise = np.random.default_rng(8).uniform(-1.0, 1.0, len(t))
    high_noise = np.diff(noise, prepend=0) * 0.5
    kick = np.sin(2 * np.pi * (48 * t + 6 * (1 - np.exp(-t * 26)))) * np.exp(-t * 12)
    cymbal = high_noise * 0.75
    for frequency in (1973, 2467, 3313):
        cymbal += 0.09 * np.sin(2 * np.pi * frequency * t)
    cymbal *= np.exp(-t * 6.5)
    snare = noise * np.exp(-t * 19) + 0.4 * np.sin(2 * np.pi * 175 * t) * np.exp(-t * 33)
    big_beat = (
        np.sin(2 * np.pi * (39 * t + 5 * (1 - np.exp(-t * 34)))) * np.exp(-t * 7.5)
        + 0.35 * np.sin(2 * np.pi * 78 * t) * np.exp(-t * 15)
        + 0.22 * high_noise * np.exp(-t * 42)
    )
    guitar = np.zeros_like(t)
    for frequency in (146.83, 196.00, 246.94):
        for harmonic, gain in enumerate((1.0, 0.38, 0.19, 0.12, 0.07), start=1):
            guitar += gain * np.sin(2 * np.pi * frequency * harmonic * t) * np.exp(-t * (3 + harmonic))
    attack = np.minimum(t * 500, 1)
    release = np.minimum((t[-1] - t) * 60, 1)
    waves = (kick, cymbal, snare, guitar, big_beat)
    volumes = (0.58, 0.35, 0.46, 0.49, 0.72)
    for instrument, wave, volume in zip(instruments, waves, volumes):
        wave *= attack * release
        samples = np.int16(wave * (32767 * volume / np.max(np.abs(wave))))
        if channels > 1:
            samples = np.repeat(samples[:, None], channels, axis=1)
        instrument.sound = pygame.sndarray.make_sound(samples)


def shadow_geometry(objects, radius, size):
    shadows = []
    width, height = size
    edge_strip = SEAM_GUARD_RADII * radius
    screen = pygame.Rect(0, 0, width, height)
    for obj in objects:
        contour = np.asarray(obj["polygon_pygame"], dtype=np.int32)
        if len(contour) >= 3:
            bounds = pygame.Rect(cv2.boundingRect(contour))
            visible = bounds.clip(screen)
            if not visible.width or not visible.height:
                continue
            # Camera spill can create a thin projected shadow along an entire edge.
            vertical_strip = (visible.width <= edge_strip
                              and (visible.left <= radius or visible.right >= width - radius))
            horizontal_strip = (visible.height <= edge_strip
                                and (visible.top <= radius or visible.bottom >= height - radius))
            if not (vertical_strip or horizontal_strip):
                shadows.append((contour, bounds.inflate(2 * radius + 2, 2 * radius + 2)))
    return shadows


def draw_background(size, instruments):
    surface = pygame.Surface(size)
    surface.fill((127, 130, 137))
    width, height = size
    for x in range(0, width, max(22, width // 40)):
        pygame.draw.line(surface, (132, 135, 143), (x, 0), (x, height))
    for y in range(0, height, max(22, width // 40)):
        pygame.draw.line(surface, (132, 135, 143), (0, y), (width, y))
    for x in range(0, width, 32):
        pygame.draw.line(surface, (151, 156, 166), (x, height // 2), (x + 12, height // 2), 2)
    for instrument in instruments:
        center = instrument.center
        radius = instrument.radius
        pygame.draw.circle(surface, instrument.color, center, radius)
        pygame.draw.circle(surface, (227, 225, 215), center, radius, max(2, radius // 18))
        pygame.draw.circle(surface, (202, 204, 204), center, radius - max(4, radius // 6), 1)
        font = pygame.font.SysFont("Segoe UI", max(14, round(radius * 0.38)), bold=True)
        label = font.render(instrument.name, True, (246, 245, 229))
        surface.blit(label, label.get_rect(center=center))
    return surface


def draw_balls(screen, balls, radius):
    width, height = screen.get_size()
    for ball in balls:
        x, y = ball.position
        dx = width if x < radius else -width if x > width - radius else 0
        dy = height if y < radius else -height if y > height - radius else 0
        for offset_x in (0, dx) if dx else (0,):
            for offset_y in (0, dy) if dy else (0,):
                center = (round(x + offset_x), round(y + offset_y))
                pygame.draw.circle(screen, (255, 219, 114), center, radius)
                pygame.draw.circle(screen, (255, 249, 220), center, radius, 2)


def show_launch_prompt():
    _, monitor, _ = get_target_monitor(0)
    size = (485, 226)
    window = Window(f"{TITLE} · Controls", size=size,
                    position=(monitor["left"] + 55, monitor["top"] + 80))
    renderer = Renderer(window)
    surface = pygame.Surface(size)
    surface.fill((25, 35, 47))
    pygame.draw.line(surface, (243, 191, 99), (24, 105), (461, 105), 2)
    for text, y, pixels, color in (
        (TITLE, 20, 32, (244, 246, 246)),
        ("Press Space to launch a ball", 62, 22, (243, 191, 99)),
        ("Use shadows to bounce balls into the instruments.", 130, 17, (219, 227, 230)),
        ("Backspace clears balls  ·  Esc exits", 166, 17, (219, 227, 230)),
    ):
        font = pygame.font.SysFont("Segoe UI", pixels, bold=pixels >= 22)
        surface.blit(font.render(text, True, color), (24, y))
    texture = Texture.from_surface(renderer, surface)
    window.focus()
    return window, renderer, texture


def main(cached_calibration=False, camera_index=1):
    pygame.mixer.pre_init(44100, -16, 1, 512)
    pygame.init()
    stream = None
    prompt = None
    try:
        screen, width, height, *_ = create_fullscreen_display(TARGET_DISPLAY_INDEX)
        pygame.display.set_caption(TITLE)
        game_window = Window.from_display_module()
        if cached_calibration:
            calibration = load_calibration(CALIBRATION_CACHE)
            if calibration is None:
                print("Run Shadow Calibration before loading cached calibration.")
                return
            H, camera_mask, a_map, b_map = calibration
        else:
            cap = get_camera_capture(camera_index)
            try:
                H, camera_mask, _, a_map, b_map = calibrate_system(
                    cap=cap, screen=screen, WIDTH=width, HEIGHT=height
                )
            finally:
                cap.release()
            save_calibration(CALIBRATION_CACHE, H, camera_mask, a_map, b_map)
        H_inv = np.linalg.inv(H)
        stream = VideoStream(camera_index)
        size = (width, height)
        radius = max(6, round(min(size) * 0.013))
        launch_x = min(width / 4, SEAM_GUARD_RADII * radius + 1)
        instruments = make_instruments(size)
        make_sounds(instruments)
        background = draw_background(size, instruments)
        h_cam, w_cam = camera_mask.shape
        # The moving sprites only brighten the static artwork. Using the static
        # projection as reference avoids mistaking delayed ball images for shadows.
        expected = gain_offset_calibration(
            screen_to_camera_space(background, H, camera_mask, h_cam, w_cam), a_map, b_map
        )
        screen.blit(background, (0, 0))
        pygame.display.flip()
        prompt = show_launch_prompt()
        prompt_visible = True
        balls = []
        shadows = []
        clock = pygame.time.Clock()
        running = True
        while running:
            dt = min(clock.tick(60) / 1000, 0.05)
            for event in pygame.event.get():
                if event.type == pygame.WINDOWCLOSE:
                    window = event.window
                    if window is not None and window.id == prompt[0].id:
                        prompt[0].hide()
                        prompt_visible = False
                        game_window.focus()
                    elif window is not None and window.id == game_window.id:
                        running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_BACKSPACE:
                        balls.clear()
                    elif event.key == pygame.K_SPACE:
                        balls.append(Ball(pygame.Vector2(launch_x, height / 2),
                                          pygame.Vector2(width * 0.42, 0)))
                        if len(balls) > MAX_BALLS:
                            balls.pop(0)
                        if prompt_visible:
                            prompt[0].hide()
                            prompt_visible = False
                            game_window.focus()
            if not running:
                break

            ok, frame = stream.read()
            if ok and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                masked = cv2.bitwise_and(gray, gray, mask=camera_mask)
                objects, _, _ = shadow_object_detection_calibrated(masked, expected, b_map, None, H_inv)
                shadows = shadow_geometry(objects, radius, size)

            for instrument in instruments:
                instrument.flash = max(0.0, instrument.flash - dt)
            for ball in balls:
                ball.update(dt, size, radius, shadows, instruments)

            screen.blit(background, (0, 0))
            for instrument in instruments:
                if instrument.flash:
                    ring = instrument.radius + round(3 + 10 * (1 - instrument.flash / 0.24))
                    pygame.draw.circle(screen, (255, 233, 154), instrument.center, ring, 4)
            draw_balls(screen, balls, radius)
            pygame.display.flip()
            if prompt_visible:
                prompt[1].blit(prompt[2])
                prompt[1].present()
    finally:
        if stream is not None:
            stream.release()
        if prompt is not None:
            window = prompt[0]
            prompt = None
            window.destroy()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--cached-calibration", action="store_true")
    main(**vars(parser.parse_args()))
