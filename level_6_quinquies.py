"""Windward, a multiplayer sailing arena on the Water Wakes simulation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
import pygame

import level_5 as controllers
import level_6_bis as water


TITLE = "Windward"
CONTROLLER_PATH = "/windward/controller"
WIND = pygame.Vector2(1, 0)
MAX_HP = 3
FIRE_INTERVAL = 0.7
SPLASH_TIME = 0.52
RESPAWN_TIME = 2.0
SHIP_COLORS = (
    (255, 205, 145), (197, 238, 177), (236, 189, 248),
    (168, 233, 247), (251, 193, 206), (243, 229, 164),
)


class SailControllerHandler(controllers.TrisControllerRequestHandler):
    controller_path = CONTROLLER_PATH
    controller_html = (
        controllers.TRIS_CONTROLLER_HTML.replace("Level 5 Parachutes Controller", f"{TITLE} Controller")
        .replace("LEVEL 5 PARACHUTES", TITLE.upper())
        .replace(
            '<button class="button action-button stone" data-button="b" data-vibe="stone">ATTACK<br>THROW</button>',
            '<button class="button action-button jump" data-button="a">FIRE LEFT</button>\n'
            '      <button class="button action-button stone" data-button="b">FIRE RIGHT</button>',
        )
    )
    controller_css = controllers.TRIS_CONTROLLER_CSS.replace(
        "grid-template-rows: 1fr;", "grid-template-rows: 1fr 1fr;"
    )
    setup_instructions = (
        "Use the joystick to steer. Sail with the wind for speed. "
        "FIRE LEFT and FIRE RIGHT launch from the two sides of the bow."
    )


def wrap(point: pygame.Vector2, size: tuple[int, int]) -> pygame.Vector2:
    return pygame.Vector2(point.x % size[0], point.y % size[1])


def wrapped_distance_squared(first, second, size):
    dx = abs(first.x - second.x)
    dy = abs(first.y - second.y)
    return min(dx, size[0] - dx) ** 2 + min(dy, size[1] - dy) ** 2


def visible_copies(point, size, margin):
    width, height = size
    x, y = point
    dx = width if x < margin else -width if x > width - margin else 0
    dy = height if y < margin else -height if y > height - margin else 0
    for ox in (0, dx) if dx else (0,):
        for oy in (0, dy) if dy else (0,):
            yield pygame.Vector2(x + ox, y + oy)


def sail_power(heading: pygame.Vector2) -> float:
    return max(0.0, (heading.dot(WIND) + 0.67) / 1.67) ** 0.68


@dataclass
class Ship:
    position: pygame.Vector2
    spawn: pygame.Vector2
    color: tuple[int, int, int]
    heading: pygame.Vector2 = field(default_factory=lambda: pygame.Vector2(WIND))
    speed: float = 0.0
    hp: int = MAX_HP
    respawn: float = 0.0
    left_wait: float = 0.0
    right_wait: float = 0.0
    hit_flash: float = 0.0

    def steer(self, joystick, dt, size):
        stick = pygame.Vector2(joystick["x"], joystick["y"])
        throttle = min(1.0, stick.length())
        if throttle > 0.14:
            turn = self.heading.angle_to(stick)
            turn = (turn + 180) % 360 - 180
            limit = 255 * dt
            self.heading.rotate_ip(max(-limit, min(limit, turn)))
        else:
            throttle = 0.0
        power = sail_power(self.heading)
        target_speed = min(size) * 0.29 * throttle * power
        self.speed += (target_speed - self.speed) * min(1.0, 5 * dt)
        if not power:
            self.speed = 0.0
        self.position = wrap(self.position + self.heading * (self.speed * dt), size)

    def update(self, joystick, dt, size):
        self.left_wait = max(0.0, self.left_wait - dt)
        self.right_wait = max(0.0, self.right_wait - dt)
        self.hit_flash = max(0.0, self.hit_flash - dt)
        if self.hp:
            self.steer(joystick, dt, size)
        else:
            self.respawn -= dt
            if self.respawn <= 0.0:
                self.position = self.spawn.copy()
                self.heading = pygame.Vector2(WIND)
                self.hp = MAX_HP
                self.speed = 0.0

    def fire(self, side, owner, size, scale, rng):
        timer = "left_wait" if side < 0 else "right_wait"
        if getattr(self, timer) > 0.0:
            return None
        setattr(self, timer, FIRE_INTERVAL)
        bow = self.position + self.heading * (13 * scale)
        sideways = self.heading.rotate(side * 90)
        origin = bow + sideways * (5 * scale)
        direction = self.heading.rotate(side * 82 + rng.uniform(-16, 16))
        distance = size[0] * rng.uniform(0.14, 0.26)
        return Cannonball(owner, origin, direction * distance, rng.uniform(0.52, 0.7))

    def damage(self, amount):
        self.hp = max(0, self.hp - amount)
        self.hit_flash = 0.22
        if not self.hp:
            self.respawn = RESPAWN_TIME


@dataclass
class Cannonball:
    owner: str
    origin: pygame.Vector2
    travel: pygame.Vector2
    duration: float
    age: float = 0.0


@dataclass
class Splash:
    center: pygame.Vector2
    age: float = 0.0


def advance_cannonballs(cannonballs, splashes, ships, dt, size, scale):
    damage = {}
    airborne = []
    radius_squared = (33 * scale) ** 2
    for shot in cannonballs:
        shot.age += dt
        if shot.age < shot.duration:
            airborne.append(shot)
            continue
        landing = wrap(shot.origin + shot.travel, size)
        splashes.append(Splash(landing))
        for player_id, ship in ships.items():
            if (player_id != shot.owner and ship.hp
                    and wrapped_distance_squared(landing, ship.position, size) <= radius_squared):
                damage[player_id] = damage.get(player_id, 0) + 1
    for player_id, amount in damage.items():
        ships[player_id].damage(amount)
    return airborne


def sync_ships(ships, snapshots, size):
    for player_id in ships.keys() - snapshots.keys():
        del ships[player_id]
    for player_id, snapshot in snapshots.items():
        if player_id in ships:
            continue
        index = int(snapshot["color_index"])
        angle = index * math.tau * 0.381966
        spawn = pygame.Vector2(size[0] / 2, size[1] / 2)
        spawn += pygame.Vector2(math.cos(angle) * 0.32 * size[0],
                                math.sin(angle) * 0.30 * size[1])
        ships[player_id] = Ship(spawn.copy(), spawn, SHIP_COLORS[index % len(SHIP_COLORS)])


def draw_ship(screen, ship, size, scale):
    forward = ship.heading
    right = forward.rotate(90)
    trim = sail_power(forward)
    for center in visible_copies(ship.position, size, 31 * scale):
        def point(along, across):
            return center + (forward * along + right * across) * scale

        hull = [point(x, y) for x, y in ((-14, -7), (6, -9), (21, 0), (6, 9), (-14, 7), (-19, 0))]
        pygame.draw.polygon(screen, (250, 221, 215) if ship.hit_flash else ship.color, hull)
        pygame.draw.polygon(screen, (249, 242, 223), hull, max(1, round(scale)))
        rear, mast, clew = point(-12, -3), point(11, 0), point(-6, 8)
        # The middle of the sail curves downwind in screen coordinates as it fills.
        belly = (rear + clew) / 2 + WIND * (13 * scale * trim)
        curve = [rear.lerp(belly, t).lerp(belly.lerp(clew, t), t)
                 for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
        pygame.draw.polygon(screen, (253, 247, 220), [mast, *curve])
        pygame.draw.line(screen, (238, 213, 171), mast, point(-12, -3), max(1, round(scale)))
        for mark in range(ship.hp):
            pygame.draw.circle(screen, (246, 248, 225), point(-23, (mark - 1) * 5), max(1, round(1.5 * scale)))


def draw_effects(screen, cannonballs, splashes, size, scale):
    for shot in cannonballs:
        progress = shot.age / shot.duration
        arc = 4 * progress * (1 - progress)
        position = wrap(shot.origin + shot.travel * progress - pygame.Vector2(0, arc * 23 * scale), size)
        radius = max(2, round((2 + arc * 1.6) * scale))
        for center in visible_copies(position, size, radius):
            pygame.draw.circle(screen, (255, 239, 187), center, radius)
    for splash in splashes:
        progress = splash.age / SPLASH_TIME
        ring = max(2, round((4 + 30 * progress) * scale))
        color = tuple(round(start + (end - start) * progress)
                      for start, end in zip((247, 252, 251), (86, 154, 190)))
        for center in visible_copies(splash.center, size, ring + 4):
            pygame.draw.circle(screen, color, center, ring, max(1, round(2 * scale)))
            for index in range(6):
                drop = center + pygame.Vector2(ring * 0.8, 0).rotate(index * 60 + 17)
                pygame.draw.circle(screen, color, drop, max(1, round(2 * scale * (1 - progress))))


def main(cached_calibration=False, bind_host="0.0.0.0", public_host=None,
         http_port=8000, ws_port=8765, camera_index=1):
    stream = http = http_thread = websocket = None
    pygame.init()
    try:
        screen, width, height, *_ = water.base.create_fullscreen_display(water.TARGET_DISPLAY_INDEX)
        pygame.display.set_caption(TITLE)
        size = (width, height)
        scale = max(0.55, min(size) / 720)
        if cached_calibration:
            calibration = water.base.load_calibration(water.CALIBRATION_CACHE)
            if calibration is None:
                print("Run Shadow Calibration before launching Windward.")
                return
            H, camera_mask, a_map, b_map = calibration
        else:
            cap = water.get_camera_capture(camera_index)
            try:
                H, camera_mask, _, a_map, b_map = water.calibrate_system(
                    cap=cap, screen=screen, WIDTH=width, HEIGHT=height
                )
            finally:
                cap.release()
            water.base.save_calibration(water.CALIBRATION_CACHE, H, camera_mask, a_map, b_map)
        H_inv = np.linalg.inv(H)
        stream = water.VideoStream(camera_index)

        state = controllers.TrisControllerState()
        config = controllers.ServerConfig(
            bind_host, public_host or controllers.detect_public_host(), http_port, ws_port,
            app_title=TITLE, controller_path=CONTROLLER_PATH,
        )
        http, http_thread = controllers.start_tris_http_server(state, config, SailControllerHandler)
        websocket = controllers.TrisControllerWebSocketServer(state, bind_host, ws_port)
        websocket.start()
        print(f"Controller page: {config.controller_url}")

        flow = water.PressureFlow(water.GRID_H, water.GRID_W)
        bubbles = water.FlowBubbles(water.BUBBLE_COUNT, width, height)
        solid = np.zeros((water.GRID_H, water.GRID_W), dtype=bool)
        cell_w, cell_h = width / water.GRID_W, height / water.GRID_H
        background = np.empty((*solid.shape, 3), dtype=np.uint8)
        water.render_flow(flow, background, solid)
        base_color = tuple(background[0, 0])
        screen.fill(base_color)
        pygame.display.flip()
        camera_h, camera_w = camera_mask.shape
        projection = water.screen_to_camera_space(screen, H, camera_mask, camera_h, camera_w)
        expected = water.gain_offset_calibration(projection, a_map, b_map)
        ships = {}
        cannonballs = []
        splashes = []
        rng = random.Random()
        clock = pygame.time.Clock()
        while True:
            dt = min(clock.tick(water.FPS) / 1000, 1 / 30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return

            ok, frame = stream.read()
            if ok and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.bitwise_and(gray, gray, mask=camera_mask)
                objects, _ = water.base.shadow_object_detection_calibrated(gray, expected, b_map, None, H_inv)
                shadow_screen = water.base.objects_to_mask(objects, height, width)
                solid = cv2.resize(shadow_screen, (water.GRID_W, water.GRID_H),
                                   interpolation=cv2.INTER_NEAREST) > 0
            flow.step(dt, solid, True)
            bubbles.step(flow, solid, dt, cell_w, cell_h)

            snapshots = {str(item["client_id"]): item for item in state.drain()["players"]}
            sync_ships(ships, snapshots, size)
            for player_id, ship in ships.items():
                snapshot = snapshots[player_id]
                ship.update(snapshot["joystick"], dt, size)
                if ship.hp:
                    pressed = controllers.build_pressed_state(snapshot["events"])
                    for side, button in ((-1, "a"), (1, "b")):
                        if pressed[button] or snapshot["buttons"][button]:
                            shot = ship.fire(side, player_id, size, scale, rng)
                            if shot is not None:
                                cannonballs.append(shot)
            for splash in splashes:
                splash.age += dt
            splashes = [splash for splash in splashes if splash.age < SPLASH_TIME]
            cannonballs = advance_cannonballs(cannonballs, splashes, ships, dt, size, scale)

            screen.fill(base_color)
            bubbles.draw(screen)
            for ship in ships.values():
                if ship.hp:
                    draw_ship(screen, ship, size, scale)
            draw_effects(screen, cannonballs, splashes, size, scale)
            pygame.display.flip()
    finally:
        if websocket is not None:
            websocket.stop()
        if http is not None:
            http.shutdown()
            http.server_close()
        if http_thread is not None:
            http_thread.join(timeout=3)
        if stream is not None:
            stream.release()
        pygame.quit()


if __name__ == "__main__":
    main(**vars(controllers.parse_args(TITLE)))
