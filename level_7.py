from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np
import pygame

import level_4 as platformer
import level_5 as combat


TITLE = "Level 7 Shadow Platforms"
BACKGROUND_COLOR = (128, 128, 128)
DROP_GROUND_LIFETIME = 0.5
SHADOW_STEP_TOLERANCE = 4


class PlatformControllerHandler(combat.TrisControllerRequestHandler):
    controller_path = "/shadow-platforms/controller"
    controller_html = (
        combat.TRIS_CONTROLLER_HTML.replace("Level 5 Parachutes Controller", f"{TITLE} Controller")
        .replace("LEVEL 5 PARACHUTES", TITLE.upper())
        .replace(
            '<button class="button action-button stone"',
            '<button class="button action-button jump" data-button="a" data-vibe="jump">JUMP</button>\n'
            '      <button class="button action-button stone"',
        )
    )
    controller_css = combat.TRIS_CONTROLLER_CSS.replace(
        "grid-template-rows: 1fr;", "grid-template-rows: 1fr 1fr;"
    )
    setup_instructions = (
        "Scan with each phone. Use the joystick to move and aim, JUMP to jump, "
        "and ATTACK / THROW to use your weapon. Press JUMP to rejoin after defeat."
    )


@dataclass
class Arena:
    floor: pygame.Rect
    shadow_tops: np.ndarray

    def set_shadows(self, shadow_mask: np.ndarray) -> None:
        np.not_equal(shadow_mask, 0, out=self.shadow_tops)
        self.shadow_tops[1:] &= shadow_mask[:-1] == 0

    def landing_y(
        self, rect: pygame.Rect, bottom: float, next_bottom: float, grounded: bool = False
    ) -> int | None:
        if next_bottom < bottom:
            return None

        tolerance = SHADOW_STEP_TOLERANCE if grounded else 0
        first_row = max(0, math.ceil(bottom - tolerance))
        last_row = min(self.floor.top - 1, math.floor(next_bottom + tolerance))
        left = max(0, rect.left)
        right = min(self.floor.width, rect.right)
        if first_row <= last_row and left < right:
            rows = np.flatnonzero(self.shadow_tops[first_row:last_row + 1, left:right].any(axis=1))
            if rows.size:
                return first_row + int(rows[0])
        if next_bottom >= self.floor.top:
            return self.floor.top
        return None


@dataclass
class FallingWeapon(combat.WeaponDrop):
    ground_time_left: float | None = None

    def get_collision_rect(self) -> pygame.Rect:
        return self.rect.inflate(6, 6)

    def fall(self, dt: float, arena: Arena) -> bool:
        self.vy = min(self.vy + platformer.PLAYER_GRAVITY * dt, combat.STONE_MAX_FALL_SPEED)
        bottom = self.y + combat.WEAPON_DROP_SIZE[1]
        distance = self.vy * dt
        landing = arena.landing_y(self.rect, bottom, bottom + distance)
        ground_elapsed = dt
        if landing is None:
            self.y += distance
        else:
            self.y = float(landing - self.rect.height)
            if self.ground_time_left is None:
                self.ground_time_left = DROP_GROUND_LIFETIME
                ground_elapsed -= max(0.0, landing - bottom) / self.vy
            self.vy = 0.0

        if self.ground_time_left is None:
            return True
        self.ground_time_left -= ground_elapsed
        return self.ground_time_left > 1e-9


@dataclass
class PlatformPlayer(combat.RemotePlayer):
    on_ground: bool = True
    parachute_open: bool = False

    def reset(self) -> None:
        super().reset()
        self.parachute_open = False
        self.on_ground = True

    def get_combat_rect(self) -> pygame.Rect:
        return self.rect

    def drop_weapon(self) -> FallingWeapon:
        return FallingWeapon(
            self.weapon,
            self.x + (self.width - combat.WEAPON_DROP_SIZE[0]) / 2,
            self.y,
        )

    def move(
        self,
        dt: float,
        held: dict[str, bool],
        pressed: dict[str, bool],
        joystick: dict[str, float],
        arena: Arena,
    ) -> combat.StoneProjectile | None:
        if pressed["start"] or (pressed["a"] and not self.alive and not self.falling):
            self.reset()
            return None

        self.stone_cooldown = max(0.0, self.stone_cooldown - dt)
        self.accolade_flash = max(0.0, self.accolade_flash - dt * 2.4)
        self.attack_flash = max(0.0, self.attack_flash - dt)
        if not self.alive:
            if self.falling:
                self.x += self.vx * dt
                platformer.clamp_remote_player(self, arena.floor.width)
                if self.fall(dt, arena):
                    self.falling = False
            return None

        move_dir = int(held["right"]) - int(held["left"])
        if move_dir:
            self.facing = move_dir
        self.aim_x, self.aim_y = joystick["x"], joystick["y"]
        projectile = self.attack() if pressed["b"] else None

        if pressed["a"] and (self.on_ground or self.coyote_timer > 0):
            self.vy = platformer.PLAYER_JUMP_SPEED
            self.on_ground = False
            self.coyote_timer = 0.0

        self.vx = move_dir * platformer.PLAYER_MOVE_SPEED
        self.x += self.vx * dt
        platformer.clamp_remote_player(self, arena.floor.width)
        self.on_ground = self.fall(dt, arena)
        if self.on_ground:
            self.coyote_timer = platformer.PLAYER_COYOTE_TIME
        else:
            self.coyote_timer = max(0.0, self.coyote_timer - dt)
        return projectile

    def fall(self, dt: float, arena: Arena) -> bool:
        self.vy = min(self.vy + platformer.PLAYER_GRAVITY * dt, platformer.PLAYER_MAX_FALL_SPEED)
        bottom = self.y + self.height
        landing = arena.landing_y(self.rect, bottom, bottom + self.vy * dt, self.on_ground)
        if landing is not None:
            self.y = float(landing - self.height)
            self.vy = 0.0
            return True
        self.y += self.vy * dt
        return False

    def kill(self) -> None:
        super().kill()
        self.on_ground = False


def main(
    cached_calibration: bool = False,
    bind_host: str = "0.0.0.0",
    public_host: str | None = None,
    http_port: int = 8000,
    ws_port: int = 8765,
    camera_index: int = 1,
) -> None:
    cap_stream = http_server = http_thread = ws_server = None
    try:
        pygame.init()
        screen, width, height, *_ = combat.create_fullscreen_display(combat.TARGET_DISPLAY_INDEX)
        pygame.display.set_caption(TITLE)
        clock = pygame.time.Clock()
        label_font = pygame.font.SysFont("Consolas", 13, bold=True)

        if cached_calibration:
            calibration = combat.load_calibration(combat.CALIBRATION_CACHE)
            if calibration is None:
                print(f"No calibration cache found at {combat.CALIBRATION_CACHE}.")
                return
            H, mask, a_map, b_map = calibration
        else:
            cap = combat.get_camera_capture(camera_index)
            try:
                H, mask, _, a_map, b_map = combat.calibrate_system(
                    cap=cap, screen=screen, WIDTH=width, HEIGHT=height
                )
                combat.save_calibration(combat.CALIBRATION_CACHE, H, mask, a_map, b_map)
            finally:
                cap.release()

        H_inv = np.linalg.inv(H)
        cap_stream = combat.VideoStream(camera_index)
        controller_state = combat.TrisControllerState()
        config = combat.ServerConfig(
            bind_host=bind_host,
            public_host=public_host or combat.detect_public_host(),
            http_port=http_port,
            ws_port=ws_port,
            app_title=TITLE,
            controller_path=PlatformControllerHandler.controller_path,
            helper_note_html="Joystick moves and aims<br>JUMP jumps<br>ATTACK / THROW uses your weapon",
        )
        http_server, http_thread = combat.start_tris_http_server(
            controller_state, config, PlatformControllerHandler
        )
        ws_server = combat.TrisControllerWebSocketServer(controller_state, bind_host, ws_port)
        ws_server.start()
        print(f"Setup page: {config.setup_url}")
        print(f"Controller page: {config.controller_url}")
        print("Local controls: ESC quits, V toggles the shadow overlay.")

        floor = pygame.Rect(0, height - combat.PLATFORM_HEIGHT, width, combat.PLATFORM_HEIGHT)
        arena = Arena(floor, np.zeros((height, width), dtype=bool))
        spawn_points = platformer.build_spawn_points(width, floor)
        players: dict[str, PlatformPlayer] = {}
        projectiles: list[combat.StoneProjectile] = []
        drops: list[FallingWeapon] = []
        drop_timer = 2.5
        show_overlay = False
        shadows = []
        exclusion = pygame.Surface(screen.get_size())
        previous_mask = None

        def draw_scene() -> None:
            screen.fill(BACKGROUND_COLOR)
            if show_overlay:
                for obj in shadows:
                    pygame.draw.polygon(screen, (103, 118, 135), obj["polygon_pygame"])
            combat.draw_platforms(screen, [floor])

        def update_shadows() -> None:
            nonlocal shadows, previous_mask
            ret, frame = cap_stream.read()
            if not ret or frame is None:
                return
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h_cam, w_cam = gray.shape
            projected = combat.screen_to_camera_space(screen, H, mask, h_cam, w_cam)
            expected = combat.gain_offset_calibration(projected, a_map, b_map)
            exclusion.fill((0, 0, 0))
            for player in players.values():
                if player.alive or player.falling:
                    pygame.draw.rect(exclusion, (255, 255, 255), player.rect)
            player_mask = combat.surface_to_camera_gray_masked(exclusion, H, mask, h_cam, w_cam) > 0
            excluded = player_mask if previous_mask is None else player_mask | previous_mask
            shadows, contours_vis, shadow_mask = combat.shadow_object_detection_calibrated(
                cv2.bitwise_and(gray, gray, mask=mask), expected, b_map, excluded, H_inv
            )
            arena.set_shadows(combat.objects_to_mask(shadows, height, width))
            previous_mask = player_mask

            if combat.SHOW_DEBUG_WINDOWS:
                cv2.imshow("Camera View", frame)
                cv2.imshow("Expected Camera Image", expected)
                cv2.imshow("Contours", contours_vis)
                cv2.imshow("Shadow Mask", shadow_mask)
                cv2.waitKey(1)

        def draw_countdown_frame(_dt: float) -> bool:
            update_shadows()
            draw_scene()
            return True

        draw_scene()
        pygame.display.flip()
        if not combat.run_countdown(
            screen, clock, combat.COUNTDOWN_SECONDS, "SHADOW PLATFORMS",
            pygame.font.SysFont("Arial", 64, bold=True),
            pygame.font.SysFont("Arial", 120, bold=True),
            draw_frame=draw_countdown_frame,
        ):
            return
        draw_scene()
        pygame.display.flip()

        while True:
            dt = min(clock.tick(combat.FPS) / 1000.0, 0.1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key == pygame.K_v:
                        show_overlay = not show_overlay

            update_shadows()
            snapshots = {
                str(player["client_id"]): player for player in controller_state.drain()["players"]
            }
            combat.sync_remote_players(players, snapshots, spawn_points, PlatformPlayer)
            player_list = list(players.values())

            drop_timer -= dt
            if drop_timer <= 0.0:
                weapon = random.choice(("stone", "sword", "bow"))
                drop_x = random.uniform(0, max(0, width - combat.WEAPON_DROP_SIZE[0]))
                drops.append(FallingWeapon(weapon, drop_x, float(-combat.WEAPON_DROP_SIZE[1])))
                drop_timer = combat.WEAPON_DROP_INTERVAL * random.uniform(0.75, 1.25)

            for player_id in sorted(snapshots):
                snapshot = snapshots[player_id]
                projectile = players[player_id].move(
                    dt, snapshot["buttons"], combat.build_pressed_state(snapshot["events"]),
                    snapshot["joystick"], arena,
                )
                if projectile is not None:
                    projectiles.append(projectile)
            platformer.resolve_player_collisions(
                [player for player in player_list if player.alive], width
            )
            drops = [
                drop for drop in drops
                if drop.fall(dt, arena) and not combat.try_pick_up_weapon(drop, player_list)
            ]
            combat.resolve_sword_attacks(player_list, drops)
            projectiles = combat.update_stones(projectiles, players, [floor], dt, width, height, drops)

            draw_scene()
            for projectile in projectiles:
                combat.draw_stone(screen, projectile)
            for drop in drops:
                combat.draw_weapon_pickup(screen, drop.rect, drop.weapon)
            for player in sorted(player_list, key=lambda player: player.rect.bottom):
                combat.draw_remote_player(screen, player, label_font)
            pygame.display.flip()
    finally:
        if cap_stream is not None:
            cap_stream.release()
        pygame.quit()
        cv2.destroyAllWindows()
        if ws_server is not None:
            ws_server.stop()
        if http_server is not None:
            http_server.shutdown()
            http_server.server_close()
        if http_thread is not None:
            http_thread.join(timeout=3.0)


if __name__ == "__main__":
    main(**vars(combat.parse_args("Level 7 - shadow platformer")))
