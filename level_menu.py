from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from ipaddress import IPv4Address
from queue import Empty, SimpleQueue
from threading import Thread
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pygame
import qrcode
from pygame._sdl2.video import Renderer, Texture, Window

from modules.windows_hotspot import Hotspot, HotspotAddressUnavailable, read_hotspot


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "calibration_cache.npz"
HTTP_PORT = 8000
WS_PORT = 8765

BG = "#10171d"
PANEL = "#1b2730"
HOVER = "#2c3e47"
INK = "#edf4f3"
MUTED = "#b2c0c7"
MINT = "#81e5bb"
GOLD = "#f5cb83"
FONT = "Segoe UI"
MENU_SIZE = (936, 590)
PHONE_SIZE = (624, 496)


@dataclass(frozen=True)
class Level:
    script: str
    title: str
    description: str
    color: str
    phone: bool = False
    controller_path: str = "/controller"


LEVELS = (
    Level("level_1", "Shadow Harbor", "Steer a boat through shadow-driven waves.", "#83d4f2"),
    Level("level_7", "Shadow Platforms", "Jump across shadows and collect falling weapons.", "#a4df9b",
          phone=True, controller_path="/shadow-platforms/controller"),
    Level("level_5", "Parachute Arena", "Ride rising heat and battle with falling weapons.", "#ffb38d", True),
    Level("level_6_bis", "Water Wakes", "Watch a current swirl behind your shadows.", "#72ded9"),
)
EXPERIMENTAL = (
    Level("level_1_bis", "Wind Dock", "Navigate the dock with a wind-driven backdrop.", "#83d4f2"),
    Level("level_2", "Gravity Well", "Bend the star field and send objects flying.", "#bdadfd"),
    Level("level_3", "Heat Glider", "Ride heat with a keyboard-controlled glider.", "#ffb38d"),
    Level("level_4", "Glider Meetup", "Share a heat-powered arena with other players.", "#ffb38d", True),
    Level("level_4_bis", "Glider Skirmish", "Throw stones in a wraparound glider arena.", "#ffb38d", True),
    Level("level_4_tris", "Glider Battle", "Aim and throw with joystick phone controls.", "#ffb38d", True),
    Level("level_6", "Pressure Stream", "Trace the current around shadows with bubbles.", "#72ded9"),
    Level("level_6_tris", "Wind Wakes", "Follow streaks through a turbulent wind field.", "#72ded9"),
    Level("level_8", "Shadow Percussion", "Bounce balls off shadows to play instruments.", "#f5cb83"),
)


def cache_status(camera_index: int) -> tuple[bool, str]:
    if not CACHE.is_file():
        return False, "Run Shadow Calibration to set up the camera and projector."
    try:
        with np.load(CACHE) as data:
            if not {"H", "mask", "a_map", "b_map"}.issubset(data.files):
                raise ValueError("Missing calibration data")
            if "camera_index" in data and int(data["camera_index"]) != camera_index:
                return False, f"Run Shadow Calibration for camera {camera_index}."
    except (OSError, ValueError, TypeError, BadZipFile):
        return False, "The calibration cache could not be read. Run Shadow Calibration again."
    return True, f"Calibration available  •  Camera {camera_index}"


def calibrate(camera_index: int) -> None:
    import cv2
    import pygame

    from level_1 import TARGET_DISPLAY_INDEX, create_fullscreen_display
    from modules.calibration import calibrate_system
    from modules.videostream import get_camera_capture

    pygame.init()
    try:
        screen, width, height, *_ = create_fullscreen_display(TARGET_DISPLAY_INDEX)
        pygame.display.set_caption("Shadow Calibration")
        cap = get_camera_capture(camera_index)
        try:
            H, mask, _, a_map, b_map = calibrate_system(
                cap=cap, screen=screen, WIDTH=width, HEIGHT=height
            )
        finally:
            cap.release()

        with tempfile.NamedTemporaryFile(dir=CACHE.parent, suffix=".npz", delete=False) as file:
            staged = Path(file.name)
        try:
            np.savez_compressed(staged, H=H, mask=mask, a_map=a_map, b_map=b_map,
                                camera_index=camera_index)
            staged.replace(CACHE)
        finally:
            staged.unlink(missing_ok=True)
    finally:
        cv2.destroyAllWindows()
        pygame.quit()


def wifi_payload(ssid: str, password: str) -> str:
    def escape(value: str) -> str:
        return "".join("\\" + char if char in '\\;,:"' else char for char in value)

    return f"WIFI:T:WPA;S:{escape(ssid)};P:{escape(password)};;"


def qr_image(payload: str) -> pygame.Surface:
    qr = qrcode.QRCode(border=4)
    qr.add_data(payload)
    matrix = qr.get_matrix()
    size = len(matrix)
    image = pygame.Surface((size, size), depth=8)
    image.set_palette([(0, 0, 0), (255, 255, 255)])
    pygame.surfarray.blit_array(image, np.asarray(matrix, dtype=np.uint8).T ^ 1)
    scale = max(1, 196 // size)
    return pygame.transform.scale(image, (size * scale, size * scale))


def wifi_qr_from_image(path: str) -> pygame.Surface | None:
    import cv2

    try:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        payload, _, _ = cv2.QRCodeDetector().detectAndDecode(image)
    except (OSError, cv2.error):
        return None
    if not payload.startswith("WIFI:") or len(payload) > 512:
        return None
    if not any(field.startswith("S:") for field in payload[5:].split(";")):
        return None
    return qr_image(payload)


def level_command(level: Level, camera_index: int, host: str | None = None) -> list[str]:
    command = [sys.executable, f"{level.script}.py", "--cached-calibration",
               "--camera-index", str(camera_index)]
    if host is not None:
        command.extend(("--bind-host", host, "--public-host", host,
                        "--http-port", str(HTTP_PORT), "--ws-port", str(WS_PORT)))
    return command


class View:
    def __init__(self, surface: pygame.Surface):
        self.surface = surface
        self.mouse = (-1, -1)
        self.buttons = []
        self.fonts = {}

    def clear(self):
        self.surface.fill(BG)
        self.buttons.clear()

    def text(self, text, position, size=18, color=INK, bold=False):
        key = (size, bold)
        if key not in self.fonts:
            self.fonts[key] = pygame.font.SysFont(FONT, size, bold=bold)
        image = self.fonts[key].render(text, True, color)
        self.surface.blit(image, position)

    def button(self, rect, text, action, accent=None):
        rect = pygame.Rect(rect)
        over = action is not None and rect.collidepoint(self.mouse)
        color = accent or PANEL
        if over:
            color = INK if accent else HOVER
        pygame.draw.rect(self.surface, color, rect, border_radius=8)
        self.text(text, (rect.x + 14, rect.y + (rect.height - 22) // 2),
                  17, BG if accent else (INK if action is not None else MUTED), bold=True)
        if action is not None:
            self.buttons.append((rect, action))

    def click(self, position):
        for rect, action in reversed(self.buttons):
            if rect.collidepoint(position):
                action()
                return


class PhoneSetup:
    def __init__(self, menu: Menu, level: Level):
        self.menu = menu
        self.level = level
        menu_x, menu_y = menu.window.position
        self.window = Window(
            f"{level.title} · Phone setup", size=PHONE_SIZE,
            position=(menu_x + 30, menu_y + 42),
        )
        self.renderer = Renderer(self.window)
        self.view = View(pygame.Surface(PHONE_SIZE))
        self.message = "Checking Windows Mobile Hotspot…"
        self.hotspot_hint = ""
        self.host = ""
        self.url = ""
        self.codes = []
        self.using_pc_hotspot = False
        self.phone_wifi_qr = menu.phone_wifi_qr
        self.qr_notice = ""
        self.server_ready = False
        self.check_at = 0.0
        self.hotspot_check_at = 0.0
        self.hotspot_result: SimpleQueue[Hotspot | None | Exception] | None = None
        self.hotspot_checked = False
        self.launch_failed = False
        self.window.focus()

    def close(self):
        del self.renderer
        self.window.destroy()

    def open_windows_settings(self):
        try:
            os.startfile("ms-settings:network-mobilehotspot")
        except OSError:
            self.message = "Open Network & internet, then Mobile hotspot."

    def check_hotspot(self):
        if self.launch_failed:
            return
        if self.hotspot_result is None:
            if time.monotonic() >= self.hotspot_check_at:
                self.hotspot_result = SimpleQueue()
                Thread(target=self.query_hotspot, args=(self.hotspot_result,), daemon=True).start()
            return
        try:
            result = self.hotspot_result.get_nowait()
        except Empty:
            return
        self.hotspot_result = None
        self.hotspot_checked = True
        self.hotspot_check_at = time.monotonic() + 3
        self.hotspot_hint = ""
        if isinstance(result, HotspotAddressUnavailable):
            self.message = "The hotspot is on, but Windows has not assigned it an IPv4 address."
            self.hotspot_hint = "You can use the PC’s current network while Windows sorts out sharing."
        elif isinstance(result, Exception):
            self.message = "Waiting for Windows to report a Wi-Fi hotspot."
        elif result is None:
            self.message = "The PC’s Mobile Hotspot is off."
            self.hotspot_hint = "If your phone provides the hotspot, use the current Wi-Fi."
        else:
            self.start(result)

    @staticmethod
    def query_hotspot(results):
        try:
            results.put(read_hotspot())
        except HotspotAddressUnavailable as error:
            results.put(error)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            results.put(OSError())

    def start(self, hotspot: Hotspot):
        wifi = qr_image(wifi_payload(hotspot.ssid, hotspot.password))
        self.start_controller(hotspot.address, wifi)

    def use_current_network(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                # Choose the Windows default route without sending any traffic.
                probe.connect(("192.0.2.1", 80))
                address = IPv4Address(probe.getsockname()[0])
            if not address.is_private or address.is_loopback or address.is_link_local:
                raise OSError
        except OSError:
            self.hotspot_hint = "Connect the PC to a local Wi-Fi network and try again."
            return
        self.start_controller(str(address))
        if self.codes and self.phone_wifi_qr is not None:
            self.show_phone_wifi_qr(self.phone_wifi_qr)

    def start_controller(self, address: str, wifi: pygame.Surface | None = None):
        url = f"http://{address}:{HTTP_PORT}{self.level.controller_path}"
        codes = [("Open the controller", qr_image(url))]
        if wifi is not None:
            codes = [("1  Join the PC hotspot", wifi), ("2  Open the controller", codes[0][1])]
        command = level_command(self.level, self.menu.camera_index, address)
        if not self.menu.launch(command, self.level.title):
            self.message = self.menu.status
            self.launch_failed = True
            return
        self.host = address
        self.url = url
        self.codes = codes
        self.using_pc_hotspot = wifi is not None
        self.window.title = f"{self.level.title} · Controller QR codes"
        self.menu.show_phone()
        self.keep_visible()

    def import_wifi_qr(self, path: str):
        if not self.codes or self.using_pc_hotspot:
            return
        wifi = wifi_qr_from_image(path)
        if wifi is None:
            self.qr_notice = "Drop a screenshot containing the phone hotspot’s Wi-Fi QR code."
            return
        self.show_phone_wifi_qr(wifi)

    def show_phone_wifi_qr(self, wifi: pygame.Surface):
        self.codes = [("1  Join the phone hotspot", wifi),
                      ("2  Open the controller", self.codes[-1][1])]
        self.qr_notice = ""

    def keep_visible(self):
        if sys.platform != "win32":
            return
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32")
        user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
        user32.FindWindowW.restype = wintypes.HWND
        handle = user32.FindWindowW(None, self.window.title)
        owner = wintypes.DWORD()
        if not handle or not user32.GetWindowThreadProcessId(wintypes.HWND(handle), ctypes.byref(owner)):
            return
        if owner.value == os.getpid():
            # HWND_TOPMOST, keeping position, size and activation unchanged.
            user32.SetWindowPos(wintypes.HWND(handle), wintypes.HWND(-1), 0, 0, 0, 0, 0x0013)

    def check_server(self):
        now = time.monotonic()
        if now < self.check_at:
            return
        self.check_at = now + 1
        try:
            for port in (HTTP_PORT, WS_PORT):
                with socket.create_connection((self.host, port), timeout=0.08):
                    pass
        except OSError:
            self.server_ready = False
            return
        self.server_ready = True

    def draw(self):
        if not self.codes and self.phone_wifi_qr is None:
            self.check_hotspot()
        view = self.view
        view.clear()
        view.text("Connect your phones", (24, 20), 34, bold=True)
        view.text(self.level.title, (25, 67), 17, MINT)
        if self.codes:
            self.draw_codes()
        else:
            self.draw_hotspot_notice()
        texture = Texture.from_surface(self.renderer, view.surface)
        self.renderer.blit(texture)
        self.renderer.present()

    def draw_hotspot_notice(self):
        view = self.view
        view.text(self.message, (25, 109), 18, MUTED)
        if self.hotspot_hint:
            view.text(self.hotspot_hint, (25, 142), 16, MUTED)
        if sys.platform == "win32" and self.hotspot_checked and not self.launch_failed:
            view.button((24, 179 if self.hotspot_hint else 153, 300, 43),
                        "Open Mobile Hotspot settings", self.open_windows_settings, MINT)
            view.button((24, 235, 300, 43), "Use current Wi-Fi", self.use_current_network)
            view.text("To share your phone’s hotspot QR, choose Use current Wi-Fi first.",
                      (25, 294), 16, MUTED)

    def draw_codes(self):
        self.check_server()
        view = self.view
        using_hotspot = len(self.codes) == 2
        instruction = ("Scan 1 to join Wi-Fi. Then scan 2 once the controller is ready." if using_hotspot
                       else "On the phone hosting the hotspot, scan the controller code.")
        view.text(instruction, (25, 108), 18, MUTED)
        for index, (title, image) in enumerate(self.codes):
            x = 24 + index * 294 if using_hotspot else (PHONE_SIZE[0] - 282) // 2
            rect = pygame.Rect(x, 146, 282, 265)
            pygame.draw.rect(view.surface, PANEL, rect, border_radius=10)
            view.text(title, (x + 18, 157), 19, MINT, bold=True)
            view.surface.blit(image, image.get_rect(center=(rect.centerx, 276)))
            if index == len(self.codes) - 1:
                view.text(self.url, (x + 18, 382), 15, MUTED)
        view.text("Controller ready" if self.server_ready else "Starting the controller server…",
                  (25, 429), 18, MINT if self.server_ready else GOLD)
        if not self.using_pc_hotspot:
            view.text(self.qr_notice or "Drop a screenshot of your phone’s hotspot QR here.",
                      (25, 461), 16, GOLD if self.qr_notice else MUTED)


class LevelControls:
    def __init__(self, title, stop, menu_position):
        width, height = 376, 158
        menu_x, menu_y = menu_position
        self.window = Window("Shadow Arcade · Level controls", size=(width, height),
                             position=(menu_x + MENU_SIZE[0] - width, menu_y + 92))
        self.renderer = Renderer(self.window)
        self.view = View(pygame.Surface((width, height)))
        self.title = title
        self.stop = stop
        self.message = ""

    def close(self):
        del self.renderer
        self.window.destroy()

    def show(self):
        self.window.show()
        self.window.focus()

    def handle(self, event):
        if event.type == pygame.WINDOWCLOSE:
            self.window.hide()
        elif event.type == pygame.MOUSEMOTION:
            self.view.mouse = event.pos
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.view.click(event.pos)

    def draw(self, stopping):
        view = self.view
        view.clear()
        view.text(self.title, (20, 17), 24, bold=True)
        view.text(self.message or "Stop the game and return to the menu.", (20, 54), 16, MUTED)
        view.button((20, 95, 150, 42), "Stop Level", None if stopping else self.stop,
                    None if stopping else GOLD)
        texture = Texture.from_surface(self.renderer, view.surface)
        self.renderer.blit(texture)
        self.renderer.present()


class Menu:
    def __init__(self, camera_index: int, phone_wifi_qr: pygame.Surface | None = None):
        self.camera_index = camera_index
        self.phone_wifi_qr = phone_wifi_qr
        self.child: subprocess.Popen | None = None
        self.child_title = ""
        self.phone: PhoneSetup | None = None
        self.level_controls = None
        self.stop_requested = False
        self.phone_focused = False
        self.experimental = False
        self.calibration_prompt = False
        self.running = True
        self.status = cache_status(camera_index)[1]
        self.view = View(pygame.display.set_mode(MENU_SIZE))
        self.window = Window.from_display_module()
        pygame.display.set_caption("Shadow Arcade")
        pygame.key.stop_text_input()

    def launch(self, command, title, *, calibration=False):
        if self.child is not None:
            return False
        try:
            self.child = subprocess.Popen(command, cwd=ROOT)
        except OSError as error:
            self.status = f"Could not launch {title}. See the terminal."
            print(error, file=sys.stderr)
            return False
        self.child_title = title
        self.stop_requested = False
        self.status = f"{title} is open. Close it to return here."
        if not calibration:
            self.level_controls = LevelControls(title, self.stop_level, self.window.position)
        return True

    def stop_level(self):
        if self.stop_requested or self.child is None or self.child.poll() is not None:
            return
        try:
            self.child.terminate()
        except OSError as error:
            if self.child.poll() is None:
                self.level_controls.message = f"Could not stop {self.child_title}."
                print(error, file=sys.stderr)
            return
        self.stop_requested = True
        self.level_controls.message = "Stopping the level…"

    def poll_child(self):
        if self.child is None:
            return
        code = self.child.poll()
        if code is None:
            return
        self.child = None
        self.close_phone()
        if self.level_controls is not None:
            self.level_controls.close()
            self.level_controls = None
        if self.stop_requested:
            self.status = f"{self.child_title} stopped."
        elif code:
            self.status = f"{self.child_title} ended with an error. See the terminal."
        else:
            self.status = cache_status(self.camera_index)[1]
        self.window.show()
        self.window.focus()

    def start_calibration(self):
        self.calibration_prompt = False
        self.close_phone()
        self.launch([sys.executable, "level_menu.py", "--calibrate",
                     "--camera-index", str(self.camera_index)], "Shadow Calibration", calibration=True)

    def select_level(self, level: Level):
        ready, status = cache_status(self.camera_index)
        if not ready:
            self.status = status
            self.calibration_prompt = True
            return
        self.close_phone()
        if level.phone:
            self.phone = PhoneSetup(self, level)
            self.phone_focused = True
            if self.phone_wifi_qr is not None:
                self.phone.message = "Use the PC’s Wi-Fi connection to your phone’s hotspot."
                self.phone.hotspot_checked = True
                self.phone.use_current_network()
        else:
            self.launch(level_command(level, self.camera_index), level.title)

    def close_phone(self):
        if self.phone is not None:
            self.phone.close()
            self.phone = None
            self.phone_focused = False

    def dismiss_phone(self):
        if self.child is None:
            self.close_phone()
        else:
            self.phone.window.hide()
            self.phone_focused = False

    def show_phone(self):
        self.phone.window.show()
        self.phone.window.focus()
        self.phone_focused = True

    def toggle_section(self):
        self.experimental = not self.experimental

    def open_calibration_prompt(self):
        self.calibration_prompt = True

    def quit(self):
        if self.child is None:
            self.close_phone()
            self.running = False
        else:
            self.status = "Close the active level or calibration first."

    def draw(self):
        view = self.view
        view.clear()
        view.text("Experimental levels" if self.experimental else "Shadow Arcade",
                  (29, 24), 44, bold=True)
        view.button((675, 31, 231, 43),
                    "←  Main menu" if self.experimental else "Experimental levels  →",
                    self.toggle_section)
        pygame.draw.line(view.surface, HOVER, (29, 89), (906, 89))
        levels = EXPERIMENTAL if self.experimental else LEVELS
        for index, level in enumerate(levels):
            self.draw_card(index, level.title, level.description, level.color,
                           "Multiplayer · Phone setup  →" if level.phone else "Launch  →",
                           lambda selected=level: self.select_level(selected))
        if not self.experimental:
            self.draw_card(len(levels), "Shadow Calibration",
                           "Align the camera and projector for every level.", GOLD,
                           "Calibrate  →", self.open_calibration_prompt)
        pygame.draw.line(view.surface, HOVER, (29, 526), (906, 526))
        view.text(self.status, (29, 546), 17, MUTED)
        if self.phone is not None:
            label = "QR codes" if self.phone.codes else "Phone setup"
            view.button((612, 539, 128, 36), label, self.show_phone)
        if self.level_controls is not None:
            view.button((754, 539, 152, 36), "Level controls", self.level_controls.show)
        if self.calibration_prompt:
            self.draw_calibration_prompt()
        pygame.display.flip()

    def draw_card(self, index, title, description, color, action_label, action):
        column, row = index % 3, index // 3
        card_count = len(EXPERIMENTAL) if self.experimental else len(LEVELS) + 1
        compact = card_count > 6
        height, spacing = (124, 137) if compact else (181, 195)
        x, y = 29 + column * 296, 105 + row * spacing
        rect = pygame.Rect(x, y, 285, height)
        view = self.view
        enabled = self.child is None
        fill = HOVER if enabled and rect.collidepoint(view.mouse) else PANEL
        pygame.draw.rect(view.surface, fill, rect, border_radius=11)
        title_y = y + (10 if compact else 19)
        pygame.draw.rect(view.surface, color, (x + 14, title_y + 3, 4, 25), border_radius=2)
        view.text(title, (x + 27, title_y), 23, INK, bold=True)
        words = description.split()
        description_y = title_y + (35 if compact else 45)
        for line, split in enumerate((words[:5], words[5:])):
            view.text(" ".join(split), (x + 18, description_y + line * 19), 16, MUTED)
        view.text(action_label, (x + 18, rect.bottom - 29), 16, color if enabled else MUTED, bold=True)
        if enabled:
            view.buttons.append((rect, action))

    def draw_calibration_prompt(self):
        view = self.view
        view.buttons.clear()
        shade = pygame.Surface(MENU_SIZE, pygame.SRCALPHA)
        shade.fill((0, 0, 0, 165))
        view.surface.blit(shade, (0, 0))
        pygame.draw.rect(view.surface, "#202c36", (147, 150, 642, 277), border_radius=12)
        view.text("Shadow Calibration", (175, 173), 32, GOLD, bold=True)
        for line, text in enumerate((
            "Keep the entire checkerboard and its red top-left dot visible.",
            "When the checkerboard appears, focus the projector window and press Space.",
            "Then focus each OpenCV preview and press a key to advance",
            "through the checkerboard, black and white captures, and final maps.",
        )):
            view.text(text, (175, 231 + 30 * line), 17, INK)
        view.button((490, 370, 117, 40), "Cancel", self.close_calibration_prompt)
        view.button((620, 370, 141, 40), "Calibrate", self.start_calibration, GOLD)

    def close_calibration_prompt(self):
        self.calibration_prompt = False

    def run(self):
        clock = pygame.time.Clock()
        while self.running:
            self.poll_child()
            self.draw()
            if self.phone is not None:
                self.phone.draw()
            if self.level_controls is not None:
                self.level_controls.draw(self.stop_requested)
            for event in pygame.event.get():
                event_window = getattr(event, "window", None)
                window_id = event_window.id if event_window is not None else None
                targets_menu = window_id == self.window.id
                phone = self.phone
                targets_phone = phone is not None and window_id == phone.window.id
                controls = self.level_controls
                if controls is not None and window_id == controls.window.id:
                    if event.type == pygame.WINDOWFOCUSGAINED:
                        self.phone_focused = False
                    controls.handle(event)
                    continue
                if event.type == pygame.WINDOWFOCUSGAINED:
                    self.phone_focused = targets_phone
                if event_window is None and event.type == pygame.KEYDOWN:
                    targets_phone = phone is not None and self.phone_focused
                if event.type == pygame.WINDOWCLOSE and targets_menu:
                    self.quit()
                    if not self.running:
                        return
                elif targets_phone:
                    if event.type == pygame.WINDOWCLOSE:
                        self.dismiss_phone()
                    elif event.type == pygame.MOUSEMOTION:
                        phone.view.mouse = event.pos
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        phone.view.click(event.pos)
                    elif event.type == pygame.DROPFILE:
                        phone.import_wifi_qr(event.file)
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.dismiss_phone()
                elif event.type == pygame.MOUSEMOTION and targets_menu:
                    self.view.mouse = event.pos
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and targets_menu:
                    self.view.click(event.pos)
                    if not self.running:
                        return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and (targets_menu or window_id is None):
                    if self.calibration_prompt:
                        self.close_calibration_prompt()
                    elif self.experimental:
                        self.toggle_section()
            clock.tick(30)


def main():
    parser = argparse.ArgumentParser(description="Choose a Shadow Arcade level or calibrate the projector.")
    parser.add_argument("--camera-index", type=int, default=os.environ.get("CAMERA_INDEX", "1"),
                        help="OpenCV camera index used by the menu and all its levels (default: 1).")
    parser.add_argument("--calibrate", action="store_true", help="Run shadow calibration directly.")
    parser.add_argument("--wifi-qr", metavar="IMAGE", help="Use a screenshot of the phone hotspot’s Wi-Fi QR.")
    args = parser.parse_args()
    if args.calibrate:
        calibrate(args.camera_index)
    else:
        pygame.display.init()
        pygame.font.init()
        try:
            phone_wifi_qr = wifi_qr_from_image(args.wifi_qr) if args.wifi_qr else None
            if args.wifi_qr and phone_wifi_qr is None:
                parser.error("the image does not contain a readable Wi-Fi QR code")
            Menu(args.camera_index, phone_wifi_qr).run()
        finally:
            pygame.quit()


if __name__ == "__main__":
    main()


