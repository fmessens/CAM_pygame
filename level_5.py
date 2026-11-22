from __future__ import annotations

import argparse
import html
import io
import json
import random
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import cv2
import numpy as np
import pygame

from level_4 import (
    BUTTON_NAMES,
    CALIBRATION_CACHE,
    COUNTDOWN_SECONDS,
    FPS,
    GRID_H,
    GRID_W,
    LIFT_GLOW_COLOR,
    PARACHUTE_CANOPY_HEIGHT,
    PARACHUTE_CANOPY_WIDTH,
    PARACHUTE_GRAVITY_SCALE,
    PARACHUTE_LIFT_ACCEL,
    PARACHUTE_LIFT_THRESHOLD,
    PARACHUTE_MAX_FALL_SPEED,
    PARACHUTE_RISE_SPEED_MAX,
    PARACHUTE_RISE_SPEED_MIN,
    PLATFORM_HEIGHT,
    PLAYER_COYOTE_TIME,
    PLAYER_GRAVITY,
    PLAYER_HEAD_COLOR,
    PLAYER_JUMP_SPEED,
    PLAYER_MAX_FALL_SPEED,
    PLAYER_MOVE_SPEED,
    PLAYER_OUTLINE_COLOR,
    PLAYER_SIZE,
    SHOW_DEBUG_WINDOWS,
    TARGET_DISPLAY_INDEX,
    ControllerState,
    ControllerWebSocketServer,
    PLAYER_COLLISION_PASSES,
    ServerConfig,
    VideoStream,
    apply_heat_tint,
    build_pressed_state,
    calibrate_system,
    clamp,
    compute_heat_distortion,
    create_fullscreen_display,
    detect_public_host,
    draw_platforms,
    gain_offset_calibration,
    get_camera_capture,
    lighten_color,
    load_calibration,
    next_spawn_slot,
    objects_to_mask,
    palette_for_index,
    run_countdown,
    sample_heat_region,
    save_calibration,
    screen_to_camera_space,
    shadow_object_detection_calibrated,
    surface_to_camera_gray_masked,
    update_heat_field,
)

try:
    import qrcode
    import qrcode.image.svg
except ImportError:  # pragma: no cover - handled at runtime
    qrcode = None


STONE_THROW_SPEED = 420.0
STONE_THROW_LIFT = -150.0
STONE_GRAVITY = 760.0
STONE_MAX_FALL_SPEED = 520.0
STONE_RADIUS = 9
STONE_LIFETIME = 3.2
STONE_COOLDOWN = 0.55
ACCOLADE_BASE_COLOR = (214, 162, 52)
ACCOLADE_TIP_COLOR = (255, 232, 168)
ACCOLADE_MAX_VISIBLE = 5
AIM_DEADZONE = 0.35
JOYSTICK_DEADZONE = 0.14
WEAPON_DROP_INTERVAL = 7.0
WEAPON_DROP_SPEED = 58.0
WEAPON_DROP_SIZE = (38, 32)
BOW_PROJECTILE_SPEED = 760.0
BOW_COOLDOWN = 0.32
SWORD_COOLDOWN = 0.45
SWORD_REACH = 92
FALLING_PLAYER_GRAVITY = 900.0


TRIS_CONTROLLER_CSS = """
:root {
  color-scheme: dark;
  --bg: #10130f;
  --panel: #202820;
  --panel-2: #2c382c;
  --edge: #5d7055;
  --ink: #f1f4e6;
  --muted: #a9b69c;
  --stick: #dfe6cd;
  --stick-base: #313d31;
  --stone: #9b9187;
  --chute: #5ea5cf;
  --jump: #d66f3f;
  --shadow: rgba(0, 0, 0, 0.36);
}

* {
  box-sizing: border-box;
  -webkit-tap-highlight-color: transparent;
  touch-action: none;
  user-select: none;
}

html,
body {
  width: 100%;
  height: 100%;
  margin: 0;
  overflow: hidden;
}

body {
  font-family: Consolas, "Lucida Console", monospace;
  color: var(--ink);
  background:
    linear-gradient(120deg, rgba(94, 165, 207, 0.18), transparent 42%),
    linear-gradient(180deg, #171d16 0%, var(--bg) 100%);
}

.layout {
  width: 100%;
  height: 100%;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(136px, 28vw);
  gap: 12px;
  padding: 12px 12px calc(12px + env(safe-area-inset-bottom));
}

.play-zone {
  min-width: 0;
  min-height: 0;
  display: grid;
  grid-template-rows: auto 1fr;
  gap: 10px;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  min-height: 44px;
}

.title {
  font-size: clamp(0.8rem, 1.6vw, 1rem);
  letter-spacing: 0.16em;
  color: var(--muted);
  white-space: nowrap;
}

.status {
  min-width: 108px;
  padding: 8px 10px;
  border: 2px solid var(--edge);
  border-radius: 8px;
  text-align: center;
  font-size: clamp(0.72rem, 1.6vw, 0.88rem);
  letter-spacing: 0.12em;
  background: rgba(255, 255, 255, 0.08);
}

.status.online {
  color: #dcf1cf;
  background: rgba(108, 166, 86, 0.24);
}

.status.offline {
  color: #f4d2a5;
  background: rgba(194, 119, 52, 0.24);
}

.rotate-warning {
  display: none;
}

.joystick-panel {
  min-height: 0;
  display: grid;
  place-items: center;
  border: 3px solid var(--edge);
  border-radius: 8px;
  background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
  box-shadow: 0 18px 42px var(--shadow);
}

.joystick {
  position: relative;
  width: min(58vh, 58vw);
  height: min(58vh, 58vw);
  min-width: 180px;
  min-height: 180px;
  max-width: 430px;
  max-height: 430px;
  border-radius: 50%;
  border: 4px solid rgba(255, 255, 255, 0.16);
  background:
    linear-gradient(90deg, transparent calc(50% - 1px), rgba(255, 255, 255, 0.18) 50%, transparent calc(50% + 1px)),
    linear-gradient(0deg, transparent calc(50% - 1px), rgba(255, 255, 255, 0.18) 50%, transparent calc(50% + 1px)),
    radial-gradient(circle, #3a493a 0%, var(--stick-base) 64%, #1b221b 100%);
}

.joystick::before {
  content: "";
  position: absolute;
  inset: 22%;
  border: 2px dashed rgba(255, 255, 255, 0.18);
  border-radius: 50%;
}

.stick {
  position: absolute;
  left: 50%;
  top: 50%;
  width: 34%;
  height: 34%;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  background:
    radial-gradient(circle at 35% 28%, rgba(255, 255, 255, 0.38), transparent 24%),
    linear-gradient(180deg, #f7f8ed 0%, var(--stick) 100%);
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.34);
}

.action-rail {
  min-height: 0;
  display: grid;
  grid-template-rows: 1fr;
  gap: 10px;
}

.button {
  border: 0;
  outline: 0;
  cursor: pointer;
  font: inherit;
  color: var(--ink);
}

.action-button {
  width: 100%;
  min-height: 0;
  border-radius: 8px;
  border: 3px solid rgba(255, 255, 255, 0.18);
  display: grid;
  place-items: center;
  padding: 8px;
  text-align: center;
  font-weight: 700;
  line-height: 1.05;
  letter-spacing: 0.08em;
  font-size: clamp(0.86rem, 2vw, 1.3rem);
  box-shadow:
    inset 0 2px 0 rgba(255, 255, 255, 0.18),
    0 14px 30px var(--shadow);
}

.action-button.stone {
  background: linear-gradient(180deg, #c7beb5 0%, var(--stone) 100%);
  color: #28231f;
}

.action-button.chute {
  background: linear-gradient(180deg, #79c4ec 0%, var(--chute) 100%);
  color: #102636;
}

.action-button.jump {
  background: linear-gradient(180deg, #f08a55 0%, var(--jump) 100%);
  color: #2b130b;
  font-size: clamp(1.05rem, 2.6vw, 1.65rem);
}

.is-active.action-button,
.joystick.is-active .stick {
  filter: brightness(0.86) saturate(1.08);
}

.is-active.action-button {
  transform: translateY(3px);
}

@media (orientation: portrait) {
  .rotate-warning {
    position: fixed;
    inset: 0;
    z-index: 20;
    display: grid;
    place-items: center;
    padding: 24px;
    background: rgba(16, 19, 15, 0.9);
    backdrop-filter: blur(8px);
  }

  .rotate-card {
    max-width: 320px;
    padding: 24px;
    border: 3px solid var(--edge);
    border-radius: 8px;
    background: linear-gradient(180deg, var(--panel-2) 0%, var(--panel) 100%);
    text-align: center;
    box-shadow: 0 18px 42px var(--shadow);
  }

  .rotate-card h1 {
    margin: 0 0 10px;
    font-size: 1.35rem;
  }

  .rotate-card p {
    margin: 0;
    color: var(--muted);
    line-height: 1.35;
  }
}
"""


TRIS_CONTROLLER_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover">
  <title>Level 5 Parachutes Controller</title>
  <link rel="stylesheet" href="/static/tris-controller.css">
</head>
<body>
  <script src="/config.js"></script>
  <div class="rotate-warning">
    <div class="rotate-card">
      <h1>Rotate to landscape</h1>
      <p>The controller is built for two-thumb play.</p>
    </div>
  </div>
  <main class="layout">
    <section class="play-zone">
      <header class="topbar">
        <span class="title" id="app-title">LEVEL 5 PARACHUTES</span>
        <div class="status" id="connection-status">CONNECTING</div>
      </header>
      <section class="joystick-panel">
        <div class="joystick" id="joystick" aria-label="Movement joystick">
          <div class="stick" id="stick"></div>
        </div>
      </section>
    </section>

    <section class="action-rail">
      <button class="button action-button stone" data-button="b" data-vibe="stone">ATTACK<br>THROW</button>
    </section>
  </main>
  <script src="/static/tris-controller.js"></script>
</body>
</html>
"""


TRIS_CONTROLLER_JS = """
const config = window.CONTROLLER_CONFIG || {};
const buttonNames = config.buttonNames || ["up", "down", "left", "right", "a", "b", "start", "select"];

const appTitle = document.getElementById("app-title");
const statusEl = document.getElementById("connection-status");
const joystick = document.getElementById("joystick");
const stick = document.getElementById("stick");
const actionButtons = Array.from(document.querySelectorAll("[data-button]"));

let socket = null;
let reconnectTimer = null;
let assignedLabel = null;
let joystickPointerId = null;
const deviceId = getDeviceId();
const heldButtons = new Map(buttonNames.map((button) => [button, false]));
const activePointers = new Map();

appTitle.textContent = (config.appTitle || "LEVEL 5 PARACHUTES").toUpperCase();

function socketUrl() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.hostname}:${config.wsPort}?device=${encodeURIComponent(deviceId)}`;
}

function createDeviceId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `tris-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getDeviceId() {
  const storageKey = "mobile-webpad-device-id";
  try {
    const existing = window.localStorage.getItem(storageKey);
    if (existing) {
      return existing;
    }
    const created = createDeviceId();
    window.localStorage.setItem(storageKey, created);
    return created;
  } catch (error) {
    return createDeviceId();
  }
}

function vibrate(pattern) {
  if (navigator.vibrate) {
    navigator.vibrate(pattern);
  }
}

function vibrationFor(element) {
  const kind = element.dataset.vibe;
  if (kind === "stone") {
    return [18, 28, 24];
  }
  if (kind === "chute") {
    return [45];
  }
  return [80];
}

function setStatus(text, className) {
  statusEl.textContent = text;
  statusEl.classList.remove("online", "offline");
  if (className) {
    statusEl.classList.add(className);
  }
}

function onlineStatusText() {
  return assignedLabel ? `${assignedLabel} ONLINE` : "ONLINE";
}

function sendMessage(message) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    return;
  }
  socket.send(JSON.stringify(message));
}

function updateButtonVisual(buttonName, active) {
  for (const element of document.querySelectorAll(`[data-button="${buttonName}"]`)) {
    element.classList.toggle("is-active", active);
  }
}

function pressButton(buttonName) {
  if (heldButtons.get(buttonName)) {
    return;
  }
  heldButtons.set(buttonName, true);
  updateButtonVisual(buttonName, true);
  sendMessage({ type: "button", button: buttonName, pressed: true });
}

function releaseButton(buttonName) {
  if (!heldButtons.get(buttonName)) {
    return;
  }
  heldButtons.set(buttonName, false);
  updateButtonVisual(buttonName, false);
  sendMessage({ type: "button", button: buttonName, pressed: false });
}

function setButton(buttonName, active) {
  if (active) {
    pressButton(buttonName);
  } else {
    releaseButton(buttonName);
  }
}

function setJoystickButtons(x, y) {
  const horizontal = 0.28;
  setButton("left", x < -horizontal);
  setButton("right", x > horizontal);
}

function sendJoystickVector(x, y) {
  sendMessage({ type: "joystick", x, y });
}

function resetJoystick() {
  joystickPointerId = null;
  joystick.classList.remove("is-active");
  stick.style.transform = "translate(-50%, -50%)";
  setJoystickButtons(0, 0);
  sendJoystickVector(0, 0);
}

function updateJoystick(event) {
  const rect = joystick.getBoundingClientRect();
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  const radius = Math.min(rect.width, rect.height) / 2;
  let x = (event.clientX - cx) / radius;
  let y = (event.clientY - cy) / radius;
  const length = Math.hypot(x, y);
  if (length > 1) {
    x /= length;
    y /= length;
  }
  if (length < 0.14) {
    x = 0;
    y = 0;
  }
  stick.style.transform = `translate(calc(-50% + ${x * radius * 0.58}px), calc(-50% + ${y * radius * 0.58}px))`;
  setJoystickButtons(x, y);
  sendJoystickVector(x, y);
}

function releasePointer(pointerId) {
  const buttonName = activePointers.get(pointerId);
  if (!buttonName) {
    return;
  }
  activePointers.delete(pointerId);
  const stillHeld = Array.from(activePointers.values()).includes(buttonName);
  if (!stillHeld) {
    releaseButton(buttonName);
  }
}

function releaseAll(reason = "release_all") {
  activePointers.clear();
  resetJoystick();
  for (const buttonName of buttonNames) {
    if (heldButtons.get(buttonName)) {
      heldButtons.set(buttonName, false);
      updateButtonVisual(buttonName, false);
    }
  }
  sendMessage({ type: "release_all", reason });
}

function connect() {
  setStatus("CONNECTING", "offline");
  socket = new WebSocket(socketUrl());

  socket.addEventListener("open", () => {
    assignedLabel = null;
    setStatus("ONLINE", "online");
    sendMessage({
      type: "hello",
      deviceId,
      userAgent: navigator.userAgent,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
      },
    });
  });

  socket.addEventListener("message", (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type === "hello") {
        assignedLabel = message.label || null;
        if (Array.isArray(message.buttons)) {
          for (const buttonName of message.buttons) {
            if (!heldButtons.has(buttonName)) {
              heldButtons.set(buttonName, false);
            }
          }
        }
        setStatus(onlineStatusText(), "online");
      }
    } catch (error) {
      return;
    }
  });

  socket.addEventListener("close", () => {
    assignedLabel = null;
    setStatus("RECONNECTING", "offline");
    releaseAll("socket_closed");
    window.clearTimeout(reconnectTimer);
    reconnectTimer = window.setTimeout(connect, 800);
  });

  socket.addEventListener("error", () => {
    setStatus("ERROR", "offline");
  });
}

joystick.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  joystickPointerId = event.pointerId;
  joystick.setPointerCapture(event.pointerId);
  joystick.classList.add("is-active");
  vibrate(20);
  updateJoystick(event);
});

joystick.addEventListener("pointermove", (event) => {
  if (event.pointerId !== joystickPointerId) {
    return;
  }
  event.preventDefault();
  updateJoystick(event);
});

const releaseJoystick = (event) => {
  if (event.pointerId !== joystickPointerId) {
    return;
  }
  event.preventDefault();
  resetJoystick();
};

joystick.addEventListener("pointerup", releaseJoystick);
joystick.addEventListener("pointercancel", releaseJoystick);
joystick.addEventListener("lostpointercapture", releaseJoystick);

for (const element of actionButtons) {
  element.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    const buttonName = element.dataset.button;
    activePointers.set(event.pointerId, buttonName);
    element.setPointerCapture(event.pointerId);
    vibrate(vibrationFor(element));
    pressButton(buttonName);
  });

  const onRelease = (event) => {
    event.preventDefault();
    releasePointer(event.pointerId);
  };

  element.addEventListener("pointerup", onRelease);
  element.addEventListener("pointercancel", onRelease);
  element.addEventListener("lostpointercapture", onRelease);
}

window.addEventListener("blur", () => releaseAll("window_blur"));
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "visible") {
    releaseAll("visibility_change");
  }
});
window.addEventListener("contextmenu", (event) => event.preventDefault());
connect();
"""


def wrap_scalar(position: float, size: int, limit: int) -> float:
    span = float(limit + size)
    while position >= limit:
        position -= span
    while position + size <= 0.0:
        position += span
    return position


def wrap_player_position(player: "RemotePlayer", screen_width: int, screen_height: int) -> None:
    player.x = wrap_scalar(player.x, player.width, screen_width)
    player.y = wrap_scalar(player.y, player.height, screen_height)


def build_spawn_points(width: int, height: int) -> list[tuple[float, float]]:
    """Spread parachutes through open air; level 5 has no platforms."""
    return [
        (
            float(width * x_ratio - PLAYER_SIZE[0] / 2.0),
            float(height * y_ratio),
        )
        for y_ratio in (0.22, 0.52)
        for x_ratio in (0.14, 0.38, 0.62, 0.86)
    ]


class TrisControllerState(ControllerState):
    def __init__(self) -> None:
        super().__init__()
        self._aim_by_client: dict[str, tuple[float, float]] = {}

    def apply_joystick(
        self,
        client_id: str,
        x: float,
        y: float,
        *,
        connection_id: int | None = None,
    ) -> bool:
        x = clamp(float(x), -1.0, 1.0)
        y = clamp(float(y), -1.0, 1.0)
        length = (x * x + y * y) ** 0.5
        if length > 1.0:
            x /= length
            y /= length
        if length < JOYSTICK_DEADZONE:
            x = 0.0
            y = 0.0

        with self._lock:
            session = self._sessions.get(client_id)
            if session is None:
                return False
            if connection_id is not None and session.connection_id != connection_id:
                return False
            self._aim_by_client[client_id] = (x, y)
            session.last_input_time = time.monotonic()
            return True

    def release_all(
        self,
        client_id: str,
        source: str = "system",
        *,
        connection_id: int | None = None,
    ) -> bool:
        released = super().release_all(
            client_id,
            source=source,
            connection_id=connection_id,
        )
        self.apply_joystick(client_id, 0.0, 0.0, connection_id=connection_id)
        return released

    def client_disconnected(self, client_id: str, *, connection_id: int | None = None) -> int:
        count = super().client_disconnected(client_id, connection_id=connection_id)
        with self._lock:
            self._aim_by_client.pop(client_id, None)
        return count

    def drain(self) -> dict[str, object]:
        snapshot = super().drain()
        with self._lock:
            for player in snapshot["players"]:
                client_id = str(player["client_id"])
                aim_x, aim_y = self._aim_by_client.get(client_id, (0.0, 0.0))
                player["joystick"] = {"x": aim_x, "y": aim_y}
        return snapshot


class TrisControllerWebSocketServer(ControllerWebSocketServer):
    async def _handle_message(
        self,
        client_id: str,
        message: dict[str, object],
        *,
        connection_id: int,
    ) -> None:
        kind = str(message.get("type", "")).lower()
        if kind == "joystick" and isinstance(self.controller_state, TrisControllerState):
            try:
                x = float(message.get("x", 0.0))
                y = float(message.get("y", 0.0))
            except (TypeError, ValueError):
                return
            self.controller_state.apply_joystick(
                client_id,
                x,
                y,
                connection_id=connection_id,
            )
            return

        await super()._handle_message(
            client_id,
            message,
            connection_id=connection_id,
        )


class TrisControllerHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        config: ServerConfig,
        controller_state: ControllerState,
        request_handler: type[BaseHTTPRequestHandler] | None = None,
    ) -> None:
        super().__init__(server_address, request_handler or TrisControllerRequestHandler)
        self.config = config
        self.controller_state = controller_state


class TrisControllerRequestHandler(BaseHTTPRequestHandler):
    server: TrisControllerHttpServer
    server_version = "MobileTrisPad/0.1"
    controller_html = TRIS_CONTROLLER_HTML
    controller_css = TRIS_CONTROLLER_CSS
    controller_path = "/controller"
    setup_instructions = "Scan with each phone. The controller uses a joystick plus three large action buttons."

    def do_GET(self) -> None:  # noqa: N802 - stdlib interface
        path = urlsplit(self.path).path

        if path in {"/", "/setup"}:
            self._send_bytes(self._build_setup_page().encode("utf-8"), "text/html; charset=utf-8")
            return
        if path.rstrip("/") in {"/controller", self.controller_path}:
            self._send_bytes(self.controller_html.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/config.js":
            self._send_bytes(self._build_config_js().encode("utf-8"), "application/javascript; charset=utf-8")
            return
        if path == "/status.json":
            snapshot = self.server.controller_state.snapshot()
            self._send_bytes(json.dumps(snapshot).encode("utf-8"), "application/json; charset=utf-8")
            return
        if path == "/qr.svg":
            qr_svg = self._build_qr_svg()
            if qr_svg is None:
                self.send_error(503, "QR generation unavailable.")
                return
            self._send_bytes(qr_svg, "image/svg+xml")
            return
        if path == "/static/tris-controller.css":
            self._send_bytes(self.controller_css.encode("utf-8"), "text/css; charset=utf-8")
            return
        if path == "/static/tris-controller.js":
            self._send_bytes(TRIS_CONTROLLER_JS.encode("utf-8"), "application/javascript; charset=utf-8")
            return

        self.send_error(404, "Not found")

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _build_setup_page(self) -> str:
        config = self.server.config
        controller_url = html.escape(config.controller_url)
        setup_url = html.escape(config.setup_url)
        qr_block = (
            '<img class="qr" src="/qr.svg" alt="QR code for the controller page">'
            if qrcode is not None
            else '<div class="qr-fallback">Install `qrcode` to enable the QR preview.</div>'
        )
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(config.app_title)} Setup</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #121712;
      --panel: #232d22;
      --ink: #eef4df;
      --accent: #8fd2ef;
      --muted: #adbaa1;
      --edge: #61735a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
      font-family: Consolas, "Lucida Console", monospace;
      background: linear-gradient(180deg, #182017 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .card {{
      width: min(900px, 100%);
      display: grid;
      grid-template-columns: 300px 1fr;
      gap: 24px;
      padding: 24px;
      border: 3px solid var(--edge);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: 0 28px 80px rgba(0, 0, 0, 0.35);
    }}
    .qr-wrap {{
      display: grid;
      place-items: center;
      min-height: 300px;
      border: 2px solid var(--edge);
      border-radius: 8px;
      background: #f5f7eb;
      padding: 18px;
    }}
    .qr {{
      width: 100%;
      max-width: 240px;
      height: auto;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: clamp(1.6rem, 4vw, 2.4rem);
      line-height: 1.05;
    }}
    p {{
      margin: 0 0 14px;
      color: var(--muted);
      line-height: 1.5;
    }}
    .link-box {{
      margin: 18px 0;
      padding: 14px;
      border-radius: 8px;
      border: 2px solid var(--edge);
      background: rgba(0, 0, 0, 0.15);
      word-break: break-all;
    }}
    .link-box strong {{
      display: block;
      margin-bottom: 8px;
      color: var(--ink);
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    .status {{
      margin-top: 18px;
      padding: 14px;
      border-radius: 8px;
      border: 1px solid var(--edge);
      background: rgba(0, 0, 0, 0.18);
    }}
    .status-line {{
      margin: 6px 0;
      color: var(--muted);
    }}
    .qr-fallback {{
      color: #24311a;
      text-align: center;
      line-height: 1.4;
    }}
    @media (max-width: 720px) {{
      .card {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="card">
    <section class="qr-wrap">{qr_block}</section>
    <section>
      <h1>{html.escape(config.app_title)}</h1>
      <p>{self.setup_instructions}</p>
      <div class="link-box"><strong>Controller URL</strong><a href="{controller_url}">{controller_url}</a></div>
      <div class="link-box"><strong>Setup page URL</strong><a href="{setup_url}">{setup_url}</a></div>
      <div class="status">
        <div class="status-line" id="connection-status">Players: checking...</div>
        <div class="status-line" id="active-players">Joined: none</div>
        <div class="status-line" id="last-event">Last event: waiting</div>
      </div>
    </section>
  </main>
  <script>
    async function refreshStatus() {{
      try {{
        const response = await fetch('/status.json', {{ cache: 'no-store' }});
        const data = await response.json();
        const labels = (data.active_players || []).map((player) => player.label).join(', ') || 'none';
        document.getElementById('connection-status').textContent = `Players: ${{data.connected_clients}} joined`;
        document.getElementById('active-players').textContent = `Joined: ${{labels}}`;
        document.getElementById('last-event').textContent = `Last event: ${{data.last_event}}`;
      }} catch (error) {{
        document.getElementById('connection-status').textContent = 'Players: unavailable';
      }}
    }}
    refreshStatus();
    setInterval(refreshStatus, 1000);
  </script>
</body>
</html>
"""

    def _build_config_js(self) -> str:
        config = {
            "appTitle": self.server.config.app_title,
            "helperNoteHtml": self.server.config.helper_note_html,
            "wsPort": self.server.config.ws_port,
            "buttonNames": BUTTON_NAMES,
        }
        return f"window.CONTROLLER_CONFIG = {json.dumps(config)};"

    def _build_qr_svg(self) -> bytes | None:
        if qrcode is None:
            return None
        qr = qrcode.QRCode(
            border=2,
            box_size=8,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
        )
        qr.add_data(self.server.config.controller_url)
        qr.make(fit=True)
        image = qr.make_image(image_factory=qrcode.image.svg.SvgPathImage)
        output = io.BytesIO()
        image.save(output)
        return output.getvalue()


def start_tris_http_server(
    controller_state: ControllerState,
    config: ServerConfig,
    request_handler: type[BaseHTTPRequestHandler] | None = None,
) -> tuple[TrisControllerHttpServer, threading.Thread]:
    server = TrisControllerHttpServer(
        (config.bind_host, config.http_port), config, controller_state, request_handler
    )
    thread = threading.Thread(target=server.serve_forever, name="tris-controller-http", daemon=True)
    thread.start()
    return server, thread


@dataclass
class StoneProjectile:
    owner_id: str
    x: float
    y: float
    vx: float
    vy: float
    radius: int = STONE_RADIUS
    lifetime: float = STONE_LIFETIME
    kind: str = "stone"

    @property
    def rect(self) -> pygame.Rect:
        diameter = self.radius * 2
        return pygame.Rect(
            round(self.x - self.radius),
            round(self.y - self.radius),
            diameter,
            diameter,
        )

    def update(
        self,
        dt_sec: float,
        platforms: list[pygame.Rect],
        screen_width: int,
        screen_height: int,
    ) -> bool:
        self.lifetime -= dt_sec
        if self.lifetime <= 0.0:
            return False

        if self.kind == "stone":
            self.vy = min(self.vy + STONE_GRAVITY * dt_sec, STONE_MAX_FALL_SPEED)
        self.x += self.vx * dt_sec
        self.y += self.vy * dt_sec

        left = wrap_scalar(self.x - self.radius, self.radius * 2, screen_width)
        top = wrap_scalar(self.y - self.radius, self.radius * 2, screen_height)
        self.x = left + self.radius
        self.y = top + self.radius

        stone_rect = self.rect
        for platform in platforms:
            if stone_rect.colliderect(platform):
                return False
        return True


@dataclass
class WeaponDrop:
    weapon: str
    x: float
    y: float
    vy: float = WEAPON_DROP_SPEED
    heat_value: float = 0.0
    heat_lift: float = 0.0

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(round(self.x), round(self.y), *WEAPON_DROP_SIZE)

    def get_canopy_rect(self) -> pygame.Rect:
        return pygame.Rect(
            round(self.x + WEAPON_DROP_SIZE[0] / 2 - PARACHUTE_CANOPY_WIDTH / 2),
            round(self.y - PARACHUTE_CANOPY_HEIGHT - 10),
            PARACHUTE_CANOPY_WIDTH,
            PARACHUTE_CANOPY_HEIGHT,
        )

    def get_collision_rect(self) -> pygame.Rect:
        return self.rect.union(self.get_canopy_rect()).inflate(6, 6)

    def update(
        self,
        dt_sec: float,
        heat_field: np.ndarray,
        shadow_mask: np.ndarray,
        screen_width: int,
        screen_height: int,
    ) -> bool:
        probe = self.get_collision_rect()
        self.heat_value = sample_heat_region(
            heat_field, probe, screen_width, screen_height
        )
        self.heat_lift = max(0.0, self.heat_value - PARACHUTE_LIFT_THRESHOLD)
        self.vy = min(
            self.vy + PLAYER_GRAVITY * PARACHUTE_GRAVITY_SCALE * dt_sec,
            PARACHUTE_MAX_FALL_SPEED,
        )
        if self.heat_lift > 0.0:
            self.vy -= self.heat_lift * PARACHUTE_LIFT_ACCEL * dt_sec
            lift_strength = clamp(
                self.heat_lift / max(1e-6, 1.0 - PARACHUTE_LIFT_THRESHOLD),
                0.0,
                1.0,
            )
            target_rise_speed = -(
                PARACHUTE_RISE_SPEED_MIN
                + lift_strength * (PARACHUTE_RISE_SPEED_MAX - PARACHUTE_RISE_SPEED_MIN)
            )
            self.vy = min(self.vy, target_rise_speed)

        old_y = self.y
        old_overlap = rect_shadow_count(
            self.get_collision_rect(), shadow_mask, screen_width, screen_height
        )
        self.y += self.vy * dt_sec
        new_overlap = rect_shadow_count(
            self.get_collision_rect(), shadow_mask, screen_width, screen_height
        )
        if new_overlap > old_overlap:
            self.y = old_y
            self.vy = 0.0
        return self.get_canopy_rect().top < screen_height


@dataclass
class RemotePlayer:
    player_id: str
    label: str
    color_index: int
    spawn_slot: int
    spawn_x: float
    spawn_y: float
    palette: dict[str, tuple[int, int, int]]
    x: float
    y: float
    width: int = PLAYER_SIZE[0]
    height: int = PLAYER_SIZE[1]
    vx: float = 0.0
    vy: float = 0.0
    on_ground: bool = False
    coyote_timer: float = 0.0
    parachute_open: bool = True
    heat_value: float = 0.0
    heat_lift: float = 0.0
    facing: int = 1
    aim_x: float = 1.0
    aim_y: float = 0.0
    alive: bool = True
    stone_cooldown: float = 0.0
    kill_count: int = 0
    accolade_flash: float = 0.0
    weapon: str | None = "stone"
    attack_flash: float = 0.0
    falling: bool = False

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(round(self.x), round(self.y), self.width, self.height)

    def reset(self) -> None:
        self.x = self.spawn_x
        self.y = self.spawn_y
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False
        self.coyote_timer = 0.0
        self.parachute_open = True
        self.heat_value = 0.0
        self.heat_lift = 0.0
        self.alive = True
        self.stone_cooldown = 0.0
        self.accolade_flash = 0.0
        self.weapon = "stone"
        self.attack_flash = 0.0
        self.falling = False

    def kill(self) -> None:
        self.alive = False
        self.falling = True
        self.vx *= 0.35
        self.vy = max(90.0, self.vy)
        self.parachute_open = False
        self.heat_value = 0.0
        self.heat_lift = 0.0

    def register_kill(self) -> None:
        self.kill_count += 1
        self.accolade_flash = 1.0

    def try_throw_stone(self) -> StoneProjectile | None:
        if self.stone_cooldown > 0.0:
            return None
        cooldown = BOW_COOLDOWN if self.weapon == "bow" else STONE_COOLDOWN
        self.stone_cooldown = cooldown
        aim_x, aim_y = self.normalized_aim()
        spawn_x = self.x + self.width / 2.0 + aim_x * (self.width * 0.58)
        spawn_y = self.y + self.height * 0.34 + aim_y * (self.height * 0.20)
        speed = BOW_PROJECTILE_SPEED if self.weapon == "bow" else STONE_THROW_SPEED
        return StoneProjectile(
            owner_id=self.player_id,
            x=spawn_x,
            y=spawn_y,
            vx=aim_x * speed,
            vy=aim_y * speed,
            radius=5 if self.weapon == "bow" else STONE_RADIUS,
            lifetime=2.2 if self.weapon == "bow" else STONE_LIFETIME,
            kind="arrow" if self.weapon == "bow" else "stone",
        )

    def normalized_aim(self) -> tuple[float, float]:
        length = (self.aim_x * self.aim_x + self.aim_y * self.aim_y) ** 0.5
        if length <= AIM_DEADZONE:
            return float(self.facing), 0.0
        return self.aim_x / length, self.aim_y / length

    def attack(self) -> StoneProjectile | None:
        if self.weapon != "sword":
            return self.try_throw_stone()
        if self.stone_cooldown <= 0.0:
            self.stone_cooldown = SWORD_COOLDOWN
            self.attack_flash = 0.16
        return None

    def drop_weapon(self) -> WeaponDrop:
        canopy = self.get_canopy_rect()
        return WeaponDrop(
            self.weapon,
            float(canopy.centerx - WEAPON_DROP_SIZE[0] / 2),
            float(canopy.top),
        )

    def get_canopy_rect(self) -> pygame.Rect:
        left = int(round(self.x + self.width / 2.0 - PARACHUTE_CANOPY_WIDTH / 2.0))
        top = int(round(self.y - PARACHUTE_CANOPY_HEIGHT - 10))
        return pygame.Rect(left, top, PARACHUTE_CANOPY_WIDTH, PARACHUTE_CANOPY_HEIGHT)

    def get_combat_rect(self) -> pygame.Rect:
        return self.rect.union(self.get_canopy_rect()) if self.alive else self.rect

    def get_heat_probe_rect(self) -> pygame.Rect:
        canopy_rect = self.get_canopy_rect()
        probe_left = canopy_rect.left + 8
        probe_width = max(12, canopy_rect.width - 16)
        probe_height = max(16, self.rect.bottom - canopy_rect.top)
        return pygame.Rect(probe_left, canopy_rect.top, probe_width, probe_height)

    def get_exclusion_rect(self) -> pygame.Rect:
        if self.falling:
            return self.rect.inflate(12, 12)
        if not self.alive:
            return pygame.Rect(0, 0, 0, 0)
        if not self.parachute_open:
            return self.rect.inflate(10, 10)
        return self.rect.union(self.get_canopy_rect()).inflate(14, 14)

    def update(
        self,
        dt_sec: float,
        held: dict[str, bool],
        pressed: dict[str, bool],
        joystick: dict[str, float],
        shadow_mask: np.ndarray,
        heat_field: np.ndarray,
        screen_width: int,
        screen_height: int,
    ) -> StoneProjectile | None:
        if pressed["start"]:
            self.reset()
            return None

        self.stone_cooldown = max(0.0, self.stone_cooldown - dt_sec)
        self.accolade_flash = max(0.0, self.accolade_flash - dt_sec * 2.4)
        self.attack_flash = max(0.0, self.attack_flash - dt_sec)

        if not self.alive:
            return None

        thrown_stone = self.attack() if pressed["b"] else None

        move_dir = 0
        if held["left"]:
            move_dir -= 1
        if held["right"]:
            move_dir += 1

        if move_dir:
            self.facing = move_dir

        aim_x = float(joystick.get("x", 0.0))
        aim_y = float(joystick.get("y", 0.0))
        aim_length = (aim_x * aim_x + aim_y * aim_y) ** 0.5
        if aim_length > AIM_DEADZONE:
            self.aim_x = aim_x / aim_length
            self.aim_y = aim_y / aim_length
        else:
            self.aim_x = float(self.facing)
            self.aim_y = 0.0

        self.vx = move_dir * PLAYER_MOVE_SPEED * 0.78

        gravity = PLAYER_GRAVITY * PARACHUTE_GRAVITY_SCALE
        max_fall_speed = PARACHUTE_MAX_FALL_SPEED
        self.heat_value = 0.0
        self.heat_lift = 0.0

        self.heat_value = sample_heat_region(
            heat_field,
            self.get_heat_probe_rect(),
            screen_width,
            screen_height,
        )
        self.heat_lift = max(0.0, self.heat_value - PARACHUTE_LIFT_THRESHOLD)

        self.vy = min(self.vy + gravity * dt_sec, max_fall_speed)

        if self.parachute_open and self.heat_lift > 0.0:
            self.vy -= self.heat_lift * PARACHUTE_LIFT_ACCEL * dt_sec
            lift_strength = clamp(
                self.heat_lift / max(1e-6, 1.0 - PARACHUTE_LIFT_THRESHOLD),
                0.0,
                1.0,
            )
            target_rise_speed = -(
                PARACHUTE_RISE_SPEED_MIN
                + lift_strength * (PARACHUTE_RISE_SPEED_MAX - PARACHUTE_RISE_SPEED_MIN)
            )
            if self.vy > target_rise_speed:
                self.vy = target_rise_speed

        old_x = self.x
        old_overlap = shadow_overlap(self, shadow_mask, screen_width, screen_height)
        self.x += self.vx * dt_sec
        self.x = wrap_scalar(self.x, self.width, screen_width)
        new_overlap = shadow_overlap(self, shadow_mask, screen_width, screen_height)
        if new_overlap > old_overlap:
            self.x = old_x
            self.vx = 0.0

        old_y = self.y
        old_overlap = shadow_overlap(self, shadow_mask, screen_width, screen_height)
        self.y += self.vy * dt_sec
        self.y = wrap_scalar(self.y, self.height, screen_height)
        new_overlap = shadow_overlap(self, shadow_mask, screen_width, screen_height)
        if new_overlap > old_overlap:
            self.y = old_y
            self.vy = 0.0

        return thrown_stone


def shadow_overlap(
    player: RemotePlayer,
    shadow_mask: np.ndarray,
    screen_width: int,
    screen_height: int,
) -> int:
    """Count shadow pixels under the body/canopy collision footprint."""
    footprint = player.rect.union(player.get_canopy_rect()).inflate(6, 6)
    clipped = footprint.clip(pygame.Rect(0, 0, screen_width, screen_height))
    if clipped.width <= 0 or clipped.height <= 0:
        return 0
    mask_height, mask_width = shadow_mask.shape
    left = max(0, int(clipped.left * mask_width / screen_width))
    right = min(mask_width, int(np.ceil(clipped.right * mask_width / screen_width)))
    top = max(0, int(clipped.top * mask_height / screen_height))
    bottom = min(mask_height, int(np.ceil(clipped.bottom * mask_height / screen_height)))
    return int(np.count_nonzero(shadow_mask[top:bottom, left:right]))


def rect_shadow_count(
    rect: pygame.Rect,
    shadow_mask: np.ndarray,
    screen_width: int,
    screen_height: int,
) -> int:
    clipped = rect.clip(pygame.Rect(0, 0, screen_width, screen_height))
    if clipped.width <= 0 or clipped.height <= 0:
        return 0
    mask_height, mask_width = shadow_mask.shape
    left = max(0, int(clipped.left * mask_width / screen_width))
    right = min(mask_width, int(np.ceil(clipped.right * mask_width / screen_width)))
    top = max(0, int(clipped.top * mask_height / screen_height))
    bottom = min(mask_height, int(np.ceil(clipped.bottom * mask_height / screen_height)))
    return int(np.count_nonzero(shadow_mask[top:bottom, left:right]))


def rect_shadow_overlap(
    rect: pygame.Rect,
    shadow_mask: np.ndarray,
    screen_width: int,
    screen_height: int,
) -> bool:
    return rect_shadow_count(rect, shadow_mask, screen_width, screen_height) > 0


def defeat_player(player: RemotePlayer, weapon_drops: list[WeaponDrop]) -> None:
    if not player.alive:
        return
    if player.weapon is not None:
        weapon_drops.append(player.drop_weapon())
        player.weapon = None
    player.kill()


def update_falling_players(
    players: list[RemotePlayer],
    dt_sec: float,
    shadow_mask: np.ndarray,
    screen_width: int,
    screen_height: int,
) -> None:
    for player in players:
        if not player.falling:
            continue
        player.vy = min(player.vy + FALLING_PLAYER_GRAVITY * dt_sec, STONE_MAX_FALL_SPEED)
        player.x += player.vx * dt_sec
        player.y += player.vy * dt_sec
        if player.rect.top >= screen_height or rect_shadow_overlap(
            player.rect, shadow_mask, screen_width, screen_height
        ):
            player.falling = False


def try_pick_up_weapon(drop: WeaponDrop, players: list[RemotePlayer]) -> bool:
    pickup_rect = drop.get_collision_rect()
    for player in players:
        if player.alive and player.get_combat_rect().colliderect(pickup_rect):
            player.weapon = drop.weapon
            return True
    return False


def update_weapon_drops(
    drops: list[WeaponDrop],
    players: list[RemotePlayer],
    dt_sec: float,
    heat_field: np.ndarray,
    shadow_mask: np.ndarray,
    screen_width: int,
    screen_height: int,
) -> list[WeaponDrop]:
    active: list[WeaponDrop] = []
    for drop in drops:
        if not drop.update(
            dt_sec, heat_field, shadow_mask, screen_width, screen_height
        ):
            continue
        if not try_pick_up_weapon(drop, players):
            active.append(drop)
    return active


def resolve_sword_attacks(
    players: list[RemotePlayer],
    weapon_drops: list[WeaponDrop],
) -> None:
    for attacker in players:
        if not attacker.alive or attacker.weapon != "sword" or attacker.attack_flash <= 0.0:
            continue
        reach = pygame.Rect(
            attacker.rect.centerx,
            attacker.rect.top - 38,
            SWORD_REACH,
            attacker.rect.height + 76,
        )
        if attacker.facing >= 0:
            reach.left = attacker.rect.centerx
        else:
            reach.right = attacker.rect.centerx
        for target in players:
            if target is attacker or not target.alive:
                continue
            if reach.colliderect(target.get_combat_rect()):
                defeat_player(target, weapon_drops)
                attacker.register_kill()
                attacker.attack_flash = 0.0
                break


def build_remote_player(
    snapshot: dict[str, object],
    spawn_slot: int,
    spawn_points: list[tuple[float, float]],
    player_type: type[RemotePlayer] = RemotePlayer,
) -> RemotePlayer:
    spawn_x, spawn_y = spawn_points[spawn_slot]
    color_index = int(snapshot["color_index"])
    return player_type(
        player_id=str(snapshot["client_id"]),
        label=str(snapshot["label"]),
        color_index=color_index,
        spawn_slot=spawn_slot,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
        palette=palette_for_index(color_index),
        x=spawn_x,
        y=spawn_y,
    )


def sync_remote_players(
    remote_players: dict[str, RemotePlayer],
    player_snapshots: dict[str, dict[str, object]],
    spawn_points: list[tuple[float, float]],
    player_type: type[RemotePlayer] = RemotePlayer,
) -> None:
    active_ids = set(player_snapshots)
    stale_ids = [player_id for player_id in remote_players if player_id not in active_ids]
    for player_id in stale_ids:
        remote_players.pop(player_id, None)

    for player_id, snapshot in player_snapshots.items():
        if player_id in remote_players:
            remote_players[player_id].label = str(snapshot["label"])
            continue

        spawn_slot = next_spawn_slot(list(remote_players.values()), len(spawn_points))
        remote_players[player_id] = build_remote_player(snapshot, spawn_slot, spawn_points, player_type)


def resolve_player_collisions(
    players: list[RemotePlayer],
    screen_width: int,
    screen_height: int,
) -> None:
    alive_players = [player for player in players if player.alive]
    for _ in range(PLAYER_COLLISION_PASSES):
        collided = False
        for index, first in enumerate(alive_players):
            for second in alive_players[index + 1:]:
                first_rect = first.rect
                second_rect = second.rect
                if not first_rect.colliderect(second_rect):
                    continue

                overlap_x = min(first_rect.right, second_rect.right) - max(first_rect.left, second_rect.left)
                overlap_y = min(first_rect.bottom, second_rect.bottom) - max(first_rect.top, second_rect.top)
                if overlap_x <= 0 or overlap_y <= 0:
                    continue

                collided = True
                if overlap_y <= overlap_x:
                    if first_rect.centery <= second_rect.centery:
                        first.y -= overlap_y
                        if first.vy > 0.0:
                            first.vy = 0.0
                        first.on_ground = True
                        first.coyote_timer = PLAYER_COYOTE_TIME
                    else:
                        second.y -= overlap_y
                        if second.vy > 0.0:
                            second.vy = 0.0
                        second.on_ground = True
                        second.coyote_timer = PLAYER_COYOTE_TIME
                else:
                    shift = overlap_x / 2.0
                    if first_rect.centerx <= second_rect.centerx:
                        first.x -= shift
                        second.x += shift
                    else:
                        first.x += shift
                        second.x -= shift
                    first.vx = 0.0
                    second.vx = 0.0
                    wrap_player_position(first, screen_width, screen_height)
                    wrap_player_position(second, screen_width, screen_height)

        if not collided:
            break


def update_stones(
    stones: list[StoneProjectile],
    remote_players: dict[str, RemotePlayer],
    platforms: list[pygame.Rect],
    dt_sec: float,
    screen_width: int,
    screen_height: int,
    weapon_drops: list[WeaponDrop],
) -> list[StoneProjectile]:
    active_stones: list[StoneProjectile] = []
    for stone in stones:
        if not stone.update(dt_sec, platforms, screen_width, screen_height):
            continue

        hit_player = None
        stone_rect = stone.rect
        for player in remote_players.values():
            if not player.alive or player.player_id == stone.owner_id:
                continue
            if stone_rect.colliderect(player.get_combat_rect()):
                hit_player = player
                break

        if hit_player is not None:
            defeat_player(hit_player, weapon_drops)
            killer = remote_players.get(stone.owner_id)
            if killer is not None:
                killer.register_kill()
            continue

        active_stones.append(stone)
    return active_stones


def draw_accolade(surface: pygame.Surface, player: RemotePlayer) -> None:
    visible_count = min(player.kill_count, ACCOLADE_MAX_VISIBLE)
    if visible_count <= 0:
        return

    flash_mix = clamp(player.accolade_flash, 0.0, 1.0)
    plume_color = lighten_color(ACCOLADE_BASE_COLOR, 0.18 + flash_mix * 0.24)
    plume_highlight = lighten_color(ACCOLADE_TIP_COLOR, flash_mix * 0.20)
    body_rect = player.rect
    direction = -1 if player.facing >= 0 else 1
    base_x = body_rect.left + 7 if direction == -1 else body_rect.right - 7
    base_y = body_rect.top + 20

    for index in range(visible_count):
        feather_base = (base_x + direction * index * 6, base_y - index * 5)
        feather_tip = (
            feather_base[0] + direction * (9 + index),
            feather_base[1] - (20 + index * 3),
        )
        feather_inner = (feather_base[0] - direction * 2, feather_base[1] - 6)
        feather_outer = (feather_base[0] + direction * 5, feather_base[1] - 11)
        pygame.draw.polygon(
            surface,
            plume_color,
            [feather_base, feather_outer, feather_tip, feather_inner],
        )
        pygame.draw.polygon(
            surface,
            plume_highlight,
            [feather_base, feather_outer, feather_tip, feather_inner],
            1,
        )


def draw_stone(surface: pygame.Surface, stone: StoneProjectile) -> None:
    center = (round(stone.x), round(stone.y))
    if stone.kind == "arrow":
        direction = pygame.Vector2(stone.vx, stone.vy)
        if direction.length_squared() > 0:
            direction.scale_to_length(18)
        start = (round(stone.x - direction.x), round(stone.y - direction.y))
        pygame.draw.line(surface, (92, 56, 28), start, center, 4)
        pygame.draw.circle(surface, (220, 220, 205), center, 3)
        return
    pygame.draw.circle(surface, (90, 84, 80), center, stone.radius)
    pygame.draw.circle(surface, (150, 142, 136), center, max(2, stone.radius - 4))
    highlight_pos = (center[0] - 3, center[1] - 3)
    pygame.draw.circle(surface, (208, 204, 196), highlight_pos, 2)


def draw_aim_arrow(surface: pygame.Surface, player: RemotePlayer) -> None:
    aim_x, aim_y = player.normalized_aim()
    body_rect = player.rect
    start = (
        int(round(body_rect.centerx + aim_x * (player.width * 0.62))),
        int(round(body_rect.centery + aim_y * (player.height * 0.38))),
    )
    end = (
        int(round(start[0] + aim_x * 30)),
        int(round(start[1] + aim_y * 30)),
    )
    normal_x = -aim_y
    normal_y = aim_x
    head = [
        end,
        (
            int(round(end[0] - aim_x * 10 + normal_x * 5)),
            int(round(end[1] - aim_y * 10 + normal_y * 5)),
        ),
        (
            int(round(end[0] - aim_x * 10 - normal_x * 5)),
            int(round(end[1] - aim_y * 10 - normal_y * 5)),
        ),
    ]
    arrow_color = lighten_color(player.palette["accent"], 0.32)
    shadow_color = (20, 24, 20)
    pygame.draw.line(surface, shadow_color, start, end, 5)
    pygame.draw.polygon(surface, shadow_color, head)
    pygame.draw.line(surface, arrow_color, start, end, 3)
    pygame.draw.polygon(surface, arrow_color, head)


def draw_remote_player(
    surface: pygame.Surface,
    player: RemotePlayer,
    label_font: pygame.font.Font,
) -> None:
    if not player.alive and not player.falling:
        return

    palette = player.palette
    outline_color = palette["chip"]
    canopy_color = palette["accent"]
    canopy_highlight = lighten_color(canopy_color, 0.4)

    if player.alive and player.parachute_open:
        canopy_rect = player.get_canopy_rect()
        if player.heat_lift > 0.0:
            glow_strength = clamp(player.heat_lift * 2.8, 0.0, 1.0)
            glow_rect = canopy_rect.union(player.rect).inflate(36, 44)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            alpha = int(90 * glow_strength)
            glow_color = lighten_color(LIFT_GLOW_COLOR, 0.1)
            pygame.draw.ellipse(
                glow_surface,
                (*glow_color, alpha),
                glow_surface.get_rect(),
            )
            surface.blit(glow_surface, glow_rect.topleft)

        canopy_points = [
            (canopy_rect.left, canopy_rect.bottom - 2),
            (canopy_rect.left + 12, canopy_rect.top + 6),
            (canopy_rect.centerx, canopy_rect.top),
            (canopy_rect.right - 12, canopy_rect.top + 6),
            (canopy_rect.right, canopy_rect.bottom - 2),
        ]
        pygame.draw.polygon(surface, canopy_color, canopy_points)
        pygame.draw.polygon(surface, canopy_highlight, canopy_points, 2)

        harness_top = player.rect.top + 6
        line_targets = [
            (player.rect.left + 6, harness_top),
            (player.rect.centerx, harness_top + 2),
            (player.rect.right - 6, harness_top),
        ]
        line_sources = [
            canopy_points[0],
            (canopy_rect.centerx, canopy_rect.bottom - 3),
            canopy_points[-1],
        ]
        for source, target in zip(line_sources, line_targets):
            pygame.draw.line(surface, canopy_highlight, source, target, 2)

    if player.alive:
        draw_accolade(surface, player)
        draw_aim_arrow(surface, player)

    body_rect = player.rect
    pygame.draw.rect(surface, palette["body"], body_rect, border_radius=12)
    pygame.draw.rect(surface, outline_color, body_rect, 2, border_radius=12)

    head_center = (body_rect.centerx, body_rect.top + 12)
    pygame.draw.circle(surface, PLAYER_HEAD_COLOR, head_center, 11)
    pygame.draw.circle(surface, outline_color, head_center, 11, 2)

    torso_top = body_rect.top + 18
    torso_bottom = body_rect.bottom - 8
    pygame.draw.line(
        surface,
        PLAYER_OUTLINE_COLOR,
        (body_rect.centerx, torso_top),
        (body_rect.centerx, torso_bottom),
        3,
    )
    arm_reach = 11 if player.stone_cooldown > STONE_COOLDOWN - 0.12 else 7
    arm_height = body_rect.top + 30
    pygame.draw.line(
        surface,
        PLAYER_OUTLINE_COLOR,
        (body_rect.centerx, body_rect.top + 26),
        (body_rect.left + arm_reach, arm_height),
        3,
    )
    pygame.draw.line(
        surface,
        PLAYER_OUTLINE_COLOR,
        (body_rect.centerx, body_rect.top + 26),
        (body_rect.right - arm_reach, arm_height),
        3,
    )
    pygame.draw.line(
        surface,
        PLAYER_OUTLINE_COLOR,
        (body_rect.centerx, torso_bottom),
        (body_rect.left + 8, body_rect.bottom - 2),
        3,
    )
    pygame.draw.line(
        surface,
        PLAYER_OUTLINE_COLOR,
        (body_rect.centerx, torso_bottom),
        (body_rect.right - 8, body_rect.bottom - 2),
        3,
    )

    label_surf = label_font.render(player.label, True, PLAYER_HEAD_COLOR)
    chip_rect = label_surf.get_rect(center=(body_rect.centerx, body_rect.centery + 6))
    chip_rect.inflate_ip(10, 4)
    pygame.draw.rect(surface, palette["chip"], chip_rect, border_radius=8)
    pygame.draw.rect(surface, canopy_highlight, chip_rect, 2, border_radius=8)
    surface.blit(label_surf, label_surf.get_rect(center=chip_rect.center))

    if player.alive and player.weapon == "sword":
        hand = (body_rect.centerx + player.facing * 8, body_rect.centery)
        if player.attack_flash > 0.0:
            slash_points = []
            for angle_degrees in range(-72, 73, 12):
                angle = np.radians(angle_degrees)
                slash_points.append(
                    (
                        round(hand[0] + player.facing * np.cos(angle) * SWORD_REACH),
                        round(hand[1] + np.sin(angle) * SWORD_REACH),
                    )
                )
            pygame.draw.lines(surface, (255, 245, 190), False, slash_points, 14)
            pygame.draw.lines(surface, (225, 230, 235), False, slash_points, 7)
            tip = slash_points[-1]
        else:
            tip = (hand[0] + player.facing * 34, hand[1] - 15)
        pygame.draw.line(surface, (225, 230, 235), hand, tip, 7)
        pygame.draw.line(surface, (92, 66, 35), (hand[0], hand[1] - 6), (hand[0], hand[1] + 6), 4)
    elif player.alive and player.weapon == "bow":
        bow_rect = pygame.Rect(0, 0, 18, 38)
        bow_rect.center = (body_rect.centerx + player.facing * 18, body_rect.centery)
        pygame.draw.arc(surface, (126, 74, 32), bow_rect, -1.45, 1.45, 4)


def draw_weapon_drop(surface: pygame.Surface, drop: WeaponDrop) -> None:
    rect = drop.rect
    canopy = drop.get_canopy_rect()
    canopy_points = [
        (canopy.left, canopy.bottom - 2),
        (canopy.left + 12, canopy.top + 6),
        (canopy.centerx, canopy.top),
        (canopy.right - 12, canopy.top + 6),
        (canopy.right, canopy.bottom - 2),
    ]
    pygame.draw.polygon(surface, (235, 194, 82), canopy_points)
    pygame.draw.polygon(surface, (255, 232, 150), canopy_points, 2)
    pygame.draw.line(surface, (230, 220, 170), canopy_points[0], (rect.left + 6, rect.top + 5), 2)
    pygame.draw.line(surface, (230, 220, 170), canopy_points[-1], (rect.right - 6, rect.top + 5), 2)
    if drop.heat_lift > 0.0:
        pygame.draw.ellipse(surface, (255, 145, 55), canopy.inflate(12, 10), 3)
    draw_weapon_pickup(surface, rect, drop.weapon)


def draw_weapon_pickup(surface: pygame.Surface, rect: pygame.Rect, weapon: str) -> None:
    if weapon == "sword":
        pygame.draw.line(surface, (225, 230, 235), (rect.centerx - 9, rect.bottom - 3), (rect.centerx + 9, rect.top + 3), 6)
    elif weapon == "bow":
        pygame.draw.arc(surface, (126, 74, 32), pygame.Rect(rect.centerx - 11, rect.top + 2, 22, 28), -1.4, 1.4, 4)
    else:
        pygame.draw.circle(surface, (105, 98, 92), rect.center, 9)
        pygame.draw.circle(surface, (190, 184, 176), (rect.centerx - 3, rect.centery - 3), 3)


def draw_world(
    screen: pygame.Surface,
    frame_surface: pygame.Surface,
    grid_image: np.ndarray,
    platforms: list[pygame.Rect],
    players: list[RemotePlayer],
    stones: list[StoneProjectile],
    weapon_drops: list[WeaponDrop],
    label_font: pygame.font.Font,
) -> None:
    pygame.surfarray.blit_array(frame_surface, grid_image.swapaxes(0, 1))
    pygame.transform.scale(frame_surface, screen.get_size(), screen)
    draw_platforms(screen, platforms)
    for stone in stones:
        draw_stone(screen, stone)
    for drop in weapon_drops:
        draw_weapon_drop(screen, drop)
    for player in sorted((item for item in players if item.alive or item.falling), key=lambda item: item.rect.bottom):
        draw_remote_player(screen, player, label_font)


def parse_args(description="Level 5 - shadow-collision parachutes"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--cached-calibration",
        action="store_true",
        help="Use cached calibration from calibration_cache.npz.",
    )
    parser.add_argument(
        "--bind-host",
        default="0.0.0.0",
        help="Address to bind the local controller servers to.",
    )
    parser.add_argument(
        "--public-host",
        default=None,
        help="LAN IP or hostname shown in the QR code. Auto-detected by default.",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8000,
        help="HTTP port for the setup page and phone controller page.",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket port for mobile input.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=1,
        help="OpenCV camera index to use. This setup detected the secondary webcam at index 1.",
    )
    return parser.parse_args()


def main(
    cached_calibration: bool = False,
    bind_host: str = "0.0.0.0",
    public_host: str | None = None,
    http_port: int = 8000,
    ws_port: int = 8765,
    camera_index: int = 1,
) -> None:
    cap_stream = None
    http_server = None
    http_thread = None
    ws_server = None

    try:
        pygame.init()
        screen, width, height, display_index, monitor, monitor_count = create_fullscreen_display(
            TARGET_DISPLAY_INDEX
        )
        pygame.display.set_caption("Level 5 - Shadow Parachutes")
        clock = pygame.time.Clock()
        ready_font = pygame.font.SysFont("Arial", 64, bold=True)
        countdown_font = pygame.font.SysFont("Arial", 120, bold=True)
        label_font = pygame.font.SysFont("Consolas", 13, bold=True)

        if display_index != TARGET_DISPLAY_INDEX:
            print(
                f"Display {TARGET_DISPLAY_INDEX} not found; using display {display_index}."
            )
        print(
            "Display:",
            f"{monitor_count} total,",
            f"using index {display_index} at {monitor['width']}x{monitor['height']}",
            f"pos {monitor['left']},{monitor['top']}",
        )

        calibration = None
        if cached_calibration:
            calibration = load_calibration(CALIBRATION_CACHE)
            if calibration is None:
                print(f"No calibration cache found at {CALIBRATION_CACHE}.")
                pygame.quit()
                return
            H, mask, a_map, b_map = calibration
            H_inv = np.linalg.inv(H)
        else:
            cap = get_camera_capture(camera_index)
            H, mask, _, a_map, b_map = calibrate_system(
                cap=cap, screen=screen, WIDTH=width, HEIGHT=height
            )
            H_inv = np.linalg.inv(H)
            save_calibration(CALIBRATION_CACHE, H, mask, a_map, b_map)
            cap.release()

        cap_stream = VideoStream(camera_index)

        public_host = public_host or detect_public_host()
        controller_state = TrisControllerState()
        config = ServerConfig(
            bind_host=bind_host,
            public_host=public_host,
            http_port=http_port,
            ws_port=ws_port,
            app_title="Level 5 Shadow Parachutes",
            helper_note_html="Joystick moves/aims<br>ATTACK / THROW uses your weapon",
        )
        http_server, http_thread = start_tris_http_server(controller_state, config)
        ws_server = TrisControllerWebSocketServer(controller_state, bind_host, ws_port)
        ws_server.start()

        print(f"Setup page:      {config.setup_url}")
        print(f"Controller page: {config.controller_url}")
        print(f"WebSocket port:  ws://{config.public_host}:{config.ws_port}")
        print("Phone controls: joystick moves/aims; ATTACK/THROW uses sword, bow, or stone.")
        print("Arena: shadow collision, no platforms, wraparound on every edge.")
        print("Local controls: ESC quit, V toggle shadow overlay, C clear heat.")
        if public_host == "127.0.0.1":
            print(
                "Warning: LAN IP auto-detection fell back to 127.0.0.1. "
                "Use --public-host if phones cannot connect."
            )

        bg_img = np.full((GRID_H, GRID_W, 3), 128, dtype=np.uint8)
        grid_x, grid_y = np.meshgrid(
            np.arange(GRID_W, dtype=np.float32),
            np.arange(GRID_H, dtype=np.float32),
        )
        heat_field = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        frame_surface = pygame.Surface((GRID_W, GRID_H))
        last_stage_img = bg_img.copy()
        platforms: list[pygame.Rect] = []
        spawn_points = build_spawn_points(width, height)
        remote_players: dict[str, RemotePlayer] = {}
        stones: list[StoneProjectile] = []
        weapon_drops: list[WeaponDrop] = []
        weapon_drop_timer = 2.5
        shadow_mask = np.zeros((GRID_H, GRID_W), dtype=bool)
        prev_player_cam_mask = None
        show_overlay = False
        frame_count = 0
        t_prev = time.time()
        debug_object_count = 0
        last_remote_event = "waiting"
        connected_clients = 0

        draw_world(
            screen,
            frame_surface,
            last_stage_img,
            platforms,
            list(remote_players.values()),
            stones,
            weapon_drops,
            label_font,
        )
        pygame.display.flip()

        def draw_heat_background(dt_sec: float) -> bool:
            nonlocal heat_field, last_stage_img, prev_player_cam_mask, debug_object_count, shadow_mask

            ret, cam_frame = cap_stream.read()
            if not ret or cam_frame is None:
                return True

            cam_gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
            cam_gray_masked = cv2.bitwise_and(cam_gray, cam_gray, mask=mask)

            h_cam, w_cam = cam_gray.shape[:2]
            warped_proj_gray_masked = screen_to_camera_space(screen, H, mask, h_cam, w_cam)
            expected = gain_offset_calibration(warped_proj_gray_masked, a_map, b_map)

            player_mask_surface = pygame.Surface(screen.get_size())
            player_mask_surface.fill((0, 0, 0))
            for player in remote_players.values():
                if not player.alive:
                    continue
                pygame.draw.rect(
                    player_mask_surface,
                    (255, 255, 255),
                    player.get_exclusion_rect(),
                    border_radius=18,
                )

            player_cam_mask = surface_to_camera_gray_masked(
                player_mask_surface,
                H,
                mask,
                h_cam,
                w_cam,
            ) > 0

            mask_diff = player_cam_mask
            if prev_player_cam_mask is not None:
                mask_diff = np.logical_or(mask_diff, prev_player_cam_mask)

            objects, contours_vis, shadow_mask_u8 = shadow_object_detection_calibrated(
                cam_gray_masked, expected, b_map, mask_diff, H_inv
            )

            shadow_screen = objects_to_mask(objects, height, width)
            shadow_small = cv2.resize(
                shadow_screen, (GRID_W, GRID_H), interpolation=cv2.INTER_NEAREST
            )
            shadow_mask = shadow_small > 0

            heat_field = update_heat_field(heat_field, shadow_mask, dt_sec)
            map_x, map_y = compute_heat_distortion(
                grid_x, grid_y, heat_field, time.perf_counter()
            )

            warped_img = cv2.remap(
                bg_img,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            apply_heat_tint(warped_img, heat_field, shadow_mask)

            if show_overlay:
                warped_img[shadow_mask] = (255, 110, 20)

            last_stage_img = warped_img
            pygame.surfarray.blit_array(frame_surface, warped_img.swapaxes(0, 1))
            pygame.transform.scale(frame_surface, screen.get_size(), screen)
            prev_player_cam_mask = player_cam_mask
            debug_object_count = len(objects)

            if SHOW_DEBUG_WINDOWS:
                cv2.imshow("Camera View", cam_frame)
                cv2.imshow("Expected Camera Image", expected)
                cv2.imshow("Contours", contours_vis)
                cv2.imshow("Shadow Mask", shadow_mask_u8)
                cv2.waitKey(1)

            return True

        if not run_countdown(
            screen,
            clock,
            COUNTDOWN_SECONDS,
            "SHADOW PARACHUTES",
            ready_font,
            countdown_font,
            draw_frame=draw_heat_background,
        ):
            return

        draw_world(
            screen,
            frame_surface,
            last_stage_img,
            platforms,
            list(remote_players.values()),
            stones,
            weapon_drops,
            label_font,
        )
        pygame.display.flip()

        running = True
        while running:
            dt_sec = min(clock.tick(FPS) / 1000.0, 0.1)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_v:
                        show_overlay = not show_overlay
                    elif event.key == pygame.K_c:
                        heat_field[:] = 0.0

            remote_snapshot = controller_state.drain()
            connected_clients = int(remote_snapshot["connected_clients"])
            player_snapshots = {
                str(snapshot["client_id"]): snapshot
                for snapshot in remote_snapshot["players"]
            }
            sync_remote_players(remote_players, player_snapshots, spawn_points)

            if str(remote_snapshot["last_event"]) != last_remote_event:
                last_remote_event = str(remote_snapshot["last_event"])
                if last_remote_event != "waiting":
                    print(f"Controllers: {last_remote_event}")

            if not draw_heat_background(dt_sec):
                running = False
                continue

            weapon_drop_timer -= dt_sec
            if weapon_drop_timer <= 0.0:
                weapon = random.choice(("stone", "sword", "bow"))
                drop_x = random.uniform(20.0, max(21.0, width - WEAPON_DROP_SIZE[0] - 20.0))
                weapon_drops.append(WeaponDrop(weapon, drop_x, float(-WEAPON_DROP_SIZE[1])))
                weapon_drop_timer = WEAPON_DROP_INTERVAL * random.uniform(0.75, 1.25)

            for player_id in sorted(player_snapshots):
                snapshot = player_snapshots[player_id]
                held = {
                    button: bool(snapshot["buttons"].get(button, False))
                    for button in BUTTON_NAMES
                }
                pressed = build_pressed_state(snapshot["events"])
                joystick = snapshot.get("joystick", {"x": 0.0, "y": 0.0})
                if not isinstance(joystick, dict):
                    joystick = {"x": 0.0, "y": 0.0}
                thrown_stone = remote_players[player_id].update(
                    dt_sec,
                    held,
                    pressed,
                    joystick,
                    shadow_mask,
                    heat_field,
                    width,
                    height,
                )
                if thrown_stone is not None:
                    stones.append(thrown_stone)
            resolve_player_collisions(list(remote_players.values()), width, height)
            weapon_drops = update_weapon_drops(
                weapon_drops,
                list(remote_players.values()),
                dt_sec,
                heat_field,
                shadow_mask,
                width,
                height,
            )
            resolve_sword_attacks(list(remote_players.values()), weapon_drops)
            stones = update_stones(
                stones, remote_players, platforms, dt_sec, width, height, weapon_drops
            )
            update_falling_players(
                list(remote_players.values()), dt_sec, shadow_mask, width, height
            )

            draw_world(
                screen,
                frame_surface,
                last_stage_img,
                platforms,
                list(remote_players.values()),
                stones,
                weapon_drops,
                label_font,
            )
            pygame.display.flip()

            if frame_count % 60 == 0:
                heat_peak = float(np.max(heat_field))
                fps = 1.0 / max(1e-6, time.time() - t_prev)
                alive_players = sum(1 for player in remote_players.values() if player.alive)
                print(
                    f"FPS: {fps:.1f} | Phones: {connected_clients} | Players: {len(remote_players)} "
                    f"| Alive: {alive_players} | Stones: {len(stones)} | Shadows: {debug_object_count} "
                    f"| Heat peak: {heat_peak:.2f}"
                )
                t_prev = time.time()
            frame_count += 1
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
    args = parse_args()
    main(
        cached_calibration=args.cached_calibration,
        bind_host=args.bind_host,
        public_host=args.public_host,
        http_port=args.http_port,
        ws_port=args.ws_port,
        camera_index=args.camera_index,
    )
