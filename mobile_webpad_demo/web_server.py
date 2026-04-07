from __future__ import annotations

from dataclasses import dataclass
import html
import io
from pathlib import Path
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from controller_state import BUTTON_NAMES, ControllerState

try:
    import qrcode
    import qrcode.image.svg
except ImportError:  # pragma: no cover - handled at runtime
    qrcode = None


STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass(slots=True)
class ServerConfig:
    bind_host: str
    public_host: str
    http_port: int
    ws_port: int
    app_title: str = "Mobile WebPad Demo"
    helper_note_html: str = "A / UP JUMP<br>B DASH"

    @property
    def setup_url(self) -> str:
        return f"http://{self.public_host}:{self.http_port}/"

    @property
    def controller_url(self) -> str:
        return f"http://{self.public_host}:{self.http_port}/controller"


class ControllerHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        config: ServerConfig,
        controller_state: ControllerState,
    ) -> None:
        super().__init__(server_address, ControllerRequestHandler)
        self.config = config
        self.controller_state = controller_state
        self.static_dir = STATIC_DIR


class ControllerRequestHandler(BaseHTTPRequestHandler):
    server: ControllerHttpServer
    server_version = "MobileWebPad/0.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib interface
        path = urlsplit(self.path).path

        if path in {"/", "/setup"}:
            self._send_bytes(self._build_setup_page().encode("utf-8"), "text/html; charset=utf-8")
            return
        if path in {"/controller", "/controller/"}:
            self._serve_static_file("controller.html", "text/html; charset=utf-8")
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
        if path.startswith("/static/"):
            relative_path = path.removeprefix("/static/")
            self._serve_static_file(relative_path)
            return

        self.send_error(404, "Not found")

    def log_message(self, format: str, *args: object) -> None:
        return

    def _serve_static_file(self, relative_path: str, content_type: str | None = None) -> None:
        base = self.server.static_dir.resolve()
        target = (base / relative_path).resolve()

        try:
            target.relative_to(base)
        except ValueError:
            self.send_error(404, "Not found")
            return

        if not target.is_file():
            self.send_error(404, "Not found")
            return

        mime = content_type or self._guess_mime_type(target)
        self._send_bytes(target.read_bytes(), mime)

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
      --bg: #151f13;
      --panel: #26351d;
      --panel-2: #314325;
      --ink: #e3f2b7;
      --accent: #bfe36a;
      --muted: #95a872;
      --edge: #50633b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
      font-family: Consolas, "Lucida Console", monospace;
      background:
        radial-gradient(circle at top, #314625 0%, rgba(49, 70, 37, 0.15) 30%, transparent 60%),
        linear-gradient(180deg, #0c120b 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .card {{
      width: min(920px, 100%);
      display: grid;
      grid-template-columns: 300px 1fr;
      gap: 24px;
      padding: 24px;
      border: 3px solid var(--edge);
      border-radius: 22px;
      background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
      box-shadow: 0 28px 80px rgba(0, 0, 0, 0.35);
    }}
    .qr-wrap {{
      display: grid;
      place-items: center;
      min-height: 300px;
      border: 2px solid var(--edge);
      border-radius: 18px;
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
    .pill {{
      display: inline-block;
      margin-bottom: 12px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--edge);
      color: var(--accent);
      background: rgba(0, 0, 0, 0.18);
    }}
    .link-box {{
      margin: 18px 0;
      padding: 14px;
      border-radius: 14px;
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
      border-radius: 14px;
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
      .card {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="card">
    <section class="qr-wrap">
      {qr_block}
    </section>
    <section>
      <div class="pill">Phone controller over local Wi-Fi</div>
      <h1>{html.escape(config.app_title)}</h1>
      <p>Each phone that scans the QR joins as its own player. Keep every device on the same Wi-Fi network.</p>
      <div class="link-box">
        <strong>Controller URL</strong>
        <a href="{controller_url}">{controller_url}</a>
      </div>
      <div class="link-box">
        <strong>Setup page URL</strong>
        <a href="{setup_url}">{setup_url}</a>
      </div>
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
        document.getElementById('connection-status').textContent =
          `Players: ${{
            data.connected_clients
          }} joined`;
        document.getElementById('active-players').textContent =
          `Joined: ${{
            labels
          }}`;
        document.getElementById('last-event').textContent =
          `Last event: ${{
            data.last_event
          }}`;
      }} catch (error) {{
        document.getElementById('connection-status').textContent =
          'Players: unavailable';
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

    @staticmethod
    def _guess_mime_type(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".html":
            return "text/html; charset=utf-8"
        if suffix == ".css":
            return "text/css; charset=utf-8"
        if suffix == ".js":
            return "application/javascript; charset=utf-8"
        if suffix == ".svg":
            return "image/svg+xml"
        if suffix == ".json":
            return "application/json; charset=utf-8"
        return "application/octet-stream"


def start_http_server(
    controller_state: ControllerState, config: ServerConfig
) -> tuple[ControllerHttpServer, threading.Thread]:
    server = ControllerHttpServer((config.bind_host, config.http_port), config, controller_state)
    thread = threading.Thread(target=server.serve_forever, name="controller-http", daemon=True)
    thread.start()
    return server, thread
