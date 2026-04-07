from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import struct
import threading
from urllib.parse import parse_qs, urlsplit

from controller_state import ControllerState


WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _build_accept_key(client_key: str) -> str:
    raw = f"{client_key}{WS_GUID}".encode("ascii")
    return base64.b64encode(hashlib.sha1(raw).digest()).decode("ascii")


class ControllerWebSocketServer:
    def __init__(self, controller_state: ControllerState, bind_host: str, port: int) -> None:
        self.controller_state = controller_state
        self.bind_host = bind_host
        self.port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: asyncio.base_events.Server | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, name="controller-websocket", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._server = self._loop.run_until_complete(
            asyncio.start_server(self._handle_client, host=self.bind_host, port=self.port)
        )
        self._ready.set()

        try:
            self._loop.run_forever()
        finally:
            self._server.close()
            self._loop.run_until_complete(self._server.wait_closed())
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        client_id = ""
        connection_id = 0
        try:
            request_line, headers = await self._read_http_headers(reader)
            upgrade = headers.get("upgrade", "").lower()
            connection = headers.get("connection", "").lower()
            key = headers.get("sec-websocket-key")

            if upgrade != "websocket" or "upgrade" not in connection or not key:
                await self._send_http_error(writer, 400, "Expected WebSocket upgrade.")
                return

            accept_key = _build_accept_key(key)
            handshake = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept_key}\r\n\r\n"
            )
            writer.write(handshake.encode("ascii"))
            await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError, UnicodeDecodeError):
            writer.close()
            await writer.wait_closed()
            return

        device_id = self._extract_device_id(request_line)
        session_info = self.controller_state.client_connected(device_id=device_id)
        client_id = str(session_info["client_id"])
        connection_id = int(session_info["connection_id"])
        await self._send_json(writer, {"type": "hello", **session_info})

        try:
            while True:
                opcode, payload = await self._read_frame(reader)

                if opcode == 0x8:
                    break
                if opcode == 0x9:
                    await self._send_frame(writer, payload, opcode=0xA)
                    continue
                if opcode != 0x1:
                    continue

                message = json.loads(payload.decode("utf-8"))
                await self._handle_message(
                    client_id=client_id,
                    message=message,
                    connection_id=connection_id,
                )
        except (
            asyncio.IncompleteReadError,
            ConnectionError,
            json.JSONDecodeError,
            struct.error,
            UnicodeDecodeError,
        ):
            pass
        finally:
            if client_id:
                self.controller_state.client_disconnected(
                    client_id,
                    connection_id=connection_id,
                )
            try:
                await self._send_frame(writer, b"", opcode=0x8)
            except ConnectionError:
                pass
            writer.close()
            await writer.wait_closed()

    async def _handle_message(
        self,
        client_id: str,
        message: dict[str, object],
        *,
        connection_id: int,
    ) -> None:
        kind = str(message.get("type", "")).lower()
        if kind == "hello":
            viewport = message.get("viewport")
            parsed_viewport = None
            if isinstance(viewport, dict):
                width = viewport.get("width")
                height = viewport.get("height")
                if isinstance(width, int) and isinstance(height, int):
                    parsed_viewport = {"width": width, "height": height}
            self.controller_state.update_client_meta(
                client_id,
                connection_id=connection_id,
                user_agent=str(message.get("user_agent", "") or message.get("userAgent", "")),
                viewport=parsed_viewport,
            )
            return

        if kind == "button":
            button = str(message.get("button", "")).lower()
            pressed = bool(message.get("pressed", False))
            self.controller_state.apply_button(
                client_id=client_id,
                button=button,
                pressed=pressed,
                source="remote",
                connection_id=connection_id,
            )
            return

        if kind == "release_all":
            self.controller_state.release_all(
                client_id=client_id,
                source="remote",
                connection_id=connection_id,
            )
            return

    async def _read_http_headers(
        self, reader: asyncio.StreamReader
    ) -> tuple[str, dict[str, str]]:
        raw_headers = await reader.readuntil(b"\r\n\r\n")
        header_text = raw_headers.decode("iso-8859-1")
        lines = header_text.split("\r\n")
        request_line = lines[0]
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if not line or ":" not in line:
                continue
            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()
        return request_line, headers

    @staticmethod
    def _extract_device_id(request_line: str) -> str | None:
        parts = request_line.split(" ", 2)
        if len(parts) < 2:
            return None
        query = parse_qs(urlsplit(parts[1]).query)
        raw_device_ids = query.get("device")
        if not raw_device_ids:
            return None
        device_id = str(raw_device_ids[0]).strip()
        return device_id or None

    async def _read_frame(
        self, reader: asyncio.StreamReader
    ) -> tuple[int, bytes]:
        first_byte, second_byte = await reader.readexactly(2)
        fin = bool(first_byte & 0x80)
        opcode = first_byte & 0x0F
        masked = bool(second_byte & 0x80)
        payload_len = second_byte & 0x7F

        if not fin:
            raise ConnectionError("Fragmented frames are not supported.")
        if not masked:
            raise ConnectionError("Browser frames must be masked.")

        if payload_len == 126:
            payload_len = struct.unpack("!H", await reader.readexactly(2))[0]
        elif payload_len == 127:
            payload_len = struct.unpack("!Q", await reader.readexactly(8))[0]

        mask = await reader.readexactly(4)
        payload = await reader.readexactly(payload_len) if payload_len else b""
        decoded = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        return opcode, decoded

    async def _send_http_error(
        self, writer: asyncio.StreamWriter, status: int, body: str
    ) -> None:
        encoded_body = body.encode("utf-8")
        response = (
            f"HTTP/1.1 {status} Error\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(encoded_body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii") + encoded_body
        writer.write(response)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def _send_json(
        self, writer: asyncio.StreamWriter, payload: dict[str, object]
    ) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        await self._send_frame(writer, encoded, opcode=0x1)

    async def _send_frame(
        self, writer: asyncio.StreamWriter, payload: bytes, opcode: int
    ) -> None:
        frame = bytearray()
        frame.append(0x80 | (opcode & 0x0F))

        length = len(payload)
        if length < 126:
            frame.append(length)
        elif length < 65536:
            frame.append(126)
            frame.extend(struct.pack("!H", length))
        else:
            frame.append(127)
            frame.extend(struct.pack("!Q", length))

        frame.extend(payload)
        writer.write(frame)
        await writer.drain()
