from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
import time


BUTTON_NAMES = ("up", "down", "left", "right", "a", "b", "start", "select")


@dataclass(slots=True, frozen=True)
class ButtonEvent:
    client_id: str
    button: str
    pressed: bool
    source: str
    timestamp: float

    @property
    def action(self) -> str:
        return "press" if self.pressed else "release"


@dataclass(slots=True)
class ClientSession:
    client_id: str
    label: str
    color_index: int
    connected_at: float
    device_id: str | None = None
    connection_id: int = 0
    buttons: dict[str, bool] = field(default_factory=lambda: {button: False for button in BUTTON_NAMES})
    events: deque[ButtonEvent] = field(default_factory=deque)
    last_event: str = "joined"
    last_input_time: float = 0.0
    user_agent: str | None = None
    viewport: dict[str, int] | None = None


class ControllerState:
    def __init__(self) -> None:
        self._sessions: dict[str, ClientSession] = {}
        self._device_to_client: dict[str, str] = {}
        self._session_counter = 0
        self._connection_counter = 0
        self._last_event = "waiting"
        self._lock = Lock()

    def client_connected(self, *, device_id: str | None = None) -> dict[str, object]:
        now = time.monotonic()
        with self._lock:
            self._connection_counter += 1
            connection_id = self._connection_counter

            if device_id:
                existing_client_id = self._device_to_client.get(device_id)
                existing_session = self._sessions.get(existing_client_id or "")
                if existing_session is not None:
                    self._release_all_locked(
                        session=existing_session,
                        now=now,
                        source="reconnect",
                    )
                    existing_session.connection_id = connection_id
                    existing_session.last_event = "reconnected"
                    existing_session.last_input_time = now
                    self._last_event = f"{existing_session.label} reconnected"
                    return {
                        "client_id": existing_session.client_id,
                        "label": existing_session.label,
                        "color_index": existing_session.color_index,
                        "buttons": BUTTON_NAMES,
                        "connection_id": connection_id,
                    }

            self._session_counter += 1
            session_number = self._session_counter
            client_id = f"player-{session_number}"
            label = f"P{session_number}"
            session = ClientSession(
                client_id=client_id,
                label=label,
                color_index=session_number,
                connected_at=now,
                device_id=device_id,
                connection_id=connection_id,
            )
            self._sessions[client_id] = session
            if device_id:
                self._device_to_client[device_id] = client_id
            self._last_event = f"{label} joined"
            return {
                "client_id": client_id,
                "label": label,
                "color_index": session.color_index,
                "buttons": BUTTON_NAMES,
                "connection_id": connection_id,
            }

    def client_disconnected(self, client_id: str, *, connection_id: int | None = None) -> int:
        now = time.monotonic()
        with self._lock:
            session = self._sessions.get(client_id)
            if session is None:
                return len(self._sessions)
            if connection_id is not None and session.connection_id != connection_id:
                return len(self._sessions)

            self._sessions.pop(client_id, None)
            if session.device_id:
                mapped_client_id = self._device_to_client.get(session.device_id)
                if mapped_client_id == client_id:
                    self._device_to_client.pop(session.device_id, None)

            self._release_all_locked(session=session, now=now, source="disconnect")
            self._last_event = f"{session.label} left"
            return len(self._sessions)

    def update_client_meta(
        self,
        client_id: str,
        *,
        connection_id: int | None = None,
        user_agent: str | None = None,
        viewport: dict[str, int] | None = None,
    ) -> None:
        with self._lock:
            session = self._sessions.get(client_id)
            if session is None:
                return
            if connection_id is not None and session.connection_id != connection_id:
                return
            if user_agent:
                session.user_agent = user_agent
            if viewport:
                session.viewport = viewport

    def apply_button(
        self,
        client_id: str,
        button: str,
        pressed: bool,
        source: str = "remote",
        *,
        connection_id: int | None = None,
    ) -> bool:
        if button not in BUTTON_NAMES:
            return False

        now = time.monotonic()
        with self._lock:
            session = self._sessions.get(client_id)
            if session is None:
                return False
            if connection_id is not None and session.connection_id != connection_id:
                return False

            previous = session.buttons[button]
            if previous == pressed:
                session.last_input_time = now
                return False

            session.buttons[button] = pressed
            event = ButtonEvent(
                client_id=client_id,
                button=button,
                pressed=pressed,
                source=source,
                timestamp=now,
            )
            session.events.append(event)
            session.last_event = f"{button}:{event.action}"
            session.last_input_time = now
            self._last_event = f"{session.label} {session.last_event}"
            return True

    def release_all(
        self,
        client_id: str,
        source: str = "system",
        *,
        connection_id: int | None = None,
    ) -> bool:
        now = time.monotonic()
        with self._lock:
            session = self._sessions.get(client_id)
            if session is None:
                return False
            if connection_id is not None and session.connection_id != connection_id:
                return False
            return self._release_all_locked(session=session, now=now, source=source)

    def _release_all_locked(self, session: ClientSession, now: float, source: str) -> bool:
        changed = False
        for button, pressed in session.buttons.items():
            if not pressed:
                continue
            session.buttons[button] = False
            session.events.append(
                ButtonEvent(
                    client_id=session.client_id,
                    button=button,
                    pressed=False,
                    source=source,
                    timestamp=now,
                )
            )
            changed = True

        if changed:
            session.last_event = "release_all"
            session.last_input_time = now
            self._last_event = f"{session.label} release_all"
        return changed

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            players = [
                self._session_snapshot(session=session, clear_events=False, include_events=False)
                for session in self._sessions.values()
            ]
            return {
                "connected_clients": len(self._sessions),
                "last_event": self._last_event,
                "active_players": players,
            }

    def drain(self) -> dict[str, object]:
        with self._lock:
            players = [
                self._session_snapshot(session=session, clear_events=True, include_events=True)
                for session in self._sessions.values()
            ]
            return {
                "connected_clients": len(self._sessions),
                "last_event": self._last_event,
                "players": players,
            }

    @staticmethod
    def _session_snapshot(
        session: ClientSession,
        *,
        clear_events: bool,
        include_events: bool,
    ) -> dict[str, object]:
        age = None
        if session.last_input_time:
            age = round(time.monotonic() - session.last_input_time, 3)

        events = list(session.events) if include_events else []
        if clear_events:
            session.events.clear()

        return {
            "client_id": session.client_id,
            "label": session.label,
            "color_index": session.color_index,
            "connected_at": round(session.connected_at, 3),
            "device_id": session.device_id,
            "buttons": session.buttons.copy(),
            "events": events,
            "last_event": session.last_event,
            "last_input_age": age,
            "viewport": session.viewport,
        }
