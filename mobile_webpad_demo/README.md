# Mobile WebPad Demo

Standalone prototype: multiple phones load the same touch controller page over local Wi-Fi and each phone controls its own spawned sprite in a small Pygame scene over WebSocket.

## Files

- `run_demo.py`: starts the HTTP setup page, WebSocket input server, and multiplayer Pygame demo.
- `controller_state.py`: per-phone session state and button edge events.
- `web_server.py`: serves the setup page, QR code, and controller frontend.
- `websocket_server.py`: minimal WebSocket server for browser input.
- `static/`: controller HTML/CSS/JS.

## Requirements

Use the project venv or install:

```powershell
python -m pip install pygame qrcode
```

## Run

From the repo root:

```powershell
.\.w_venv\Scripts\python.exe .\mobile_webpad_demo\run_demo.py
```

Then:

1. Open the printed setup URL on your PC browser.
2. Scan the QR code with one or more phones.
3. Each phone joins as its own player in landscape mode.

## Controls

- Each connected phone spawns a separate player sprite.
- D-pad `LEFT` / `RIGHT`: move
- D-pad `DOWN`: crouch on ground, fast-fall in air
- `A` or D-pad `UP`: jump
- `B`: dash
- `START`: reset that player

Keyboard-only extras:

- `Tab`: toggle debug panel

## Notes

- The phone must be on the same Wi-Fi network as the PC.
- Disconnecting a phone removes that player from the scene.
- Player sprites collide with each other and with the platforms.
- If LAN auto-detection fails, pass `--public-host 192.168.x.x`.
- For a quick smoke test without a visible window:

```powershell
.\.w_venv\Scripts\python.exe .\mobile_webpad_demo\run_demo.py --headless --duration 2
```
