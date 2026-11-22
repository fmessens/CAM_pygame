# CAM_pygame

Interactivity with games through a camera + projector setup.

![ball game](images/visual.jpg)

## Setup

You need a projector or beamer and a USB camera pointed at the projected screen. The default display target is screen 2 (`TARGET_DISPLAY_INDEX = 1` in the level scripts).

On this Windows setup, OpenCV camera index `1` is used for the secondary webcam in the updated levels.

![Screen setup](images/screen_setup.png)

## Install

```powershell
.\.w_venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Games

Root scripts:

- `level_1.py` - Dock level with shadow-driven waves and a boat.
- `level_2.py` - Gravity well with warped background and flying objects.
- `level_3.py` - Single-player heat glider with keyboard controls.
- `level_4.py` - Multiplayer heat gliders with QR phone controllers.
- `level_4_bis.py` - Multiplayer wraparound heat gliders with floor gaps and throwable stones.
- `level_5.py` - Multiplayer always-open parachutes with stones, non-sticky shadow collision, and no platforms.
  Stone, sword, and bow pickups periodically descend on full heat-reactive parachutes;
  defeated players fall away while their held weapon continues descending as a pickup.
- `level_7.py` - Multiplayer platformer with one-way shadow platforms, jumping, and Level 5 combat.
- `level_8.py` - Shadow Percussion. Press Space to launch a ball through the Big Beat
  in the middle. Balls wrap at all four edges. Bounce them off shadows to play the
  kick, snare, cymbal, or guitar in the corners. Press Backspace to clear the balls.
- `main_levels.py` - Runs Level 1, then Level 2 after success.

Experimental scripts live in `experiments/`.

## Run

### Shadow Arcade menu

```powershell
python .\level_menu.py --camera-index 1
```

Choose **Shadow Calibration** to align the camera and projector. Keep the red dot
in the checkerboard’s top-left square visible to the camera. Focus the projected
checkerboard and press **Space**, then advance through the OpenCV previews with a key.
The preview highlights the detected top edge in yellow.
The menu writes the shared `calibration_cache.npz` and launches every level using it.
Recalibrate if the camera or projection geometry changes. The other levels are under
**Experimental levels**. Use **Stop Level** in the separate **Level Controls**
window to end the running game. The menu can reopen that window during play.

**Tech info** explains calibration, shadow detection, the Shadow Harbor wave
solver, and the Water Wakes fluid solver. With three extended displays connected,
use **Third screen** or **Shift+F11** to open a fullscreen copy of the menu on display 3.
It follows the launcher page, switches to the QR setup for multiplayer, and
returns to the launcher page when the level stops.
The launcher and controller QR windows can each be dragged by their title bar
and resized. Press **F11** in either window to fill its current display, then
press **F11** or **Esc** to return to windowed mode.

**Vortex Bubbles** in Experimental levels uses the Water Wakes solver. Green
bubbles mark detected vortices, and the number is the live count.

**Windward** in Experimental levels is a multiplayer sailing game on the same
water simulation. Steer with the phone joystick, sail downwind for speed, and
use **Fire Left** and **Fire Right** to launch cannonballs from the bow.
Ships, projectiles, and impacts wrap at all four edges.

Phone-controlled levels use the PC’s Windows Mobile Hotspot. If it is off,
click **Open Mobile Hotspot settings**, set **Share over** to **Wi-Fi**, and turn it on.
The level starts as soon as the launcher detects the hotspot and shows two QR
codes. Scan **1** to join Wi-Fi, then **2** to open the controller once it is ready.
If the PC is connected to a phone’s hotspot, select **Use current Wi-Fi** instead
and scan the controller QR. To share the phone hotspot with other players, drop
a screenshot of the phone’s Wi-Fi QR into the controller QR window. Players can
then scan the phone hotspot QR first and the controller QR second.
The controllers communicate locally over TCP ports `8000` and `8765`.

### Standalone levels

Level 1:

```powershell
.\.w_venv\Scripts\python.exe .\level_1.py --camera-index 1
```

Level 2:

```powershell
.\.w_venv\Scripts\python.exe .\level_2.py --image images\galaxy-1489300667bhk.jpg --camera-index 1
```

Level 3:

```powershell
.\.w_venv\Scripts\python.exe .\level_3.py --camera-index 1
```

Level 4:

```powershell
.\.w_venv\Scripts\python.exe .\level_4.py
```

Level 4 Bis:

```powershell
.\.w_venv\Scripts\python.exe .\level_4_bis.py --camera-index 1
```

Level 5:

```powershell
.\.w_venv\Scripts\python.exe .\level_5.py --camera-index 1
```

Level 7:

```powershell
python .\level_7.py --camera-index 1
```

Scan the QR code on the printed setup page to join from a phone. The joystick moves and aims.
Use **Jump** to reach shadow platforms and **Attack / Throw** to use your weapon.
Stone, sword, and bow pickups disappear half a second after landing on the floor or a shadow.
Press **Jump** to rejoin after falling when defeated.

Standalone extended calibration experiment:

```powershell
.\.w_venv\Scripts\python.exe .\calibration_experiment.py --camera-index 1
.\.w_venv\Scripts\python.exe .\level_5.py --camera-index 1 --cached-calibration
```

The experiment holds black, Level 5 gray, and white for several seconds and combines
multiple camera frames. Adjust it with `--settle-seconds`, `--sample-seconds`,
`--statistic mean`, or a camera-supported `--camera-exposure` value.

Combined Level 1 -> Level 2 flow:

```powershell
.\.w_venv\Scripts\python.exe .\main_levels.py --image images\galaxy-1489300667bhk.jpg --camera-index 1
```

## Camera Selection

`level_1.py`, `level_2.py`, `level_4_bis.py`, and `main_levels.py` support:

```powershell
--camera-index 1
```

Use `--camera-index 0` if you want the primary webcam instead.

## Phone Controllers

`level_4.py` and `level_4_bis.py` start a local setup page and QR controller page. Phones must be on the same Wi-Fi network as the computer.

Useful options:

```powershell
--public-host 192.168.x.x
--http-port 8000
--ws-port 8765
```

Use `--public-host` if auto-detection prints `127.0.0.1` or phones cannot connect.

## Calibration Cache

Level scripts support `--cached-calibration`.

- Without it, the script runs camera/projector calibration and writes `calibration_cache.npz`.
- With it, the script loads `calibration_cache.npz` and skips calibration.
