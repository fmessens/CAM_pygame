import math
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pygame

from modules.calibration import calibrate_system
from modules.camera_magic import gain_offset_calibration, screen_to_camera_space
from modules.settings import HEIGHT, WIDTH
from modules.videostream import VideoStream, get_camera_capture

try:
    import sounddevice as sd
except ImportError:
    sd = None


GRID_H = 180
GRID_W = 240
FPS = 60

DT = 1.0
WAVE_SPEED = 0.45
DAMPING = 0.005
C2 = (WAVE_SPEED * DT) ** 2

SOURCE_X = 8
WAVE_BRUSH_RADIUS = 6

SHADOW_RATIO = 0.6
SHADOW_MIN_DELTA = 15.0
SHADOW_MIN_AREA = 200

AUDIO_PATH = "C:\\Users\\ferdi\\Music\\Sima Deep - Contagious 062.mp3"
AUDIO_SAMPLE_RATE = 44100
AUDIO_BLOCK_SIZE = 1024
LOWPASS_CUTOFF_HZ = 180.0
LOW_BAND_HZ = 220.0
AUDIO_TRACK_GAIN = 0.85
AUDIO_DRIVE_GAIN = 1.6
AUDIO_DRIVE_CLAMP = 2.2

MIC_TONE_FREQS = (110.0, 165.0, 220.0, 330.0)
MIC_TONE_GAIN = 0.22
MIC_LEVEL_SMOOTH = 0.2
MIC_HEIGHT_GAIN = 4.2
MIC_VEL_GAIN = 2.2
MIC_NOISE_FLOOR = 0.03

REVERB_WET_BASE = 0.0
REVERB_WET_RANGE = 0.5
REVERB_FEEDBACK_BASE = 0.6
REVERB_FEEDBACK_RANGE = 0.3
REVERB_SMOOTH = 0.2
REVERB_SHADOW_BASE = 0.15
REVERB_SHADOW_BOOST = 0.85
REVERB_SEND_GAIN = 1.1

MIC_COLORS = (
    (70, 220, 200),
    (255, 170, 70),
    (190, 120, 255),
    (80, 220, 130),
)


def make_kernel(radius):
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    d2 = xx * xx + yy * yy
    kernel = np.exp(-d2 / (radius * radius * 0.5))
    kernel /= kernel.max()
    return kernel


def add_kernel(field, cy, cx, kernel, scale):
    r = kernel.shape[0] // 2
    y0 = max(0, cy - r)
    x0 = max(0, cx - r)
    y1 = min(field.shape[0], cy + r + 1)
    x1 = min(field.shape[1], cx + r + 1)

    ky0 = y0 - (cy - r)
    kx0 = x0 - (cx - r)
    ky1 = ky0 + (y1 - y0)
    kx1 = kx0 + (x1 - x0)

    field[y0:y1, x0:x1] += kernel[ky0:ky1, kx0:kx1] * scale


def step(u, u_prev, u_next, solid, c2, damping):
    lap = (
        u[1:-1, 1:-1] * -4.0
        + u[1:-1, 2:]
        + u[1:-1, :-2]
        + u[2:, 1:-1]
        + u[:-2, 1:-1]
    )
    u_next[1:-1, 1:-1] = (
        (2.0 - damping) * u[1:-1, 1:-1]
        - (1.0 - damping) * u_prev[1:-1, 1:-1]
        + c2 * lap
    )
    u_next[0, :] = 0.0
    u_next[-1, :] = 0.0
    u_next[:, 0] = 0.0
    u_next[:, -1] = 0.0
    u_next[solid] = 0.0


def render(u, img, color_scale, overlay=None):
    c = np.clip(u * color_scale, -127, 127)
    img[..., 0] = (128 + c).astype(np.uint8)
    img[..., 1] = 128
    img[..., 2] = (128 - c).astype(np.uint8)
    if overlay is not None:
        img[overlay] = (30, 30, 30)


def objects_to_mask(objects, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for obj in objects:
        pts = np.array(obj["polygon_pygame"], dtype=np.int32)
        if pts.shape[0] < 3:
            continue
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
    return mask


def shadow_object_detection_calibrated(
    cam_gray_masked,
    expected,
    b_map,
    mask_diff,
    H_inv,
    ratio=SHADOW_RATIO,
    min_delta=SHADOW_MIN_DELTA,
    min_area=SHADOW_MIN_AREA,
):
    cam_f = cam_gray_masked.astype(np.float32)
    expected_f = expected.astype(np.float32)
    shadow_thresh = b_map + (expected_f - b_map) * ratio
    shadow_mask = (cam_f < shadow_thresh) & ((expected_f - cam_f) > min_delta)
    if mask_diff is not None:
        shadow_mask[mask_diff] = False

    shadow_mask_u8 = shadow_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        shadow_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    objects = []
    contours_vis = cv2.cvtColor(shadow_mask_u8, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        cv2.drawContours(contours_vis, [cnt], -1, (0, 255, 0), 2)
        pts_cam = cnt.astype(np.float32)
        pts_screen = cv2.perspectiveTransform(pts_cam, H_inv)
        pygame_polygon = []
        for p in pts_screen:
            x_cv, y_cv = p[0]
            pygame_polygon.append((int(x_cv), int(y_cv)))
        objects.append({
            "contour_cam": cnt,
            "polygon_pygame": pygame_polygon
        })

    return objects, contours_vis


def sample_bilinear(field, x, y):
    h, w = field.shape
    x = max(0.0, min(w - 1.0, x))
    y = max(0.0, min(h - 1.0, y))

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)

    tx = x - x0
    ty = y - y0

    a = (1.0 - tx) * field[y0, x0] + tx * field[y0, x1]
    b = (1.0 - tx) * field[y1, x0] + tx * field[y1, x1]
    return (1.0 - ty) * a + ty * b


def clamp(value, vmin, vmax):
    return vmin if value < vmin else vmax if value > vmax else value


def sample_shadow_strength(mask, gx, gy, radius=1):
    x = int(gx)
    y = int(gy)
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(mask.shape[1], x + radius + 1)
    y1 = min(mask.shape[0], y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    return float(mask[y0:y1, x0:x1].mean())


def compute_lowpass_signal(audio_mono, sample_rate, cutoff_hz):
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / float(sample_rate)
    alpha = dt / (rc + dt)
    out = np.empty_like(audio_mono)
    state = 0.0
    for idx, sample in enumerate(audio_mono):
        state += alpha * (sample - state)
        out[idx] = state
    return out


def compute_lowband_envelope(audio_mono, sample_rate, block_size, cutoff_hz):
    if audio_mono.size == 0:
        return np.zeros(1, dtype=np.float32)
    hop = block_size
    block_count = int(math.ceil(audio_mono.size / float(block_size)))
    window = np.hanning(block_size).astype(np.float32)
    max_bin = max(1, int(cutoff_hz / sample_rate * block_size))
    env = np.zeros(block_count, dtype=np.float32)
    for idx in range(block_count):
        start = idx * hop
        block = audio_mono[start:start + block_size]
        if block.size < block_size:
            pad = np.zeros(block_size, dtype=np.float32)
            pad[:block.size] = block
            block = pad
        spectrum = np.fft.rfft(block * window)
        mags = np.abs(spectrum[:max_bin])
        env[idx] = float(np.sqrt(np.mean(mags * mags)))
    peak = float(env.max())
    if peak > 1e-6:
        env /= peak
    return env


def load_audio_array(path, sample_rate):
    if not path.is_file():
        return None, None, f"Audio file not found: {path}"
    mixer_started = False
    try:
        if pygame.mixer.get_init() is None:
            try:
                pygame.mixer.init(frequency=sample_rate, channels=2, size=-16)
            except pygame.error:
                pygame.mixer.init()
            mixer_started = True
        sound = pygame.mixer.Sound(str(path))
        raw = pygame.sndarray.array(sound)
        freq, _, _ = pygame.mixer.get_init()
    except pygame.error as exc:
        if mixer_started:
            pygame.mixer.quit()
        return None, None, f"pygame mixer error: {exc}"

    if mixer_started:
        pygame.mixer.quit()

    if raw.ndim == 1:
        raw = raw[:, None]
    if raw.shape[1] == 1:
        raw = np.repeat(raw, 2, axis=1)
    elif raw.shape[1] > 2:
        raw = raw[:, :2]

    if np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        scale = float(max(abs(info.min), info.max))
        audio = raw.astype(np.float32) / scale
    else:
        audio = raw.astype(np.float32)

    return audio, int(freq), None


class CombFilter:
    def __init__(self, delay):
        self.buf = np.zeros(delay, dtype=np.float32)
        self.idx = 0

    def process(self, sample, feedback):
        out = self.buf[self.idx]
        self.buf[self.idx] = sample + out * feedback
        self.idx += 1
        if self.idx >= self.buf.size:
            self.idx = 0
        return out


class AllPassFilter:
    def __init__(self, delay, feedback):
        self.buf = np.zeros(delay, dtype=np.float32)
        self.idx = 0
        self.feedback = feedback

    def process(self, sample):
        buf = self.buf[self.idx]
        out = -sample + buf
        self.buf[self.idx] = sample + buf * self.feedback
        self.idx += 1
        if self.idx >= self.buf.size:
            self.idx = 0
        return out


class SimpleReverb:
    def __init__(self, sample_rate):
        scale = sample_rate / 44100.0
        comb_delays = [int(d * scale) for d in (1116, 1188, 1277, 1356)]
        allpass_delays = [int(d * scale) for d in (556, 441)]
        self.comb_gain = 1.0 / len(comb_delays)
        self.allpass_feedback = 0.5
        self.combs = []
        self.allpasses = []
        for ch in range(2):
            offset = ch * 23
            self.combs.append([CombFilter(d + offset) for d in comb_delays])
            self.allpasses.append([
                AllPassFilter(d + offset, self.allpass_feedback) for d in allpass_delays
            ])

    def process(self, block, feedback):
        frames = block.shape[0]
        out = np.zeros_like(block)
        for ch in range(2):
            combs = self.combs[ch]
            allpasses = self.allpasses[ch]
            for i in range(frames):
                sample = block[i, ch]
                acc = 0.0
                for comb in combs:
                    acc += comb.process(sample, feedback)
                acc *= self.comb_gain
                for ap in allpasses:
                    acc = ap.process(acc)
                out[i, ch] = acc
        return out


class AudioEngine:
    def __init__(self, audio, sample_rate, block_size, mic_freqs):
        self.audio = audio
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.mic_freqs = np.array(mic_freqs, dtype=np.float32)
        self.mic_levels = np.zeros(len(mic_freqs), dtype=np.float32)
        self.phases = np.zeros(len(mic_freqs), dtype=np.float32)
        self.low_signal = None
        self.low_env = None
        self.drive_sample = 0.0
        self.drive_env = 0.0
        self.audio_level = 0.0
        self.reverb_wet = 0.0
        self.reverb_feedback = REVERB_FEEDBACK_BASE
        self.pos = 0
        self.lock = threading.Lock()
        self.stream = None
        self.error = None
        self.reverb = SimpleReverb(sample_rate)

        mono = audio.mean(axis=1)
        self.low_signal = compute_lowpass_signal(mono, sample_rate, LOWPASS_CUTOFF_HZ)
        self.low_env = compute_lowband_envelope(mono, sample_rate, block_size, LOW_BAND_HZ)

    def start(self):
        if sd is None:
            self.error = "sounddevice is not installed"
            return False
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=2,
                dtype="float32",
                callback=self._callback,
            )
            self.stream.start()
            return True
        except Exception as exc:
            self.error = str(exc)
            return False

    def _callback(self, outdata, frames, time_info, status):
        audio_len = self.audio.shape[0]
        end = self.pos + frames
        if end <= audio_len:
            chunk = self.audio[self.pos:end]
        else:
            tail = audio_len - self.pos
            head = end - audio_len
            chunk = np.vstack((self.audio[self.pos:], self.audio[:head]))
        chunk = chunk * AUDIO_TRACK_GAIN

        indices = np.arange(frames, dtype=np.float32)
        tone = np.zeros(frames, dtype=np.float32)
        with self.lock:
            mic_levels = self.mic_levels.copy()
            reverb_wet = self.reverb_wet
            reverb_feedback = self.reverb_feedback

        for idx, freq in enumerate(self.mic_freqs):
            amp = mic_levels[idx] * MIC_TONE_GAIN
            if amp <= 1e-6:
                continue
            phase = self.phases[idx]
            omega = 2.0 * math.pi * freq / self.sample_rate
            tone += amp * np.sin(phase + omega * indices)
            phase = phase + omega * frames
            self.phases[idx] = phase - math.floor(phase / (2.0 * math.pi)) * (2.0 * math.pi)

        mixed = chunk + tone[:, None]
        if reverb_wet > 1e-5:
            wet = self.reverb.process(mixed, reverb_feedback)
            mixed = mixed + wet * reverb_wet
        mixed = np.clip(mixed, -1.0, 1.0)
        outdata[:] = mixed

        block_idx = (self.pos // self.block_size) % len(self.low_env)
        drive_sample = self.low_signal[self.pos % len(self.low_signal)]
        drive_env = self.low_env[block_idx]
        rms = float(np.sqrt(np.mean(chunk[:, 0] * chunk[:, 0])))
        with self.lock:
            self.drive_sample = drive_sample
            self.drive_env = drive_env
            self.audio_level = 0.9 * self.audio_level + 0.1 * rms

        self.pos = end % audio_len

    def set_mic_levels(self, levels):
        with self.lock:
            self.mic_levels[:] = levels

    def set_reverb_amount(self, amount):
        amount = clamp(amount, 0.0, 1.0)
        wet = REVERB_WET_BASE + REVERB_WET_RANGE * amount
        feedback = REVERB_FEEDBACK_BASE + REVERB_FEEDBACK_RANGE * amount
        feedback = clamp(feedback, 0.1, 0.95)
        with self.lock:
            self.reverb_wet = wet
            self.reverb_feedback = feedback

    def get_drive(self):
        with self.lock:
            return self.drive_sample, self.drive_env, self.audio_level

    def stop(self):
        if self.stream is None:
            return
        self.stream.stop()
        self.stream.close()
        self.stream = None


class WaveMic:
    def __init__(self, gx, gy, color):
        self.gx = float(gx)
        self.gy = float(gy)
        self.color = color
        self.level = 0.0
        self.shadow = 0.0
        self.reverb = 0.0


def main():
    audio_path = Path(AUDIO_PATH)
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])

    audio_data, audio_rate, audio_error = load_audio_array(audio_path, AUDIO_SAMPLE_RATE)
    if audio_error:
        print(audio_error)
        return

    audio_engine = AudioEngine(audio_data, audio_rate, AUDIO_BLOCK_SIZE, MIC_TONE_FREQS)
    audio_ok = audio_engine.start()
    if audio_ok:
        print(f"Audio output: {audio_path} @ {audio_rate} Hz")
    else:
        print("Audio output failed.")
        if audio_engine.error:
            print(audio_engine.error)
        return

    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Soundscape - Waves + Shadows + MP3")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    cap = get_camera_capture()
    H, mask, _, a_map, b_map = calibrate_system(
        cap=cap, screen=screen, WIDTH=WIDTH, HEIGHT=HEIGHT
    )
    H_inv = np.linalg.inv(H)
    cap.release()

    cap_stream = VideoStream(0)

    u = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    u_prev = np.zeros_like(u)
    u_next = np.zeros_like(u)
    solid = np.zeros((GRID_H, GRID_W), dtype=bool)

    source_mask = np.zeros_like(solid)
    source_mask[:, SOURCE_X] = True

    wave_kernel = make_kernel(WAVE_BRUSH_RADIUS)
    img = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
    frame = pygame.Surface((GRID_W, GRID_H))

    scale_x = WIDTH / GRID_W
    scale_y = HEIGHT / GRID_H

    mic_positions = [
        (GRID_W * 0.3, GRID_H * 0.3),
        (GRID_W * 0.7, GRID_H * 0.3),
        (GRID_W * 0.3, GRID_H * 0.7),
        (GRID_W * 0.7, GRID_H * 0.7),
    ]
    mics = [WaveMic(x, y, MIC_COLORS[i % len(MIC_COLORS)]) for i, (x, y) in enumerate(mic_positions)]
    mic_levels = np.zeros(len(mics), dtype=np.float32)

    running = True
    paused = False
    audio_drive_on = True
    show_overlay = False
    reverb_amount = 0.0
    frame_count = 0
    t_prev = pygame.time.get_ticks()

    print("Controls: left click=wave, right click=move mic,")
    print("C=clear, M=toggle audio drive, P=pause, V=overlay, ESC=quit")

    while running:
        dt_sec = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_c:
                    u[:] = 0.0
                    u_prev[:] = 0.0
                elif event.key == pygame.K_m:
                    audio_drive_on = not audio_drive_on
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_v:
                    show_overlay = not show_overlay

        ret, cam_frame = cap_stream.read()
        if not ret:
            continue

        cam_gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
        cam_gray_masked = cv2.bitwise_and(cam_gray, cam_gray, mask=mask)

        h_cam, w_cam = cam_gray.shape[:2]
        warped_proj_gray_masked = screen_to_camera_space(screen, H, mask, h_cam, w_cam)
        expected = gain_offset_calibration(warped_proj_gray_masked, a_map, b_map)

        mask_diff = np.zeros_like(cam_gray_masked, dtype=bool)
        objects, contours_vis = shadow_object_detection_calibrated(
            cam_gray_masked, expected, b_map, mask_diff, H_inv
        )

        shadow_screen = objects_to_mask(objects, HEIGHT, WIDTH)
        shadow_small = cv2.resize(
            shadow_screen, (GRID_W, GRID_H), interpolation=cv2.INTER_NEAREST
        )
        solid = shadow_small > 0
        u[solid] = 0.0
        u_prev[solid] = 0.0

        mx, my = pygame.mouse.get_pos()
        gx = int(mx / scale_x)
        gy = int(my / scale_y)
        gx = int(clamp(gx, 0, GRID_W - 1))
        gy = int(clamp(gy, 0, GRID_H - 1))
        buttons = pygame.mouse.get_pressed(3)
        if buttons[0]:
            add_kernel(u, gy, gx, wave_kernel, 2.0)
        if buttons[2]:
            closest = min(mics, key=lambda m: (m.gx - gx) ** 2 + (m.gy - gy) ** 2)
            closest.gx = gx
            closest.gy = gy

        if not paused:
            step(u, u_prev, u_next, solid, C2, DAMPING)
            u_prev, u, u_next = u, u_next, u_prev

            if audio_drive_on:
                drive_sample, drive_env, _ = audio_engine.get_drive()
                drive = drive_sample * (0.35 + 1.35 * drive_env) * AUDIO_DRIVE_GAIN
                drive = clamp(drive, -AUDIO_DRIVE_CLAMP, AUDIO_DRIVE_CLAMP)
                mask_src = source_mask & ~solid
                u[mask_src] = drive

        reverb_sum = 0.0
        for idx, mic in enumerate(mics):
            height = sample_bilinear(u, mic.gx, mic.gy)
            prev = sample_bilinear(u_prev, mic.gx, mic.gy)
            energy = abs(height) * MIC_HEIGHT_GAIN + abs(height - prev) * MIC_VEL_GAIN
            level = clamp(energy, 0.0, 1.0)
            if level < MIC_NOISE_FLOOR:
                level = 0.0
            mic.level += (level - mic.level) * MIC_LEVEL_SMOOTH
            mic_levels[idx] = mic.level
            mic.shadow = sample_shadow_strength(shadow_small, mic.gx, mic.gy, radius=1)
            mic.reverb = mic.level * (REVERB_SHADOW_BASE + REVERB_SHADOW_BOOST * mic.shadow)
            reverb_sum += mic.reverb

        audio_engine.set_mic_levels(mic_levels)
        reverb_target = clamp(reverb_sum * REVERB_SEND_GAIN / max(1, len(mics)), 0.0, 1.0)
        reverb_amount += (reverb_target - reverb_amount) * REVERB_SMOOTH
        audio_engine.set_reverb_amount(reverb_amount)

        render(u, img, color_scale=220.0, overlay=solid if show_overlay else None)
        pygame.surfarray.blit_array(frame, img.swapaxes(0, 1))
        pygame.transform.scale(frame, screen.get_size(), screen)

        for mic in mics:
            sx = int(mic.gx * scale_x)
            sy = int(mic.gy * scale_y)
            boost = int(100 * mic.level)
            shadow_boost = int(70 * mic.shadow)
            reverb_boost = int(90 * mic.reverb)
            color = (
                min(255, mic.color[0] + boost + shadow_boost + reverb_boost),
                min(255, mic.color[1] + boost + shadow_boost + reverb_boost),
                min(255, mic.color[2] + boost + shadow_boost + reverb_boost),
            )
            pygame.draw.circle(screen, color, (sx, sy), 8)
            pygame.draw.circle(screen, (20, 20, 20), (sx, sy), 8, 1)
            if mic.shadow > 0.1:
                pygame.draw.circle(screen, (220, 220, 220), (sx, sy), 12, 1)

        _, _, audio_level = audio_engine.get_drive()
        bar_w = int(WIDTH * 0.28)
        bar_h = 8
        bar_x = 18
        bar_y = HEIGHT - 18
        pygame.draw.rect(screen, (25, 25, 25), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(
            screen,
            (70, 210, 160),
            (bar_x, bar_y, int(bar_w * clamp(audio_level * 2.5, 0.0, 1.0)), bar_h),
        )
        label = font.render(f"Audio: {audio_path.name}", True, (220, 220, 220))
        screen.blit(label, (bar_x, bar_y - 20))

        pygame.display.flip()

        if frame_count % 60 == 0:
            now = pygame.time.get_ticks()
            fps = 1000.0 / max(1.0, now - t_prev)
            print(f"FPS: {fps:.1f} | Shadows: {len(objects)}")
            t_prev = now
        frame_count += 1

        cv2.imshow("Camera View", cam_frame)
        cv2.imshow("Expected Camera Image", expected)
        cv2.imshow("Contours", contours_vis)
        cv2.imshow("Shadow Mask (Screen)", shadow_screen)
        cv2.waitKey(1)

    cap_stream.release()
    audio_engine.stop()
    pygame.quit()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
