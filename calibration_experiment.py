"""Standalone projector/camera calibration experiment for level_5.py.

Projects a checkerboard for geometry, then holds black, Level 5 gray, and
white long enough for camera auto-exposure to settle. Multiple frames are
combined per level and a per-pixel linear gain/offset model is saved in the
same format consumed by level_5.py.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import pygame

from level_3 import CALIBRATION_CACHE, TARGET_DISPLAY_INDEX, create_fullscreen_display
from modules.calibration import (
    build_projection_mask,
    compute_projection_polygon,
    compute_screen_object_points,
    detect_checkerboard,
    orient_checkerboard,
    preview_homography,
    project_checkerboard,
)
from modules.videostream import get_camera_capture


LEVELS = (0, 128, 255)


def pump_or_quit() -> None:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            raise KeyboardInterrupt


def capture_geometry(cap, screen, cols: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    width, height = screen.get_size()
    project_checkerboard(cols, rows, screen)
    print("Geometry: keep the checkerboard and its red top-left dot visible, then press SPACE.")
    pattern_inner = (cols - 1, rows - 1)

    while True:
        pump_or_quit()
        ok, frame = cap.read()
        if not ok:
            continue
        found, corners = detect_checkerboard(frame, pattern_inner)
        preview = frame.copy()
        if found:
            cv2.drawChessboardCorners(preview, pattern_inner, corners, True)
        cv2.putText(
            preview,
            "SPACE: capture  ESC: cancel",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (40, 220, 40) if found else (40, 40, 230),
            2,
        )
        cv2.imshow("Calibration geometry", preview)
        key = cv2.waitKey(1) & 0xFF
        space_pressed = key == 32 or pygame.key.get_pressed()[pygame.K_SPACE]
        if not space_pressed:
            continue
        if not found:
            print("Checkerboard was not detected; adjust the camera/projector and retry.")
            time.sleep(0.25)
            continue

        screen_points = compute_screen_object_points(cols, rows, width, height)
        oriented = orient_checkerboard(frame, corners, pattern_inner, screen_points, width, height)
        if oriented is None:
            print("Red orientation dot not detected. Make it visible and press SPACE again.")
            time.sleep(0.25)
            continue
        homography, corners = oriented
        camera_corners = compute_projection_polygon(homography, width, height)
        preview_homography(frame, pattern_inner, corners, camera_corners)
        return homography, build_projection_mask(frame, camera_corners)


def capture_level(
    cap,
    screen,
    mask: np.ndarray,
    level: int,
    settle_seconds: float,
    sample_seconds: float,
    statistic: str,
) -> np.ndarray:
    screen.fill((level, level, level))
    pygame.display.flip()
    pygame.event.pump()
    label = "BLACK" if level == 0 else "WHITE" if level == 255 else f"GRAY {level}"
    print(f"{label}: settling for {settle_seconds:.1f}s...")

    settle_end = time.monotonic() + settle_seconds
    last_frame = None
    while time.monotonic() < settle_end:
        pump_or_quit()
        ok, frame = cap.read()
        if ok:
            last_frame = frame
            cv2.imshow(f"Live photometric capture - {label}", frame)
        cv2.waitKey(1)

    print(f"{label}: sampling for {sample_seconds:.1f}s...")
    sample_end = time.monotonic() + sample_seconds
    samples: list[np.ndarray] = []
    while time.monotonic() < sample_end:
        pump_or_quit()
        ok, frame = cap.read()
        if not ok:
            continue
        last_frame = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        samples.append(gray)
        cv2.imshow(f"Live photometric capture - {label}", frame)
        cv2.waitKey(1)

    if not samples and last_frame is not None:
        samples.append(cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY))
    if not samples:
        raise RuntimeError(f"No camera frames captured for {label}.")

    stack = np.stack(samples).astype(np.float32)
    combined = np.median(stack, axis=0) if statistic == "median" else np.mean(stack, axis=0)
    combined[mask == 0] = 0.0
    print(
        f"{label}: combined {len(samples)} frames; "
        f"projection mean={combined[mask > 0].mean():.1f}"
    )
    return combined


def fit_gain_offset(captures: list[np.ndarray], mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(LEVELS, dtype=np.float32)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered * x_centered))
    y = np.stack(captures).astype(np.float32)
    y_mean = np.mean(y, axis=0)
    gain = np.sum(x_centered[:, None, None] * (y - y_mean), axis=0) / denominator
    offset = y_mean - gain * float(x.mean())
    gain = np.clip(gain, 0.001, None)
    gain[mask == 0] = 0.001
    offset[mask == 0] = 0.0
    return gain.astype(np.float32), offset.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-exposure Level 5 calibration experiment")
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--settle-seconds", type=float, default=4.0)
    parser.add_argument("--sample-seconds", type=float, default=3.0)
    parser.add_argument("--statistic", choices=("median", "mean"), default="median")
    parser.add_argument("--camera-exposure", type=float, default=None)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--output", default=CALIBRATION_CACHE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cap = None
    pygame.init()
    try:
        screen, width, height, display_index, monitor, monitor_count = create_fullscreen_display(
            TARGET_DISPLAY_INDEX
        )
        pygame.display.set_caption("Level 5 Calibration Experiment")
        print(
            f"Using display {display_index}/{monitor_count - 1}: "
            f"{monitor['width']}x{monitor['height']} at {monitor['left']},{monitor['top']}"
        )
        cap = get_camera_capture(args.camera_index)
        if args.camera_exposure is not None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, args.camera_exposure)
            print(
                f"Requested manual exposure {args.camera_exposure}; "
                f"camera reports {cap.get(cv2.CAP_PROP_EXPOSURE):.3f}"
            )

        homography, mask = capture_geometry(cap, screen, args.cols, args.rows)
        captures = [
            capture_level(
                cap,
                screen,
                mask,
                level,
                args.settle_seconds,
                args.sample_seconds,
                args.statistic,
            )
            for level in LEVELS
        ]
        gain, offset = fit_gain_offset(captures, mask)
        np.savez_compressed(
            args.output,
            H=homography,
            mask=mask,
            a_map=gain,
            b_map=offset,
            calibration_levels=np.asarray(LEVELS, dtype=np.uint8),
            captured_levels=np.stack(captures).astype(np.float32),
        )
        projected = mask > 0
        print(f"Saved Level 5-compatible calibration to {args.output}")
        print(
            f"Gain mean/range: {gain[projected].mean():.4f} / "
            f"{gain[projected].min():.4f}..{gain[projected].max():.4f}"
        )
        print(
            f"Offset mean/range: {offset[projected].mean():.1f} / "
            f"{offset[projected].min():.1f}..{offset[projected].max():.1f}"
        )
        print("Run level_5.py with --cached-calibration to use it.")
        cv2.waitKey(1500)
    except KeyboardInterrupt:
        print("Calibration cancelled; no new cache was written.")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    main()
