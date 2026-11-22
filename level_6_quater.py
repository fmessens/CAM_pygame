"""Vortex Bubbles, a minimal view of the Water Wakes simulation."""

import argparse

from level_6_bis import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vortex Bubbles")
    parser.add_argument("--cached-calibration", action="store_true")
    parser.add_argument("--camera-index", type=int, default=1)
    args = parser.parse_args()
    main(args.cached_calibration, args.camera_index, show_vortices=True)
