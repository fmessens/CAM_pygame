import argparse
import os

import level_1
import level_2


def main(image_path, camera_index=1):
    success = level_1.main(
        cached_calibration=False,
        return_on_success=True,
        camera_index=camera_index,
    )
    if not success:
        return
    level_2.main(image_path, cached_calibration=False, camera_index=camera_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Level 1 then Level 2 on success.")
    parser.add_argument(
        "--image",
        default=os.path.join("images", "galaxy-1489300667bhk.jpg"),
        help="Path to the background image for level 2.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=1,
        help="OpenCV camera index to use. This setup detected the secondary webcam at index 1.",
    )
    args = parser.parse_args()
    main(args.image, camera_index=args.camera_index)
