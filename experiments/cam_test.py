import cv2
import time
import argparse

RES_LIST = [
    (640, 480),
    (1280, 720),
    (1920, 1080),
    (1280, 800),
]

HELP = """
Keys:
  q         Quit
  n / p     Next / Previous camera index
  r         Cycle resolution preset
  s         Save snapshot to disk
  a         Toggle autofocus (if supported)
  [ / ]     Decrease / Increase exposure (if supported)
"""

def open_cam(idx):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    return cap

def set_res(cap, w, h):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0, help='initial camera index')
    args = parser.parse_args()

    cam_index = args.index
    cap = open_cam(cam_index)
    if cap is None:
        print(f"Failed to open camera {cam_index}")
        return

    res_i = 0
    set_res(cap, *RES_LIST[res_i])

    last = time.time()
    fcnt = 0
    fps = 0.0
    exp_val = cap.get(cv2.CAP_PROP_EXPOSURE)
    af_val = cap.get(cv2.CAP_PROP_AUTOFOCUS)

    print("Camera Test. Press 'q' to quit.")
    print(HELP)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed. Trying to reopen…")
            cap.release()
            cap = open_cam(cam_index)
            if cap is None:
                print(f"Cannot reopen camera {cam_index}")
                break
            set_res(cap, *RES_LIST[res_i])
            continue

        # FPS calc
        fcnt += 1
        now = time.time()
        if now - last >= 1.0:
            fps = fcnt / (now - last)
            fcnt = 0
            last = now

        h, w = frame.shape[:2]
        overlay = frame.copy()
        text = f"Cam {cam_index}  {w}x{h}  {fps:4.1f} FPS"
        cv2.rectangle(overlay, (8,8), (8+420, 8+80), (0,0,0), -1)
        alpha = 0.45
        frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
        y = 30
        for line in [text, "n/p: camera  r: res  a: autofocus  [/]: exposure", "s: snapshot  q: quit"]:
            cv2.putText(frame, line, (16,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            y += 24

        cv2.imshow('Cam Test', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            cam_index += 1
            cap.release(); cap = open_cam(cam_index)
            if cap is None:
                print(f"No camera at index {cam_index}, rolling back")
                cam_index -= 1
                cap = open_cam(cam_index)
            else:
                set_res(cap, *RES_LIST[res_i])
        elif key == ord('p'):
            cam_index = max(0, cam_index - 1)
            cap.release(); cap = open_cam(cam_index)
            if cap is None:
                print("Failed to open previous camera")
                cam_index += 1
                cap = open_cam(cam_index)
            else:
                set_res(cap, *RES_LIST[res_i])
        elif key == ord('r'):
            res_i = (res_i + 1) % len(RES_LIST)
            set_res(cap, *RES_LIST[res_i])
        elif key == ord('s'):
            ts = time.strftime('%Y%m%d_%H%M%S')
            fname = f'snapshot_{ts}.png'
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")
        elif key == ord('a'):
            try:
                cur = cap.get(cv2.CAP_PROP_AUTOFOCUS)
                new = 0 if cur > 0.5 else 1
                cap.set(cv2.CAP_PROP_AUTOFOCUS, new)
                print(f"Autofocus set to {new}")
            except Exception:
                print("Autofocus property not supported on this camera.")
        elif key == ord('['):
            try:
                cur = cap.get(cv2.CAP_PROP_EXPOSURE)
                if cur == 0:
                    cur = -6  # some drivers use negatives
                cap.set(cv2.CAP_PROP_EXPOSURE, cur - 1)
                print(f"Exposure -> {cur - 1}")
            except Exception:
                print("Exposure control not supported.")
        elif key == ord(']'):
            try:
                cur = cap.get(cv2.CAP_PROP_EXPOSURE)
                if cur == 0:
                    cur = -6
                cap.set(cv2.CAP_PROP_EXPOSURE, cur + 1)
                print(f"Exposure -> {cur + 1}")
            except Exception:
                print("Exposure control not supported.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()