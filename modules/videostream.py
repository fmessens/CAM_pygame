import os
import sys

import cv2
import time
import threading


def _preferred_backend():
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def _open_camera(src=0):
    cap = cv2.VideoCapture(src, _preferred_backend())
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open camera index {src}.")

    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError(f"Camera index {src} opened but did not return frames.")

    backend_name = cap.getBackendName() if hasattr(cap, "getBackendName") else "unknown"
    print(f"Camera: using index {src} via {backend_name}")
    return cap


def get_camera_capture(src=None):
    camera_index = int(os.environ.get("CAMERA_INDEX", "0")) if src is None else src
    return _open_camera(camera_index)
#===========================================================
#  VIDEO STREAM CLASS (BACKGROUND THREAD)
#===========================================================

class VideoStream:
    def __init__(self, src=None):
        camera_index = int(os.environ.get("CAMERA_INDEX", "0")) if src is None else src
        self.cap = _open_camera(camera_index)
        
        # Read the first frame to avoid errors on startup
        self.ret, self.frame = self.cap.read()
        
        self.stopped = False
        self.lock = threading.Lock()
        
        # Start the background thread
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True # Thread dies when main program dies
        self.t.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            # We only care about the latest frame. 
            # We use a lock to ensure we don't read while writing.
            with self.lock:
                self.ret, self.frame = ret, frame
            
            # Tiny sleep to not hog 100% of a CPU core if camera waits
            time.sleep(0.001)

    def read(self):
        # Return the most recent frame available
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.stopped = True
        self.t.join()
        self.cap.release()
