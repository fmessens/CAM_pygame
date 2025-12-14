import cv2
import time
import threading


def get_camera_capture():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap
#===========================================================
#  VIDEO STREAM CLASS (BACKGROUND THREAD)
#===========================================================

class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Still keep this low
        
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