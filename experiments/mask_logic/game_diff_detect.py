import cv2
import numpy as np
import pygame
import time

CAM_INDEX = 1
THRESH = 25
MIN_AREA = 400
REFRESH_INTERVAL = 1.5

# Checkerboard pattern size (inner corners)
CB_COLS, CB_ROWS = 8, 6

pygame.init()
pygame.display.set_caption("Game / Diff Detection (Checkerboard Ref)")
screen = pygame.display.set_mode((960, 720))
font = pygame.font.SysFont(None, 22)
clock = pygame.time.Clock()

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("Cannot open camera")
    raise SystemExit

# ------------------------------------------------
# Helpers
# ------------------------------------------------
def cv_to_surface(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return pygame.image.frombuffer(rgb.tobytes(), (w, h), 'RGB')

def compute_mask(frame, background_ref):
    if frame.shape != background_ref.shape:
        return np.zeros(frame.shape[:2], np.uint8)
    gray_ref = cv2.cvtColor(background_ref, cv2.COLOR_BGR2GRAY)
    gray_cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_ref, gray_cur)
    _, mask = cv2.threshold(diff, THRESH, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 5)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask

# ------------------------------------------------
# Wait until checkerboard reference is visible
# ------------------------------------------------
roi = None
background_ref = None
print("Waiting for checkerboard reference pattern...")

waiting = True
while waiting:
    ok, frame = cap.read()
    if not ok:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (CB_COLS, CB_ROWS), None)

    debug = frame.copy()
    if ret:
        cv2.drawChessboardCorners(debug, (CB_COLS, CB_ROWS), corners, ret)
        # Bounding box for visualization
        pts = corners.reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 255, 0), 3)
        msg = "Checkerboard detected — press SPACE to confirm"
        cv2.putText(debug, msg, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(debug, "Looking for checkerboard pattern...",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    surf = cv_to_surface(debug)
    surf = pygame.transform.smoothscale(surf, screen.get_size())
    screen.blit(surf, (0, 0))
    pygame.display.flip()
    clock.tick(30)

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            raise SystemExit
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE and ret:
            background_ref = frame.copy()
            roi = (x, y, w, h)
            waiting = False
            print("Reference confirmed.")

# ------------------------------------------------
# Main diff-detection loop
# ------------------------------------------------
print("Running diff detection...")
last_refresh = time.time()
running = True

while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    ok, frame = cap.read()
    if not ok:
        continue

    # Periodic background refresh
    if time.time() - last_refresh > REFRESH_INTERVAL:
        background_ref = frame.copy()
        last_refresh = time.time()

    mask = compute_mask(frame, background_ref)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        approx = cv2.approxPolyDP(c, 3, True)
        pts = [(int(x), int(y)) for [[x, y]] in approx]
        polys.append(pts)

    # Visualization
    display = frame.copy()
    cv2.rectangle(display, (roi[0], roi[1]),
                  (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)
    for pts in polys:
        cv2.polylines(display, [np.array(pts, np.int32)],
                      True, (255, 255, 0), 2)

    surf = cv_to_surface(display)
    surf = pygame.transform.smoothscale(surf, screen.get_size())
    screen.blit(surf, (0, 0))

    txt = f"Objects: {len(polys)} | Press Q to quit"
    t = font.render(txt, True, (255, 255, 255))
    screen.blit(t, (10, 10))
    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()
