import cv2
import numpy as np
import pygame
import time

# ----------------------------
# Config
# ----------------------------
CAM_INDEX = 0
PROJ_SIZE = (800, 600)
GAME_SIZE = (800, 600)

REFRESH_INTERVAL = 1.5     # seconds between background updates
THRESH = 25                # diff threshold
MIN_AREA = 400             # minimum contour area
BORDER_COLOR = (255, 0, 255)
REF_COLOR = (180, 180, 180)

# ----------------------------
# Init
# ----------------------------
pygame.init()
pygame.display.set_caption("Projection + Game Dual Windows")

proj_screen = pygame.display.set_mode(PROJ_SIZE, pygame.NOFRAME)  # projector window
game_screen = pygame.display.set_mode(GAME_SIZE)                  # game window
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 22)

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("Cannot open camera")
    pygame.quit()
    raise SystemExit

# ----------------------------
# Reference surface (projected image)
# ----------------------------
ref_surface = pygame.Surface(PROJ_SIZE)
ref_surface.fill(REF_COLOR)
pygame.draw.rect(ref_surface, BORDER_COLOR, ref_surface.get_rect(), 10)

# Convert reference surface to BGR for OpenCV baseline
ref_string = pygame.image.tostring(ref_surface, "RGB")
ref_bgr = np.frombuffer(ref_string, np.uint8).reshape(PROJ_SIZE[1], PROJ_SIZE[0], 3)
ref_bgr = cv2.cvtColor(ref_bgr, cv2.COLOR_RGB2BGR)
background_ref = ref_bgr.copy()
last_refresh = time.time()

# ----------------------------
# Helper: compute difference mask
# ----------------------------
def compute_mask(frame, background_ref):
    gray_ref = cv2.cvtColor(background_ref, cv2.COLOR_BGR2GRAY)
    gray_cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_ref, gray_cur)
    _, mask = cv2.threshold(diff, THRESH, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 5)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask

# ----------------------------
# Main loop
# ----------------------------
running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    # ---- Projector window: show static reference ----
    proj_screen.blit(ref_surface, (0, 0))
    pygame.display.update()  # projector side refresh

    # ---- Capture camera frame ----
    ok, frame = cap.read()
    if not ok:
        continue

    # ---- Refresh background every N seconds ----
    if time.time() - last_refresh > REFRESH_INTERVAL:
        background_ref = frame.copy()
        last_refresh = time.time()

    # ---- Compute diff mask ----
    mask = compute_mask(frame, background_ref)

    # ---- Find contours (objects) ----
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        approx = cv2.approxPolyDP(c, 3, True)
        pts = [(int(x), int(y)) for [[x, y]] in approx]
        polys.append(pts)

    # ---- Game window visualization ----
    game_screen.fill((0, 0, 0))
    for pts in polys:
        pygame.draw.polygon(game_screen, (255, 255, 0), pts, 2)

    text = f"Objects: {len(polys)}"
    t = font.render(text, True, (255, 255, 255))
    game_screen.blit(t, (10, 10))

    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()
