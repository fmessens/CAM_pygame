import cv2
import numpy as np

# Checkerboard parameters (must match Script B)
COLS, ROWS = 8, 6        # inner corners
SQUARE_SIZE = 100         # pixels per square
MARGIN = 80               # border margin in pixels

# Create checkerboard image
w = COLS * SQUARE_SIZE + 2*MARGIN
h = ROWS * SQUARE_SIZE + 2*MARGIN
img = np.ones((h, w, 3), np.uint8) * 255

for y in range(ROWS + 1):
    for x in range(COLS + 1):
        if (x + y) % 2 == 0:
            x0 = MARGIN + x*SQUARE_SIZE
            y0 = MARGIN + y*SQUARE_SIZE
            cv2.rectangle(img, (x0, y0),
                          (x0+SQUARE_SIZE, y0+SQUARE_SIZE),
                          (0, 0, 0), -1)

cv2.namedWindow("Checkerboard Reference", cv2.WINDOW_NORMAL)
cv2.imshow("Checkerboard Reference", img)
cv2.setWindowProperty("Checkerboard Reference",
                      cv2.WND_PROP_TOPMOST, 1)

print("Drag to projector display. Press F for fullscreen, ESC to quit.")
fullscreen = False
while True:
    key = cv2.waitKey(50) & 0xFF
    if key == 27:  # ESC
        break
    elif key in (ord('f'), ord('F')):
        fullscreen = not fullscreen
        prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty("Checkerboard Reference",
                              cv2.WND_PROP_FULLSCREEN, prop)

cv2.destroyAllWindows()
