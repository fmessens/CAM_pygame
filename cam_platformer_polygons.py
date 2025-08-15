import cv2
import numpy as np
import pygame
import time
import math

# ----------------------------
# Config
# ----------------------------
CAM_INDEX = 1
WINDOW_SCALE = 1.0

# Morphology / contouring
BLUR_KSIZE = 5
MORPH_K = 5
MIN_AREA_FRAC = 0.004      # ignore tiny blobs (fraction of frame area)
IOU_MERGE_BBOX = 0.15       # merge hull bounding boxes (coarse prune)
MAX_OBSTACLES = 20

# Gameplay
PLAYER_W, PLAYER_H = 34, 48
MOVE_SPEED = 3.6
JUMP_VEL = -10.2
GRAVITY = 0.60
MAX_FALL = 16

TRANSPORT_TOUCH_PAD = 3     # extra pixels to count as touching for transport
SMOOTH_ALPHA = 0.35         # EMA for centroid smoothing (velocity stability)

# ----------------------------
# Utils
# ----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# simple IoU on axis-aligned boxes [x1,y1,x2,y2]
def iou(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    ua=(a[2]-a[0])*(a[3]-a[1]); ub=(b[2]-b[0])*(b[3]-b[1])
    return inter/(ua+ub-inter+1e-9)

def merge_overlaps(boxes, thr=IOU_MERGE_BBOX):
    boxes=boxes[:]
    changed=True
    while changed and len(boxes)>1:
        changed=False; used=[False]*len(boxes); out=[]
        for i in range(len(boxes)):
            if used[i]: continue
            base=list(boxes[i])
            for j in range(i+1,len(boxes)):
                if used[j]: continue
                if iou(base, boxes[j])>=thr:
                    b=boxes[j]
                    base=[min(base[0],b[0]),min(base[1],b[1]),
                          max(base[2],b[2]),max(base[3],b[3])]
                    used[j]=True; changed=True
            used[i]=True; out.append(base)
        boxes=out
    return boxes

# ----------------------------
# Auto-calibration (HSV)
# ----------------------------
def auto_calibrate(cap, seconds=1.0):
    """
    Sample a centered ROI for ~seconds to profile the wall. We classify 'non-wall' pixels as:
      colored (S >= S_MIN) OR dark (V <= V_WHITE_MIN - 1)
    Thresholds are derived from percentiles of the wall ROI.
    """
    s_vals = []
    v_vals = []
    t_end = time.time() + seconds
    ok, frame = cap.read()
    if not ok:
        return 40, 210  # fallbacks

    H, W = frame.shape[:2]
    x0 = int(W*0.30); x1 = int(W*0.70)
    y0 = int(H*0.30); y1 = int(H*0.70)

    while time.time() < t_end:
        ok, f = cap.read()
        if not ok: break
        hsv = cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        if BLUR_KSIZE>1:
            hsv = cv2.medianBlur(hsv, BLUR_KSIZE)
        _, Sc, Vc = cv2.split(hsv)
        s_vals.append(Sc[::8, ::8].astype(np.uint16))
        v_vals.append(Vc[::8, ::8].astype(np.uint16))

    if not s_vals:
        return 40, 210

    S = np.concatenate([a.reshape(-1) for a in s_vals])
    V = np.concatenate([a.reshape(-1) for a in v_vals])

    s95 = float(np.percentile(S, 95))
    v85 = float(np.percentile(V, 85))
    S_MIN = int(clamp(round(s95 + 8), 20, 140))
    V_WHITE_MIN = int(clamp(round(v85), 140, 252))
    return S_MIN, V_WHITE_MIN

# ----------------------------
# Polygon obstacle tracker
# ----------------------------
class PolyTracker:
    def __init__(self, S_MIN, V_WHITE_MIN):
        self.S_MIN = S_MIN
        self.V_WHITE_MIN = V_WHITE_MIN  # used when dynamic darkness is OFF
        # Dynamic darkness reference (initialized from calibration)
        self.V_REF = float(V_WHITE_MIN)
        self.DARK_DYNAMIC = True   # when True, compute dark threshold relative to V_REF
        self.DARK_DELTA = 18       # pixels darker than V_REF to count as dark
        self.V_REF_BETA = 0.20     # EMA factor for updating V_REF from live frames

        self.prev = {}     # id -> {centroid:(x,y)}
        self.next_id = 1

    def set_thresholds(self, smin, vmin):
        self.S_MIN = smin; self.V_WHITE_MIN = vmin
        # also reset reference toward current absolute threshold
        self.V_REF = float(vmin)

    def set_dark_mode(self, dynamic: bool):
        self.DARK_DYNAMIC = dynamic

    def set_dark_delta(self, delta: int):
        self.DARK_DELTA = int(clamp(delta, 0, 255))

    def _contours_to_polys(self, frame_bgr):
        h,w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        if BLUR_KSIZE>1:
            hsv = cv2.medianBlur(hsv, BLUR_KSIZE)
        Hc, Sc, Vc = cv2.split(hsv)
        mask_color = (Sc >= self.S_MIN)

        # Darkness mask: dynamic relative threshold (default) or absolute
        if self.DARK_DYNAMIC:
            h2, w2 = frame_bgr.shape[:2]
            x0 = int(w2*0.30); x1 = int(w2*0.70)
            y0 = int(h2*0.30); y1 = int(h2*0.70)
            V_roi = Vc[y0:y1, x0:x1]
            # 85th percentile of ROI approximates wall brightness
            v85 = float(np.percentile(V_roi, 85)) if V_roi.size else float(self.V_REF)
            # EMA update for stability
            self.V_REF = (1.0 - self.V_REF_BETA) * self.V_REF + self.V_REF_BETA * v85
            v_thresh = self.V_REF - self.DARK_DELTA
            v_thresh = np.clip(v_thresh, 0, 255)
            mask_dark = (Vc <= v_thresh)
        else:
            mask_dark  = (Vc <= self.V_WHITE_MIN - 1)

        mask = np.logical_or(mask_color, mask_dark).astype(np.uint8)*255

        if MORPH_K>0:
            k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(MORPH_K,MORPH_K))
            mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        min_area = MIN_AREA_FRAC * (w*h)
        boxes = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            hull = cv2.convexHull(c)  # ensure convex for robust SAT
            peri = cv2.arcLength(hull, True)
            eps = 0.02 * peri  # simplify hull
            approx = cv2.approxPolyDP(hull, eps, True)
            if len(approx) < 3:
                continue
            pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
            # bounding rect (for pruning & transport contact checks)
            x,y,w_,h_ = cv2.boundingRect(np.array(pts, dtype=np.float32))
            boxes.append([x,y,x+w_,y+h_])
            polys.append({
                "points": pts,
                "bbox": (x,y,w_,h_),
            })
        # coarse merge by bbox IoU to prevent duplicates / fragments
        boxes = merge_overlaps(boxes, IOU_MERGE_BBOX)
        # keep only polys whose bbox matches a merged box (within tolerance)
        filtered = []
        for p in polys:
            x,y,w_,h_ = p["bbox"]
            bb = [x,y,x+w_,y+h_]
            keep=False
            for m in boxes:
                if iou(bb,m) >= 0.5:
                    keep=True; break
            if keep:
                filtered.append(p)
        # sort large first and limit
        filtered.sort(key=lambda p: p["bbox"][2]*p["bbox"][3], reverse=True)
        return filtered[:MAX_OBSTACLES], mask

    def _centroid(self, pts):
        xs = [p[0] for p in pts]; ys=[p[1] for p in pts]
        return (sum(xs)/len(xs), sum(ys)/len(ys))

    def detect(self, frame_bgr):
        polys, mask = self._contours_to_polys(frame_bgr)
        # assign IDs by nearest centroid
        prev_c = {pid: self.prev[pid]["centroid"] for pid in self.prev}
        used=set()
        tracked=[]
        for p in polys:
            c = self._centroid(p["points"])
            best=None; bestd=1e12
            for pid, pc in prev_c.items():
                if pid in used: continue
                d=(c[0]-pc[0])**2 + (c[1]-pc[1])**2
                if d<bestd:
                    bestd=d; best=pid
            if best is None or bestd > 80**2:
                pid = self.next_id; self.next_id += 1
                vx=vy=0.0
                scx, scy = c
            else:
                pid = best; used.add(best)
                pcx, pcy = self.prev[pid]["centroid"]
                vx = (c[0]-pcx); vy = (c[1]-pcy)
                # EMA smoothing on centroid (for velocity stability)
                a=SMOOTH_ALPHA
                scx = a*c[0] + (1-a)*pcx
                scy = a*c[1] + (1-a)*pcy
            tracked.append({
                "id": pid,
                "points": p["points"],
                "centroid": (scx, scy),
                "vx": vx,
                "vy": vy,
                "bbox": p["bbox"],
            })
        # update prev map
        self.prev = {t["id"]: {"centroid": t["centroid"]} for t in tracked}
        return tracked, mask

# ----------------------------
# SAT collision helpers (AABB vs convex polygon)
# ----------------------------

def rect_points(rect: pygame.Rect):
    return [
        (rect.left, rect.top),
        (rect.right, rect.top),
        (rect.right, rect.bottom),
        (rect.left, rect.bottom)
    ]

def normalize(vx, vy):
    mag = math.hypot(vx, vy)
    if mag < 1e-9:
        return (0.0, 0.0)
    return (vx/mag, vy/mag)

def project_points(ax, ay, pts):
    # ax,ay should be unit length for meaningful MTV
    dots = [p[0]*ax + p[1]*ay for p in pts]
    return (min(dots), max(dots))

def interval_overlap(a_min, a_max, b_min, b_max):
    return min(a_max, b_max) - max(a_min, b_min)

def sat_rect_poly_MTV(rect: pygame.Rect, poly_pts):
    """Return (mtv_x, mtv_y) to separate rect from convex polygon.
       If no collision, return (0,0) and False. Otherwise (mtv, True).
    """
    axes = [(1.0, 0.0), (0.0, 1.0)]  # AABB axes
    # polygon edge normals
    for i in range(len(poly_pts)):
        x1,y1 = poly_pts[i]
        x2,y2 = poly_pts[(i+1)%len(poly_pts)]
        ex,ey = (x2-x1, y2-y1)
        nx,ny = normalize(-ey, ex)  # perpendicular
        if nx==0 and ny==0:
            continue
        axes.append((nx,ny))

    rect_pts = rect_points(rect)

    smallest_overlap = 1e12
    smallest_axis = (0.0, 0.0)

    # centers along axes (for MTV direction)
    rcx = (rect.left + rect.right)/2.0
    rcy = (rect.top + rect.bottom)/2.0
    pcx = sum(p[0] for p in poly_pts)/len(poly_pts)
    pcy = sum(p[1] for p in poly_pts)/len(poly_pts)

    for ax,ay in axes:
        r_min, r_max = project_points(ax, ay, rect_pts)
        p_min, p_max = project_points(ax, ay, poly_pts)
        ov = interval_overlap(r_min, r_max, p_min, p_max)
        if ov <= 0:
            return (0.0, 0.0), False  # separating axis found
        # choose minimal overlap axis
        if ov < smallest_overlap:
            smallest_overlap = ov
            # MTV direction: move rect away from polygon center along axis
            r_c = rcx*ax + rcy*ay
            p_c = pcx*ax + pcy*ay
            direction = -1.0 if (p_c - r_c) > 0 else 1.0
            smallest_axis = (ax*direction, ay*direction)

    mtv = (smallest_axis[0]*smallest_overlap, smallest_axis[1]*smallest_overlap)
    return mtv, True

# ----------------------------
# Player with polygon collisions
# ----------------------------
class Player:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, PLAYER_W, PLAYER_H)
        self.vx = 0.0; self.vy = 0.0
        self.on_ground = False

    def handle_input(self):
        k=pygame.key.get_pressed()
        self.vx=0.0
        if k[pygame.K_LEFT] or k[pygame.K_a]:  self.vx=-MOVE_SPEED
        if k[pygame.K_RIGHT] or k[pygame.K_d]: self.vx= MOVE_SPEED
        if (k[pygame.K_SPACE] or k[pygame.K_w] or k[pygame.K_UP]) and self.on_ground:
            self.vy=JUMP_VEL; self.on_ground=False

    def apply_gravity(self):
        self.vy = min(self.vy + GRAVITY, MAX_FALL)

    def transport_by_obstacles(self, obstacles):
        pad = TRANSPORT_TOUCH_PAD
        me = self.rect.inflate(pad*2, pad*2)
        dx=dy=0.0
        for o in obstacles:
            x,y,w,h = o["bbox"]
            obb = pygame.Rect(int(x),int(y),int(w),int(h))
            if me.colliderect(obb):
                dx += o["vx"]; dy += o["vy"]
        if dx or dy:
            self.rect.x += int(round(dx))
            self.rect.y += int(round(dy))

    def move_and_resolve(self, obstacles):
        # integrate velocity
        self.rect.x += int(self.vx)
        self.rect.y += int(self.vy)

        self.on_ground = False
        grounded_vx = grounded_vy = 0.0

        # resolve against each polygon sequentially
        for o in obstacles:
            poly = o["points"]
            mtv, hit = sat_rect_poly_MTV(self.rect, poly)
            if not hit:
                continue
            # apply MTV
            self.rect.x += int(round(mtv[0]))
            self.rect.y += int(round(mtv[1]))
            # vertical implications
            if mtv[1] < 0:  # pushed up => landed
                self.vy = 0.0
                self.on_ground = True
                grounded_vx += o["vx"]; grounded_vy += o["vy"]
            elif mtv[1] > 0:  # head bump
                self.vy = 0.0
            # side collisions naturally push via mtv[0]

        if self.on_ground:
            self.rect.x += int(round(grounded_vx))
            self.rect.y += int(round(grounded_vy))

# ----------------------------
# OpenCV <-> Pygame
# ----------------------------
def cv_to_surface(bgr):
    rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h,w=rgb.shape[:2]
    return pygame.image.frombuffer(rgb.tobytes(), (w,h), 'RGB')

# ----------------------------
# Main
# ----------------------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Cannot open camera"); return

    print("Calibrating wall… hold still for ~1 second.")
    S_MIN, V_WHITE_MIN = auto_calibrate(cap, seconds=1.0)
    print(f"Calibration: S_MIN={S_MIN}, V_WHITE_MIN={V_WHITE_MIN}")

    ok,frame=cap.read()
    if not ok:
        print("No frame"); return

    H,W=frame.shape[:2]
    win_w,win_h=int(W*WINDOW_SCALE),int(H*WINDOW_SCALE)

    pygame.init()
    pygame.display.set_caption("Camera Obstacles — Polygon SAT + Auto-calibration")
    screen=pygame.display.set_mode((win_w,win_h))
    font=pygame.font.SysFont(None,22)
    clock=pygame.time.Clock()

    tracker = PolyTracker(S_MIN, V_WHITE_MIN)
    player = Player(win_w//2-PLAYER_W//2, 10)

    show_mask=False
    running=True
    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: running=False
            elif e.type==pygame.KEYDOWN:
                if e.key==pygame.K_q: running=False
                elif e.key==pygame.K_m: show_mask=not show_mask
                elif e.key==pygame.K_c:
                    print("Recalibrating…")
                    S_MIN, V_WHITE_MIN = auto_calibrate(cap, seconds=1.0)
                    tracker.set_thresholds(S_MIN, V_WHITE_MIN)
                    tracker.V_REF = float(V_WHITE_MIN)
                    print(f"Calibration → S_MIN={S_MIN}, V_WHITE_MIN={V_WHITE_MIN}, V_REF={int(tracker.V_REF)}")
                elif e.key==pygame.K_LEFTBRACKET:   # [
                    if tracker.DARK_DYNAMIC:
                        tracker.set_dark_delta(tracker.DARK_DELTA - 1)
                    else:
                        V_WHITE_MIN = max(0, V_WHITE_MIN - 2)
                        tracker.set_thresholds(S_MIN, V_WHITE_MIN)
                elif e.key==pygame.K_RIGHTBRACKET:  # ]
                    if tracker.DARK_DYNAMIC:
                        tracker.set_dark_delta(tracker.DARK_DELTA + 1)
                    else:
                        V_WHITE_MIN = min(255, V_WHITE_MIN + 2)
                        tracker.set_thresholds(S_MIN, V_WHITE_MIN)
                elif e.key==pygame.K_MINUS:         # -  lower S_MIN (more pixels count as colored)
                    S_MIN = max(0, S_MIN - 2)
                    tracker.set_thresholds(S_MIN, V_WHITE_MIN)
                elif e.key==pygame.K_EQUALS:        # =  raise S_MIN (stricter color requirement)
                    S_MIN = min(255, S_MIN + 2)
                    tracker.set_thresholds(S_MIN, V_WHITE_MIN)
                elif e.key==pygame.K_t:             # T toggle dynamic/absolute darkness
                    tracker.set_dark_mode(not tracker.DARK_DYNAMIC)

        ok,frame=cap.read()
        if not ok: break

        obstacles_raw, mask = tracker.detect(frame)
        # Scale polygons to window size (if WINDOW_SCALE != 1)
        scale = WINDOW_SCALE
        obstacles=[]
        for ob in obstacles_raw:
            pts = [(p[0]*scale, p[1]*scale) for p in ob["points"]]
            x,y,w_,h_ = ob["bbox"]
            obstacles.append({
                "points": pts,
                "vx": ob["vx"]*scale,
                "vy": ob["vy"]*scale,
                "bbox": (x*scale,y*scale,w_*scale,h_*scale)
            })

        # Background
        bg = frame if not show_mask else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        surf=cv_to_surface(bg)
        if scale!=1.0:
            surf=pygame.transform.smoothscale(surf,(win_w,win_h))
        screen.blit(surf,(0,0))

        # Draw polygons
        for o in obstacles:
            pts_int = [(int(x),int(y)) for x,y in o["points"]]
            pygame.draw.polygon(screen, (255,255,0), pts_int, width=2)

        # Transport first
        player.transport_by_obstacles(obstacles)
        # Controls & physics
        player.handle_input()
        player.apply_gravity()
        player.move_and_resolve(obstacles)

        # Bounds / respawn
        player.rect.left = clamp(player.rect.left,0,win_w-player.rect.width)
        if player.rect.bottom>win_h:
            player.rect.x=win_w//2-PLAYER_W//2; player.rect.y=10
            player.vx=player.vy=0.0; player.on_ground=False

        pygame.draw.rect(screen,(255,255,255),player.rect,0)

        # HUD
        if tracker.DARK_DYNAMIC:
            dark_line = f"S_MIN={S_MIN}  Dark: dynamic  V_ref={int(tracker.V_REF)}  Δ={int(tracker.DARK_DELTA)}   [/] Δ  -/= S   T:mode  (C=recal, M=mask, Q=quit)"
        else:
            dark_line = f"S_MIN={S_MIN}  Dark: absolute  V_WHITE_MIN={V_WHITE_MIN}   [/] V  -/= S   T:mode  (C=recal, M=mask, Q=quit)"
        lines=[
            f"FPS: {clock.get_fps():4.1f}   Obstacles: {len(obstacles)}",
            dark_line,
            "Polygon SAT collisions. Moving objects carry/push the player.",
        ]
        y0=6
        for line in lines:
            t=font.render(line,True,(0,0,0)); screen.blit(t,(8,y0+1))
            t=font.render(line,True,(255,255,255)); screen.blit(t,(7,y0)); y0+=18

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()

if __name__=="__main__":
    main()