"""
Structured-light variant of the platformer using a projector pattern.
Workflow:
  1) Run projector_pattern.py fullscreen on the beamer.
  2) Run this script. Aim the camera next to the projector lens.
  3) Press R to capture a clean reference of the wall-with-pattern.
  4) Move objects near the wall: occlusions/distortions become obstacles.

Segmentation: per-frame grayscale vs reference difference with adaptive
threshold (mean + k*std) + optional darkness polarity. Extract convex
polygons (hulls) and do SAT collisions; moving polygons carry/push player.

Hotkeys:
  Q  quit    M  toggle mask view    R  recapture reference
  ,/.  lower/raise diff_k (sensitivity)
  ;/'  lower/raise delta_dark (require objects to be this much darker than ref)
"""
import cv2, numpy as np, pygame, time, math

# Camera/game config
CAM_INDEX = 0
WINDOW_SCALE = 1.0
BLUR_KSIZE = 5
MORPH_K = 5
MIN_AREA_FRAC = 0.003
MAX_OBSTACLES = 20

# Diff segmentation params
DIFF_K = 2.2          # threshold = mean(diff) + DIFF_K * std(diff)
DELTA_DARK = 8        # require gray <= ref_gray - DELTA_DARK (helps reject flashes)

# Player physics
PLAYER_W, PLAYER_H = 34, 48
MOVE_SPEED = 3.6
JUMP_VEL = -10.2
GRAVITY = 0.60
MAX_FALL = 16
TRANSPORT_TOUCH_PAD = 3

# ---- geometry / SAT helpers ----

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def rect_points(r: pygame.Rect):
    return [(r.left, r.top), (r.right, r.top), (r.right, r.bottom), (r.left, r.bottom)]

def normalize(vx, vy):
    m = math.hypot(vx, vy)
    if m < 1e-9: return (0.0, 0.0)
    return (vx/m, vy/m)

def project(ax, ay, pts):
    dots = [p[0]*ax + p[1]*ay for p in pts]
    return (min(dots), max(dots))

def overlap(a0, a1, b0, b1):
    return min(a1, b1) - max(a0, b0)

def sat_rect_poly(rect: pygame.Rect, poly_pts):
    axes = [(1.0,0.0),(0.0,1.0)]
    for i in range(len(poly_pts)):
        x1,y1 = poly_pts[i]; x2,y2 = poly_pts[(i+1)%len(poly_pts)]
        nx,ny = normalize(-(y2-y1), (x2-x1))
        if nx==0 and ny==0: continue
        axes.append((nx,ny))
    rpts = rect_points(rect)
    rcx = (rect.left+rect.right)/2; rcy=(rect.top+rect.bottom)/2
    pcx = sum(p[0] for p in poly_pts)/len(poly_pts)
    pcy = sum(p[1] for p in poly_pts)/len(poly_pts)
    best_ov = 1e9; best_axis=(0,0)
    for ax,ay in axes:
        r0,r1 = project(ax,ay,rpts); p0,p1 = project(ax,ay,poly_pts)
        ov = overlap(r0,r1,p0,p1)
        if ov <= 0: return (0.0,0.0), False
        if ov < best_ov:
            best_ov = ov
            r_c = rcx*ax + rcy*ay; p_c = pcx*ax + pcy*ay
            dir = -1.0 if (p_c - r_c) > 0 else 1.0
            best_axis = (ax*dir, ay*dir)
    return (best_axis[0]*best_ov, best_axis[1]*best_ov), True

# ---- polygon extraction from mask ----

def mask_to_polygons(mask, min_area_px):
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys=[]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area_px: continue
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        eps = 0.02 * peri
        approx = cv2.approxPolyDP(hull, eps, True)
        if len(approx) >= 3:
            pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
            x,y,w,h = cv2.boundingRect(np.array(pts, dtype=np.float32))
            polys.append({"points": pts, "bbox": (x,y,w,h)})
    polys.sort(key=lambda p: p["bbox"][2]*p["bbox"][3], reverse=True)
    return polys[:MAX_OBSTACLES]

# ---- player ----
class Player:
    def __init__(self,x,y):
        self.rect=pygame.Rect(x,y,PLAYER_W,PLAYER_H)
        self.vx=0.0; self.vy=0.0
        self.on_ground=False
    def input(self):
        k=pygame.key.get_pressed(); self.vx=0.0
        if k[pygame.K_LEFT] or k[pygame.K_a]: self.vx=-MOVE_SPEED
        if k[pygame.K_RIGHT] or k[pygame.K_d]: self.vx= MOVE_SPEED
        if (k[pygame.K_SPACE] or k[pygame.K_w] or k[pygame.K_UP]) and self.on_ground:
            self.vy=JUMP_VEL; self.on_ground=False
    def gravity(self):
        self.vy=min(self.vy+GRAVITY, MAX_FALL)
    def transport(self, obstacles):
        pad=TRANSPORT_TOUCH_PAD; me=self.rect.inflate(pad*2,pad*2)
        dx=dy=0.0
        for o in obstacles:
            x,y,w,h=o["bbox"]; ob=pygame.Rect(int(x),int(y),int(w),int(h))
            if me.colliderect(ob): dx+=o["vx"]; dy+=o["vy"]
        self.rect.x+=int(round(dx)); self.rect.y+=int(round(dy))
    def collide(self, obstacles):
        self.rect.x += int(self.vx)
        self.rect.y += int(self.vy)
        self.on_ground=False; gx=gy=0.0
        for o in obstacles:
            mtv,hit = sat_rect_poly(self.rect, o["points"])
            if not hit: continue
            self.rect.x += int(round(mtv[0])); self.rect.y += int(round(mtv[1]))
            if mtv[1] < 0: self.vy=0.0; self.on_ground=True; gx+=o["vx"]; gy+=o["vy"]
            elif mtv[1] > 0: self.vy=0.0
        if self.on_ground:
            self.rect.x += int(round(gx)); self.rect.y += int(round(gy))

# ---- main ----

def cv_to_surface(bgr):
    rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h,w=rgb.shape[:2]
    return pygame.image.frombuffer(rgb.tobytes(), (w,h), 'RGB')


def main():
    global DIFF_K, DELTA_DARK
    cap=cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print('Cannot open camera'); return

    ok,frame=cap.read()
    if not ok: print('No frame'); return
    H,W=frame.shape[:2]
    win_w,win_h=int(W*WINDOW_SCALE),int(H*WINDOW_SCALE)

    pygame.init()
    pygame.display.set_caption('Projector Platformer — Structured Light')
    screen=pygame.display.set_mode((win_w,win_h))
    font=pygame.font.SysFont(None,22)
    clock=pygame.time.Clock()

    ref_gray=None
    player=Player(win_w//2-PLAYER_W//2, 10)

    # tracking of obstacle centroids → velocity
    prev_centroids={}

    show_mask=False
    running=True
    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: running=False
            elif e.type==pygame.KEYDOWN:
                if e.key==pygame.K_q: running=False
                elif e.key==pygame.K_m: show_mask=not show_mask
                elif e.key==pygame.K_r:
                    # capture averaged reference
                    print('Capturing reference… hold still')
                    n=20; acc=None
                    for i in range(n):
                        ok,f=cap.read()
                        if not ok: break
                        g=cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                        if BLUR_KSIZE>1: g=cv2.GaussianBlur(g,(BLUR_KSIZE|1,BLUR_KSIZE|1),0)
                        if acc is None: acc=g.astype(np.float32)
                        else: acc=acc*0.9 + g*0.1
                        time.sleep(0.01)
                    if acc is not None:
                        ref_gray=cv2.convertScaleAbs(acc)
                        print('Reference updated')
                elif e.key==pygame.K_COMMA:   # ,
                    DIFF_K=max(0.5, DIFF_K-0.1)
                elif e.key==pygame.K_PERIOD:  # .
                    DIFF_K=min(5.0, DIFF_K+0.1)
                elif e.key==pygame.K_SEMICOLON:   # ;
                    DELTA_DARK=max(0, DELTA_DARK-1)
                elif e.key==pygame.K_QUOTE:       # '
                    DELTA_DARK=min(50, DELTA_DARK+1)

        ok,frame=cap.read()
        if not ok: break

        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if BLUR_KSIZE>1:
            gray=cv2.GaussianBlur(gray,(BLUR_KSIZE|1,BLUR_KSIZE|1),0)

        if ref_gray is None:
            # ask user to press R
            mask=np.zeros_like(gray)
        else:
            d=cv2.absdiff(gray, ref_gray)
            thr = d.mean() + DIFF_K * d.std()
            _, m = cv2.threshold(d, max(5, int(thr)), 255, cv2.THRESH_BINARY)
            # enforce darker-than-ref to focus on occlusions/shadows
            darker = (gray.astype(np.int16) <= ref_gray.astype(np.int16) - int(DELTA_DARK)).astype(np.uint8)*255
            mask = cv2.bitwise_and(m, darker)
            if MORPH_K>0:
                k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(MORPH_K,MORPH_K))
                mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
                mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        # polygons from mask
        min_area = MIN_AREA_FRAC * (W*H)
        polys = mask_to_polygons(mask, min_area)

        # track velocities by centroid match
        obstacles=[]
        used_prev=set()
        cur_centroids=[]
        for p in polys:
            pts=p["points"]
            cx=sum(x for x,_ in pts)/len(pts); cy=sum(y for _,y in pts)/len(pts)
            cur_centroids.append((cx,cy))
            # nearest prev
            best=-1; bestd=1e12
            for i,(px,py) in enumerate(prev_centroids.get('pts', [])):
                if i in used_prev: continue
                d=(cx-px)**2+(cy-py)**2
                if d<bestd: best=i; bestd=d
            if best==-1 or bestd>80**2:
                vx=vy=0.0
            else:
                vx=cx - prev_centroids['pts'][best][0]
                vy=cy - prev_centroids['pts'][best][1]
                used_prev.add(best)
            x,y,w,h=p["bbox"]
            obstacles.append({"points":pts, "bbox":(x,y,w,h), "vx":vx, "vy":vy})
        prev_centroids={'pts': cur_centroids}

        # render
        bg = frame if not show_mask else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        surf=cv_to_surface(bg)
        if WINDOW_SCALE!=1.0:
            surf=pygame.transform.smoothscale(surf,(win_w,win_h))
        screen.blit(surf,(0,0))
        for o in obstacles:
            pts=[(int(x*WINDOW_SCALE), int(y*WINDOW_SCALE)) for x,y in o['points']]
            pygame.draw.polygon(screen,(255,255,0),pts,2)

        # physics
        player.transport(obstacles)
        player.input(); player.gravity(); player.collide(obstacles)
        player.rect.left = clamp(player.rect.left,0,win_w-player.rect.width)
        if player.rect.bottom>win_h:
            player.rect.x=win_w//2-PLAYER_W//2; player.rect.y=10
            player.vx=player.vy=0.0; player.on_ground=False
        pygame.draw.rect(screen,(255,255,255),player.rect,0)

        # HUD
        lines=[
            f"FPS: {clock.get_fps():4.1f}  Obstacles: {len(obstacles)}  (Q quit, M mask)",
            f"Ref: {'OK' if ref_gray is not None else 'MISSING — press R'}  diff_k={DIFF_K:.1f}  Δdark={DELTA_DARK}",
        ]
        y0=6
        for line in lines:
            t=font.render(line,True,(0,0,0)); screen.blit(t,(8,y0+1))
            t=font.render(line,True,(255,255,255)); screen.blit(t,(7,y0)); y0+=18

        pygame.display.flip(); clock.tick(60)

    cap.release(); pygame.quit()

if __name__=='__main__':
    main()