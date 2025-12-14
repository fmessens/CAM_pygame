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

BLUR_KSIZE = 5
MORPH_K = 5
MIN_AREA_FRAC = 0.004
IOU_MERGE_BBOX = 0.15
MAX_OBSTACLES = 20

PLAYER_W, PLAYER_H = 34, 48
MOVE_SPEED = 3.6
JUMP_VEL = -10.2
GRAVITY = 0.60
MAX_FALL = 16

TRANSPORT_TOUCH_PAD = 3
SMOOTH_ALPHA = 0.35

# ----------------------------
# Utils
# ----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

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
# Polygon tracker (simplified for precomputed masks)
# ----------------------------
class PolyTracker:
    def __init__(self):
        self.prev = {}
        self.next_id = 1

    def _centroid(self, pts):
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
        return (sum(xs)/len(xs), sum(ys)/len(ys))

    def detect_from_mask(self, mask):
        cnts,_=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys=[]
        min_area=MIN_AREA_FRAC*(mask.shape[0]*mask.shape[1])
        boxes=[]
        for c in cnts:
            a=cv2.contourArea(c)
            if a<min_area: continue
            hull=cv2.convexHull(c)
            approx=cv2.approxPolyDP(hull,0.02*cv2.arcLength(hull,True),True)
            pts=[(float(p[0][0]),float(p[0][1])) for p in approx]
            x,y,w,h=cv2.boundingRect(np.array(pts,np.float32))
            boxes.append([x,y,x+w,y+h])
            polys.append({"points":pts,"bbox":(x,y,w,h)})
        boxes=merge_overlaps(boxes)
        filtered=[]
        for p in polys:
            x,y,w,h=p["bbox"]; bb=[x,y,x+w,y+h]
            keep=False
            for m in boxes:
                if iou(bb,m)>=0.5: keep=True; break
            if keep: filtered.append(p)

        # --- tracking between frames ---
        prev_c={pid:self.prev[pid]["centroid"] for pid in self.prev}
        used=set()
        tracked=[]
        for p in filtered:
            c=self._centroid(p["points"])
            best=None; bestd=1e12
            for pid,pc in prev_c.items():
                if pid in used: continue
                d=(c[0]-pc[0])**2+(c[1]-pc[1])**2
                if d<bestd: bestd=d; best=pid
            if best is None or bestd>80**2:
                pid=self.next_id; self.next_id+=1; vx=vy=0.0; scx,scy=c
            else:
                pid=best; used.add(best)
                pcx,pcy=self.prev[pid]["centroid"]
                vx=(c[0]-pcx); vy=(c[1]-pcy)
                a=SMOOTH_ALPHA
                scx=a*c[0]+(1-a)*pcx; scy=a*c[1]+(1-a)*pcy
            tracked.append({
                "id":pid,"points":p["points"],"centroid":(scx,scy),
                "vx":vx,"vy":vy,"bbox":p["bbox"]
            })
        self.prev={t["id"]:{ "centroid":t["centroid"]} for t in tracked}
        return tracked

# ----------------------------
# SAT helpers + Player (unchanged)
# ----------------------------
def rect_points(r):
    return [(r.left,r.top),(r.right,r.top),(r.right,r.bottom),(r.left,r.bottom)]
def normalize(vx,vy):
    m=math.hypot(vx,vy)
    return (0,0) if m<1e-9 else (vx/m,vy/m)
def project_points(ax,ay,pts):
    dots=[p[0]*ax+p[1]*ay for p in pts]
    return (min(dots),max(dots))
def interval_overlap(a_min,a_max,b_min,b_max):
    return min(a_max,b_max)-max(a_min,b_min)
def sat_rect_poly_MTV(rect,poly):
    axes=[(1,0),(0,1)]
    for i in range(len(poly)):
        x1,y1=poly[i]; x2,y2=poly[(i+1)%len(poly)]
        ex,ey=(x2-x1,y2-y1); nx,ny=normalize(-ey,ex)
        if nx==ny==0: continue
        axes.append((nx,ny))
    rect_pts=rect_points(rect)
    rcx=(rect.left+rect.right)/2; rcy=(rect.top+rect.bottom)/2
    pcx=sum(p[0] for p in poly)/len(poly)
    pcy=sum(p[1] for p in poly)/len(poly)
    smallest=1e12; s_axis=(0,0)
    for ax,ay in axes:
        rmin,rmax=project_points(ax,ay,rect_pts)
        pmin,pmax=project_points(ax,ay,poly)
        ov=interval_overlap(rmin,rmax,pmin,pmax)
        if ov<=0: return (0,0),False
        if ov<smallest:
            smallest=ov
            r_c=rcx*ax+rcy*ay; p_c=pcx*ax+pcy*ay
            dir=-1 if (p_c-r_c)>0 else 1
            s_axis=(ax*dir,ay*dir)
    mtv=(s_axis[0]*smallest,s_axis[1]*smallest)
    return mtv,True

class Player:
    def __init__(self,x,y):
        self.rect=pygame.Rect(x,y,PLAYER_W,PLAYER_H)
        self.vx=self.vy=0.0; self.on_ground=False
    def handle_input(self):
        k=pygame.key.get_pressed(); self.vx=0
        if k[pygame.K_LEFT] or k[pygame.K_a]: self.vx=-MOVE_SPEED
        if k[pygame.K_RIGHT] or k[pygame.K_d]: self.vx=MOVE_SPEED
        if (k[pygame.K_SPACE] or k[pygame.K_UP]) and self.on_ground:
            self.vy=JUMP_VEL; self.on_ground=False
    def apply_gravity(self):
        self.vy=min(self.vy+GRAVITY,MAX_FALL)
    def transport_by_obstacles(self,obs):
        pad=TRANSPORT_TOUCH_PAD; me=self.rect.inflate(pad*2,pad*2)
        dx=dy=0
        for o in obs:
            x,y,w,h=o["bbox"]; obb=pygame.Rect(int(x),int(y),int(w),int(h))
            if me.colliderect(obb):
                dx+=o["vx"]; dy+=o["vy"]
        if dx or dy:
            self.rect.x+=int(round(dx)); self.rect.y+=int(round(dy))
    def move_and_resolve(self,obs):
        self.rect.x+=int(self.vx); self.rect.y+=int(self.vy)
        self.on_ground=False; gvx=gvy=0
        for o in obs:
            mtv,hit=sat_rect_poly_MTV(self.rect,o["points"])
            if not hit: continue
            self.rect.x+=int(round(mtv[0])); self.rect.y+=int(round(mtv[1]))
            if mtv[1]<0:
                self.vy=0; self.on_ground=True; gvx+=o["vx"]; gvy+=o["vy"]
            elif mtv[1]>0: self.vy=0
        if self.on_ground:
            self.rect.x+=int(round(gvx)); self.rect.y+=int(round(gvy))

# ----------------------------
# Difference-based detection helpers
# ----------------------------
def detect_projection_boundary(frame_bgr):
    """Detects the magenta frame drawn by the projector."""
    hsv=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2HSV)
    lower=np.array([140,100,100]); upper=np.array([170,255,255])
    mask=cv2.inRange(hsv,lower,upper)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    x,y,w,h=cv2.boundingRect(max(cnts,key=cv2.contourArea))
    return (x,y,w,h)

def capture_background_reference(cap,roi):
    x,y,w,h=roi
    ok,frame=cap.read()
    if not ok: return None
    return frame[y:y+h,x:x+w].copy()

def detect_foreground_objects(frame,background_ref,roi,thresh=35):
    x,y,w,h=roi
    cur=cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
    bg=cv2.cvtColor(background_ref,cv2.COLOR_BGR2GRAY)
    diff=cv2.absdiff(bg,cur)
    _,mask=cv2.threshold(diff,thresh,255,cv2.THRESH_BINARY)
    mask=cv2.medianBlur(mask,5)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k)
    return mask

def cv_to_surface(bgr):
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    h,w=rgb.shape[:2]
    return pygame.image.frombuffer(rgb.tobytes(),(w,h),'RGB')

# ----------------------------
# Main
# ----------------------------
def main():
    cap=cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Cannot open camera"); return

    pygame.init()
    pygame.display.set_caption("Projection Diff Tracker")
    ret,frame=cap.read()
    if not ret:
        print("No frame"); return

    H,W=frame.shape[:2]
    win_w,win_h=int(W*WINDOW_SCALE),int(H*WINDOW_SCALE)
    screen=pygame.display.set_mode((win_w,win_h))
    font=pygame.font.SysFont(None,22)
    clock=pygame.time.Clock()

    tracker=PolyTracker()
    player=Player(win_w//2-PLAYER_W//2,10)

    # --- Calibration ---
    print("Show magenta border projection. Detecting...")
    roi=None
    while roi is None:
        ok,frame=cap.read()
        if not ok: continue
        roi=detect_projection_boundary(frame)
        pygame.draw.rect(screen,(255,0,255),screen.get_rect(),8)
        pygame.display.flip()
    print(f"Projection ROI: {roi}")
    print("Capturing clean background...")
    time.sleep(0.5)
    background_ref=capture_background_reference(cap,roi)
    print("Ready!")

    show_mask=False; running=True
    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: running=False
            elif e.type==pygame.KEYDOWN:
                if e.key==pygame.K_q: running=False
                elif e.key==pygame.K_m: show_mask=not show_mask
                elif e.key==pygame.K_c:
                    print("Recalibrating...")
                    roi=detect_projection_boundary(frame)
                    background_ref=capture_background_reference(cap,roi)
                    print("Recalibrated.")

        ok,frame=cap.read()
        if not ok: break

        mask=detect_foreground_objects(frame,background_ref,roi)
        obstacles_raw=tracker.detect_from_mask(mask)

        # Scale polygons to window size
        scale=WINDOW_SCALE
        obstacles=[]
        for ob in obstacles_raw:
            pts=[(p[0]*scale,p[1]*scale) for p in ob["points"]]
            x,y,w,h=ob["bbox"]
            obstacles.append({
                "points":pts,
                "vx":ob["vx"]*scale,"vy":ob["vy"]*scale,
                "bbox":(x*scale,y*scale,w*scale,h*scale)
            })

        bg=frame if not show_mask else cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        surf=cv_to_surface(bg)
        if scale!=1.0: surf=pygame.transform.smoothscale(surf,(win_w,win_h))
        screen.blit(surf,(0,0))
        pygame.draw.rect(screen,(255,0,255),screen.get_rect(),8)  # border

        for o in obstacles:
            pts_int=[(int(x),int(y)) for x,y in o["points"]]
            pygame.draw.polygon(screen,(255,255,0),pts_int,2)

        player.transport_by_obstacles(obstacles)
        player.handle_input()
        player.apply_gravity()
        player.move_and_resolve(obstacles)

        player.rect.left=clamp(player.rect.left,0,win_w-player.rect.width)
        if player.rect.bottom>win_h:
            player.rect.x=win_w//2-PLAYER_W//2; player.rect.y=10
            player.vx=player.vy=0; player.on_ground=False

        pygame.draw.rect(screen,(255,255,255),player.rect,0)

        t=font.render(f"FPS: {clock.get_fps():.1f} | Obstacles: {len(obstacles)} | M:mask  C:recal  Q:quit",
                      True,(255,255,255))
        screen.blit(t,(10,10))
        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()

if __name__=="__main__":
    main()
