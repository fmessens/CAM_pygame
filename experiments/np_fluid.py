import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

H, W = 100, 160
dt = 0.3
damping = 0.15
g = 0.6                      # gravity strength

water = np.zeros((H, W), dtype=np.float32)
solid = np.zeros((H, W), dtype=bool)

vy = np.zeros((H, W), dtype=np.float32)
vx = np.zeros((H, W), dtype=np.float32)

solid[60:65, 30:130] = True
water[0:4, 70:90] = 1.0


def grad_y(a):
    g = np.zeros_like(a)
    g[:-1, :] = a[1:, :] - a[:-1, :]
    return g

def grad_x(a):
    g = np.zeros_like(a)
    g[:, :-1] = a[:, 1:] - a[:, :-1]
    return g

def advect(field, vy, vx):
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    y0 = np.clip((y - vy), 0, H-1).astype(np.int32)
    x0 = np.clip((x - vx), 0, W-1).astype(np.int32)

    return field[y0, x0]


def step(water, vy, vx, solid):
    # pressure from water height
    Fy = -grad_y(water)
    Fx = -grad_x(water)

    # velocity update (THIS was missing gravity)
    vy[:] = (1 - damping) * vy + dt * Fy + g
    #Fx[:-1,:] = Fx[:-1,:]*(vy[:-1,:]-vy[1:,:])  # prevent water from going up through solid
    vx[:] = (1 - damping) * vx + dt * Fx

    # block solids
    vy[solid] = 0
    vx[solid] = 0

    # advect water
    water[:] = advect(water, vy, vx)
    water[:] = np.clip(water, 0, 1)
    water[solid] = 0
    return Fx, Fy, vy, vx


while True:
    water[0:4, 70:90] = 1.0
    Fx, Fy, vy, vx = step(water, vy, vx, solid)
    clear_output(wait=True)
    dvy = grad_y(vy)
    plt.imshow(Fx + solid, cmap="Blues")
    plt.axis("off")
    plt.show()
    time.sleep(0.03)