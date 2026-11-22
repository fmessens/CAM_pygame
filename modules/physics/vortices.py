"""Identify compact rotating regions in a two-dimensional velocity grid."""

import cv2
import numpy as np


def detect_vortices(velocity: np.ndarray, solid: np.ndarray) -> tuple[np.ndarray, int]:
    u, v = cv2.GaussianBlur(velocity, (5, 5), 0.8).transpose(2, 0, 1)
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    # A complex eigenvalue pair of the velocity gradient indicates local rotation.
    # Pure shear can have high curl without a complex eigenvalue pair.
    spin_squared = -(((du_dx - dv_dy) / 2) ** 2 + du_dy * dv_dx)
    curl = dv_dx - du_dy
    rotating = (spin_squared > 0.035 ** 2) & ~solid
    rotating[[0, -1], :] = False
    rotating[:, [0, -1]] = False

    mask = np.zeros_like(solid, dtype=bool)
    count = 0
    for direction in (-1, 1):
        candidates = (rotating & (direction * curl > 0)).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= 4
        keep[0] = False
        count += int(np.count_nonzero(keep))
        mask |= keep[labels]
    return mask, count
