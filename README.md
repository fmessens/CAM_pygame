# CAM_pygame
Interactivity with games through a camera + projector setup.
![ball game](images/visual.jpg)

## SETUP

You need a beamer and a USB camera (eg. webcam) pointed at the beamer screen. The default setup projects to screen 2.
![Screen setup](images/screen_setup.png)

## SCRIPTS

Root scripts:
- `level_1.py` Dock level (shadow-driven wave + boat).
- `level_2.py` Gravity well with warped background + flying objects.
- `main_levels.py` Runs level 1, then level 2 on success.

Experimental scripts live in `experiments/`.

## RUN

Level 1:
`python level_1.py`

Level 2:
`python level_2.py --image images/galaxy-1489300667bhk.jpg`

Combined flow:
`python main_levels.py --image images/galaxy-1489300667bhk.jpg`

## CALIBRATION CACHE

Both level scripts support `--cached-calibration`.
- If enabled: load `calibration_cache.npz`.
- If not enabled: run calibration and write `calibration_cache.npz`.


