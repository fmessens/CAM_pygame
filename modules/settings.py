import os
import random

WIDTH, HEIGHT = 800, 600
lagbuffer_ball=10
lagbuffer_water=10
os.environ['SDL_VIDEO_WINDOW_POS'] = "1280,0"

start_velocity = [4, 3]

initial_water_speed = (6, 9)

WATER_GRAVITY = 0.2
WATER_DRAG = 0.95