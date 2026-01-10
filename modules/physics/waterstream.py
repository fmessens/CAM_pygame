import pygame
import random
import math

from modules.poly_utils import point_in_poly, polygon_normal
from modules.settings import WATER_GRAVITY, WATER_DRAG

GRAVITY = pygame.Vector2(0, WATER_GRAVITY)
DRAG = WATER_DRAG

class WaterParticle:
    __slots__ = ("pos", "vel", "life")

    def __init__(self, pos, vel):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.life = random.randint(90, 150)

    def update(self):
        self.vel += GRAVITY
        self.vel *= DRAG
        self.pos += self.vel
        self.life -= 1

class WaterStream:
    def __init__(self, origin):
        self.origin = pygame.Vector2(origin)
        self.particles = []

    def spawn(self, rate=8):
        base_angle = math.radians(90)  # straight down
        spread = math.radians(8)

        for _ in range(rate):
            a = base_angle + random.uniform(-spread, spread)
            speed = random.uniform(6, 9)
            vel = pygame.Vector2(
                math.cos(a) * speed,
                math.sin(a) * speed
            )
            self.particles.append(WaterParticle(self.origin, vel))

    def update(self, objects, WIDTH, HEIGHT):
        for p in self.particles:
            p.update()

            # wall bounds
            if p.pos.x < 0 or p.pos.x > WIDTH:
                p.vel.x *= -0.5
            if p.pos.y > HEIGHT:
                p.life = 0

            # shadow collision
            for obj in objects:
                poly = obj["polygon_pygame"]
                if point_in_poly(p.pos, poly):
                    # estimate surface normal from closest edge
                    normal = polygon_normal(p.pos, poly)
                    p.vel.reflect_ip(normal)
                    p.vel *= 0.6
                    break

        self.particles = [p for p in self.particles if p.life > 0]

    def draw(self, screen):
        for p in self.particles:
            pygame.draw.circle(
                screen,
                (60, 150, 255),
                (int(p.pos.x), int(p.pos.y)),
                2
            )
