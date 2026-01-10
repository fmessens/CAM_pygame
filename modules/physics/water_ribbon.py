import pygame
import numpy as np

class WaterRibbon:
    def __init__(self, x, top=0, bottom=600, step=12, width=14):
        self.step = step
        self.width = width
        self.top = top
        self.bottom = bottom
        self.span = bottom - top
        self.points = [
            pygame.Vector2(x, y)
            for y in range(top, bottom, step)
        ]
        self.speed = 3.2

    def update(self, objects):
        for i, p in enumerate(self.points):
            # Flow downward
            p.y += self.speed

            # Shadow interaction
            for obj in objects:
                for ox, oy in obj["polygon_pygame"]:
                    dy = abs(oy - p.y)
                    if dy < self.step * 1.5:
                        dx = p.x - ox
                        if abs(dx) < self.width * 2:
                            p.x += np.sign(dx) * 2.0

        # Wrap points that leave the bottom back to the top so the ribbon never vanishes
        for p in self.points:
            p.y = self.top + (p.y - self.top) % self.span

        # Keep points ordered from top to bottom for smooth polygon tangents
        self.points.sort(key=lambda v: v.y)

        # Smooth sideways motion
        for i in range(1, len(self.points) - 1):
            self.points[i].x = (
                self.points[i-1].x +
                self.points[i].x +
                self.points[i+1].x
            ) / 3.0

    def polygon(self):
        left = []
        right = []

        for i in range(len(self.points)):
            if i == 0:
                tangent = self.points[i+1] - self.points[i]
            else:
                tangent = self.points[i] - self.points[i-1]

            normal = pygame.Vector2(-tangent.y, tangent.x)
            if normal.length() > 0:
                normal.scale_to_length(self.width)

            left.append(self.points[i] + normal)
            right.append(self.points[i] - normal)

        return left + right[::-1]

    def draw(self, screen):
        poly = self.polygon()
        pygame.draw.polygon(screen, (70, 150, 255), poly)
