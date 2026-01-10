import pygame

def point_in_poly(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)

    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]

        if ((y1 > y) != (y2 > y)) and \
           (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-6) + x1):
            inside = not inside
    return inside


def polygon_normal(pt, poly):
    # find closest edge
    min_dist = 1e9
    best_normal = pygame.Vector2(0, -1)

    for i in range(len(poly)):
        p1 = pygame.Vector2(poly[i])
        p2 = pygame.Vector2(poly[(i + 1) % len(poly)])
        edge = p2 - p1
        normal = pygame.Vector2(-edge.y, edge.x).normalize()

        dist = abs((pt - p1).dot(normal))
        if dist < min_dist:
            min_dist = dist
            best_normal = normal

    return best_normal
