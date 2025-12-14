

def ball_physics(ball, ball_vel, objects, 
                 WIDTH, HEIGHT):
    # Update ball position
    ball.x += ball_vel[0]
    ball.y += ball_vel[1]

    # Bounce off walls
    if ball.left < 0 or ball.right > WIDTH:
        ball_vel[0] = -ball_vel[0]
    if ball.top < 0 or ball.bottom > HEIGHT:
        ball_vel[1] = -ball_vel[1]

    for obj in objects:
        for px, py in obj["polygon_pygame"]:
            if ball.collidepoint(px, py):
                ball_vel[0] *= -1
                ball_vel[1] *= -1
                break

    return ball, ball_vel