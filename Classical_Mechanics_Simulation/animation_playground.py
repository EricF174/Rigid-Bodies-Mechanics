import pygame
from bodies import *

TICK = 60  # updates 60 times per second

ball1 = body()
ball1.velocity = [2, 0]
ball1.com = [100, 500]
ball1.draw_equilateral(30, radius=50)
# ball1.draw_custom_shape(np.array([[0, 0], [0, 100], [100, 100], [100, 0]]))
ball1.body_colour = (255, 0, 0)
ball1.mass = 10
ball1.add_force([0, 9.81 * ball1.mass])

ball2 = body()
ball2.velocity = [0, 0]
ball2.com = [500, 1000]
# ball2.draw_equilateral(4, radius=50)
ball2.draw_custom_shape(np.array([[0, 0], [0, 1000], [100, 1000], [100, 0]]))
ball2.body_colour = (0, 0, 255)
ball2.mass = 1e100
# ball2.add_force([0, 9.81*ball2.mass])
# ball.add_force([-10, 0])

ball3 = body()
ball3.velocity = [-1, 0]
ball3.com = [900, 500]
ball3.draw_equilateral(30, radius=50)
# ball2.draw_custom_shape(np.array([[0, 0], [0, 100], [100, 100], [100, 0]]))
ball3.body_colour = (0, 255, 0)
ball3.mass = 10
ball3.add_force([0, 9.81*ball3.mass])
# ball.add_force([-10, 0])

pygame.init()
display_surface = pygame.display.set_mode((1000, 1000))
pygame.display.set_caption('learning pygame')
display_surface.fill((255, 255, 255))
clock = pygame.time.Clock()

objects = [ball3, ball1, ball2]


engine_running = True
while engine_running:
    # This chunk makes it so window close stops program
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            engine_running = False

    display_surface.fill((255, 255, 255))

    for obj in objects:
        obj.eom()
        pygame.draw.polygon(display_surface, obj.body_colour, obj.vertices)

    collision_objs = check_collision(objects)
    if collision_objs != 0:
        # pygame.draw.circle(display_surface, (255, 0, 0), [100, 100], 50)
        collision_response(collision_objs)
    pygame.display.update()
    clock.tick(TICK)
pygame.quit()
