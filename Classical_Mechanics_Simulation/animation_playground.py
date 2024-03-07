import pygame
from bodies import *

TICK = 100  # updates 60 times per second. LIMITATION: lower ticks increases inaccuracy in calculations due to cumulative
# time-step misrepresenting proper integral calculations
ball1 = body()
ball1.velocity = [3, 0]
ball1.com = [500, 600]
# ball1.draw_equilateral(20, radius=80)
ball1.draw_custom_shape(np.array([[0, 0], [0, 100], [100, 100], [100, 0]]))
ball1.body_colour = (255, 0, 0)
ball1.mass = 10
ball1.add_force([0, -9.81 * ball1.mass])

# ball2 = body()
# ball2.velocity = [0, 0]
# ball2.com = [500, 800]
# # ball2.draw_equilateral(16, radius=40)
# ball2.draw_custom_shape(np.array([[0, 0], [0, 100], [100, 100], [100, 0]]))
# ball2.body_colour = (0, 255, 255)
# ball2.mass = 10
# ball2.add_force([0, -9.81*ball2.mass])
# # ball.add_force([-10, 0])

ball3 = body()
ball3.velocity = [3, 4]
ball3.com = [500, 400]
# ball3.draw_equilateral(20, radius=80)
ball3.draw_custom_shape(np.array([[0, 0], [0, 100], [100, 100], [100, 0]]))
ball3.body_colour = (0, 255, 0)
ball3.mass = 10
ball3.add_force([0, -9.81*ball3.mass])
# ball.add_force([-10, 0])

wallb = body()
wallb.velocity = [0, 0]
wallb.com = [500, 0]
wallb.draw_custom_shape(np.array([[0, 0], [0, 1000], [100, 1000], [100, 0]]))
wallb.body_colour = (0, 0, 255)
wallb.mass = 1e200

wallr = body()
wallr.velocity = [0, 0]
wallr.com = [1000, 500]
wallr.draw_custom_shape(np.array([[0, 0], [899, 0], [899, 100], [0, 100]]))
wallr.body_colour = (0, 0, 255)
wallr.mass = 1e200

walll = body()
walll.velocity = [0, 0]
walll.com = [0, 500]
walll.draw_custom_shape(np.array([[0, 0], [899, 0], [899, 100], [0, 100]]))
walll.body_colour = (0, 0, 255)
walll.mass = 1e200

wallu = body()
wallu.velocity = [0, 0]
wallu.com = [500, 1000]
wallu.draw_custom_shape(np.array([[0, 0], [0, 1000], [100, 1000], [100, 0]]))
wallu.body_colour = (0, 0, 255)
wallu.mass = 1e200

pygame.init()
display_surface = pygame.display.set_mode((1000, 1000))
pygame.display.set_caption('Rigid Body Sim')
display_surface.fill((255, 255, 255))
clock = pygame.time.Clock()


objects = [ball3, wallb, wallr, walll, wallu]
engine_running = True
ball1.coords = None
last_tick_collided_objects = 0
while engine_running:
    # This chunk makes it so window close stops program
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            engine_running = False

    display_surface.fill((255, 255, 255))
    display_surface.blit(pygame.transform.flip(display_surface, False, True), dest=(0, 0))

    for obj in objects:
        obj.eom()
        pygame.draw.polygon(display_surface, obj.body_colour, obj.vertices)

    collision_objs = check_collision(objects)
    if len(collision_objs) != 0:
        # pygame.draw.circle(display_surface, (255, 0, 0), [200, 900], 60)
        # print(ball1_coords)
        for c_objs in collision_objs:
            collision_response(c_objs, last_tick_collided_objects)

    last_tick_collided_objects = collision_objs
    display_surface.blit(pygame.transform.flip(display_surface, False, True), dest=(0, 0))
    ball1_coords = ball1.vertices[1, 1] - 50
    pygame.display.update()
    clock.tick(TICK)
pygame.quit()
