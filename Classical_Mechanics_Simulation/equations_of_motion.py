from bodies import *


def kinematics(obj, time_step):
    """
    :param obj:
    :param time_step:
    :return:
    """
    # delta d = v_initial * t + 1/2 * a * t^2
    delta_d = obj.velocity * time_step + 0.5 * obj.acceleration * time_step**2
    obj.com = obj.com + delta_d

    obj.velocity = obj.velocity + obj.acceleration * time_step
    return


ball = particle()
ball.velocity = ball.com = ball.acceleration = [0, 0]
ball.draw_equilateral(3,radius=2)
ball.mass = 10
ball.add_force([1, 1])
print(ball.edges)

