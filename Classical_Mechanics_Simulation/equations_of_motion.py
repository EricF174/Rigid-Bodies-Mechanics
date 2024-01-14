from bodies import *


def eom(obj):
    """
    :param obj: particle object
    :param forces: numpy array
    :return: acceleration
    """
    # sum f = ma
    obj.acceleration = sum(obj.forces) / obj.mass
    return


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
ball.mass = 10
ball.forces = np.append(ball.forces, [[0, -9.81 * ball.mass]], axis=0)
ball.forces = np.append(ball.forces, [[10, 0]], axis=0)
for i in range(99):
    eom(ball)
    kinematics(ball, 1)
    # print(ball.velocity, ball.com)
