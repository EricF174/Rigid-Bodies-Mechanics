from bodies import *


def eom(obj, forces):
    """
    :param obj: particle object
    :param forces: numpy array
    :return: acceleration
    """
    # sum f = ma
    obj.acceleration = sum(forces) / obj.mass
    return


def kinematics(obj, time_step):
    """
    :param obj:
    :param time_step:
    :return:
    """
    # delta d = v_initial * t + 1/2 * a * t^2
    delta_d = obj.velocity * time_step + 0.5 * obj.acceleration * time_step**2
    obj.position = obj.position + delta_d
    return

