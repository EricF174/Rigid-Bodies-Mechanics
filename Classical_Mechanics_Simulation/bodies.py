import numpy as np
import math

TICK = 60  # updates 60 times per second. LIMITATION: lower ticks increases inaccuracy in calculations due to cumulative
# time-step misrepresenting proper integral calculations


class body:
    # basic shapes are limited to polygons
    def __init__(self):
        """
        body attributes:
        com: center of mass (m)
        velocity: a 1x2 array representing vector (m/s)
        acceleration: a 1x2 array representing vector (m/s^2)
        forces: nx4 array of 1x4 vector arrays representing forces acting on body (N) where first two indices represent
                magnitude and direction, last two represent point of force exertion
        mass: mass of body (kg)
        vertices: arrays of points that connect with each other, where [0,0] is the center point (m)
        area: area of body (m^2)
        """
        self.com = np.array([])
        self.velocity = np.array([])
        self.acceleration = np.array([])

        self.mass = None
        self.forces = np.array([[0 ,0]])

        self.vertices = np.array([])
        self.area = None

        self.body_colour = ()
    def add_force(self, force, point=np.array([0, 0])):
        """
        :param point:
        :param force: 1x2 vector representing force (N)
        :return:
        """
        self.forces = np.append(self.forces, [force], axis=0)
    def draw_custom_shape(self, points):
        """
        Manually enter points and the function will adjust coordinates such that (0,0) is the center of mass
        :param points: n x 2 numpy array
        :return: n x 2 numpy array of adjusted set of points which draws a polygon
        """
        # clear current points
        self.vertices = np.array([])

        # auto adjust so [0,0] is center of mass
        # https://en.wikipedia.org/wiki/Centroid: find area first using shoelace formula
        tot = 0
        iteration = len(points) - 1
        for i in range(iteration):
            tot += (points[i, 0] * points[i + 1, 1]) - (points[i + 1, 0] * points[i, 1])
        A = tot / 2
        self.area = abs(A)
        # find centroid
        tot = 0
        for i in range(iteration):
            tot += (points[i, 0] + points[i + 1, 0]) * (
                    points[i, 0] * points[i + 1, 1] - points[i + 1, 0] * points[i, 1])
        cx = tot / (6 * A)

        tot = 0
        for i in range(iteration):
            tot += (points[i, 1] + points[i + 1, 1]) * (
                    points[i, 0] * points[i + 1, 1] - points[i + 1, 0] * points[i, 1])
        cy = tot / (6 * A)
        # re-adjust vertices so points are relative to window
        self.vertices = np.transpose(
            np.append([points[:, 1] - cy + self.com[0]], [points[:, 0] - cx + self.com[1]], axis=0))


    def draw_equilateral(self, vertices, radius):
        """
        This method creates equilateral polygons
        :param vertices: number of vertices
        :param radius: radius
        :return: n x 2 numpy array of points which draws a polygon
        """
        # clear current points
        self.vertices = np.array([])

        # https://stackoverflow.com/questions/3436453/calculate-coordinates-of-a-regular-polygons-vertices
        # initialise points
        points = np.empty((0, 2))
        for i in range(vertices):
            x = radius * math.cos(2 * math.pi * i / vertices)
            y = radius * math.sin(2 * math.pi * i / vertices)
            points = np.append(points, [[x + self.com[0], y + self.com[1]]], axis=0)
        self.vertices = points

    def eom(self):
        """
        This method computates the acceleration and angular acceleration of the bodies using the equations of motion
        """
        # sum f = ma
        self.acceleration = np.array([sum(self.forces[:, 0]), sum(self.forces[:, 1])]) / self.mass
        # convert to real time
        acceleration = self.acceleration / TICK

        # we now find out our velocity in next time step using Improved Euler's Method, which improves accuracy over
        # regular eulers method
        velocity = self.velocity + acceleration
        velocity_ini = velocity
        acceleration_plus1 = (np.array([sum(self.forces[:, 0]), sum(self.forces[:, 1])]) / self.mass) / TICK
        velocity_plus1 = self.velocity + acceleration_plus1
        self.velocity = (velocity_ini + velocity_plus1) / 2
        self.com = self.com + self.velocity
        self.vertices = np.transpose(
            np.append([self.vertices[:, 0] + self.velocity[0]], [self.vertices[:, 1] + self.velocity[1]], axis=0))




        return

class rope:
    pass


class spring:
    pass


def check_collision(objects):
    collided_objects = np.empty((0, 2))
    """
    Using seperating axis theorem https://research.ncl.ac.uk/game/mastersdegree/gametechnologies/previousinformation/physics4collisiondetection/2017%20Tutorial%204%20-%20Collision%20Detection.pdf
    on every axis of all shapes check if they have collisions, collision being if shape A max > shape B min AND shape A
    min < shape B max
    :param objects: all bodies in system
    :return:
    """
    # using vector dot product/projection, vector a projected into vector b = a * unit vector of b
    # create all projection vectors
    p_vectors = np.empty((0, 2))
    for obj in objects:
        # find the vector of each edge and divide by its length to get its unit vector
        p_vector = np.divide([obj.vertices[-1, 0] - obj.vertices[0, 0], obj.vertices[-1, 1] - obj.vertices[0, 1]], (
                    (obj.vertices[-1, 0] - obj.vertices[0, 0]) ** 2 + (obj.vertices[-1, 1] -
                                                                       obj.vertices[0, 1]) ** 2) ** 0.5)
        p_vectors = np.append(p_vectors, [p_vector], axis=0)
        for i in range(len(obj.vertices)-1):
            p_vector = np.divide([obj.vertices[i + 1, 0] - obj.vertices[i, 0], obj.vertices[i + 1, 1] -
                                  obj.vertices[i, 1]], ((obj.vertices[i + 1, 0] - obj.vertices[i, 0]) ** 2 +
                                                        (obj.vertices[i + 1, 1] - obj.vertices[i, 1]) ** 2) ** 0.5)
            p_vectors = np.append(p_vectors, [p_vector], axis=0)  # squaring removes negative -1 into 1
    # remove repeating unit vectors to reduce processing time
    p_vectors = np.unique(p_vectors, axis=0)

    # now project every object points into the unit vectors and find min and max of objects
    for i in range(len(objects)):
        obj1 = objects[i]

        for j in range(i + 1, len(objects)):
            obj2 = objects[j]

            gap_detect = 0
            for project in p_vectors:
                # project object 1
                obj1_proj_points = np.array(obj1.vertices[:, 0] * project[0] + obj1.vertices[:, 1] * project[1])
                [obj1_min, obj1_max] = [min(obj1_proj_points), max(obj1_proj_points)]
                # project object 2
                obj2_proj_points = np.array(obj2.vertices[:, 0] * project[0] + obj2.vertices[:, 1] * project[1])
                [obj2_min, obj2_max] = [min(obj2_proj_points), max(obj2_proj_points)]

                # gap conditions
                if obj2_max < obj1_min or obj1_max < obj2_min:
                    # no_collision detected
                    gap_detect += 1

            if gap_detect == 0:
                collided_objects = np.append(collided_objects, np.array([[obj1, obj2]]), axis=0)
                break

    return collided_objects  # no collision


def collision_response(collided_objects, last_tick_collided_objects):
    if collided_objects in last_tick_collided_objects:
        return
    # two possible scenarios: the vertice collide or the edge collide

    # Assign the two colliding objects
    obj1 = collided_objects[0]
    obj2 = collided_objects[1]

    # Calculate the point of contact, by calculating the shortest distance of every vertices in objects 1 to every
    # line segment in object 2 and vice versa

    # initialise the shortest distance to infinity
    shortest_distance = math.inf
    shortest_point = None

    for point1 in obj1.vertices:
        for i in range(len(obj2.vertices) - 1):
            # using https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
            # and https://math.stackexchange.com/questions/2248617/shortest-distance-between-a-point-and-a-line-segment

            # Calculate whether the shortest distance from the point to line segment is at the ends of the segment or
            # in between
            t = (-1 * ((obj2.vertices[i, 0] - point1[0]) * (obj2.vertices[i+1, 0] - obj2.vertices[i, 0]) +
                      (obj2.vertices[i, 1] - point1[1]) * (obj2.vertices[i+1, 1] - obj2.vertices[i, 1])) /
                 ((obj2.vertices[i+1, 0] - obj2.vertices[i, 0]) ** 2 + (obj2.vertices[i+1, 1] -
                                                                        obj2.vertices[i, 1]) ** 2))

            # in between the vertices
            if 0 <= t <= 1:
                potential_shortest_distance = (abs((obj2.vertices[i+1, 0] - obj2.vertices[i, 0]) *
                                                  (obj2.vertices[i, 1] - point1[1]) - (obj2.vertices[i, 0] - point1[0])
                                                  * (obj2.vertices[i+1, 1] - obj2.vertices[i, 1])) /
                                               (((obj2.vertices[i+1, 0] - obj2.vertices[i, 0]) ** 2 +
                                                 (obj2.vertices[i+1, 1] - obj2.vertices[i, 1]) ** 2) ** 0.5))
            else:
                # distance between point and vertices
                distance_to_vertices = [
                    ((obj2.vertices[i, 0] - point1[0]) ** 2 + (obj2.vertices[i, 1] - point1[1]) ** 2) ** 0.5,
                    ((obj2.vertices[i + 1, 0] - point1[0]) ** 2 + (obj2.vertices[i + 1, 1] - point1[1]) ** 2) ** 0.5]
                potential_shortest_distance = min(distance_to_vertices)

            if potential_shortest_distance < shortest_distance:
                shortest_distance = potential_shortest_distance
                # normal = surface of contact - now we find the vector of the contact surface
                normal = [obj2.vertices[i + 1, 0] - obj2.vertices[i, 0], obj2.vertices[i + 1, 1] - obj2.vertices[i, 1]]
                shortest_point = point1

    # now do the opposite in repeat
    for point2 in obj2.vertices:
        for i in range(len(obj1.vertices) - 1):
            # distance to vertices

            t = (-1 * ((obj1.vertices[i, 0] - point2[0]) * (obj1.vertices[i+1, 0] - obj1.vertices[i, 0]) +
                      (obj1.vertices[i, 1] - point2[1]) * (obj1.vertices[i+1, 1] - obj1.vertices[i, 1])) /
                 ((obj1.vertices[i+1, 0] - obj1.vertices[i, 0]) ** 2 + (obj1.vertices[i+1, 1] -
                                                                        obj1.vertices[i, 1]) ** 2))

            if 0 <= t <= 1:
                potential_shortest_distance = (abs((obj1.vertices[i+1, 0] - obj1.vertices[i, 0]) *
                                                  (obj1.vertices[i, 1] - point2[1]) - (obj1.vertices[i, 0] - point2[0])
                                                  * (obj1.vertices[i+1, 1] - obj1.vertices[i, 1])) /
                                               (((obj1.vertices[i+1, 0] - obj1.vertices[i, 0]) ** 2 +
                                                 (obj1.vertices[i+1, 1] - obj1.vertices[i, 1]) ** 2) ** 0.5))
            else:
                # distance between point and vertices
                distance_to_vertices = [
                    ((obj1.vertices[i, 0] - point2[0]) ** 2 + (obj1.vertices[i, 1] - point2[1]) ** 2) ** 0.5,
                    ((obj1.vertices[i + 1, 0] - point2[0]) ** 2 + (obj1.vertices[i + 1, 1] - point2[1]) ** 2) ** 0.5]
                potential_shortest_distance = min(distance_to_vertices)

            if potential_shortest_distance < shortest_distance:
                shortest_distance = potential_shortest_distance
                normal = [obj1.vertices[i + 1, 0] - obj1.vertices[i, 0], obj1.vertices[i + 1, 1] - obj1.vertices[i, 1]]
                shortest_point = point2

    # normalise the vector
    unit_normal = np.divide(normal, (normal[0] ** 2 + normal[1] ** 2) ** 0.5)

    # now we check if the point is inside or outside the polygon using ray casting:
    # source from https://www.youtube.com/watch?v=RSXM9bgqxJM&ab_channel=Insidecode



    # Linear Momentum
    # object momentum along tangential direction is conserved
    # system momentum along normal direction is conserved
    # using the equation for the coefficient of restitution, e = 1, and system momentum along the normal direction
    # first we must rotate coordinate system along normal axis
    theta = math.atan(unit_normal[1] / unit_normal[0])

    # apply transformation to normal tangential
    v1_t = obj1.velocity[0]*math.cos(theta) + obj1.velocity[1]*math.sin(theta)
    v2_t = obj2.velocity[0]*math.cos(theta) + obj2.velocity[1]*math.sin(theta)

    v1_n = -1 * obj1.velocity[0]*math.sin(theta) + obj1.velocity[1]*math.cos(theta)
    v2_n = -1 * obj2.velocity[0]*math.sin(theta) + obj2.velocity[1]*math.cos(theta)
    # v1_n - v2_n = v2_n_new - v1_n_new
    # m1*v1_n + m2*v2_n = m1*v1_n_new + m2*v2_n_new
    # rearrange to get
    v1_n_new = ((obj1.mass - obj2.mass) * v1_n + 2 * obj2.mass*v2_n) / (obj1.mass + obj2.mass)
    v2_n_new = v1_n - v2_n + v1_n_new
    # now convert back to regular x-y coordinate system
    v1_x = v1_t*math.cos(theta) - v1_n_new*math.sin(theta)
    v2_x = v2_t*math.cos(theta) - v2_n_new*math.sin(theta)
    v1_y = v1_t*math.sin(theta) + v1_n_new*math.cos(theta)
    v2_y = v2_t*math.sin(theta) + v2_n_new*math.cos(theta)

    # reassign new velocities
    obj1.velocity = np.array([v1_x, v1_y])
    obj2.velocity = np.array([v2_x, v2_y])
    # print(shortest_point, shortest_distance)
    return

