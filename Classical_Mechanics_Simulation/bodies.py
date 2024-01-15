import numpy as np
import math


class particle:
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
        edges: arrays of points that connect with each other, where [0,0] is the center point (m)
        area: area of body (m^2)
        """
        self.com = np.array([])
        self.velocity = np.array([])
        self.acceleration = np.array([])

        self.mass = None
        self.forces = np.empty((0, 2))

        self.edges = np.array([])
        self.area = None

    def add_force(self, force, point=np.array([0, 0])):
        """
        :param point:
        :param force: 1x2 vector representing force (N)
        :return:
        """
        self.forces = np.append(self.forces, [force, point])

    def draw_custom_shape(self, points):
        """
        Manually enter points and the function will adjust coordinates such that (0,0) is the center of mass
        :param points: n x 2 numpy array
        :return: n x 2 numpy array of adjusted set of points which draws a polygon
        """
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
        # re-adjust edges so points are relative to window
        self.edges = np.transpose(
            np.append([points[:, 1] - cy + self.com[0]], [points[:, 0] - cx + self.com[1]], axis=0))
        pass

    def draw_equilateral(self, vertices, radius):
        """
        This method creates equilateral polygons
        :param vertices: number of vertices
        :param radius: radius
        :return: n x 2 numpy array of points which draws a polygon
        """
        # https://stackoverflow.com/questions/3436453/calculate-coordinates-of-a-regular-polygons-vertices
        # initialise points
        points = np.empty((0, 2))
        for i in range(vertices):
            x = radius * math.cos(2 * math.pi * i / vertices)
            y = radius * math.sin(2 * math.pi * i / vertices)
            points = np.append(points, [[x, y]], axis=0)
        self.edges = points

    def eom(self):
        """
        This method computates the acceleration and angular acceleration of the bodies using the equations of motion
        """
        # sum f = ma
        self.acceleration = sum(self.forces[0:2]) / self.mass
        return


class rope:
    pass


class spring:
    pass


def check_collision(objects):
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
        p_vector = np.divide([obj.edges[-1, 0] - obj.edges[0, 0], obj.edges[-1, 1] - obj.edges[0, 1]], (
                    (obj.edges[-1, 0] - obj.edges[0, 0]) ** 2 + (obj.edges[-1, 1] - obj.edges[0, 1]) ** 2) ** 0.5)
        p_vectors = np.append(p_vectors, p_vector, axis=0)
        for i in range(len(obj.edges - 1)):
            p_vector = np.divide([obj.edges[i + 1, 0] - obj.edges[i, 0], obj.edges[i + 1, 1] - obj.edges[i, 1]], ((obj.edges[i + 1, 0] - obj.edges[i, 0]) ** 2 + (obj.edges[i + 1, 1] - obj.edges[i, 1]) ** 2) ** 0.5)
            p_vectors = np.append(p_vectors, p_vector, axis=0)
    # remove repeating unit vectors to reduce processing time
    p_vectors = np.unique(p_vectors, axis=0)

    # now project every object points into the unit vectors and find min and max of objects
    for project in p_vectors:
        for obj1 in objects:
            # project object 1
            obj1_proj_points = obj1.edges[:, 0] * project[0] + obj1.edges[:, 1] * project[1]
            [obj1_min, obj1_max] = [min(obj1_proj_points[:, 0]), max(obj1_proj_points[:, 1])]
            for obj2 in objects:
                if obj1 != obj2:
                    # project object 2
                    obj2_proj_points = obj2.edges[:, 0] * project[0] + obj2.edges[:, 1] * project[1]
                    [obj2_min, obj2_max] = [min(obj2_proj_points[:, 0]), max(obj2_proj_points[:, 1])]

                    if obj1_max <= obj2_min or obj1_min >= obj2_max:
                        return 0  # no collision
        return [obj1, obj2]  # collision


def collision_response(collided_objects):
    obj1 = collided_objects[0]
    obj2 = collided_objects[1]
    shortest_distance = math.inf
    for point1 in obj1.edges:
        for i in range(len(obj2.edges) - 1):
            # using https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
            # y = mx + k
            # m = (y1-y0)/(x1-x0)
            m = (obj2.edges[i + 1, 1] - obj2.edges[i, 1]) / (obj2.edges[i + 1, 0] - obj2.edges[i, 0])
            k = obj2.edges[i + 1] - m * obj2.edges[i + 1]
            potential_shortest_distance = (k + m * point1[0] - point1[1]) / (1 + m ** 2) ** 0.5
            if potential_shortest_distance <= shortest_distance:
                shortest_distance = potential_shortest_distance
                # find line
                x = (point1[0] + m * point1[1] - m * k) / (m ** 2 + 1)
                y = m * ((point1[0] + m * point1[1] - m * k) / (m ** 2 + 1)) + k
                normal = [y - point1[1], x - point1[0]]
    for point2 in obj2.edges:
        for i in range(len(obj1.edges) - 1):
            m = (obj1.edges[i + 1, 1] - obj1.edges[i, 1]) / (obj1.edges[i + 1, 0] - obj1.edges[i, 0])
            k = obj1.edges[i + 1] - m * obj1.edges[i + 1]
            potential_shortest_distance = (k + m * point2[0] - point2[1]) / (1 + m ** 2) ** 0.5
            if potential_shortest_distance < shortest_distance:
                shortest_distance = potential_shortest_distance
                # find line
                x = (point2[0] + m * point2[1] - m * k) / (m ** 2 + 1)
                y = m * ((point2[0] + m * point2[1] - m * k) / (m ** 2 + 1)) + k
                normal = [y - point2[1], x - point2[0]]
    unit_normal = np.divide(normal, (normal[0] ** 2 + normal[1] ** 2) ** 0.5)

    # object momentum along tangential direction is conserved
    # system momentum along normal direction is conserved
