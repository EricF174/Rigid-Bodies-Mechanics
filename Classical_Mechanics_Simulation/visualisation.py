import pyglet
from equations_of_motion import *
from pyglet import shapes
from pyglet import clock

obj = particle()
obj.com = np.array([1, 1])
obj.mass = 10
obj.draw_custom_shape(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))


pyglet.app.run()
