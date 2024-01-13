import pyglet
from pyglet import shapes
from pyglet import clock

window = pyglet.window.Window()

label = pyglet.text.Label('0',
                          font_name='Times New Roman',
                          font_size=window.width // 10,
                          x=window.width // 2, y=window.height // 2,
                          anchor_x='center', anchor_y='center')

circle = shapes.Circle(x=100, y=150, radius=100, color=(50, 225, 30))
circle.dx = 10.0

def move_circle(dt):
    circle.x += circle.dx * dt
def tick(dt):
    label.text = str(int(label.text) + 1)

pyglet.clock.schedule_interval(move_circle, 1/60.0)  # update at 60Hz
pyglet.clock.schedule_interval(tick, 1)  # update at 60Hz

@window.event
def on_draw():
    window.clear()
    label.draw()
    circle.draw()


pyglet.app.run()
