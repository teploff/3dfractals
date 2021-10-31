from ursina import Ursina, camera, window, Light, color, scene, Entity, held_keys, time, Mesh, EditorCamera, invoke
import math
from ursina import Mesh


from calculations.one_phase import calculate
from visualization.entity import Builder

MAX_DEPTH = 5
LIMIT_VALUE = 2.0

# ONE PHASE CONSTANTS
ITER_COUNT = 30

# SEVERAL PHASES CONSTANTS
ITER_TETRAHEDRON_COUNT = 30
ITER_TRIANGLE_COUNT = 10

ACCURACY = 0.001


class Game(Ursina):
    def __init__(self):
        super().__init__()

        window.color = color.black
        window.fullscreen_size = 1920, 1080
        window.fullscreen = False

        Light(type='ambient', color=(0.5, 0.5, 0.5, 1))
        Light(type='directional', color=(0.5, 0.5, 0.5, 1), direction=(1, 1, 1))

        self.fractal = Builder(calculate(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH))
        # self.fractal = several_phases_build(ITER_TETRAHEDRON_COUNT, ITER_TRIANGLE_COUNT, LIMIT_VALUE)
        self.state = 0

        EditorCamera()

    @staticmethod
    def new_game():
        scene.clear()

    def update(self):
        pass

    def input(self, key):
        if key == 'q':
            if self.state == 0:
                return
            self.state -= 1

            scene.clear()
            self.fractal.gen(self.state).model.generate()
        elif key == 'e':
            if self.state == (len(self.fractal.sequence) - 1):
                return
            self.state += 1

            scene.clear()
            self.fractal.gen(self.state).model.generate()

        super().input(key)


if __name__ == '__main__':
    game = Game()
    update = game.update
    game.run()
