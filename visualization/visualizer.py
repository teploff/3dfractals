from ursina import Ursina, camera, window, Light, color, scene, Entity, held_keys, time, Mesh, EditorCamera, invoke

from calculations.classic_one_phase import calculate as classic_one_phase_calc
from calculations.one_phase import calculate as one_phase_calc
from calculations.several_phases import calculate as several_phase_calc
# from calculations.stochasticity import calculate as stochastic_calc
from visualization.entity import Builder

MAX_DEPTH = 7
LIMIT_VALUE = 2.0

# ONE PHASE CONSTANTS
ITER_COUNT = 20

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

        self.fractal = Builder(classic_one_phase_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, only_for_metrics=False))
        # self.fractal = Builder(one_phase_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, only_for_metrics=False))
        # self.fractal = Builder(several_phase_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, only_for_metrics=True))
        # self.fractal = Builder(stochastic_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH))
        self.state = 0

        EditorCamera()

    @staticmethod
    def new_game():
        scene.clear()

    def update(self):
        pass

    def _behaviour(self, key):
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
        elif key == "w":
            self.state = len(self.fractal.sequence) - 1

            scene.clear()
            self.fractal.gen(self.state).model.generate()
        elif key == "s":
            self.state = 0

            scene.clear()
            self.fractal.gen(self.state).model.generate()

    def input_hold(self, key):
        self._behaviour(key)

        super().input_hold(key)

    def input(self, key):
        self._behaviour(key)

        super().input(key)


if __name__ == '__main__':
    game = Game()
    update = game.update
    game.run()
