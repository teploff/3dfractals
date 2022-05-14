from ursina import Ursina, camera, window, Light, color, scene, Entity, held_keys, time, Mesh, EditorCamera, invoke

from calculations.classic_one_phase import calculate as classic_one_phase_calc
from calculations.one_phase import calculate as one_phase_calc
from calculations.several_phases import calculate as several_phase_calc
from calculations.stochasticity import calculate as stochastic_calc
from calculations.functional_one_phase import calculate as functional_calc
from calculations.combined import calculate as combined_calc
from visualization.entity import Builder

# COMMON CONSTANTS
ITER_COUNT = 2003
MAX_DEPTH = 5
LIMIT_VALUE = 2.0

# SEVERAL PHASES CONSTANTS
DELTA_ITERATIONS = 200

# STOCHASTIC CONSTANTS
PROBABILITY = 1.0


class Game(Ursina):
    def __init__(self):
        super().__init__()

        window.color = color.white
        window.fullscreen_size = 1920, 1080
        window.fullscreen = False

        Light(type='ambient', color=(0.5, 0.5, 0.5, 1))
        Light(type='directional', color=(0.5, 0.5, 0.5, 1), direction=(1, 1, 1))

        # self.fractal = Builder(classic_one_phase_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, only_for_metrics=True))

        # self.fractal = Builder(one_phase_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, only_for_metrics=False))

        # self.fractal = Builder(several_phase_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, 1, only_for_metrics=True))
        # self.fractal = Builder(several_phase_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, 200, only_for_metrics=True))
        # self.fractal = Builder(several_phase_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, 400, only_for_metrics=True))

        # self.fractal = Builder(stochastic_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, 0.6, only_for_metrics=True))
        # self.fractal = Builder(stochastic_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, 0.9, only_for_metrics=True))

        # self.fractal = Builder(functional_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, only_for_metrics=True))

        self.fractal = Builder(combined_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, PROBABILITY, DELTA_ITERATIONS, only_for_metrics=True))
        # self.fractal = Builder(combined_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, 0.75, 1, only_for_metrics=True))
        # self.fractal = Builder(combined_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, 0.9, 1, only_for_metrics=True))
        # self.fractal = Builder(combined_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, 0.6, 200, only_for_metrics=True))
        # self.fractal = Builder(combined_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, 0.75, 200, only_for_metrics=True))
        # self.fractal = Builder(combined_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, 0.9, 200, only_for_metrics=True))
        # self.fractal = Builder(combined_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, 0.6, 400, only_for_metrics=True))
        # self.fractal = Builder(combined_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, 0.75, 400, only_for_metrics=True))
        # self.fractal = Builder(combined_calc(ITER_COUNT, LIMIT_VALUE, MAX_DEPTH, 0.9, 400, only_for_metrics=True))

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
