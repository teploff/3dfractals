from ursina import Ursina, window, Light, color, scene, Entity, held_keys, time, Mesh, EditorCamera
from typing import Tuple
import math


class Game(Ursina):
    def __init__(self):
        super().__init__()

        window.color = color.black
        window.fullscreen_size = 1920, 1080
        window.fullscreen = False

        Light(type='directional', color=(1, 1, 1, 1), direction=(0, 0, 1))
        Light(type='directional', color=(1, 1, 1, 1), direction=(0, 0, -1))
        # Light(type='ambient', color=(0.5, 0.5, 0.5, 1))

        vertices = [[0.0, 0.0, 0.0], [0.5, 0.8660254037844386, 0.0], [1.0, 0.0, 0.0],
                    [0.5, 0.28867513459481287, -0.816496580927726]]
        triangles = [0, 2, 3, 0, 3, 1, 2, 1, 3, 0, 2, 1]

        normal = (0.0, 0.0, -0.8660254037844386)
        #
        A1, B1, C1, _, _ = make_coef_surface(Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(0.5, 0.28867513459481287, -0.816496580927726))
        A2, B2, C2, _, _ = make_coef_surface(Point(0.0, 0.0, 0.0), Point(0.5, 0.28867513459481287, -0.816496580927726), Point(0.5, 0.8660254037844386, 0.0))
        A3, B3, C3, _, _ = make_coef_surface(Point(1.0, 0.0, 0.0), Point(0.5, 0.8660254037844386, 0.0), Point(0.5, 0.28867513459481287, -0.816496580927726))
        A4, B4, C4, _, _ = make_coef_surface(Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(0.5, 0.8660254037844386, 0.0))
        #
        # A4 *= -1
        # B4 *= -1
        # C4 *= -1
        #
        if A1 * normal[0] + B1 * normal[1] + C1 * normal[2] < 0:
            A1 *= -1
            B1 *= -1
            C1 *= -1
            triangles[0], triangles[1] = triangles[1], triangles[0]

        if A2 * normal[0] + B2 * normal[1] + C2 * normal[2] < 0:
            A2 *= -1
            B2 *= -1
            C2 *= -1
            triangles[3], triangles[4] = triangles[4], triangles[3]

        if A3 * normal[0] + B3 * normal[1] + C3 * normal[2] < 0:
            A3 *= -1
            B3 *= -1
            C3 *= -1
            triangles[6], triangles[7] = triangles[7], triangles[6]

        if A4 * normal[0] + B4 * normal[1] + C4 * normal[2] > 0:
            A4 *= -1
            B4 *= -1
            C4 *= -1
            triangles[6], triangles[7] = triangles[7], triangles[6]

        normals = [(A1, B1, C1), (A2, B2, C2), (A3, B3, C3), (A4, B4, C4)]

        self.surface = Entity(
            model=Mesh(vertices=vertices, triangles=triangles, mode='triangle', thickness=4, normals=normals), scale=2, color=color.yellow)

        EditorCamera()

    @staticmethod
    def new_game():
        scene.clear()

    def update(self):
        self.surface.rotation_y += held_keys['e'] * time.dt * 10
        self.surface.rotation_y -= held_keys['q'] * time.dt * 10

        self.surface.x += held_keys['d'] * time.dt * 5
        self.surface.x -= held_keys['a'] * time.dt * 5
        self.surface.y += held_keys['w'] * time.dt * 5
        self.surface.y -= held_keys['s'] * time.dt * 5

        self.surface.model.generate()

    def input(self, key):
        super().input(key)


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def make_coef_surface(p1: Point, p2: Point, p3: Point) -> (float, float, float, float, Tuple[float, float, float]):
    """
    Вычисление коэффициентов плоскости A, B, C, проходящую через три точки p1, p2 и p3, и вектора нормали N к этой
    плоскости.
    :param p1: Первая точка
    :param p2: Вторая точка
    :param p3: Третья точка
    :return: Кожфиициенты клоскости A, B и C и проходящий через нее вектор нормали N
    """

    A = (p2.y - p1.y) * (p3.z - p1.z) - (p3.y - p1.y) * (p2.z - p1.z)
    if A == -0.0:
        A = 0.0
    B = (p3.x - p1.x) * (p2.z - p1.z) - (p2.x - p1.x) * (p3.z - p1.z)
    if B == -0.0:
        B = 0.0
    C = (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)
    if C == -0.0:
        C = 0.0
    N = math.sqrt(math.pow(A, 2) + math.pow(B, 2) + math.pow(C, 2))
    n = (A/N, B/N, C/N)

    return A, B, C, N, n


if __name__ == '__main__':
    game = Game()
    update = game.update
    game.run()