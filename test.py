from ursina import Ursina, window, Light, color, scene, Entity, held_keys, time, Mesh, EditorCamera
from typing import Dict, Tuple, List
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

        p1 = Point(0.0, 0.0, 0.0)
        p2 = Point(0.5, 0.8660254037844386, 0.0)
        p3 = Point(1.0, 0.0, 0.0)
        p4 = Point(0.5, 0.28867513459481287, -0.816496580927726)

        vertices = [[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z], [p3.x, p3.y, p3.z], [p4.x, p4.y, p4.z]]
        triangles = [0, 2, 3, 0, 3, 1, 2, 1, 3, 0, 2, 1]

        normal = (0.0, 0.0, -0.8660254037844386)

        A1, B1, C1, N1, _ = make_coef_surface(p1, p3, p4)
        A2, B2, C2, N2, _ = make_coef_surface(p1, p4, p2)
        A3, B3, C3, N3, _ = make_coef_surface(p3, p2, p4)
        A4, B4, C4, N4, _ = make_coef_surface(p1, p3, p2)

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

        # special case
        if A4 * normal[0] + B4 * normal[1] + C4 * normal[2] > 0:
            A4 *= -1
            B4 *= -1
            C4 *= -1
            triangles[9], triangles[10] = triangles[10], triangles[9]

        normals = [(A1, B1, C1), (A2, B2, C2), (A3, B3, C3), (A4, B4, C4)]

        mp1 = calc_midpoint(p1, p3)
        mp2 = calc_midpoint(p3, p4)
        mp3 = calc_midpoint(p4, p1)
        h_new = math.sqrt(2.0 / 3.0) * calc_distance(mp1, mp2) / calc_distance(p1, p3)
        p5, p6 = median_case(mp1, mp2, mp3)

        p7 = find_p7_point(p1, p5)

        vvv = find_p4_point(A1, B1, C1, N1, h_new, p7)

        # triangles = triangles[3:]
        vertices += [[mp1.x, mp1.y, mp1.z], [mp2.x, mp2.y, mp2.z], [mp3.x, mp3.y, mp3.z], [vvv.x, vvv.y, vvv.z]]

        A11, B11, C11, _, _ = make_coef_surface(mp1, mp2, vvv)
        A12, B12, C12, _, _ = make_coef_surface(mp1, mp3, vvv)
        A13, B13, C13, _, _ = make_coef_surface(mp2, mp3, vvv)
        A14, B14, C14, _, _ = make_coef_surface(p1, mp1, mp3)
        A15, B15, C15, _, _ = make_coef_surface(mp1, p3, mp2)
        A16, B16, C16, _, _ = make_coef_surface(mp2, p4, mp3)

        trian = []
        norman = []
        # triangle = [[mp1.x, mp1.y, mp1.z], [mp2.x, mp2.y, mp2.z], [vvv.x, vvv.y, vvv.z]]
        triangle = [4, 5, 7]
        if A11 * A1 + B11 * B1 + C11 * C1 < 0:
            A11 *= -1
            B11 *= -1
            C11 *= -1
            triangle[0], triangle[1] = triangle[1], triangle[0]
        trian += triangle
        norman += [(A11, B11, C11)]

        # triangle = [[mp1.x, mp1.y, mp1.z], [mp3.x, mp3.y, mp3.z], [vvv.x, vvv.y, vvv.z]]
        triangle = [4, 6, 7]
        if A12 * A1 + B12 * B1 + C12 * C1 < 0:
            A12 *= -1
            B12 *= -1
            C12 *= -1
        trian += triangle
        norman += [(A12, B12, C12)]

        # triangle = [[mp2.x, mp2.y, mp2.z], [mp3.x, mp3.y, mp3.z], [vvv.x, vvv.y, vvv.z]]
        triangle = [5, 6, 7]
        if A13 * A1 + B13 * B1 + C13 * C1 < 0:
            A13 *= -1
            B13 *= -1
            C13 *= -1
            triangle[0], triangle[1] = triangle[1], triangle[0]
        trian += triangle
        norman += [(A13, B13, C13)]

        # triangle = [[p1.x, p1.y, p1.z], [mp1.x, mp1.y, mp1.z], [mp3.x, mp3.y, mp3.z]]
        triangle = [0, 4, 6]
        if A14 * A1 + B14 * B1 + C14 * C1 < 0:
            A14 *= -1
            B14 *= -1
            C14 *= -1
            triangle[0], triangle[1] = triangle[1], triangle[0]
        trian += triangle
        norman += [(A14, B14, C14)]

        # triangle = [[mp1.x, mp1.y, mp1.z], [p3.x, p3.y, p3.z], [mp2.x, mp2.y, mp2.z]]
        triangle = [4, 2, 5]
        if A15 * A1 + B15 * B1 + C15 * C1 < 0:
            A15 *= -1
            B15 *= -1
            C15 *= -1
            triangle[0], triangle[1] = triangle[1], triangle[0]
        trian += triangle
        norman += [(A15, B15, C15)]

        # triangle = [[mp2.x, mp2.y, mp2.z], [p4.x, p4.y, p4.z], [mp3.x, mp3.y, mp3.z]]
        triangle = [5, 3, 6]
        if A16 * A1 + B16 * B1 + C16 * C1 < 0:
            A16 *= -1
            B16 *= -1
            C16 *= -1
            triangle[0], triangle[1] = triangle[1], triangle[0]
        trian += triangle
        norman += [(A16, B16, C16)]

        triangles = trian + triangles[3:]
        normals = norman + normals[1:]

        self.surface = Entity(
            model=Mesh(vertices=vertices, triangles=triangles, mode='triangle', thickness=4, normals=normals), scale=2, color=color.yellow)

        # mp1 = calc_midpoint(p1, p4)
        # mp2 = calc_midpoint(p4, p2)
        # mp3 = calc_midpoint(p2, p1)
        # h_new = math.sqrt(2.0 / 3.0) * calc_distance(mp1, mp2) / calc_distance(p1, p4)
        # cal_tetrahedron(mp1, mp2, mp3, h_new, A2, B2, C2, N2, self.surface, thickness=4, color=color.yellow)
        #
        # mp1 = calc_midpoint(p3, p2)
        # mp2 = calc_midpoint(p2, p4)
        # mp3 = calc_midpoint(p4, p3)
        # h_new = math.sqrt(2.0 / 3.0) * calc_distance(mp1, mp2) / calc_distance(p3, p2)
        # cal_tetrahedron(mp1, mp2, mp3, h_new, A3, B3, C3, N3, self.surface, thickness=4, color=color.yellow)
        #
        # mp1 = calc_midpoint(p1, p3)
        # mp2 = calc_midpoint(p3, p2)
        # mp3 = calc_midpoint(p2, p1)
        # h_new = math.sqrt(2.0 / 3.0) * calc_distance(mp1, mp2) / calc_distance(p1, p3)
        # cal_tetrahedron(mp1, mp2, mp3, h_new, A4, B4, C4, N4, self.surface, thickness=4, color=color.yellow)

        EditorCamera()

    @staticmethod
    def new_game():
        scene.clear()

    def update(self):
        self.surface.rotation_y += held_keys['e'] * time.dt * 100
        self.surface.rotation_y -= held_keys['q'] * time.dt * 100

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


def calc_midpoint(p1: Point, p2: Point) -> Point:
    """

    :param p1:
    :param p2:
    :return:
    """

    return Point((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0, (p1.z + p2.z) / 2.0)


def calc_distance(p1: Point, p2: Point) -> float:
    """

    :param p1:
    :param p2:
    :return:
    """

    return math.sqrt(math.pow(p2.x - p1.x, 2) + math.pow(p2.y - p1.y, 2) + math.pow(p2.z - p1.z, 2))


def median_case(p1: Point, p2: Point, p3: Point) -> (Point, Point):
    """

    :param p1:
    :param p2:
    :param p3:
    :return:
    """

    p5 = Point((p2.x + p3.x) / 2.0, (p2.y + p3.y) / 2.0, (p2.z + p3.z) / 2.0)
    p6 = Point((p1.x + p3.x) / 2.0, (p1.y + p3.y) / 2.0, (p1.z + p3.z) / 2.0)

    return p5, p6


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


def find_p7_point(p1: Point, p5: Point) -> Point:
    """

    :param p1:
    :param p2:
    :param p5:
    :param p6:
    :return:
    """
    x = p1.x + ((2 * (p5.x - p1.x)) / 3.0)
    y = p1.y + ((2 * (p5.y - p1.y)) / 3.0)
    z = p1.z + ((2 * (p5.z - p1.z)) / 3.0)

    return Point(x, y, z)


def find_p4_point(a: float, b: float, c: float, n: float, h: float, p7: Point) -> Point:
    """

    :param a:
    :param b:
    :param c:
    :param n:
    :param h:
    :param p7:
    :return:
    """

    x = p7.x + (a * h) / n
    y = p7.y + (b * h) / n
    z = p7.z + (c * h) / n

    return Point(x, y, z)


def cal_tetrahedron(p1: Point, p2: Point, p3: Point, h: float, A: float, B: float, C: float, N:float,
                    thickness:int, color: color) -> (List[float], List[float], List[float]):
    """

    :param p1:
    :param p2:
    :param p3:
    :param h:
    :param A:
    :param B:
    :param C:
    :param N:
    :param parent:
    :param thickness:
    :param color:
    :return:
    """
    p5, p6 = median_case(p1, p2, p3)

    p7 = find_p7_point(p1, p5)

    p4 = find_p4_point(A, B, C, N, h, p7)

    vertices = [[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z], [p3.x, p3.y, p3.z], [p4.x, p4.y, p4.z]]
    triangles = [0, 1, 3, 0, 2, 3, 1, 2, 3, 0, 1, 2]

    A1, B1, C1, _, _ = make_coef_surface(p1, p2, p4)
    A2, B2, C2, _, _ = make_coef_surface(p1, p3, p4)
    A3, B3, C3, _, _ = make_coef_surface(p2, p3, p4)
    # A4, B4, C4, _, _ = make_coef_surface(p1, p2, p3)

    if A1 * A + B1 * B + C1 * C < 0:
        A1 *= -1
        B1 *= -1
        C1 *= -1
        triangles[0], triangles[1] = triangles[1], triangles[0]

    if A2 * A + B2 * B + C2 * C < 0:
        A2 *= -1
        B2 *= -1
        C2 *= -1
        triangles[3], triangles[4] = triangles[4], triangles[3]

    if A3 * A + B3 * B + C3 * C < 0:
        A3 *= -1
        B3 *= -1
        C3 *= -1
        triangles[6], triangles[7] = triangles[7], triangles[6]

    # if A4 * A + B4 * B + C4 * C > 0:
    #     A4 *= -1
    #     B4 *= -1
    #     C4 *= -1
    #     triangles[9], triangles[10] = triangles[10], triangles[9]

    # normals = [(A1, B1, C1), (A2, B2, C2), (A3, B3, C3), (A4, B4, C4)]
    normals = [(A1, B1, C1), (A2, B2, C2), (A3, B3, C3)]

    return vertices, triangles, normals


if __name__ == '__main__':
    game = Game()
    update = game.update
    game.run()
