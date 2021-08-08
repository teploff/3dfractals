from typing import Tuple, List, Dict
from ursina import Ursina, camera, window, Light, color, scene, Entity, held_keys, time, Mesh, EditorCamera, invoke
import math
from ursina import Mesh

MAX_DEPTH = 1
ITER_COUNT = 10
LIMIT_VALUE = 2.0
ACCURACY = 0.001


class Game(Ursina):
    def __init__(self):
        super().__init__()

        window.color = color.black
        window.fullscreen_size = 1920, 1080
        window.fullscreen = False

        Light(type='ambient', color=(0.5, 0.5, 0.5, 1))
        Light(type='directional', color=(0.5, 0.5, 0.5, 1), direction=(1, 1, 1))

        self.fractal = build(ITER_COUNT, LIMIT_VALUE)
        self.state = -1

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
            if self.state == (len(self.fractal.materials) - 1):
                return
            self.state += 1

            scene.clear()
            self.fractal.gen(self.state).model.generate()

        super().input(key)


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Line:
    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2

    def length(self):
        return math.sqrt((self.p2.x - self.p1.x) ** 2 + (self.p2.y - self.p1.y) ** 2 + (self.p2.z - self.p1.z) ** 2)


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
    B = (p3.x - p1.x) * (p2.z - p1.z) - (p2.x - p1.x) * (p3.z - p1.z)
    C = (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)
    N = math.sqrt(math.pow(A, 2) + math.pow(B, 2) + math.pow(C, 2))
    n = (A / N, B / N, C / N)

    return A, B, C, N, n


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


class MaterialState:
    def __init__(self, depth: int, vertices: List[List[float]], triangles: List[List[int]], p1: Point, p2: Point,
                 p3: Point, p4: Point = None, h: float = None):
        self.depth = depth
        self.vertices = vertices
        self.triangles = triangles
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.h = h


class Builder:
    def __init__(self):
        self.materials = []

    def append_material(self, material: List[MaterialState]):
        self.materials.append(material)

    def gen(self, item: int):
        print(f'state = {item}, capacity = {len(self.materials)}')
        if item >= len(self.materials):
            raise KeyError

        material = self.materials[item][0]
        parent_entity = Entity(model=Mesh(vertices=material.vertices, triangles=material.triangles, thickness=4,
                                          mode='line'), scale=2, color=color.yellow)

        if len(self.materials[item]) == 1:
            return parent_entity

        for material in self.materials[item][1:]:
            Entity(model=Mesh(vertices=material.vertices, triangles=material.triangles,
                                                    thickness=4, mode='line'), scale=2, color=color.yellow)

        return parent_entity


def lol(koef: float, depth: int) -> [MaterialState]:
    p1 = Point(0.0 * koef, 0.0 * koef, 0.0 * koef)
    p2 = Point(0.5 * koef, (math.sqrt(3) / 2.0) * koef, 0.0 * koef)
    p3 = Point(1.0 * koef, 0.0 * koef, 0.0 * koef)
    h = (math.sqrt(2.0 / 3.0)) * koef

    print(Line(p1, p2).length())

    A, B, C, N, n = make_coef_surface(p1, p2, p3)
    p5, p6 = median_case(p1, p2, p3)

    p7 = find_p7_point(p1, p5)

    p4 = find_p4_point(A, B, C, N, h, p7)

    vertices = [[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z], [p3.x, p3.y, p3.z], [p4.x, p4.y, p4.z]]
    triangles = [[0, 1], [1, 2], [2, 0], [0, 3], [3, 1], [3, 2]]

    return [MaterialState(depth, vertices, triangles, p1=p1, p2=p2, p3=p3, p4=p4, h=h)]


def build(iter_count: int, limit_value: float) -> Builder:
    fractal = Builder()

    coefficient = limit_value / float(iter_count)

    for i in range(iter_count):
        c = coefficient + coefficient * i
        fractal.append_material(lol(c, 0))

    last_material = fractal.materials[-1][0]
    mp11 = calc_midpoint(last_material.p1, last_material.p2)
    mp21 = calc_midpoint(last_material.p2, last_material.p3)
    mp31 = calc_midpoint(last_material.p3, last_material.p1)
    h_new_1 = last_material.h * calc_distance(mp11, mp21) / calc_distance(last_material.p1, last_material.p2)

    mp12 = calc_midpoint(last_material.p1, last_material.p2)
    mp22 = calc_midpoint(last_material.p2, last_material.p4)
    mp32 = calc_midpoint(last_material.p4, last_material.p1)
    h_new_2 = last_material.h * calc_distance(mp12, mp22) / calc_distance(last_material.p1, last_material.p2)

    mp13 = calc_midpoint(last_material.p2, last_material.p3)
    mp23 = calc_midpoint(last_material.p3, last_material.p4)
    mp33 = calc_midpoint(last_material.p4, last_material.p2)
    h_new_3 = last_material.h * calc_distance(mp13, mp23) / calc_distance(last_material.p2, last_material.p3)

    mp14 = calc_midpoint(last_material.p1, last_material.p3)
    mp24 = calc_midpoint(last_material.p3, last_material.p4)
    mp34 = calc_midpoint(last_material.p4, last_material.p1)
    h_new_4 = last_material.h * calc_distance(mp14, mp24) / calc_distance(last_material.p1, last_material.p3)

    # Коэфициент роста не для тетраедра, а для трегуольника, образовашегося путем дроблении грани, на 4 треугольника
    ordinary_coefficient = calc_distance(last_material.p1, mp11) / float(iter_count)

    c = 1
    for _ in range(iter_count):
        materials = []

        c += ordinary_coefficient

        materials.append(growth_triangle(p1=last_material.p1, p2=mp11, p3=mp31, coefficient=c, depth=1))
        materials.append(growth_triangle(p1=mp11, p2=last_material.p2, p3=mp21, coefficient=c, depth=1))
        materials.append(growth_triangle(p1=mp21, p2=last_material.p3, p3=mp31, coefficient=c, depth=1))

        materials.append(growth_triangle(p1=last_material.p1, p2=mp12, p3=mp32, coefficient=c, depth=1))
        materials.append(growth_triangle(p1=mp12, p2=last_material.p2, p3=mp22, coefficient=c, depth=1))
        materials.append(growth_triangle(p1=mp22, p2=last_material.p4, p3=mp32, coefficient=c, depth=1))

        materials.append(growth_triangle(p1=last_material.p2, p2=mp13, p3=mp33, coefficient=c, depth=1))
        materials.append(growth_triangle(p1=mp13, p2=last_material.p3, p3=mp23, coefficient=c, depth=1))
        materials.append(growth_triangle(p1=mp23, p2=last_material.p4, p3=mp33, coefficient=c, depth=1))

        materials.append(growth_triangle(p1=last_material.p1, p2=mp14, p3=mp34, coefficient=c, depth=1))
        materials.append(growth_triangle(p1=mp14, p2=last_material.p3, p3=mp24, coefficient=c, depth=1))
        materials.append(growth_triangle(p1=mp24, p2=last_material.p4, p3=mp34, coefficient=c, depth=1))

        fractal.append_material(materials)

    # edges.append(cal_tetrahedron_1(mp1, mp2, mp3, h_new, (A, B, C), self.surface, thickness=4, color=color.azure))
    # edges.append({"edges": [[p1, mp1, mp3], [mp1, p2, mp2], [mp2, p3, mp3]], "normal": (-A, -B, -C), "height": h_new})

    # edges.append(cal_tetrahedron(mp1, mp2, mp3, h_new, (A, B, C), self.surface, thickness=4, color=color.azure))
    # edges.append({"edges": [[p1, mp1, mp3], [mp1, p2, mp2], [mp2, p4, mp3]], "normal": (A, B, C), "height": h_new})

    # edges.append(cal_tetrahedron(mp1, mp2, mp3, h_new, (A, B, C), self.surface, thickness=4, color=color.azure))
    # edges.append({"edges": [[p2, mp1, mp3], [mp1, p3, mp2], [mp2, p4, mp3]], "normal": (A, B, C), "height": h_new})

    # edges.append(cal_tetrahedron(mp1, mp2, mp3, h_new, (A, B, C), self.surface, thickness=4, color=color.azure))
    # edges.append({"edges": [[p1, mp1, mp3], [mp1, p3, mp2], [mp2, p4, mp3]], "normal": (A, B, C), "height": h_new})

    return fractal


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


def cal_tetrahedron_1(p1: Point, p2: Point, p3: Point, h: float, n_prev: Tuple[float, float, float], parent: Entity,
                      thickness: int, color: color) -> Dict:
    """

    :param p1:
    :param p2:
    :param p3:
    :param h:
    :param n_prev:
    :param parent:
    :return:
    """

    A, B, C, N, n = make_coef_surface(p1, p2, p3)

    A *= -1
    B *= -1
    C *= -1

    p5, p6 = median_case(p1, p2, p3)

    p7 = find_p7_point(p1, p5)

    p4 = find_p4_point(A, B, C, N, h, p7)

    vertiti = [[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z], [p3.x, p3.y, p3.z], [p4.x, p4.y, p4.z]]
    trititi = [[0, 1, 2, 0], [0, 1, 3, 0], [0, 2, 3, 0], [1, 2, 3, 1]]

    Entity(parent=parent, model=Mesh(vertices=vertiti, triangles=trititi, mode='line', thickness=thickness),
           color=color)

    return {"edges": [[p1, p2, p4], [p1, p3, p4], [p2, p3, p4]], "normal": (A, B, C), "height": h}


def growth_triangle(p1: Point, p2: Point, p3: Point, coefficient: float, depth: int):
    p11 = Point(p1.x * coefficient, p1.y * coefficient, p1.z * coefficient)
    p22 = Point(p2.x * coefficient, p2.y * coefficient, p2.z * coefficient)
    p33 = Point(p3.x * coefficient, p3.y * coefficient, p3.z * coefficient)

    print(Line(p11, p22).length())

    vertices = [[p11.x, p11.y, p11.z], [p22.x, p22.y, p22.z], [p33.x, p33.y, p33.z]]
    triangles = [[0, 1], [1, 2], [2, 0]]

    return MaterialState(depth, vertices, triangles, p1=p11, p2=p22, p3=p33)


if __name__ == '__main__':
    game = Game()
    update = game.update
    game.run()
