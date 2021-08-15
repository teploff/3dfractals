from typing import Tuple, List, Dict
from ursina import Ursina, camera, window, Light, color, scene, Entity, held_keys, time, Mesh, EditorCamera, invoke
import math
from ursina import Mesh

MAX_DEPTH = 5
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

        self.fractal = one_phase_build(ITER_COUNT, LIMIT_VALUE)
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
                 p3: Point, p4: Point = None, h: float = None, A: float = None, B: float = None, C: float = None):
        self.depth = depth
        self.vertices = vertices
        self.triangles = triangles
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.h = h
        self.A = A
        self.B = B
        self.C = C


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


def make_basic_tetrahedron(coefficient: float, depth: int) -> [MaterialState]:
    """
    Формирование базового тетраэдра (который растет из точки)
    :param coefficient: Коэффициент представления фигуры
    :param depth: заданная глубина
    :return: Состояние фрактала на заданном коэфициенте
    """

    p1 = Point(0.0 * coefficient, 0.0 * coefficient, 0.0 * coefficient)
    p2 = Point(0.5 * coefficient, (math.sqrt(3) / 2.0) * coefficient, 0.0 * coefficient)
    p3 = Point(1.0 * coefficient, 0.0 * coefficient, 0.0 * coefficient)
    h = (math.sqrt(2.0 / 3.0)) * coefficient

    A, B, C, N, n = make_coef_surface(p1, p2, p3)
    p5, p6 = median_case(p1, p2, p3)

    p7 = find_p7_point(p1, p5)

    p4 = find_p4_point(A, B, C, N, h, p7)

    vertices = [[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z], [p3.x, p3.y, p3.z], [p4.x, p4.y, p4.z]]
    triangles = [[0, 1], [1, 2], [2, 0], [0, 3], [3, 1], [3, 2]]

    return [MaterialState(depth, vertices, triangles, p1=p1, p2=p2, p3=p3, p4=p4, h=h, A=A, B=B, C=C)]


def one_phase_build(iter_count: int, limit_value: float) -> Builder:
    """
    Формирования фрактальной структуры однофазным адгоритмом роста
    :param iter_count: Количество итераций, за которое необходимо вырастить каждый из компонентов фрактальной структуры
    :param limit_value: Предельено значние стороны, до которого необходимо осуществлять рост
    :return: Сформрованная фрактальная труктура по каждой из итераций
    """

    fractal = Builder()

    coefficient = limit_value / float(iter_count)

    for i in range(iter_count):
        c = coefficient + coefficient * i
        fractal.append_material(make_basic_tetrahedron(c, 0))

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
    for i in range(iter_count):
        materials = []

        c += ordinary_coefficient
        c2 = coefficient + coefficient * i

        materials.append(growth_triangle(p1=last_material.p1, p2=mp11, p3=mp31, h=h_new_1, n_prev=(-last_material.A, -last_material.B, -last_material.C), coefficient=c, depth=1))
        materials.append(growth_triangle(p1=mp11, p2=last_material.p2, p3=mp21, h=h_new_1,  n_prev=(-last_material.A, -last_material.B, -last_material.C), coefficient=c, depth=1))
        materials.append(growth_triangle(p1=mp21, p2=last_material.p3, p3=mp31, h=h_new_1,  n_prev=(-last_material.A, -last_material.B, -last_material.C), coefficient=c, depth=1))

        f_c = calc_centroid(
            Point(mp11.x * c, mp11.y * c, mp11.z * c),
            Point(mp21.x * c, mp21.y * c, mp21.z * c),
            Point(mp31.x * c, mp31.y * c, mp31.z * c))
        materials.append(cal_tetrahedron_1(mp11, mp21, mp31, h_new_1, coefficient=c2, depth=1, f_c=f_c))

        materials.append(growth_triangle(p1=last_material.p1, p2=mp12, p3=mp32, h=h_new_2,  n_prev=(last_material.A, last_material.B, last_material.C), coefficient=c, depth=1))
        materials.append(growth_triangle(p1=mp12, p2=last_material.p2, p3=mp22, h=h_new_2,  n_prev=(last_material.A, last_material.B, last_material.C), coefficient=c, depth=1))
        materials.append(growth_triangle(p1=mp22, p2=last_material.p4, p3=mp32, h=h_new_2,  n_prev=(last_material.A, last_material.B, last_material.C), coefficient=c, depth=1))

        f_c = calc_centroid(
            Point(mp12.x * c, mp12.y * c, mp12.z * c),
            Point(mp22.x * c, mp22.y * c, mp22.z * c),
            Point(mp32.x * c, mp32.y * c, mp32.z * c))
        materials.append(cal_tetrahedron(mp12, mp22, mp32, h_new_1, (last_material.A, last_material.B, last_material.C), coefficient=c2, depth=1, f_c=f_c))

        materials.append(growth_triangle(p1=last_material.p2, p2=mp13, p3=mp33, h=h_new_3,  n_prev=(last_material.A, last_material.B, last_material.C), coefficient=c, depth=1))
        materials.append(growth_triangle(p1=mp13, p2=last_material.p3, p3=mp23, h=h_new_3,  n_prev=(last_material.A, last_material.B, last_material.C), coefficient=c, depth=1))
        materials.append(growth_triangle(p1=mp23, p2=last_material.p4, p3=mp33, h=h_new_3,  n_prev=(last_material.A, last_material.B, last_material.C), coefficient=c, depth=1))

        f_c = calc_centroid(
            Point(mp13.x * c, mp13.y * c, mp13.z * c),
            Point(mp23.x * c, mp23.y * c, mp23.z * c),
            Point(mp33.x * c, mp33.y * c, mp33.z * c))
        materials.append(cal_tetrahedron(mp13, mp23, mp33, h_new_3, (last_material.A, last_material.B, last_material.C), coefficient=c2, depth=1, f_c=f_c))

        materials.append(growth_triangle(p1=last_material.p1, p2=mp14, p3=mp34, h=h_new_4,  n_prev=(last_material.A, last_material.B, last_material.C), coefficient=c, depth=1))
        materials.append(growth_triangle(p1=mp14, p2=last_material.p3, p3=mp24, h=h_new_4,  n_prev=(last_material.A, last_material.B, last_material.C), coefficient=c, depth=1))
        materials.append(growth_triangle(p1=mp24, p2=last_material.p4, p3=mp34, h=h_new_4,  n_prev=(last_material.A, last_material.B, last_material.C), coefficient=c, depth=1))

        f_c = calc_centroid(
            Point(mp14.x * c, mp14.y * c, mp14.z * c),
            Point(mp24.x * c, mp24.y * c, mp24.z * c),
            Point(mp34.x * c, mp34.y * c, mp34.z * c))
        materials.append(cal_tetrahedron(mp14, mp24, mp34, h_new_4, (last_material.A, last_material.B, last_material.C), coefficient=c2, depth=1, f_c=f_c))

        fractal.append_material(materials)

    # preparation
    edges = preparation_edges(fractal.materials[-1])

    current_depth = 1
    while MAX_DEPTH - current_depth != 0:
        c = 1
        for i in range(iter_count):
            materials = []

            c += ordinary_coefficient
            c2 = coefficient + coefficient * i

            for edgs in edges:
                mp1 = calc_midpoint(edgs["edges"][0], edgs["edges"][1])
                mp2 = calc_midpoint(edgs["edges"][1], edgs["edges"][2])
                mp3 = calc_midpoint(edgs["edges"][2], edgs["edges"][0])
                h_new = edgs["height"] * calc_distance(mp1, mp2) / calc_distance(edgs["edges"][0], edgs["edges"][1])

                materials.append(growth_triangle(p1=edgs["edges"][0], p2=mp1, p3=mp3, h=h_new, n_prev=edgs["normal"], coefficient=c, depth=1))
                materials.append(growth_triangle(p1=mp1, p2=edgs["edges"][1], p3=mp2, h=h_new, n_prev=edgs["normal"], coefficient=c, depth=1))
                materials.append(growth_triangle(p1=mp2, p2=edgs["edges"][2], p3=mp3, h=h_new, n_prev=edgs["normal"], coefficient=c, depth=1))

                f_c = calc_centroid(
                    Point(mp1.x * c, mp1.y * c, mp1.z * c),
                    Point(mp2.x * c, mp2.y * c, mp2.z * c),
                    Point(mp3.x * c, mp3.y * c, mp3.z * c))
                materials.append(
                    cal_tetrahedron(mp1, mp2, mp3, h_new, edgs["normal"], coefficient=c2, depth=1, f_c=f_c))
            fractal.append_material(materials)

        # preparation
        edges = preparation_edges(fractal.materials[-1])
        current_depth += 1

    return fractal


def preparation_edges(materials: List[MaterialState]) -> List[Dict]:
    edges = []

    for material in materials:
        if material.p4 == None:
            edges.append({
                "edges": [material.p1, material.p2, material.p3],
                "height": material.h,
                "normal": (material.A, material.B, material.C)
            })
        else:
            edges.append({
                "edges": [material.p1, material.p2, material.p4],
                "height": material.h,
                "normal": (material.A, material.B, material.C)
            })
            edges.append({
                "edges": [material.p1, material.p3, material.p4],
                "height": material.h,
                "normal": (material.A, material.B, material.C)
            })
            edges.append({
                "edges": [material.p2, material.p3, material.p4],
                "height": material.h,
                "normal": (material.A, material.B, material.C)
            })

    return edges


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


def cal_tetrahedron_1(p1: Point, p2: Point, p3: Point, h: float, coefficient: float, depth: int, f_c: Point) -> MaterialState:
    p11 = Point(p1.x * coefficient, p1.y * coefficient, p1.z * coefficient)
    p22 = Point(p2.x * coefficient, p2.y * coefficient, p2.z * coefficient)
    p33 = Point(p3.x * coefficient, p3.y * coefficient, p3.z * coefficient)
    h1 = h * coefficient

    s_c = calc_centroid(p11, p22, p33)

    dx = f_c.x - s_c.x
    dy = f_c.y - s_c.y
    dz = f_c.z - s_c.z

    p11.x += dx
    p22.x += dx
    p33.x += dx

    p11.y += dy
    p22.y += dy
    p33.y += dy

    p11.z += dz
    p22.z += dz
    p33.z += dz


    A, B, C, N, n = make_coef_surface(p11, p22, p33)

    A *= -1
    B *= -1
    C *= -1

    p5, p6 = median_case(p11, p22, p33)

    p7 = find_p7_point(p11, p5)

    p4 = find_p4_point(A, B, C, N, h1, p7)

    vertices = [[p11.x, p11.y, p11.z], [p22.x, p22.y, p22.z], [p33.x, p33.y, p33.z], [p4.x, p4.y, p4.z]]
    triangles = [[0, 1], [1, 2], [2, 0], [0, 3], [3, 1], [3, 2]]

    return MaterialState(depth, vertices, triangles, p1=p11, p2=p22, p3=p33, p4=p4, h=h1, A=A, B=B, C=C)


def cal_tetrahedron(p1: Point, p2: Point, p3: Point, h: float, n_prev: Tuple[float, float, float], coefficient: float, depth: int, f_c: Point) -> MaterialState:
    p11 = Point(p1.x * coefficient, p1.y * coefficient, p1.z * coefficient)
    p22 = Point(p2.x * coefficient, p2.y * coefficient, p2.z * coefficient)
    p33 = Point(p3.x * coefficient, p3.y * coefficient, p3.z * coefficient)
    h1 = h * coefficient


    s_c = calc_centroid(p11, p22, p33)

    dx = f_c.x - s_c.x
    dy = f_c.y - s_c.y
    dz = f_c.z - s_c.z

    p11.x += dx
    p22.x += dx
    p33.x += dx

    p11.y += dy
    p22.y += dy
    p33.y += dy

    p11.z += dz
    p22.z += dz
    p33.z += dz

    A, B, C, N, n = make_coef_surface(p11, p22, p33)

    if n_prev[0] * A + n_prev[1] * B + n_prev[2] * C < 0:
        A *= -1
        B *= -1
        C *= -1

    p5, p6 = median_case(p11, p22, p33)

    p7 = find_p7_point(p11, p5)

    p4 = find_p4_point(A, B, C, N, h1, p7)

    vertices = [[p11.x, p11.y, p11.z], [p22.x, p22.y, p22.z], [p33.x, p33.y, p33.z], [p4.x, p4.y, p4.z]]
    triangles = [[0, 1], [1, 2], [2, 0], [0, 3], [3, 1], [3, 2]]

    return MaterialState(depth, vertices, triangles, p1=p11, p2=p22, p3=p33, p4=p4, h=h1, A=A, B=B, C=C)


def growth_triangle(p1: Point, p2: Point, p3: Point, h: float, n_prev: Tuple[float, float, float], coefficient: float, depth: int):
    p11 = Point(p1.x * coefficient, p1.y * coefficient, p1.z * coefficient)
    p22 = Point(p2.x * coefficient, p2.y * coefficient, p2.z * coefficient)
    p33 = Point(p3.x * coefficient, p3.y * coefficient, p3.z * coefficient)
    h1 = h * coefficient

    vertices = [[p11.x, p11.y, p11.z], [p22.x, p22.y, p22.z], [p33.x, p33.y, p33.z]]
    triangles = [[0, 1], [1, 2], [2, 0]]

    return MaterialState(depth, vertices, triangles, p1=p11, p2=p22, p3=p33, h=h1, A=n_prev[0], B=n_prev[1], C=n_prev[2])


def calc_centroid(p1: Point, p2: Point, p3: Point) -> Point:
    return Point((p1.x + p2.x + p3.x) / 3.0, (p1.y + p2.y + p3.y) / 3.0, (p1.z + p2.z + p3.z) / 3.0)


if __name__ == '__main__':
    game = Game()
    update = game.update
    game.run()
