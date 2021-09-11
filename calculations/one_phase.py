import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from entity import Point, Line, Face, Tetrahedron


MAX_DEPTH = 4
LIMIT_VALUE = 2.0

# ONE PHASE CONSTANTS
ITER_COUNT = 100


class FractalState:
    def __init__(self):
        self.points = []
        self.lines = []
        self.triangles = []
        self.tetrahedron = []

    def append_points(self, point: Point):
        self.points.append(point)

    def append_lines(self, line: Line):
        for l in self.lines:
            # Если такая прямая уже существует в списке наших прямых, ничего не добавляем
            if (l.p1 == line.p1 and l.p2 == line.p2) or (l.p1 == line.p2 and l.p2 == line.p1):
                return
        self.lines.append(line)

    def append_triangles(self, face: Face):
        self.triangles.append(face)

    def append_tetrahedron(self, tetrahedron: Tetrahedron):
        self.tetrahedron.append(tetrahedron)

    @property
    def calc_line_length(self):
        length = 0

        for l in self.lines:
            length += l.length

        return length

    @property
    def calc_square(self):
        square = 0

        for t in self.triangles:
            square += t.square

        # особый случай, когда основание тетраэдра располагается на грани и для того, чтобы вычислить площадь фрактала,
        # необходимо от площади грани (родительской), где располается тетраэдр, вычесть площадь основания тетраэдра.
        # Т.к. у нас тэтраэдр и все площади тождественно равны. То досаточно найти всю площадь тетраэдра , поделить на 4
        # и вычесть из площади ролительской грани это значение.
        for tetra in self.tetrahedron:
            square += tetra.parent.square - tetra.total_square / 4.0

        return square

    @property
    def calc_volume(self):
        volume = 0

        for t in self.tetrahedron:
            volume += t.volume

        return volume


def growth_triangle(p1: Point, p2: Point, p3: Point, coefficient: float) -> (Point, Point, Point):
    new_p1 = Point(p1.x * coefficient, p1.y * coefficient, p1.z * coefficient)
    new_p2 = Point(p2.x * coefficient, p2.y * coefficient, p2.z * coefficient)
    new_p3 = Point(p3.x * coefficient, p3.y * coefficient, p3.z * coefficient)

    return new_p1, new_p2, new_p3


def make_coef_surface(p1: Point, p2: Point, p3: Point) -> (float, float, float, float):
    """
    Вычисление коэффициентов плоскости A, B, C, проходящую через три точки p1, p2 и p3, и вектора нормали N к этой
    плоскости.
    :param p1: Первая точка
    :param p2: Вторая точка
    :param p3: Третья точка
    :return: Коэффициенты клоскости A, B и C и проходящий через нее вектор нормали N
    """

    a = (p2.y - p1.y) * (p3.z - p1.z) - (p3.y - p1.y) * (p2.z - p1.z)
    b = (p3.x - p1.x) * (p2.z - p1.z) - (p2.x - p1.x) * (p3.z - p1.z)
    c = (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)

    n = math.sqrt(math.pow(a, 2) + math.pow(b, 2) + math.pow(c, 2))

    return a, b, c, n


def calc_distance(p1: Point, p2: Point) -> float:
    """

    :param p1:
    :param p2:
    :return:
    """

    return math.sqrt(math.pow(p2.x - p1.x, 2) + math.pow(p2.y - p1.y, 2) + math.pow(p2.z - p1.z, 2))

def calc_midpoint(p1: Point, p2: Point) -> Point:
    """

    :param p1:
    :param p2:
    :return:
    """

    return Point((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0, (p1.z + p2.z) / 2.0)


def calc_centroid(p1: Point, p2: Point, p3: Point) -> Point:
    return Point((p1.x + p2.x + p3.x) / 3.0, (p1.y + p2.y + p3.y) / 3.0, (p1.z + p2.z + p3.z) / 3.0)


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

def cal_tetrahedron(p1: Point, p2: Point, p3: Point, h: float, n_prev: Tuple[float, float, float], coefficient: float, f_c: Point) -> (Point, Point, Point, Point,Tuple[float, float, float]):
    p1 = Point(p1.x * coefficient, p1.y * coefficient, p1.z * coefficient)
    p2 = Point(p2.x * coefficient, p2.y * coefficient, p2.z * coefficient)
    p3 = Point(p3.x * coefficient, p3.y * coefficient, p3.z * coefficient)
    h *= coefficient

    s_c = calc_centroid(p1, p2, p3)

    dx = f_c.x - s_c.x
    dy = f_c.y - s_c.y
    dz = f_c.z - s_c.z

    p1.x += dx
    p2.x += dx
    p3.x += dx

    p1.y += dy
    p2.y += dy
    p3.y += dy

    p1.z += dz
    p2.z += dz
    p3.z += dz

    A, B, C, N = make_coef_surface(p1, p2, p3)

    if n_prev[0] * A + n_prev[1] * B + n_prev[2] * C < 0:
        A *= -1
        B *= -1
        C *= -1

    p5, p6 = median_case(p1, p2, p3)

    p7 = find_p7_point(p1, p5)

    p4 = find_p4_point(A, B, C, N, h, p7)

    return p1, p2, p3, p4, (A, B, C)


def calculate(iter_count: int, limit_value: float):
    # Начальные точки и высота
    s_p1 = Point(0.0, 0.0, 0.0)
    s_p2 = Point(0.5, (math.sqrt(3) / 2.0), 0.0)
    s_p3 = Point(1.0, 0.0, 0.0)

    # Состояние фрактала на каждой его из итераций
    fractal_states = []

    # Определяем коэффициент
    coefficient = limit_value / float(iter_count)

    # Растем треугольник
    for i in range(iter_count):
        fractal_state = FractalState()
        c = coefficient + coefficient * i

        # растим точки и высоту для будущего тетраедра, чтоб сохряналась пропорция.
        # Возможно иммет смысл высоты вычислять отдельно
        p1, p2, p3 = growth_triangle(s_p1, s_p2, s_p3, c)

        # Добавляем точки
        points = [p1, p2, p3]
        for p in points:
            fractal_state.append_points(p)

        # Добаляем прямые
        lines = [Line(p1, p2), Line(p2, p3), Line(p1, p3)]
        for l in lines:
            fractal_state.append_lines(l)

        # Добавляем треугольник
        fractal_state.append_triangles(Face(p1, p2, p3))

        # Добавляем текущей состояние в список всех состояний на конкретной итерации
        fractal_states.append(fractal_state)

    # Берем координаты выросшего треугольника треуголника
    p1, p2, p3 = fractal_states[-1].points

    # Вычисляем нормаль к плоскости [p1, p2, p3]
    a, b, c, _ = make_coef_surface(p1, p2, p3)

    active_edges = [{
        "edges": [p1, p2, p3],
        "normal": (a, b, c)
    }]

    coefficient_for_triangle = calc_distance(p1, calc_midpoint(p1, p2)) / float(iter_count)

    current_depth = 0
    prev_tetrahedron = []
    while MAX_DEPTH - current_depth != 0:
        # Коэфициент роста не для тетраедра, а для трегуольника, образовашегося путем дроблении грани, на 4 треугольника
        c1 = 1

        temp_active_edges = []
        for i in range(iter_count):
            fractal_state = FractalState()

            c1 += coefficient_for_triangle
            c2 = coefficient + coefficient * i

            temp_active_edges = []

            for edgs in active_edges:
                # Находим серединные точки для каждой из прямых [p1; p2], [p2; p3] и [p3; p1]
                # Так же находим пропорциональную высоту
                mp1 = calc_midpoint(edgs["edges"][0], edgs["edges"][1])
                mp2 = calc_midpoint(edgs["edges"][1], edgs["edges"][2])
                mp3 = calc_midpoint(edgs["edges"][2], edgs["edges"][0])
                # Знаем что высота в тетраеэдре равна такой пропорции от стороны
                h = (math.sqrt(6.0)/3) * calc_distance(mp1, mp2)

                # TODO: занимаемся треугольниками
                # Растим точки и высоту левого треугольника [p1, mp1, mp3].
                # ------------------------------------------------------------------------------------------------------
                p1, p2, p3 = growth_triangle(edgs["edges"][0], mp1, mp3, c1)

                # Добавляем точки
                points = [p1, p2, p3]
                for p in points:
                    fractal_state.append_points(p)

                # Добаляем прямые. Добавляем лишь к прямым реальные, невоображаемые. Так как в общей длине фрактала,
                # длина [p2, p3] учитываться не должна
                lines = [Line(p1, p2), Line(p1, p3)]
                for l in lines:
                    fractal_state.append_lines(l)

                # Добавляем треугольник
                fractal_state.append_triangles(Face(p1, p2, p3))

                # Добавляем в текущий список активных
                temp_active_edges.append({
                    "edges": [p1, p2, p3],
                    "normal": edgs["normal"]
                })

                # Растим точки и высоту верхнего треугольника [mp1, p2, mp2].
                # ------------------------------------------------------------------------------------------------------
                p1, p2, p3 = growth_triangle(mp1, edgs["edges"][1], mp2, c1)

                # Добавляем точки
                points = [p1, p2, p3]
                for p in points:
                    fractal_state.append_points(p)

                # Добаляем прямые. Добавляем лишь к прямым реальные, невоображаемые. Так как в общей длине фрактала,
                # длина [p2, p3] учитываться не должна
                lines = [Line(p1, p2), Line(p1, p3)]
                for l in lines:
                    fractal_state.append_lines(l)

                # Добавляем треугольник
                fractal_state.append_triangles(Face(p1, p2, p3))

                # Добавляем в текущий список активных
                temp_active_edges.append({
                    "edges": [p1, p2, p3],
                    "normal": edgs["normal"]
                })

                # Растим точки и высоту правого треугольника [mp2, p3, mp3].
                # ------------------------------------------------------------------------------------------------------
                p1, p2, p3 = growth_triangle(mp2, edgs["edges"][2], mp3, c1)

                # Добавляем точки
                points = [p1, p2, p3]
                for p in points:
                    fractal_state.append_points(p)

                # Добаляем прямые. Добавляем лишь к прямым реальные, невоображаемые. Так как в общей длине фрактала,
                # длина [p2, p3] учитываться не должна
                lines = [Line(p1, p2), Line(p1, p3)]
                for l in lines:
                    fractal_state.append_lines(l)

                # Добавляем треугольник
                fractal_state.append_triangles(Face(p1, p2, p3))

                # Добавляем в текущий список активных
                temp_active_edges.append({
                    "edges": [p1, p2, p3],
                    "normal": edgs["normal"]
                })

                # TODO: занимаемся тетраэдром
                # ------------------------------------------------------------------------------------------------------
                # Вычисляем серидинную точку
                f_c = calc_centroid(
                    Point(mp1.x * c2, mp1.y * c2, mp1.z * c2),
                    Point(mp2.x * c2, mp2.y * c2, mp2.z * c2),
                    Point(mp3.x * c2, mp3.y * c2, mp3.z * c2))
                p1, p2, p3, p4, coef_surface = cal_tetrahedron(mp1, mp2, mp3, h, edgs["normal"], coefficient=c2, f_c=f_c)

                # Добавляем точки
                points = [p1, p2, p3, p4]
                for p in points:
                    fractal_state.append_points(p)

                # Добаляем прямые
                lines = [Line(p1, p2), Line(p2, p3), Line(p1, p3), Line(p1, p4), Line(p2, p4), Line(p3, p4)]
                for l in lines:
                    fractal_state.append_lines(l)

                # Добавляем треугольники
                fractal_state.append_triangles(Face(p1, p4, p2))
                fractal_state.append_triangles(Face(p1, p4, p3))
                fractal_state.append_triangles(Face(p2, p4, p3))

                # Добавляем тетраэдр
                fractal_state.append_tetrahedron(Tetrahedron(p1, p2, p3, p4, parent=Face(mp1 * c1, mp2 * c1, mp3 * c1)))

                # Добавляем в текущий список активных
                temp_active_edges.append({
                    "edges": [p1, p4, p2],
                    "normal": coef_surface
                })

                temp_active_edges.append({
                    "edges": [p1, p4, p3],
                    "normal": coef_surface
                })

                temp_active_edges.append({
                    "edges": [p2, p4, p3],
                    "normal": coef_surface
                })

            # Так же добавляекм предыдущие, чтобы учесть предыдущий объем объем и его не потерять
            for tetrahedron in prev_tetrahedron:
                fractal_state.append_tetrahedron(tetrahedron)

            fractal_states.append(fractal_state)

        # тут необходимо подготовить данные для следующей итерации глубины фрактала, а также инкрементировать пройденную
        # глубину
        active_edges = temp_active_edges
        for tetra in fractal_states[-1].tetrahedron:
            prev_tetrahedron.append(tetra)
        current_depth += 1

    iterations = []
    line_length = []
    square = []
    volume = []

    for iteration, state in enumerate(fractal_states):
        iterations.append(iteration + 1)
        line_length.append(state.calc_line_length)
        square.append(state.calc_square)
        volume.append(state.calc_volume)
        print(f'state = {iteration}. '
              f'Total line length = {state.calc_line_length}. '
              f'Total square = {state.calc_square}. '
              f'Total volume = {state.calc_volume}')

    fig1, ax1 = plt.subplots()
    ax1.plot(iterations, line_length, 'o-', label=r'$a$', c='black', linewidth=3)
    fig2, ax2 = plt.subplots()
    ax2.plot(iterations, square, 'X-', label=r'$a$', c='black', linewidth=3)
    fig3, ax3 = plt.subplots()
    ax3.plot(iterations, volume, '*-', label=r'$a$', c='black', linewidth=3)

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    ax1.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax1.set(xlabel='Число циклов роста, ед.', ylabel='Длина фрактальной линии, ед.')

    ax2.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax2.set(xlabel='Число циклов роста, ед.', ylabel='Площадь фрактала, ед.')

    ax3.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax3.set(xlabel='Число циклов роста, ед.', ylabel='Объем фрактала, ед.')

    # setting label sizes after creation
    ax1.xaxis.label.set_size(10)
    ax1.yaxis.label.set_size(10)

    ax2.xaxis.label.set_size(10)
    ax2.yaxis.label.set_size(10)

    ax3.xaxis.label.set_size(10)
    ax3.yaxis.label.set_size(10)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)

    fig1.savefig("length.png")
    fig2.savefig("square.png")
    fig3.savefig("value.png")

    plt.show()


if __name__ == '__main__':
    calculate(ITER_COUNT, LIMIT_VALUE)
