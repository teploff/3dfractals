import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from entity import Point, Line, Face, Tetrahedron

MAX_DEPTH = 5
LIMIT_VALUE = 2.0

# ONE PHASE CONSTANTS
ITER_COUNT = 100


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


def find_tetrahedron_vertex(p1: Point, p2: Point, p3: Point, h: float) -> Point:
    a, b, c, n = make_coef_surface(p1, p2, p3)

    p5, p6 = median_case(p1, p2, p3)

    p7 = find_p7_point(p1, p5)

    p4 = find_p4_point(a, b, c, n, h, p7)

    return p4


def find_centroid(p1: Point, p2: Point, p3: Point, p4: Point) -> Point:
    """
    :param p1:
    :param p2:
    :param p3:
    :param p4:
    :return:
    """

    x_c = (1.0 / 4.0) * (p1.x + p2.x + p3.x + p4.x)
    y_c = (1.0 / 4.0) * (p1.y + p2.y + p3.y + p4.y)
    z_c = (1.0 / 4.0) * (p1.z + p2.z + p3.z + p4.z)

    return Point(x_c, y_c, z_c)


def find_step_growth(start_len: float, final_len: float, iter_count: int, p: Point, c_p: Point) -> (float, float, float):
    """

    :param start_len:
    :param final_len:
    :param iter_count:
    :param p:
    :param c_p:
    :return:
    """
    coefficient = ((final_len - start_len) / start_len)

    x = coefficient * (p.x - c_p.x) / float(iter_count)
    y = coefficient * (p.y - c_p.y) / float(iter_count)
    z = coefficient * (p.z - c_p.z) / float(iter_count)

    return x, y, z


def calculate(iter_count: int, limit_value: float):
    # Начальные точки тетраэдра и начальный коэфициент
    s_p1 = Point(0.0, 0.0, 0.0)
    s_p2 = Point(0.5, (math.sqrt(3) / 2.0), 0.0)
    s_p3 = Point(1.0, 0.0, 0.0)
    s_h = math.sqrt(2.0 / 3.0) * 1
    s_p4 = find_tetrahedron_vertex(s_p1, s_p2, s_p3, s_h)
    s_coefficient = 0.05

    # Значения предельной длины, погрешности и колечества цикла роста
    a = 2.0
    fault = 0.001
    circles_growth = 20

    # Начальные преобразования тетраэдра. Уменьшаем его четыре точки на коефициент s_coefficient
    s_p1 *= s_coefficient
    s_p2 *= s_coefficient
    s_p3 *= s_coefficient
    s_p4 *= s_coefficient

    # Высчитываем начальную длину, на основе которой будем вычислять шаг инкрементирования
    s_len = Line(s_p1, s_p2).length

    # Вычисляем центр тетраэдра и приращение для дальнейших вычилений роста
    s_p_c = find_centroid(s_p1, s_p2, s_p3, s_p4)
    delta_p1 = find_step_growth(s_len, a, circles_growth, s_p1, s_p_c)
    delta_p2 = find_step_growth(s_len, a, circles_growth, s_p2, s_p_c)
    delta_p3 = find_step_growth(s_len, a, circles_growth, s_p3, s_p_c)
    delta_p4 = find_step_growth(s_len, a, circles_growth, s_p4, s_p_c)

    # Заводим массив тетрэдров
    tetrahedrons = [Tetrahedron(s_p1, s_p2, s_p3, s_p4, None)]

    line_length = []
    square = []
    volume = []
    iterations = []
    global_i = 0
    # Если разница между текущей длиной отрезка тетраэдра и его пределом все еще больше погрешности - продолжаем
    # наращивать
    while abs(Line(s_p1, s_p2).length - a) > fault:
        # Собираем метрики
        # На первом этапе (росте одного лишь тетраэдра) - объем равен нулю. Так как начальный тетраэдр не считается
        # фракталом, а считается лишь его грань
        l = tetrahedrons[0]._face1.total_length
        s = tetrahedrons[0]._face1.square
        v = 0
        line_length.append(l)
        square.append(s)
        volume.append(v)
        ####

        s_p1 += delta_p1
        s_p2 += delta_p2
        s_p3 += delta_p3
        s_p4 += delta_p4

        global_i += 1
        iterations.append(global_i)

    #####
    l = tetrahedrons[0]._face1.total_length
    s = tetrahedrons[0]._face1.square
    v = 0
    line_length.append(l)
    square.append(s)
    volume.append(v)
    global_i += 1
    iterations.append(global_i)
    ####

    triangles = [Face(s_p1, s_p2, s_p3), Face(s_p1, s_p4, s_p2), Face(s_p1, s_p4, s_p3), Face(s_p2, s_p4, s_p3)]
    increments = [[delta_p1, delta_p2, delta_p3, delta_p4]]
    limits = [a * 2]

    current_depth = 0
    while MAX_DEPTH - current_depth != 0:
        # необхоимо обновить шаг роста, ведь финальная длина поменяется на x2
        for i, tetrahedron in enumerate(tetrahedrons):
            # Вычисляем центр тетраэдра и приращение для дальнейших вычилений роста
            s_p_c = find_centroid(tetrahedron.p1, tetrahedron.p2, tetrahedron.p3, tetrahedron.p4)
            s_len = Line(tetrahedron.p1, tetrahedron.p2).length

            # Пересчитываем дельты
            delta_p1 = find_step_growth(s_len, s_len * 2, circles_growth, tetrahedron.p1, s_p_c)
            delta_p2 = find_step_growth(s_len, s_len * 2, circles_growth, tetrahedron.p2, s_p_c)
            delta_p3 = find_step_growth(s_len, s_len * 2, circles_growth, tetrahedron.p3, s_p_c)
            delta_p4 = find_step_growth(s_len, s_len * 2, circles_growth, tetrahedron.p4, s_p_c)

            increments[i] = [delta_p1, delta_p2, delta_p3, delta_p4]

        new_triangles = []
        for triangle in triangles:
            # Находим серединные точки к прямым
            # Так же находим пропорциональную высоту
            mp1 = calc_midpoint(triangle.p1, triangle.p2)
            mp2 = calc_midpoint(triangle.p2, triangle.p3)
            mp3 = calc_midpoint(triangle.p1, triangle.p3)
            # Знаем что высота в тетраеэдре равна такой пропорции от стороны
            h = (math.sqrt(6.0) / 3) * calc_distance(mp1, mp2)
            p4 = find_tetrahedron_vertex(mp1, mp2, mp3, h)

            # Начальные преобразования тетраэдра
            mp1 *= s_coefficient
            mp2 *= s_coefficient
            mp3 *= s_coefficient
            p4 *= s_coefficient

            s_len = Line(mp1, mp2).length

            # Вычисляем центр тетраэдра и приращение для дальнейших вычилений роста
            s_p_c = find_centroid(mp1, mp2, mp3, p4)
            delta_p1 = find_step_growth(s_len, a, circles_growth, mp1, s_p_c)
            delta_p2 = find_step_growth(s_len, a, circles_growth, mp2, s_p_c)
            delta_p3 = find_step_growth(s_len, a, circles_growth, mp3, s_p_c)
            delta_p4 = find_step_growth(s_len, a, circles_growth, p4, s_p_c)

            tetrahedrons.append(Tetrahedron(mp1, mp2, mp3, p4, triangle))
            increments.append([delta_p1, delta_p2, delta_p3, delta_p4])
            limits.append(a)
            # Добавили треугольники, которые не лежат на тетраэдре
            new_triangles.append(Face(triangle.p1, mp1, mp3))
            new_triangles.append(Face(mp1, triangle.p2, mp2))
            new_triangles.append(Face(mp2, triangle.p3, mp3))
            # Добавляем треугольники, который на тетраэдре без основания
            new_triangles.append(Face(mp1, mp2, p4))
            new_triangles.append(Face(mp1, mp3, p4))
            new_triangles.append(Face(mp2, mp3, p4))

        triangles = new_triangles

        exist_no_growth = True
        while exist_no_growth:
            exist_no_growth = False

            for i, tetrahedron in enumerate(tetrahedrons):
                if abs(Line(tetrahedron.p1, tetrahedron.p2).length - limits[i]) > fault:
                    exist_no_growth = True

                    tetrahedron.p1 += increments[i][0]
                    tetrahedron.p2 += increments[i][1]
                    tetrahedron.p3 += increments[i][2]
                    tetrahedron.p4 += increments[i][3]

            # Если не сделать такую проверку задублирует метрику, потому что не найдет ни одного тетраедра, которого
            # нужно расти, а он уже вырос:)
            if exist_no_growth:
                #####
                l = 0
                s = 0
                v = 0
                for i, t in enumerate(tetrahedrons):
                    if i != 0:
                        v += t.volume
                    l += t.total_length
                    s += t.total_square

                # Так как мы строим фрактал с тетраэдра, а метрики мы должны собрнать с фрактала. Под фраткалом мы
                # понимаем одну из граней начального третраэда, котороя притерпела некоторое изменение. Т.е. метрика с
                # других граней начального тэтраэдра нас не интересует. Поэтому всю длину и площадь должны разделить на
                # четыре соотвественно, а объем оставить прежним, так как первый тетраэдр мы не учитываем
                line_length.append(l / 4.0)
                square.append(s / 4.0)
                volume.append(v)
                ####

                global_i += 1
                iterations.append(global_i)

        # Необходимо увеличить limit роста x2 для каждого из тетраэдров.
        for i, _ in enumerate(limits):
            limits[i] *= 2

        current_depth += 1

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
    #
    # setting label sizes after creation
    # ax1.xaxis.label.set_size(10)
    # ax1.yaxis.label.set_size(10)
    #
    # ax2.xaxis.label.set_size(10)
    # ax2.yaxis.label.set_size(10)
    #
    # ax3.xaxis.label.set_size(10)
    # ax3.yaxis.label.set_size(10)
    #
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.legend(fontsize=15)
    #
    fig1.savefig("length.png")
    fig2.savefig("square.png")
    fig3.savefig("value.png")

    plt.show()


if __name__ == '__main__':
    calculate(ITER_COUNT, LIMIT_VALUE)
