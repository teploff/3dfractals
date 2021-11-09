import math
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from entity import Point, Line, Face, Tetrahedron
from visualization.entity import Model


def make_interpolation(x: List[int], y: List[float]) -> np.ndarray:
    """
    Формирование интерполяции на существующем кортеже точек
    :param x:
    :param y:
    :return:
    """
    x_train = np.array(x)
    y_train = np.array(y)

    [a_y_x, b_y_x], _ = curve_fit(lambda x1, a, b: a * np.exp(b * x1), x_train, y_train, p0=[0.01285, 0.0351])

    return a_y_x * np.exp(b_y_x * x_train)


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


def find_tetrahedron_vertex(p1: Point, p2: Point, p3: Point, h: float, parent_surface_k=None, is_special_case=False) -> (Point, Tuple[float, float, float]):
    a, b, c, n = make_coef_surface(p1, p2, p3)

    if parent_surface_k is not None:
        if is_special_case:
            a *= -1
            b *= -1
            c *= -1
        else:
            if parent_surface_k[0] * a + parent_surface_k[1] * b + parent_surface_k[2] * c < 0:
                a *= -1
                b *= -1
                c *= -1

    p5, p6 = median_case(p1, p2, p3)

    p7 = find_p7_point(p1, p5)

    p4 = find_p4_point(a, b, c, n, h, p7)

    return p4, (a, b, c)


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


def undergrown_tetrahedron_exists(depths: dict) -> bool:
    """
    Проверяем есть ли еще хотя бы один тетраэдр, который не вырос?
    :param depths:
    :return:
    """
    for tetra_id in depths["maximum"]:
        if depths["maximum"][tetra_id] != depths["current"][tetra_id]:
            return True

    return False


def calculate(iter_count: int, limit_value: float, depth: int) -> List[List[Model]]:
    """
    Вычисление однофазной модели
    :param iter_count: количество итераций роста
    :param limit_value: предальное значение отрезка
    :param depth: глубина фраткальной структуры
    :return:
    """
    # Начальные точки тетраэдра, вектор нормали (с коэффициентами A, B и C), и начальный коэфициент
    # для уменьшения фигуры
    s_p1 = Point(0.0, 0.0, 0.0)
    s_p2 = Point(0.5, (math.sqrt(3) / 2.0), 0.0)
    s_p3 = Point(1.0, 0.0, 0.0)
    s_h = math.sqrt(2.0 / 3.0) * 1
    s_p4, surface_k = find_tetrahedron_vertex(s_p1, s_p2, s_p3, s_h)
    s_coefficient = 0.05

    # Значения погрешности, которое будем сопоставлять при достижении отрезка нужной длины (а)
    fault = 0.001

    # Начальные преобразования тетраэдра. Уменьшаем его четыре точки на коефициент s_coefficient
    s_p1 *= s_coefficient
    s_p2 *= s_coefficient
    s_p3 *= s_coefficient
    s_p4 *= s_coefficient

    # Высчитываем начальную длину, на основе которой будем вычислять шаг инкрементирования
    s_len = Line(s_p1, s_p2).length

    # Вычисляем центр тетраэдра и приращение для дальнейших вычилений роста
    s_p_c = find_centroid(s_p1, s_p2, s_p3, s_p4)
    delta_p1 = find_step_growth(s_len, limit_value, iter_count, s_p1, s_p_c)
    delta_p2 = find_step_growth(s_len, limit_value, iter_count, s_p2, s_p_c)
    delta_p3 = find_step_growth(s_len, limit_value, iter_count, s_p3, s_p_c)
    delta_p4 = find_step_growth(s_len, limit_value, iter_count, s_p4, s_p_c)

    # Заводим массив тетрэдров и добавляем в него бозовый. В базовом указываем точки, коэфициенты и отсуствие
    # родительской грани
    tetrahedron = Tetrahedron(s_p1, s_p2, s_p3, s_p4, surface_k, None)
    tetrahedrons = [tetrahedron]

    # Массивы для подсчета метрик: длины, площади и объема. Так же массив итераций роста.
    iterations = []
    line_length = []
    square = []
    volume = []

    # Итерация роста полной фигуры. Необходима в будущем для визуализации величин длины, площади и объема фрактала
    global_i = 0
    # Представление фигуры на заданной итерации роста. Необходимо для отображения движком ursina.
    ursina_models = []

    # Если разница между текущей длиной отрезка тетраэдра и его пределом все еще больше погрешности - продолжаем
    # наращивать
    while abs(Line(s_p1, s_p2).length - limit_value) > fault:
        # Собираем метрики
        # На первом этапе (росте одного лишь тетраэдра) - объем равен нулю. Так как начальный тетраэдр не считается
        # фракталом, а считается лишь его грань
        l = tetrahedron._face1.total_length
        s = tetrahedron._face1.square
        v = 0
        line_length.append(l)
        square.append(s)
        volume.append(v)
        ####

        # Собираем примитивы для дальнейшей визуализации движком ursina.
        v1 = [[s_p1.x, s_p1.y, s_p1.z], [s_p2.x, s_p2.y, s_p2.z], [s_p3.x, s_p3.y, s_p3.z], [s_p4.x, s_p4.y, s_p4.z]]
        t1 = [[0, 1], [1, 2], [2, 0], [0, 3], [3, 1], [3, 2]]
        ursina_models.append([Model(vertices=v1, triangles=t1)])

        # Инкрементируем каждую точку тетраэдра на величину, вычисленную до
        s_p1 += delta_p1
        s_p2 += delta_p2
        s_p3 += delta_p3
        s_p4 += delta_p4

        global_i += 1
        iterations.append(global_i)

    #####
    l = tetrahedron._face1.total_length
    s = tetrahedron._face1.square
    v = 0
    line_length.append(l)
    square.append(s)
    volume.append(v)
    global_i += 1
    iterations.append(global_i)
    ####

    # После того, как вырастили родительский тетраэдр. Формируем массив активных треугольников, на которых будем растить
    # последующие тетраэдры. Одному треугольнику даем метку, чтоб отследить тетраэдры, которые учасвтсввуют в
    # вычислении метрик общей фигуры: длины, площади и объема.
    triangles = [
        Face(s_p1, s_p2, s_p3, depth, tetrahedron, True, True, limit=limit_value),
        Face(s_p1, s_p4, s_p2, depth, tetrahedron, limit=limit_value),
        Face(s_p1, s_p4, s_p3, depth, tetrahedron, limit=limit_value),
        Face(s_p2, s_p4, s_p3, depth, tetrahedron, limit=limit_value),
    ]
    # Необходимо обновить инкрименты к базовому тетраэдру
    # Вычисляем центр тетраэдра и приращение для дальнейших вычилений роста
    s_p_c = find_centroid(tetrahedron.p1, tetrahedron.p2, tetrahedron.p3, tetrahedron.p4)
    s_len = Line(tetrahedron.p1, tetrahedron.p2).length

    # Задаем рандомное колиечество итерация роста для тетраэдра
    iters = iter_count + random.randint(1, iter_count)

    # Пересчитываем дельты
    limit = limit_value + random.uniform(0.25, 0.45)
    delta_p1 = find_step_growth(s_len, limit, iters, tetrahedron.p1, s_p_c)
    delta_p2 = find_step_growth(s_len, limit, iters, tetrahedron.p2, s_p_c)
    delta_p3 = find_step_growth(s_len, limit, iters, tetrahedron.p3, s_p_c)
    delta_p4 = find_step_growth(s_len, limit, iters, tetrahedron.p4, s_p_c)

    # Объявляем массив инкрементов для каждого из тетраэдра. В данном случае для базового тетраэдра
    # Объявляем массим пределов, до какого предела растить тетраэдр. Для базового тетраэдра необхоимо обновить шаг
    # роста, ведь финальная длина поменяется на x2
    tetrahedron_info = {
        "increments": {
            tetrahedron.id: [delta_p1, delta_p2, delta_p3, delta_p4]
        },
        "limits": {
            tetrahedron.id: limit
        },
        "iterations_count": {
            tetrahedron.id: iters
        },
        "depths": {
            "current": {
                tetrahedron.id: 0
            },
            "maximum": {
                tetrahedron.id: depth
            }
        },
        "new": {}
    }

    # Записываем вычисленные инкрименты, относящиеся к конкретному тетраэдру, в список инкрементов
    tetrahedron_info["increments"][tetrahedron.id] = [delta_p1, delta_p2, delta_p3, delta_p4]

    # Заведем словарь средних точек для того, чтобы после каждого роста фигур, пересчитать их основывася
    recalc_middle_points = {}

    # Выращиваем до тех пор пока существует хоть один тетраэдр, который не дорос до задонной глубины.
    while undergrown_tetrahedron_exists(tetrahedron_info["depths"]):
        # Необходимо прорядить список активных треугольников и добавить к ним новопоступившиеся и образовать новые
        # тетраэдры
        temp_triangles = []
        for i, triangle in enumerate(triangles):
            if abs(Line(triangle.p1, triangle.p2).length - triangle.limit) > fault:
                temp_triangles.append(triangle)
            else:
                # Тут необходима проверка. Так как выросший треугольник может быть неактуальным для дальнейшего роста
                # по причине того, что треугольник лежит на тетраэдре, глубина которого уже равна предельному значению
                # роста
                if triangle.max_depth == 0:
                    continue

                # Находим серединные точки к прямым
                # Так же находим пропорциональную высоту
                mp1 = calc_midpoint(triangle.p1, triangle.p2)
                mp2 = calc_midpoint(triangle.p2, triangle.p3)
                mp3 = calc_midpoint(triangle.p1, triangle.p3)
                # Зная что высота в тетраеэдре равна такой пропорции от стороны, вычислим ее
                h = (math.sqrt(6.0) / 3) * calc_distance(mp1, mp2)
                # Находим вершину тетраэдра
                p4, surface_k = find_tetrahedron_vertex(mp1, mp2, mp3, h,
                                                        (triangle.parent.A, triangle.parent.B, triangle.parent.C),
                                                        triangle.special)

                # Начальные преобразования найденного тетраэдра
                mp1 *= s_coefficient
                mp2 *= s_coefficient
                mp3 *= s_coefficient
                p4 *= s_coefficient

                # Высчитываем начальную длину, на основе которой будем вычислять шаг инкрементирования
                s_len = Line(mp1, mp2).length

                # Вычисляем центр тетраэдра и приращение для дальнейших вычилений роста
                s_p_c = find_centroid(mp1, mp2, mp3, p4)

                # TODO: Необходимо понимать, какой это из трегольников, если это тот, который образовался снаруже того,
                # TODO: который в центре, то глубина количество иетраций должно расчитываться уже не от родительсвого
                # TODO: а уже от тетраэдра, который поставили по середине. Иначе этот тетраэдр вырастит
                # TODO: быстрее положенного и будет искажение с параллельным переносом
                # Задаем рандомное колиечество итерация роста для тетраэдра
                iters = tetrahedron_info["iterations_count"][triangle.parent.id] + random.randint(1, tetrahedron_info["iterations_count"][triangle.parent.id])

                # Тут тонкий момент. Необходимо подобрать такой предел, чтоб онбыл : а) стохастичный б) но при этом не
                # пересекался с родительским пределом, чтобы избежать наслоения
                limit = tetrahedron_info["limits"][triangle.parent.id] * random.uniform(0.25, 0.45)
                print(f'Родительский limit = {tetrahedron_info["limits"][triangle.parent.id]}; Треугольника limit = {triangle.limit}; Дочерний limit = {limit}')
                delta_p1 = find_step_growth(s_len, limit, iters, mp1, s_p_c)
                delta_p2 = find_step_growth(s_len, limit, iters, mp2, s_p_c)
                delta_p3 = find_step_growth(s_len, limit, iters, mp3, s_p_c)
                delta_p4 = find_step_growth(s_len, limit, iters, p4, s_p_c)

                # Добавляем найденный и приобразованный тетраэдр в список всех тетраэдров
                # Если треугольник помечен, как интересущий нас для сбора метрики, то помечаем тетраэдр и все
                # произвольные от него трегольники. # Иначе считаем его обычным
                tetrahedron = Tetrahedron(mp1, mp2, mp3, p4, surface_k, triangle, triangle.mark)
                tetrahedrons.append(tetrahedron)

                # Добавляем список инриментов, относящихся к этому тетраэдру, в список общих инкрементов всех тетраэдров
                tetrahedron_info["increments"][tetrahedron.id] = [delta_p1, delta_p2, delta_p3, delta_p4]
                tetrahedron_info["limits"][tetrahedron.id] = limit

                # Так же занесем максимальную (родительская глубина - 1) и текущую глубину фрактала -1
                tetrahedron_info["depths"]["current"][tetrahedron.id] = -1
                tetrahedron_info["depths"]["maximum"][tetrahedron.id] = triangle.max_depth - 1
                tetrahedron_info["iterations_count"][tetrahedron.id] = iters
                tetrahedron_info["new"][tetrahedron.id] = True


                # Так как каждая из граней, когда вырастает до значение limit_value должна делиться поровну на 4 части
                # (четыре равных треугольника) и на одну из них мы уже поставили тетраэдр (на центарльную). Существую
                # еще три, которые расположены слева, справа и сверху над описанным. Добавим эти треугольники. Но сперва
                # необходимо пересчитать центральные точки заново, потому что передыдущие уже были модифицированы
                # операцией масштабирования
                mp11 = calc_midpoint(triangle.p1, triangle.p2)
                mp22 = calc_midpoint(triangle.p2, triangle.p3)
                mp33 = calc_midpoint(triangle.p1, triangle.p3)

                temp_triangles.append(Face(triangle.p1, mp11, mp33, triangle.max_depth - 1, triangle.parent, triangle.mark, triangle.special, limit=limit))
                temp_triangles.append(Face(mp11, triangle.p2, mp22, triangle.max_depth - 1, triangle.parent, triangle.mark, triangle.special, limit=limit))
                temp_triangles.append(Face(mp22, triangle.p3, mp33, triangle.max_depth - 1, triangle.parent, triangle.mark, triangle.special, limit=limit))

                # Теперь заносим точку в словарь, чтоб отследить ее после роста и пересчитать
                recalc_middle_points[mp11] = [triangle.p1, triangle.p2]
                recalc_middle_points[mp22] = [triangle.p2, triangle.p3]
                recalc_middle_points[mp33] = [triangle.p1, triangle.p3]

        # Объединяем отфильтрованный список активных треугольников со списком новонайденных треугольников
        triangles = temp_triangles

        # Объявлем массив примитивов, относящихся к конкретной инетарции роста по всем тетраэдрам
        ursina_curr_stage = []
        new_triangles = []
        for i, tetrahedron in enumerate(tetrahedrons):
            # Проверяем нужно ли растить данному тетраэдру или он все таки уже вырос на заданную глубину
            if tetrahedron_info["depths"]["current"][tetrahedron.id] == tetrahedron_info["depths"]["maximum"][tetrahedron.id]:
                # Не забываем про визуализацию. Даже не смотря на то, что тетраэдр уже вырос, визуализировать его нам
                # необходимо
                v1 = [[tetrahedron.p1.x, tetrahedron.p1.y, tetrahedron.p1.z],
                      [tetrahedron.p2.x, tetrahedron.p2.y, tetrahedron.p2.z],
                      [tetrahedron.p3.x, tetrahedron.p3.y, tetrahedron.p3.z],
                      [tetrahedron.p4.x, tetrahedron.p4.y, tetrahedron.p4.z]
                      ]
                t1 = [[0, 1], [1, 2], [2, 0], [0, 3], [3, 1], [3, 2]]
                ursina_curr_stage.append(Model(vertices=v1, triangles=t1))
                continue

            if abs(Line(tetrahedron.p1, tetrahedron.p2).length - tetrahedron_info["limits"][tetrahedron.id]) > fault:
                tetrahedron.p1 += tetrahedron_info["increments"][tetrahedron.id][0]
                tetrahedron.p2 += tetrahedron_info["increments"][tetrahedron.id][1]
                tetrahedron.p3 += tetrahedron_info["increments"][tetrahedron.id][2]
                tetrahedron.p4 += tetrahedron_info["increments"][tetrahedron.id][3]

                # Если это не базовый тетраэдр, необходимо сделать параллельный перенос
                if tetrahedron.parent is not None:
                    # Необходимо выполнить параллельный перенос.
                    # Для этого вычислим центр родительской грани, на которой базируется тетраэдр
                    parent_centroid = calc_centroid(tetrahedron.parent.p1, tetrahedron.parent.p2, tetrahedron.parent.p3)
                    # И вычислим центр дочернего основания тетраэдра
                    child_centroid = calc_centroid(tetrahedron.p1, tetrahedron.p2, tetrahedron.p3)
                    # и осещетсвим парарелельный перенос путем инкеремнтирваония, каждой из точек дочернего тетраэдра
                    # на dx, dx и dz соотвественно
                    dx = parent_centroid.x - child_centroid.x
                    dy = parent_centroid.y - child_centroid.y
                    dz = parent_centroid.z - child_centroid.z

                    tetrahedron.p1.x += dx
                    tetrahedron.p2.x += dx
                    tetrahedron.p3.x += dx
                    tetrahedron.p4.x += dx

                    tetrahedron.p1.y += dy
                    tetrahedron.p2.y += dy
                    tetrahedron.p3.y += dy
                    tetrahedron.p4.y += dy

                    tetrahedron.p1.z += dz
                    tetrahedron.p2.z += dz
                    tetrahedron.p3.z += dz
                    tetrahedron.p4.z += dz

            # Собираем примитивы для дальнейшей визуализации
            v1 = [[tetrahedron.p1.x, tetrahedron.p1.y, tetrahedron.p1.z],
                  [tetrahedron.p2.x, tetrahedron.p2.y, tetrahedron.p2.z],
                  [tetrahedron.p3.x, tetrahedron.p3.y, tetrahedron.p3.z],
                  [tetrahedron.p4.x, tetrahedron.p4.y, tetrahedron.p4.z]
                  ]
            t1 = [[0, 1], [1, 2], [2, 0], [0, 3], [3, 1], [3, 2]]
            ursina_curr_stage.append(Model(vertices=v1, triangles=t1))

            # Необходимо проверить вырос ли текущий тетраэдр. Если да, инкрементировать ему текущую глубину на 1,
            # увеличить предел роста на x2 и обновить инкрименты
            if abs(Line(tetrahedron.p1, tetrahedron.p2).length - tetrahedron_info["limits"][tetrahedron.id]) <= fault:
                tetrahedron_info["depths"]["current"][tetrahedron.id] += 1

                # Проверяем нужно ли вычислять данному тетраэдру инкрименты или
                # он все таки уже вырос на заданную глубину
                if tetrahedron_info["depths"]["current"][tetrahedron.id] == tetrahedron_info["depths"]["maximum"][tetrahedron.id]:
                    continue

                # Тут необходимо определить является ли это новый тетраэдр, который дорос до предельного значения или
                # является старым. Нужно это для того, чтобы, в случае нового тетраэдра - пополнить список активных
                # треугольников
                rookie = False
                if tetrahedron.id in tetrahedron_info["new"]:
                    rookie = True

                # limit = limit - random.randint(1, int(limit)) / 2.0
                # Здесь необходимо дополнить список активных треугольников. Однако есть тонкий момент. Если тетраэдр
                # впервые дорос до предельного значения limit_value, то именно он является донором новых треугольников
                # для татраэдра. Старые учитывать нельзя!
                if rookie:
                    max_depth = tetrahedron_info["depths"]["maximum"][tetrahedron.id]
                    new_triangles.append(Face(tetrahedron.p1, tetrahedron.p2, tetrahedron.p4, max_depth, tetrahedron, triangle.mark, triangle.special, limit=tetrahedron_info["limits"][tetrahedron.id]))
                    new_triangles.append(Face(tetrahedron.p2, tetrahedron.p3, tetrahedron.p4, max_depth, tetrahedron, triangle.mark, triangle.special, limit=tetrahedron_info["limits"][tetrahedron.id]))
                    new_triangles.append(Face(tetrahedron.p1, tetrahedron.p3, tetrahedron.p4, max_depth, tetrahedron, triangle.mark, triangle.special, limit=tetrahedron_info["limits"][tetrahedron.id]))

                    del tetrahedron_info["new"][tetrahedron.id]

                # Необходимо установить пределы, чтоб не было пересечений в дальнейшем росте.
                # 1) Если это новый тетраэдр или небазовый, то берем рандом от текущего предела тетраэдра и предела
                # родителя
                # 2) Если это тетраэдр базовый, продолжаем увеличивать предел
                if rookie or tetrahedron.parent is not None:
                    limit = tetrahedron_info["limits"][tetrahedron.parent.parent.id] * random.uniform(0.25, 0.45)
                else:
                    limit = tetrahedron_info["limits"][tetrahedron.id] + random.uniform(1, tetrahedron_info["limits"][tetrahedron.id])

                tetrahedron_info["limits"][tetrahedron.id] = limit
                # Вычисляем центр тетраэдра и приращение для дальнейших вычилений роста
                s_p_c = find_centroid(tetrahedron.p1, tetrahedron.p2, tetrahedron.p3, tetrahedron.p4)
                s_len = Line(tetrahedron.p1, tetrahedron.p2).length

                # Задаем рандомное колиечество итераций роста для тетраэдра, исходя из того, базовый ли это тетраэдр
                # или нет. Нужно это для того, чтобы дочерние элементы не достраивались быстрее родительских, чтоб не
                # было гонки с визуализацией
                if tetrahedron.parent is not None:
                    iters = tetrahedron_info["iterations_count"][triangle.parent.id] + random.randint(1, tetrahedron_info["iterations_count"][triangle.parent.id])
                else:
                    iters = iter_count

                # Пересчитываем дельты
                delta_p1 = find_step_growth(s_len, limit, iters, tetrahedron.p1, s_p_c)
                delta_p2 = find_step_growth(s_len, limit, iters, tetrahedron.p2, s_p_c)
                delta_p3 = find_step_growth(s_len, limit, iters, tetrahedron.p3, s_p_c)
                delta_p4 = find_step_growth(s_len, limit, iters, tetrahedron.p4, s_p_c)

                # Записываем вычисленные инкрименты, относящиеся к конкретному тетраэдру, в список инкрементов
                tetrahedron_info["increments"][tetrahedron.id] = [delta_p1, delta_p2, delta_p3, delta_p4]
                # Обновляем количество итераций роста
                tetrahedron_info["iterations_count"][tetrahedron.id] = iters

        # Добавляем найденные грани в список активных отрезков
        triangles += new_triangles
        ursina_models.append(ursina_curr_stage)

        # Пересчитываем серединные точки
        for point, points in recalc_middle_points.items():
            p = calc_midpoint(points[0], points[1])
            point.x = p.x
            point.y = p.y
            point.z = p.z

        # Собираем метрики
        l = 0
        s = 0
        v = 0
        for i, t in enumerate(tetrahedrons):
            # Если это базовый тетраэдр: то объем не учитываем, площадь равна площади одной грани, как и длина
            if i == 0:
                l = t._face1.total_length
                s = t._face1.square
            else:
                if t.mark:
                    v += t.volume
                    l += t.total_length
                    s += t.total_square
        # Сохраняем метрики. Если метрика текущая ничем не отличается от предыдущей
        # (а это означает, что все маркированные фишгуры мы просчитали), то учитывать ее не нужно.
        if global_i == 0 or volume[-1] != v:
            line_length.append(l)
            square.append(s)
            volume.append(v)
            global_i += 1
            iterations.append(global_i)
        ####

    # Вычисляем отношения S/L и V/S для обнаружения закономерностей.
    s_l = [square[i] / line_length[i] for i in range(len(iterations))]
    v_s = [volume[i] / square[i] for i in range(len(iterations))]

    # # Производим интерполяцию по найденным метрикам
    # y_length = make_interpolation(iterations, line_length)
    # y_square = make_interpolation(iterations, square)
    # y_volume = make_interpolation(iterations, volume)

    # Строим графики для найденных и апроксимируемыъ метрик.
    fig1, ax1 = plt.subplots()
    ax1.plot(iterations, line_length, 'o', label=r'$a$', c='black', linewidth=1)
    # ax1.plot(iterations, y_length, '-', label=r'$b$', c='red', linewidth=1)
    fig2, ax2 = plt.subplots()
    ax2.plot(iterations, square, 'X', label=r'$a$', c='black', linewidth=1)
    # ax2.plot(iterations, y_square, '-', label=r'$b$', c='red', linewidth=1)
    fig3, ax3 = plt.subplots()
    ax3.plot(iterations, volume, '*', label=r'$a$', c='black', linewidth=1)
    # ax3.plot(iterations, y_volume, '-', label=r'$b$', c='red', linewidth=1)
    fig4, ax4 = plt.subplots()
    ax4.plot(iterations, s_l, '*', label=r'$a$', c='black', linewidth=1)
    fig5, ax5 = plt.subplots()
    ax5.plot(iterations, v_s, '*', label=r'$a$', c='black', linewidth=1)

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    ax5.grid(True)

    ax1.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax1.set(xlabel='Число циклов роста, ед.', ylabel='Длина фрактальной линии, ед.')

    ax2.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax2.set(xlabel='Число циклов роста, ед.', ylabel='Площадь фрактала, ед.')

    ax3.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax3.set(xlabel='Число циклов роста, ед.', ylabel='Объем фрактала, ед.')

    ax4.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax4.set(xlabel='Число циклов роста, ед.', ylabel='Отношение S/L, ед.')

    ax5.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax5.set(xlabel='Число циклов роста, ед.', ylabel='Отношение V/S, ед.')
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
    fig4.savefig("s_l.png")
    fig5.savefig("v_s.png")

    plt.show()

    return ursina_models
