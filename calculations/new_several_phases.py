from datetime import datetime
import math
import pickle
import random
from typing import List, Tuple

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


def calculate(iter_count: int, limit_value: float, depth: int, only_for_metrics: bool = False) -> List[List[Model]]:
    """
    Вычисление новой многофазной модели. Отличие от существующей многофазной модели в том, что дочерние тетраэдры
    появляются раньше времни, когда родительский тетраэдр достиг значения limit_value.
    :param iter_count: количество итераций роста
    :param limit_value: предальное значение отрезка
    :param depth: глубина фраткальной структуры
    :param only_for_metrics: если мы хотим собрать лишь метрики, то нам нет необходимости забивать оперативу данными для
     визуализации. Поэтому, если флаг будет в True, ребра и треугольники для движка Ursina собираться не будут
    :return:
    """
    print(f'Начало работы вычисления нового многофазного метода: {datetime.now()}')

    # Начальные точки тетраэдра, вектор нормали (с коэффициентами A, B и C), и начальный коэфициент
    # для уменьшения фигуры
    s_p1 = Point(0.0, 0.0, 0.0)
    s_p2 = Point(0.5, (math.sqrt(3) / 2.0), 0.0)
    s_p3 = Point(1.0, 0.0, 0.0)
    s_h = math.sqrt(2.0 / 3.0) * 1
    s_p4, surface_k = find_tetrahedron_vertex(s_p1, s_p2, s_p3, s_h)
    s_coefficient = 0.05

    # Значения погрешности, которое будем сопоставлять при достижении отрезка нужной длины (limit_value)
    fault = 0.02

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

    # Создаем базовый тетраэдр. Указываем точки, коэфициенты и отсуствие родительской грани.
    tetrahedron = Tetrahedron(s_p1, s_p2, s_p3, s_p4, surface_k, None)

    # Заводим массив тетрэдров и добавляем в него бозовый.
    tetrahedrons = [tetrahedron]

    # Создаем массив активных треугольников, на которых будем растить последующие тетраэдры, и добавляем в него
    # существующие треугольники, из которых состоит тетраэдр.
    triangle1 = Face(s_p1, s_p2, s_p3, depth, tetrahedron, True, True)
    triangle2 = Face(s_p1, s_p4, s_p2, depth, tetrahedron)
    triangle3 = Face(s_p1, s_p4, s_p3, depth, tetrahedron)
    triangle4 = Face(s_p2, s_p4, s_p3, depth, tetrahedron)
    triangles = [triangle1, triangle2, triangle3, triangle4]

    # Объявляем массив инкрементов для каждого из тетраэдра. В данном случае для базового тетраэдра
    # Объявляем массим пределов, до какого предела растить тетраэдр. Для базового тетраэдра необхоимо обновить шаг
    # роста, ведь финальная длина поменяется на x2
    tetrahedron_info = {
        "increments": {
            tetrahedron.id: [delta_p1, delta_p2, delta_p3, delta_p4]
        },
        "limits": {
            tetrahedron.id: limit_value
        },
        "iterations": {
            "per_order": {
                tetrahedron.id: iter_count
            },
            "current": {
                tetrahedron.id: 0,
            },
        },
        "depths": {
            "current": {
                tetrahedron.id: -1
            },
            "maximum": {
                tetrahedron.id: depth
            }
        },
    }

    # Задаем дельту, которая будет определять - на сколько итераций раньше нужно ставить дочерний тетраэдр, до того
    # момента, как родительский вырос до значения limit_value
    delta_birth = random.randint(1, int(0.1 * iter_count))

    # Задаем словарь для отслеживания треугольников, которые растут по многофазному алгоритму. Это те треугольники,
    # которые будут являться донорами новых тетраэдров. Секция birth опредеояет, какой треугольник и когда должен
    # зараждаться.
    triangles_info = {
        "birth": {
            triangle1.id: tetrahedron_info["iterations"]["per_order"][triangle1.parent.id] - delta_birth,
            triangle2.id: tetrahedron_info["iterations"]["per_order"][triangle2.parent.id] - delta_birth,
            triangle3.id: tetrahedron_info["iterations"]["per_order"][triangle3.parent.id] - delta_birth,
            triangle4.id: tetrahedron_info["iterations"]["per_order"][triangle4.parent.id] - delta_birth,
        }
    }

    # Задаем словарь, чтоб понимать на какой глубине было выбранно определенное значение итераций роста, на котором
    # стоит начать растить дочерние тетраэдры. Нужно лишь для того, чтоб не было разных рандомов.
    help_map = {
        1: tetrahedron_info["iterations"]["per_order"][triangle1.parent.id] - delta_birth
    }

    # Массивы для подсчета метрик: длины, площади, объема, объема лишь базового тетраэдра и размаха всего фрактала,
    # представляющего собой длину ребра базового тетраэдра. Так же массив итераций роста.
    iterations = []
    line_length = []
    square = []
    volume = []
    volume_base = []
    fractal_span = []

    # Итерация роста полной фигуры. Необходима в будущем для визуализации величин длины, площади и объема фрактала
    global_i = 0
    # Представление фигуры на заданной итерации роста. Необходимо для отображения движком ursina.
    ursina_models = []

    # Заведем словарь средних точек для того, чтобы после каждого роста фигур, пересчитать их основывася
    recalc_middle_points = {}

    # Выращиваем до тех пор пока существует хоть один тетраэдр, который не дорос до задонной глубины.
    while undergrown_tetrahedron_exists(tetrahedron_info["depths"]):
        # Необходимо прорядить список активных треугольников и добавить к ним новопоступившиеся и образовать новые
        # тетраэдры
        temp_triangles = []
        for i, triangle in enumerate(triangles):
            # Если текущая итерация роста родительского тетраэдра не равна времени появления дочернего тетраэдра, то
            # ждем этого момента
            if triangles_info["birth"][triangle.id] != tetrahedron_info["iterations"]["current"][triangle.parent.id]:
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
                p4, surface_k = find_tetrahedron_vertex(mp1, mp2, mp3, h, (triangle.parent.A, triangle.parent.B, triangle.parent.C), triangle.special)

                # Начальные преобразования найденного тетраэдра
                mp1 *= s_coefficient
                mp2 *= s_coefficient
                mp3 *= s_coefficient
                p4 *= s_coefficient

                # Вычисляем центр тетраэдра и приращение для дальнейших вычилений роста
                s_p_c = find_centroid(mp1, mp2, mp3, p4)

                # Высчитываем начальную длину, на основе которой будем вычислять шаг инкрементирования
                s_len = Line(mp1, mp2).length

                # количество необходимых итераций вычисояется суммированием ожидаемого родителского значения с разницей
                # итераций, на сколько раньше начал расти дочерний
                iters = iter_count + iter_count - triangles_info["birth"][triangle.id]
                delta_p1 = find_step_growth(s_len, limit_value, iters, mp1, s_p_c)
                delta_p2 = find_step_growth(s_len, limit_value, iters, mp2, s_p_c)
                delta_p3 = find_step_growth(s_len, limit_value, iters, mp3, s_p_c)
                delta_p4 = find_step_growth(s_len, limit_value, iters, p4, s_p_c)

                # Добавляем найденный и приобразованный тетраэдр в список всех тетраэдров
                # Если треугольник помечен, как интересущий нас для сбора метрики, то помечаем тетраэдр и все
                # произвольные от него трегольники. Иначе считаем его обычным
                tetrahedron = Tetrahedron(mp1, mp2, mp3, p4, surface_k, triangle, triangle.mark)
                tetrahedrons.append(tetrahedron)

                # Добавляем нужную информацию для нового тетраэдра
                tetrahedron_info["increments"][tetrahedron.id] = [delta_p1, delta_p2, delta_p3, delta_p4]
                tetrahedron_info["limits"][tetrahedron.id] = limit_value
                tetrahedron_info["iterations"]["per_order"][tetrahedron.id] = limit_value
                tetrahedron_info["iterations"]["current"][tetrahedron.id] = 0
                tetrahedron_info["depths"]["current"][tetrahedron.id] = -1
                tetrahedron_info["depths"]["maximum"][tetrahedron.id] = triangle.max_depth - 1

                # Проверка на то, чтобы не записать в список активных треугольников те, которые больше не должны расти
                if triangle.max_depth - 1 != 0:
                    # Определяем
                    if (depth - triangle.max_depth) not in help_map:
                        birth = iter_count - random.randint(1, int(0.1 * iter_count))
                        help_map[depth - triangle.max_depth] = birth
                    else:
                        birth = help_map[depth - triangle.max_depth]

                    # Добавляем, которые на тетраэдре
                    triangle11 = Face(mp1, p4, mp2, triangle.max_depth - 1, tetrahedron, triangle.mark, triangle.special)
                    triangle22 = Face(mp1, p4, mp3, triangle.max_depth - 1, tetrahedron, triangle.mark, triangle.special)
                    triangle33 = Face(mp2, p4, mp3, triangle.max_depth - 1, tetrahedron, triangle.mark, triangle.special)

                    triangles_info["birth"][triangle11.id] = birth
                    triangles_info["birth"][triangle22.id] = birth
                    triangles_info["birth"][triangle33.id] = birth

                    temp_triangles.append(triangle11)
                    temp_triangles.append(triangle22)
                    temp_triangles.append(triangle33)

                    # Так как каждая из граней, когда вырастает до значение limit_value должна делиться поровну на 4
                    # части (четыре равных треугольника) и на одну из них мы уже поставили тетраэдр (на центарльную).
                    # Существую еще три, которые расположены слева, справа и сверху над описанным. Добавим эти
                    # треугольники. Но сперва необходимо пересчитать центральные точки заново, потому что передыдущие
                    # уже были модифицированы операцией масштабирования
                    mp11 = calc_midpoint(triangle.p1, triangle.p2)
                    mp22 = calc_midpoint(triangle.p2, triangle.p3)
                    mp33 = calc_midpoint(triangle.p1, triangle.p3)

                    triangle1 = Face(triangle.p1, mp11, mp33, triangle.max_depth - 1, tetrahedron, triangle.mark, triangle.special)
                    triangle2 = Face(mp11, triangle.p2, mp22, triangle.max_depth - 1, tetrahedron, triangle.mark, triangle.special)
                    triangle3 = Face(mp22, triangle.p3, mp33, triangle.max_depth - 1, tetrahedron, triangle.mark, triangle.special)

                    triangles_info["birth"][triangle1.id] = birth
                    triangles_info["birth"][triangle2.id] = birth
                    triangles_info["birth"][triangle3.id] = birth

                    temp_triangles.append(triangle1)
                    temp_triangles.append(triangle2)
                    temp_triangles.append(triangle3)

                    # Теперь заносим точку в словарь, чтоб отследить ее после роста и пересчитать
                    recalc_middle_points[mp11] = [triangle.p1, triangle.p2]
                    recalc_middle_points[mp22] = [triangle.p2, triangle.p3]
                    recalc_middle_points[mp33] = [triangle.p1, triangle.p3]

        # Объединяем отфильтрованный список активных треугольников со списком новонайденных треугольников
        triangles = temp_triangles

        # Объявлем массив примитивов, относящихся к конкретной инетарции роста по всем тетраэдрам
        ursina_curr_stage = []

        # Вычисляем приращение для каждого из тетраэдров
        for tetrahedron in tetrahedrons:
            # Растим тетраэдр. Инкрементируем каждую из его граней на зарнее вычисленное значение
            tetrahedron.p1 += tetrahedron_info["increments"][tetrahedron.id][0]
            tetrahedron.p2 += tetrahedron_info["increments"][tetrahedron.id][1]
            tetrahedron.p3 += tetrahedron_info["increments"][tetrahedron.id][2]
            tetrahedron.p4 += tetrahedron_info["increments"][tetrahedron.id][3]

            # Добавляем шаг роста
            tetrahedron_info["iterations"]["current"][tetrahedron.id] += 1

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
            # Если хотим собрать метрики для большого значения глубины фрактала - нет необходимости нагружать
            # оперативку данными, которые для визуализации не нужны, так как смотреть не будем на них
            if not only_for_metrics:
                ursina_curr_stage.append(Model(vertices=v1, triangles=t1))

            # Необходимо проверить вырос ли текущий тетраэдр. Если да, инкрементировать ему текущую глубину на 1,
            # увеличить предел роста на x2 и обновить инкрименты
            # print(f'current length = {Line(tetrahedron.p1, tetrahedron.p2).length}; limits = {tetrahedron_info["limits"][tetrahedron.id]}')
            if abs(Line(tetrahedron.p1, tetrahedron.p2).length - tetrahedron_info["limits"][tetrahedron.id]) <= fault:
                tetrahedron_info["depths"]["current"][tetrahedron.id] += 1

                # Проверяем нужно ли вычислять данному тетраэдру инкрименты или
                # он все таки уже вырос на заданную глубину
                if tetrahedron_info["depths"]["current"][tetrahedron.id] == tetrahedron_info["depths"]["maximum"][tetrahedron.id]:
                    continue

                tetrahedron_info["limits"][tetrahedron.id] *= 2
                # Вычисляем центр тетраэдра и приращение для дальнейших вычилений роста
                s_p_c = find_centroid(tetrahedron.p1, tetrahedron.p2, tetrahedron.p3, tetrahedron.p4)
                s_len = Line(tetrahedron.p1, tetrahedron.p2).length

                # Пересчитываем дельты
                delta_p1 = find_step_growth(s_len, s_len * 2, iter_count, tetrahedron.p1, s_p_c)
                delta_p2 = find_step_growth(s_len, s_len * 2, iter_count, tetrahedron.p2, s_p_c)
                delta_p3 = find_step_growth(s_len, s_len * 2, iter_count, tetrahedron.p3, s_p_c)
                delta_p4 = find_step_growth(s_len, s_len * 2, iter_count, tetrahedron.p4, s_p_c)

                # Записываем вычисленные инкрименты, относящиеся к конкретному тетраэдру, в список инкрементов
                tetrahedron_info["increments"][tetrahedron.id] = [delta_p1, delta_p2, delta_p3, delta_p4]

        # Пересчитываем серединные точки
        for point, points in recalc_middle_points.items():
            p = calc_midpoint(points[0], points[1])
            point.x = p.x
            point.y = p.y
            point.z = p.z

        # Если хотим собрать метрики для большого значения глубины фрактала - нет необходимости нагружать оперативку
        # данными, которые для визуализации не нужны, так как смотреть не будем на них
        if not only_for_metrics:
            ursina_models.append(ursina_curr_stage)

        # Собираем метрики проходясь по каждому из тетраэдров
        l = 0
        s = 0
        v = 0
        for i, t in enumerate(tetrahedrons):
            # Если это базовый тетраэдр: то объем не учитываем, площадь равна площади одной грани, как и длина
            if i == 0:
                volume_base.append(t.volume)
                fractal_span.append(t._line1.length)
                l = t._face1.total_length
                s = t._face1.square
            else:
                if t.mark:
                    v += t.volume
                    l += t.total_length
                    s += t.total_square
        # Сохраняем метрики
        line_length.append(l)
        square.append(s)
        volume.append(v)
        ####

        global_i += 1
        iterations.append(global_i)

    print(f'Окончание работы вычисления нового многофазного метода: {datetime.now()}')
    print()

    # Вычисляем отношения S/L и V/S для обнаружения закономерностей.
    s_l = [square[i] / line_length[i] for i in range(len(iterations))]
    v_s = [volume[i] / square[i] for i in range(len(iterations))]
    v_l = [volume[i] / line_length[i] for i in range(len(iterations))]
    v_v_base = [4 * volume[i] / volume_base[i] for i in range(len(iterations))]

    with open(f'./metrics/datasets/new_several_phases/iterations_iter_count_{iter_count}_depth_{depth}.txt', 'wb') as f:
        pickle.dump(iterations, f)

    with open(f'./metrics/datasets/new_several_phases/length_iter_count_{iter_count}_depth_{depth}.txt', 'wb') as f:
        pickle.dump(line_length, f)

    with open(f'./metrics/datasets/new_several_phases/square_iter_count_{iter_count}_depth_{depth}.txt', 'wb') as f:
        pickle.dump(square, f)

    with open(f'./metrics/datasets/new_several_phases/volume_iter_count_{iter_count}_depth_{depth}.txt', 'wb') as f:
        pickle.dump(volume, f)

    with open(f'./metrics/datasets/new_several_phases/s_l_iter_count_{iter_count}_depth_{depth}.txt', 'wb') as f:
        pickle.dump(s_l, f)

    with open(f'./metrics/datasets/new_several_phases/v_s_iter_count_{iter_count}_depth_{depth}.txt', 'wb') as f:
        pickle.dump(v_s, f)

    with open(f'./metrics/datasets/new_several_phases/v_l_iter_count_{iter_count}_depth_{depth}.txt', 'wb') as f:
        pickle.dump(v_l, f)

    with open(f'./metrics/datasets/new_several_phases/v_v_base_iter_count_{iter_count}_depth_{depth}.txt', 'wb') as f:
        pickle.dump(v_v_base, f)

    with open(f'./metrics/datasets/new_several_phases/fractal_span_iter_count_{iter_count}_depth_{depth}.txt', 'wb') as f:
        pickle.dump(fractal_span, f)

    print(f'Количество тетраэдров = {len(tetrahedrons)} в новом многофазном методе при глубине = {depth}')

    return ursina_models
