from datetime import datetime
import math
import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from entity import Point, Line, Face, Tetrahedron
from visualization.entity import Model


def real_length(fractal_depth: int, limit_value: float) -> float:
    # TODO: Пока нет формулы
    pass


def real_square(fractal_depth: int, limit_value: float) -> float:
    return (6 ** fractal_depth) * (math.sqrt(3)/4.0) * limit_value ** 2


def real_value(fractal_depth: int, limit_value: float) -> float:
    return 2 ** (3*fractal_depth) * ((limit_value ** 3) / 6 * math.sqrt(2))


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


def df(i: int, N: int):
    """
    Вычисленгие приращения логистической функции на i-ой итарации от общего количества итераций N.
    :param i: текущий номер итерации. i = 1, 2, ..., N
    :param N: общее количество итераций.
    :return: Вычисленное приращение логистической функции.
    """
    lambda_ = 1.0
    mu = -5.0

    tau_alpha = (1.0 / lambda_) * (-13.82 - mu)
    tau_i = tau_alpha + 27.64 * ((2 * i - 1) / (2 * lambda_ * N))
    e = math.e**(lambda_*tau_i + mu)

    f_i = (e / (1 + e)**2) * (27.64 / N)

    return f_i

def find_step_growth(start_len: float, final_len: float, p: Point, c_p: Point, func) -> (float, float, float):
    """

    :param start_len:
    :param final_len:
    :param p:
    :param c_p:
    :param func: логистическая функция изменения закона роста
    :return:
    """
    coefficient = ((final_len - start_len) / start_len)

    x = coefficient * (p.x - c_p.x) * func
    y = coefficient * (p.y - c_p.y) * func
    z = coefficient * (p.z - c_p.z) * func

    return x, y, z


def calculate(iter_count: int, limit_value: float, depth: int, only_for_metrics: bool = False) -> List[List[Model]]:
    """
    Вычисление однофазной модели
    :param iter_count: количество итераций роста
    :param limit_value: предальное значение отрезка
    :param depth: глубина фраткальной структуры
    :param only_for_metrics: если мы хотим собрать лишь метрики, то нам нет необходимости забивать оперативу данными для
     визуализации. Поэтому, если флаг будет в True, ребра и треугольники для движка Ursina собираться не будут
    :return:
    """
    print(f'Начало работы вычисления функционального метода: {datetime.now()}')

    # Начальные точки тетраэдра, вектор нормали (с коэффициентами A, B и C), и начальный коэфициент
    # для уменьшения фигуры
    s_p1 = Point(0.0, 0.0, 0.0)
    s_p2 = Point(0.5, (math.sqrt(3) / 2.0), 0.0)
    s_p3 = Point(1.0, 0.0, 0.0)
    s_h = math.sqrt(2.0 / 3.0) * 1
    s_p4, surface_k = find_tetrahedron_vertex(s_p1, s_p2, s_p3, s_h)
    s_coefficient = 0.05

    # Начальные преобразования тетраэдра. Уменьшаем его четыре точки на коефициент s_coefficient
    s_p1 *= s_coefficient
    s_p2 *= s_coefficient
    s_p3 *= s_coefficient
    s_p4 *= s_coefficient

    # Вычисляем центр тетраэдра и приращение для дальнейших вычилений роста
    s_p_c = find_centroid(s_p1, s_p2, s_p3, s_p4)

    # Создлаем базовый тетраэдр. Указываем точки, коэфициенты и отсуствие родительской грани
    tetrahedron = Tetrahedron(s_p1, s_p2, s_p3, s_p4, surface_k, None)

    s_len = tetrahedron._line1.length

    tetrahedron_info = {
        "centroid": {
          tetrahedron.id: s_p_c
        },
        "iteration_growth": {
            "current": {
                tetrahedron.id: 1
            },
            "total": {
                tetrahedron.id: iter_count * (depth + 1)
            }
        },
        "max_depth": {
            tetrahedron.id: depth
        },
        "max_length": {
            tetrahedron.id: limit_value ** (depth + 1)
        },
        "first_points": {
            tetrahedron.id: {
                "p1": Point(s_p1.x, s_p1.y, s_p1.z),
                "p2": Point(s_p2.x, s_p2.y, s_p2.z),
                "p3": Point(s_p3.x, s_p3.y, s_p3.z),
                "p4": Point(s_p4.x, s_p4.y, s_p4.z),
            }
        },
        "father": {
            tetrahedron.id: False
        }
    }

    # Заводим массив тетрэдров и добавляем в него бозовый. В базовом указываем точки, коэфициенты и отсуствие
    # родительской грани
    tetrahedrons = [tetrahedron]

    # Массивы для подсчета метрик: длины, площади и объема. Так же массив итераций роста.
    iterations = []
    line_length = []
    square = []
    volume = []
    volume_base = []

    # Итерация роста полной фигуры. Необходима в будущем для визуализации величин длины, площади и объема фрактала
    global_i = 0
    # Представление фигуры на заданной итерации роста. Необходимо для отображения движком ursina.
    ursina_models = []

    # Создаем массив активных треугольников, на которых будем растить последующие тетраэдры.
    triangles = []

    # Заведем словарь средних точек для того, чтобы после каждого роста фигур, пересчитать их основывася
    recalc_middle_points = {}

    # Растим до тех пор, пока базовый тетраэдр не достиг максимальной итерации роста. Тут достаточно проверить хотя бы
    # один тетраэдр (в данном случае базовый) так как это ведь однофазный рост и все должно быть синхронно.
    while tetrahedron_info["iteration_growth"]["current"][tetrahedrons[0].id] - 1 != tetrahedron_info["iteration_growth"]["total"][tetrahedrons[0].id]:
        # Необходимо прорядить список активных треугольников и добавить к ним новопоступившиеся и образовать новые
        # тетраэдры
        temp_triangles = []
        for i, triangle in enumerate(triangles):
            # Если сторона треугольника еще не достигла значения limit_value, то оставляем его в списке активных
            # треугольников. Тут особенность - нет никаких epsilon.
            if Line(triangle.p1, triangle.p2).length <= limit_value:
                temp_triangles.append(triangle)
            else:
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

                # Добавляем найденный и приобразованный тетраэдр в список всех тетраэдров
                # Если треугольник помечен, как интересущий нас для сбора метрики, то помечаем тетраэдр и все
                # произвольные от него трегольники. Иначе считаем его обычным
                tetrahedron = Tetrahedron(mp1, mp2, mp3, p4, surface_k, triangle, triangle.mark)
                tetrahedrons.append(tetrahedron)

                # Добавляем нужную информацию для нового тетраэдра
                tetrahedron_info["centroid"][tetrahedron.id] = s_p_c
                tetrahedron_info["iteration_growth"]["current"][tetrahedron.id] = 1
                tetrahedron_info["iteration_growth"]["total"][tetrahedron.id] = iter_count * triangle.max_depth
                tetrahedron_info["max_depth"][tetrahedron.id] = triangle.max_depth - 1
                tetrahedron_info["max_length"][tetrahedron.id] = limit_value ** triangle.max_depth
                tetrahedron_info["first_points"][tetrahedron.id] = {
                    "p1": Point(mp1.x, mp1.y, mp1.z),
                    "p2": Point(mp2.x, mp2.y, mp2.z),
                    "p3": Point(mp3.x, mp3.y, mp3.z),
                    "p4": Point(p4.x, p4.y, p4.z),
                }
                tetrahedron_info["father"][tetrahedron.id] = False

                # Проверка на то, чтобы не записать в список активных треугольников те, которые больше не должны расти
                if triangle.max_depth - 1 != 0:
                    # Так как каждая из граней, когда вырастает до значение limit_value должна делиться поровну на 4
                    # части (четыре равных треугольника) и на одну из них мы уже поставили тетраэдр (на центарльную).
                    # Существую еще три, которые расположены слева, справа и сверху над описанным. Добавим эти
                    # треугольники. Но сперва необходимо пересчитать центральные точки заново, потому что передыдущие
                    # уже были модифицированы операцией масштабирования
                    mp11 = calc_midpoint(triangle.p1, triangle.p2)
                    mp22 = calc_midpoint(triangle.p2, triangle.p3)
                    mp33 = calc_midpoint(triangle.p1, triangle.p3)
                    temp_triangles.append(Face(triangle.p1, mp11, mp33, triangle.max_depth - 1, triangle.parent, triangle.mark, triangle.special))
                    temp_triangles.append(Face(mp11, triangle.p2, mp22, triangle.max_depth - 1, triangle.parent, triangle.mark, triangle.special))
                    temp_triangles.append(Face(mp22, triangle.p3, mp33, triangle.max_depth - 1, triangle.parent, triangle.mark, triangle.special))

                    # Теперь заносим точку в словарь, чтоб отследить ее после роста и пересчитать
                    recalc_middle_points[mp11] = [triangle.p1, triangle.p2]
                    recalc_middle_points[mp22] = [triangle.p2, triangle.p3]
                    recalc_middle_points[mp33] = [triangle.p1, triangle.p3]

        # Объединяем отфильтрованный список активных треугольников со списком новонайденных треугольников
        triangles = temp_triangles

        # Список активных треугольников, найденных на текущей итерации роста
        stage_triangles = []

        # Объявлем массив примитивов, относящихся к конкретной инетарции роста по всем тетраэдрам
        ursina_curr_stage = []

        # Вычисляем приращение для каждого из тетраэдров
        for tetrahedron in tetrahedrons:
            # Собираем все необходимые комопненты для расчета дельт роста. Центарльную точку, текущую итерацию,
            # предельное значение длины и общее количество итераций роста для текущего тетраэдра
            s_p_c = tetrahedron_info["centroid"][tetrahedron.id]
            iteration = tetrahedron_info["iteration_growth"]["current"][tetrahedron.id]
            N = tetrahedron_info["iteration_growth"]["total"][tetrahedron.id]
            final_len = tetrahedron_info["max_length"][tetrahedron.id]

            # Пересчитываем дельты
            delta_p1 = find_step_growth(s_len, final_len, tetrahedron_info["first_points"][tetrahedron.id]["p1"], s_p_c, df(iteration, N))
            delta_p2 = find_step_growth(s_len, final_len, tetrahedron_info["first_points"][tetrahedron.id]["p2"], s_p_c, df(iteration, N))
            delta_p3 = find_step_growth(s_len, final_len, tetrahedron_info["first_points"][tetrahedron.id]["p3"], s_p_c, df(iteration, N))
            delta_p4 = find_step_growth(s_len, final_len, tetrahedron_info["first_points"][tetrahedron.id]["p4"], s_p_c, df(iteration, N))

            # Обновить текущую итерацию для текущего тетраэдра
            tetrahedron_info["iteration_growth"]["current"][tetrahedron.id] += 1

            # наращиваем текущий тетраэдр
            tetrahedron.p1 += delta_p1
            tetrahedron.p2 += delta_p2
            tetrahedron.p3 += delta_p3
            tetrahedron.p4 += delta_p4

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

            # После того, как нарастили тетраэдр. Нужно понять, если тетраэдр находится в списке тетраэдров, которые
            # junior (т.е, которые только начали свой рост и еще не вырасли до значение limit_value) и он еще не разу не
            # был родителем, но приэтом нужно сделат проверку на его длину ребра. Если все ок - то именно этот тетраэдр
            # становится донором новых треуголников, на основе которых появляются новые тетраэдры
            if not (tetrahedron_info["father"][tetrahedron.id]) and (tetrahedron._line1.length > limit_value):
                # Если это родительский тетраэдр, то это осбоый случай. Тут необходимо:
                # а) пометить один из треугольников, для сбора метрики. А метрику собираем только лишь по одному
                # б) создать не три, а уже 4 треугольника.
                if tetrahedron.parent is None:
                    stage_triangles.append(Face(tetrahedron.p1, tetrahedron.p2, tetrahedron.p3, depth, tetrahedron, True, True))
                    stage_triangles.append(Face(tetrahedron.p1, tetrahedron.p4, tetrahedron.p2, depth, tetrahedron))
                    stage_triangles.append(Face(tetrahedron.p1, tetrahedron.p4, tetrahedron.p3, depth, tetrahedron))
                    stage_triangles.append(Face(tetrahedron.p2, tetrahedron.p4, tetrahedron.p3, depth, tetrahedron))
                # Иначе - это обычный тетраэдр и количество необходимых треугольников равно трем. Т.е. все
                # треугольники кроме основания. Ведь мы не собираемся растить внутрь :)
                else:
                    max_depth = tetrahedron_info["max_depth"][tetrahedron.id]
                    if max_depth != 0:
                        stage_triangles.append(Face(tetrahedron.p1, tetrahedron.p2, tetrahedron.p4, max_depth, tetrahedron, triangle.mark, triangle.special))
                        stage_triangles.append(Face(tetrahedron.p1, tetrahedron.p3, tetrahedron.p4, max_depth, tetrahedron, triangle.mark, triangle.special))
                        stage_triangles.append(Face(tetrahedron.p2, tetrahedron.p3, tetrahedron.p4, max_depth, tetrahedron, triangle.mark, triangle.special))

                # помечаем, что данный тетраэдр стал родителем новых треугольников для последующего построения на них
                # тетраэдров
                tetrahedron_info["father"][tetrahedron.id] = True

        triangles += stage_triangles

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

        if len(tetrahedrons) > 1:
            # Собираем метрики проходясь по каждому из тетраэдров
            l = 0
            s = 0
            v = 0
            for i, t in enumerate(tetrahedrons):
                # Если это базовый тетраэдр: то объем не учитываем, площадь равна площади одной грани, как и длина
                if i == 0:
                    volume_base.append(t.volume)
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

    print(f'Окончание работы вычисления функционального однофазного метода: {datetime.now()}')
    print()

    # Вычисляем отношения S/L и V/S для обнаружения закономерностей.
    s_l = [square[i] / line_length[i] for i in range(len(iterations))]
    v_s = [volume[i] / square[i] for i in range(len(iterations))]
    v_l = [volume[i] / line_length[i] for i in range(len(iterations))]
    v_v_base = [4 * volume[i] / volume_base[i] for i in range(len(iterations))]

    # # TODO: разкомментировать по необходиомости
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
    fig6, ax6 = plt.subplots()
    ax6.plot(iterations, v_l, '*', label=r'$a$', c='black', linewidth=1)
    fig7, ax7 = plt.subplots()
    ax7.plot(iterations, v_v_base, '*', label=r'$a$', c='black', linewidth=1)

    with open(f'./metrics/datasets/functional/iterations_iter_count_{iter_count}_depth_{depth}.txt', 'wb') as f:
        pickle.dump(iterations, f)

    with open(f'./metrics/datasets/functional/length_iter_count_{iter_count}_depth_{depth}.txt', 'wb') as f:
        pickle.dump(line_length, f)

    with open(f'./metrics/datasets/functional/square_iter_count_{iter_count}_depth_{depth}.txt', 'wb') as f:
        pickle.dump(square, f)

    with open(f'./metrics/datasets/functional/volume_iter_count_{iter_count}_depth_{depth}.txt', 'wb') as f:
        pickle.dump(volume, f)

    with open(f'./metrics/datasets/functional/s_l_iter_count_{iter_count}_depth_{depth}.txt', 'wb') as f:
        pickle.dump(s_l, f)

    with open(f'./metrics/datasets/functional/v_s_iter_count_{iter_count}_depth_{depth}.txt', 'wb') as f:
        pickle.dump(v_s, f)

    with open(f'./metrics/datasets/functional/v_l_iter_count_{iter_count}_depth_{depth}.txt', 'wb') as f:
        pickle.dump(v_l, f)

    with open(f'./metrics/datasets/functional/v_v_base_iter_count_{iter_count}_depth_{depth}.txt', 'wb') as f:
        pickle.dump(v_v_base, f)

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    ax5.grid(True)
    ax6.grid(True)
    ax7.grid(True)

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

    ax6.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax6.set(xlabel='Число циклов роста, ед.', ylabel='Отношение V/L, ед.')

    ax7.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax7.set(xlabel='Число циклов роста, ед.', ylabel='Отношение 4*V1/V0, ед.')

    fig1.savefig(f'./metrics/graphics/functional/length.png')
    fig2.savefig(f'./metrics/graphics/functional/square.png')
    fig3.savefig(f'./metrics/graphics/functional/value.png')
    fig4.savefig(f'./metrics/graphics/functional/s_l.png')
    fig5.savefig(f'./metrics/graphics/functional/v_s.png')
    fig6.savefig(f'./metrics/graphics/functional/v_l.png')
    fig7.savefig(f'./metrics/graphics/functional/4v1_v0.png')

    plt.show()

    print(f'Количество тетраэдров = {len(tetrahedrons)} в функциональном методе при глубине = {depth}')

    return ursina_models
