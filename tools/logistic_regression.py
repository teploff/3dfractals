import numpy as np
import math
from typing import List

import matplotlib.pyplot as plt
import pickle


class Point2D:
    def __init__(self, x: float, y: float):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y


def yi(l0: float, li: float) -> float:
    """

    :param l0:
    :param li:
    :return:
    """
    return math.log((l0 - li) / li)


def calculate_a_b(l0: float, samples: List[Point2D]) -> (float, float):
    """

    :param l0:
    :param samples:
    :return:
    """
    sum_xi = 0
    sum_yi = 0
    sum_mul_xi_yi = 1
    sum_pow_xi = 0
    n = float(len(samples))

    for sample in samples:
        sum_xi += sample.x
        sum_yi += yi(l0, sample.y)
        sum_mul_xi_yi += sample.x * yi(l0, sample.y)
        sum_pow_xi += sample.x * sample.x

    a = (n * sum_mul_xi_yi - sum_xi * sum_yi) / (n * sum_pow_xi - sum_xi * sum_xi)
    b = (sum_yi - a * sum_xi) / n

    return a, b


def f(x: float, l0: float, a: float, b: float) -> float:
    """

    :param x:
    :param l0:
    :param a:
    :param b:
    :return:
    """
    return l0 / (1 + math.e**(a * x + b))


def thin_out(x: List[float], y: List[float], k: int):
    """

    :param x:
    :param y:
    :return:
    """

    if len(x) != len(y):
        raise "different dimensions"

    x1, y1 = [], []
    for i, _ in enumerate(x):
        if i % k == 0:
            x1.append(x[i])
            y1.append(y[i])

    return x1, y1


if __name__ == '__main__':
    with open(f'../metrics/datasets/combined/iterations_limit_value_2.0_iter_count_1000_depth_5_delta_200_l_rnd_1.0.txt', 'rb') as fp:
        x = pickle.load(fp)

    with open(f'../metrics/datasets/combined/length_limit_value_2.0_iter_count_1000_depth_5_delta_200_l_rnd_1.0.txt', 'rb') as fp:
        y = pickle.load(fp)

    x1, y1, = thin_out(x, y, 640)
    x2, y2, = thin_out(x, y, 1280)

    fig1, ax1 = plt.subplots()

    ax1.plot(x1, y1, 'o')
    ax1.plot(x2, y2, 'o')

    plt.show()
