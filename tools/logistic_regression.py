import numpy as np
from typing import List, Tuple, Union, Iterable

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pickle
import scipy.optimize as opt


def thin_out(x: List[int], y: List[float], k: int):
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


def sigmoid(x, l0, x0, k, b):
    return l0 / (1 + np.exp(-k * (x - x0))) + b


def read_from_file(iter_count: int, limit_value: float, depth: int, left_limit_rnd: float, delta_iters: int):
    with open(f'../metrics/datasets/combined/iterations_limit_value_{limit_value}_iter_count_{iter_count}_depth_{depth}_delta_{delta_iters}_l_rnd_{left_limit_rnd}.txt', 'rb') as fp:
        x = pickle.load(fp)

    with open(f'../metrics/datasets/combined/length_limit_value_{limit_value}_iter_count_{iter_count}_depth_{depth}_delta_{delta_iters}_l_rnd_{left_limit_rnd}.txt', 'rb') as fp:
        y = pickle.load(fp)

    return x, y


def func_difference(x: List[float], popt1: Union[Iterable], popt2: Union[Iterable]) -> Tuple[List[float], List[float], float]:
    """
    :param x:
    :param popt1:
    :param popt2:
    :return:
    """
    powers = []
    differences = []
    s = 0

    for x1 in x:
        y1 = sigmoid(x1, *popt1)
        y2 = sigmoid(x1, *popt2)

        differences.append(y1 - y2)
        powers.append(np.power(y1 - y2, 2))

        s += np.power(y1 - y2, 2)

    return differences, powers, s


def find_coefficients(iterations_count: int):
    x, y = read_from_file(iterations_count, 2.0, 5, 1.0, 100)
    coefficients, _ = opt.curve_fit(sigmoid, x, y, p0=p0)

    return coefficients


def binary_search(experimental_x: List[int], experimental_coefficients: Union[Iterable], l_1: int, l_2: int):
    iteration = 0
    result = {
        'final_tau': 0,
        'attempts': []
    }

    # Repeat until the pointers low and high meet each other
    while l_1 <= l_2:
        iteration = l_1 + (l_2 - l_1) // 2

        differences, powers, diff = func_difference(experimental_x, experimental_coefficients, find_coefficients(iteration))

        result['attempts'].append({
            'iteration': iteration,
            'coefficients': find_coefficients(iteration),
            'differences': differences,
            'powers': powers
        })

        _, _, left_diff = func_difference(experimental_x, experimental_coefficients, find_coefficients(iteration - 1))
        _, _, right_diff = func_difference(experimental_x, experimental_coefficients, find_coefficients(iteration + 1))

        if left_diff < right_diff:
            l_2 = iteration - 1
        else:
            l_1 = iteration + 1

    result['final_tau'] = iteration

    return result


if __name__ == '__main__':
    with open(f'../metrics/datasets/combined/iterations_limit_value_2.0_iter_count_2000_depth_5_delta_200_l_rnd_1.0.txt', 'rb') as fp:
        x = pickle.load(fp)

    with open(f'../metrics/datasets/combined/length_limit_value_2.0_iter_count_2000_depth_5_delta_200_l_rnd_1.0.txt', 'rb') as fp:
        y = pickle.load(fp)

    # x1, y1, = thin_out(x, y, 1280)
    x1, y1, = thin_out(x, y, 640)

    p0 = [max(y1), np.median(x1), 0, min(y1)]
    popt, _ = opt.curve_fit(sigmoid, x1, y1, p0=p0)

    result = binary_search(x1, popt, 256, 4096)

    x_s = np.arange(0, max(x1), 0.1)
    y_s = [sigmoid(x, *popt) for x in x_s]

    for attempt in result['attempts']:
        fig = plt.figure(tight_layout=True)
        fig.suptitle(f'Текущий tau = {attempt["iteration"]}. Пытаемся найти tau = 2000', fontsize=10)
        gs = gridspec.GridSpec(2, 2)

        y_s_1 = [sigmoid(x, *attempt['coefficients']) for x in x_s]

        ax = fig.add_subplot(gs[0, :])
        ax.plot(x_s, y_s, 'o', c='red', label='Экспериментальная кривая')
        ax.plot(x_s, y_s_1, 'o', c='green', label='Уточняющая кривая')
        ax.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
        ax.set_ylabel('Длина фраткальной структуры')
        ax.set_xlabel('Количество итераций роста (tau)')

        ax = fig.add_subplot(gs[1, 0])
        ax.set_title('f-g')
        ax.plot(x1, attempt['differences'], 'o', markersize=5)
        ax.set_ylabel('y')
        ax.set_xlabel('x')

        ax = fig.add_subplot(gs[1, 1])
        ax.set_title('(f-g)**2')
        ax.plot(x1, attempt['powers'], 'o', markersize=5)
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        fig.align_labels()

        fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()

        plt.show()
