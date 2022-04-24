import numpy as np
from typing import List

import matplotlib.pyplot as plt
import pickle
import scipy.optimize as opt


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


def sigmoid(x, l0, x0, k, b):
    return l0 / (1 + np.exp(-k * (x - x0))) + b


if __name__ == '__main__':
    with open(f'../metrics/datasets/combined/iterations_limit_value_2.0_iter_count_1500_depth_5_delta_200_l_rnd_1.0.txt', 'rb') as fp:
        x = pickle.load(fp)

    with open(f'../metrics/datasets/combined/length_limit_value_2.0_iter_count_1500_depth_5_delta_200_l_rnd_1.0.txt', 'rb') as fp:
        y = pickle.load(fp)

    x1, y1, = thin_out(x, y, 1280)
    # x1, y1, = thin_out(x, y, 640)

    p0 = [max(y1), np.median(x1), 0, min(y1)]
    popt, _ = opt.curve_fit(sigmoid, x1, y1, p0=p0)

    fig1, ax1 = plt.subplots()

    x_s = np.arange(0, max(x1), 0.01)
    y_s = [sigmoid(x, *popt) for x in x_s]

    ax1.plot(x1, y1, 'o', markersize=15)
    ax1.plot(x_s, y_s, 'o')

    plt.show()
