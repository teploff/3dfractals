import pickle
from typing import List

import matplotlib.pyplot as plt


def build_classic_one_phase(iter_count: int, depth: int):
    with open(f'../metrics/datasets/classic/iterations_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        iterations = pickle.load(fp)

    with open(f'../metrics/datasets/classic/length_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        line_length = pickle.load(fp)

    with open(f'../metrics/datasets/classic/square_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        square = pickle.load(fp)

    with open(f'../metrics/datasets/classic/volume_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        volume = pickle.load(fp)

    with open(f'../metrics/datasets/classic/s_l_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        s_l = pickle.load(fp)

    with open(f'../metrics/datasets/classic/v_s_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        v_s = pickle.load(fp)

    with open(f'../metrics/datasets/classic/v_l_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        v_l = pickle.load(fp)

    with open(f'../metrics/datasets/classic/v_v_base_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        v_v_base = pickle.load(fp)

    with open(f'../metrics/datasets/classic/fractal_span_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        fractal_span = pickle.load(fp)

    # # TODO: разкомментировать по необходиомости
    # # Производим интерполяцию по найденным метрикам
    # y_length = make_interpolation(iterations, line_length)
    # y_square = make_interpolation(iterations, square)
    # y_volume = make_interpolation(iterations, volume)

    # Строим графики для найденных и апроксимируемыъ метрик.
    fig1, ax1 = plt.subplots()
    ax1.plot(iterations, line_length, 'o', label=r'$a$', c='black', linewidth=1)
    fig2, ax2 = plt.subplots()
    ax2.plot(iterations, square, 'X', label=r'$a$', c='black', linewidth=1)
    fig3, ax3 = plt.subplots()
    ax3.plot(iterations, volume, '*', label=r'$a$', c='black', linewidth=1)
    fig4, ax4 = plt.subplots()
    ax4.plot(iterations, s_l, '*', label=r'$a$', c='black', linewidth=1)
    fig5, ax5 = plt.subplots()
    ax5.plot(iterations, v_s, '*', label=r'$a$', c='black', linewidth=1)
    fig6, ax6 = plt.subplots()
    ax6.plot(iterations, v_l, '*', label=r'$a$', c='black', linewidth=1)
    fig7, ax7 = plt.subplots()
    ax7.plot(iterations, v_v_base, '*', label=r'$a$', c='black', linewidth=1)
    fig8, ax8 = plt.subplots()
    ax8.plot(iterations, fractal_span, '*', label=r'$a$', c='black', linewidth=1)

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    ax5.grid(True)
    ax6.grid(True)
    ax7.grid(True)
    ax8.grid(True)

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

    ax8.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax8.set(xlabel='Число циклов роста, ед.', ylabel='Размах фрактала, ед.')

    fig1.savefig(f'../metrics/graphics/classic/length.png')
    fig2.savefig(f'../metrics/graphics/classic/square.png')
    fig3.savefig(f'../metrics/graphics/classic/value.png')
    fig4.savefig(f'../metrics/graphics/classic/s_l.png')
    fig5.savefig(f'../metrics/graphics/classic/v_s.png')
    fig6.savefig(f'../metrics/graphics/classic/v_l.png')
    fig7.savefig(f'../metrics/graphics/classic/4v1_v0.png')
    fig8.savefig(f'../metrics/graphics/classic/fractal_span.png')

    plt.show()


def build_one_phase(iter_count: int, depth: int):
    with open(f'../metrics/datasets/one_phase/iterations_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        iterations = pickle.load(fp)

    with open(f'../metrics/datasets/one_phase/length_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        line_length = pickle.load(fp)

    with open(f'../metrics/datasets/one_phase/square_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        square = pickle.load(fp)

    with open(f'../metrics/datasets/one_phase/volume_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        volume = pickle.load(fp)

    with open(f'../metrics/datasets/one_phase/s_l_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        s_l = pickle.load(fp)

    with open(f'../metrics/datasets/one_phase/v_s_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        v_s = pickle.load(fp)

    with open(f'../metrics/datasets/one_phase/v_l_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        v_l = pickle.load(fp)

    with open(f'../metrics/datasets/one_phase/v_v_base_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        v_v_base = pickle.load(fp)

    with open(f'../metrics/datasets/one_phase/fractal_span_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        fractal_span = pickle.load(fp)

    # Строим графики для найденных и апроксимируемыъ метрик.
    fig1, ax1 = plt.subplots()
    ax1.plot(iterations, line_length, 'o', label=r'$a$', c='black', linewidth=1)
    fig2, ax2 = plt.subplots()
    ax2.plot(iterations, square, 'X', label=r'$a$', c='black', linewidth=1)
    fig3, ax3 = plt.subplots()
    ax3.plot(iterations, volume, '*', label=r'$a$', c='black', linewidth=1)
    fig4, ax4 = plt.subplots()
    ax4.plot(iterations, s_l, '*', label=r'$a$', c='black', linewidth=1)
    fig5, ax5 = plt.subplots()
    ax5.plot(iterations, v_s, '*', label=r'$a$', c='black', linewidth=1)
    fig6, ax6 = plt.subplots()
    ax6.plot(iterations, v_l, '*', label=r'$a$', c='black', linewidth=1)
    fig7, ax7 = plt.subplots()
    ax7.plot(iterations, v_v_base, '*', label=r'$a$', c='black', linewidth=1)
    fig8, ax8 = plt.subplots()
    ax8.plot(iterations, fractal_span, '*', label=r'$a$', c='black', linewidth=1)

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    ax5.grid(True)
    ax6.grid(True)
    ax7.grid(True)
    ax8.grid(True)

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

    ax8.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax8.set(xlabel='Число циклов роста, ед.', ylabel='Размах фрактала, ед.')

    fig1.savefig(f'../metrics/graphics/one_phase/length.png')
    fig2.savefig(f'../metrics/graphics/one_phase/square.png')
    fig3.savefig(f'../metrics/graphics/one_phase/value.png')
    fig4.savefig(f'../metrics/graphics/one_phase/s_l.png')
    fig5.savefig(f'../metrics/graphics/one_phase/v_s.png')
    fig6.savefig(f'../metrics/graphics/one_phase/v_l.png')
    fig7.savefig(f'../metrics/graphics/one_phase/4v1_v0.png')
    fig8.savefig(f'../metrics/graphics/one_phase/fractal_span.png')

    plt.show()


def build_several_phases(iter_count: int, depth: int, deltas: List[int]):
    with open(f'../metrics/datasets/several_phases/iterations_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}.txt', 'rb') as fp:
        iterations1 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/length_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}.txt', 'rb') as fp:
        line_length1 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/square_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}.txt', 'rb') as fp:
        square1 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/volume_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}.txt', 'rb') as fp:
        volume1 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/s_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}.txt', 'rb') as fp:
        s_l1 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/v_s_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}.txt', 'rb') as fp:
        v_s1 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/v_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}.txt', 'rb') as fp:
        v_l1 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/v_v_base_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}.txt', 'rb') as fp:
        v_v_base1 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/fractal_span_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}.txt', 'rb') as fp:
        fractal_span1 = pickle.load(fp)


    with open(f'../metrics/datasets/several_phases/iterations_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}.txt', 'rb') as fp:
        iterations2 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/length_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}.txt', 'rb') as fp:
        line_length2 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/square_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}.txt', 'rb') as fp:
        square2 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/volume_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}.txt', 'rb') as fp:
        volume2 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/s_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}.txt', 'rb') as fp:
        s_l2 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/v_s_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}.txt', 'rb') as fp:
        v_s2 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/v_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}.txt', 'rb') as fp:
        v_l2 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/v_v_base_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}.txt', 'rb') as fp:
        v_v_base2 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/fractal_span_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}.txt', 'rb') as fp:
        fractal_span2 = pickle.load(fp)


    with open(f'../metrics/datasets/several_phases/iterations_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}.txt', 'rb') as fp:
        iterations3 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/length_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}.txt', 'rb') as fp:
        line_length3 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/square_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}.txt', 'rb') as fp:
        square3 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/volume_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}.txt', 'rb') as fp:
        volume3 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/s_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}.txt', 'rb') as fp:
        s_l3 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/v_s_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}.txt', 'rb') as fp:
        v_s3 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/v_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}.txt', 'rb') as fp:
        v_l3 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/v_v_base_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}.txt', 'rb') as fp:
        v_v_base3 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/fractal_span_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}.txt', 'rb') as fp:
        fractal_span3 = pickle.load(fp)

    # Строим графики для найденных и апроксимируемыъ метрик.
    fig1, ax1 = plt.subplots()
    ax1.plot(iterations1, line_length1, 'o', label=r'$1$', c='black', linewidth=1)
    ax1.plot(iterations2, line_length2, 'o', label=r'$200$', c='red', linewidth=1)
    ax1.plot(iterations3, line_length3, 'o', label=r'$400$', c='blue', linewidth=1)

    fig2, ax2 = plt.subplots()
    ax2.plot(iterations1, square1, 'X', label=r'$1$', c='black', linewidth=1)
    ax2.plot(iterations2, square2, 'X', label=r'$200$', c='red', linewidth=1)
    ax2.plot(iterations3, square3, 'X', label=r'$400$', c='blue', linewidth=1)

    fig3, ax3 = plt.subplots()
    ax3.plot(iterations1, volume1, '*', label=r'$1$', c='black', linewidth=1)
    ax3.plot(iterations2, volume2, '*', label=r'$200$', c='red', linewidth=1)
    ax3.plot(iterations3, volume3, '*', label=r'$400$', c='blue', linewidth=1)

    fig4, ax4 = plt.subplots()
    ax4.plot(iterations1, s_l1, '*',  label=r'$1$', c='black', linewidth=1)
    ax4.plot(iterations2, s_l2, '*', label=r'$200$', c='red', linewidth=1)
    ax4.plot(iterations3, s_l3, '*', label=r'$400$', c='blue', linewidth=1)

    fig5, ax5 = plt.subplots()
    ax5.plot(iterations1, v_s1, '*', label=r'$1$', c='black', linewidth=1)
    ax5.plot(iterations2, v_s2, '*', label=r'$200$', c='red', linewidth=1)
    ax5.plot(iterations3, v_s3, '*', label=r'$400$', c='blue', linewidth=1)

    fig6, ax6 = plt.subplots()
    ax6.plot(iterations1, v_l1, '*', label=r'$1$', c='black', linewidth=1)
    ax6.plot(iterations2, v_l2, '*', label=r'$200$', c='red', linewidth=1)
    ax6.plot(iterations3, v_l3, '*', label=r'$400$', c='blue', linewidth=1)

    fig7, ax7 = plt.subplots()
    ax7.plot(iterations1, v_v_base1, '*', label=r'$1$', c='black', linewidth=1)
    ax7.plot(iterations2, v_v_base2, '*', label=r'$200$', c='red', linewidth=1)
    ax7.plot(iterations3, v_v_base3, '*', label=r'$400$', c='blue', linewidth=1)

    fig8, ax8 = plt.subplots()
    ax8.plot(iterations1, fractal_span1, '*', label=r'$1$', c='black', linewidth=1)
    ax8.plot(iterations2, fractal_span2, '*', label=r'$200$', c='red', linewidth=1)
    ax8.plot(iterations3, fractal_span3, '*', label=r'$400$', c='blue', linewidth=1)

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    ax5.grid(True)
    ax6.grid(True)
    ax7.grid(True)
    ax8.grid(True)

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

    ax8.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax8.set(xlabel='Число циклов роста, ед.', ylabel='Размах фрактала, ед.')

    fig1.savefig(f'../metrics/graphics/several_phases/length.png')
    fig2.savefig(f'../metrics/graphics/several_phases/square.png')
    fig3.savefig(f'../metrics/graphics/several_phases/value.png')
    fig4.savefig(f'../metrics/graphics/several_phases/s_l.png')
    fig5.savefig(f'../metrics/graphics/several_phases/v_s.png')
    fig6.savefig(f'../metrics/graphics/several_phases/v_l.png')
    fig7.savefig(f'../metrics/graphics/several_phases/4v1_v0.png')
    fig8.savefig(f'../metrics/graphics/several_phases/fractal_span.png')

    plt.show()


def build_stochastic(iter_count: int, depth: int, l_rndms: List[float]):
    with open(f'../metrics/datasets/stochasticity/iterations_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
        iterations1 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/length_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
        line_length1 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/square_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
        square1 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/volume_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
        volume1 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/s_l_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
        s_l1 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/v_s_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
        v_s1 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/v_l_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
        v_l1 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/v_v_base_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
        v_v_base1 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/fractal_span_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
        fractal_span1 = pickle.load(fp)


    with open(f'../metrics/datasets/stochasticity/iterations_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
        iterations2 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/length_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
        line_length2 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/square_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
        square2 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/volume_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
        volume2 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/s_l_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
        s_l2 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/v_s_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
        v_s2 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/v_l_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
        v_l2 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/v_v_base_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
        v_v_base2 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/fractal_span_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
        fractal_span2 = pickle.load(fp)


    with open(f'../metrics/datasets/stochasticity/iterations_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
        iterations3 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/length_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
        line_length3 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/square_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
        square3 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/volume_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
        volume3 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/s_l_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
        s_l3 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/v_s_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
        v_s3 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/v_l_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
        v_l3 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/v_v_base_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
        v_v_base3 = pickle.load(fp)

    with open(f'../metrics/datasets/stochasticity/fractal_span_iter_count_{iter_count}_depth_{depth}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
        fractal_span3 = pickle.load(fp)

    # Строим графики для найденных и апроксимируемыъ метрик.
    fig1, ax1 = plt.subplots()
    ax1.plot(iterations1, line_length1, 'o', label=r'$0.6$', c='black', linewidth=1)
    ax1.plot(iterations2, line_length2, 'o', label=r'$0.75$', c='red', linewidth=1)
    ax1.plot(iterations3, line_length3, 'o', label=r'$0.9$', c='blue', linewidth=1)

    fig2, ax2 = plt.subplots()
    ax2.plot(iterations1, square1, 'X', label=r'$0.6$', c='black', linewidth=1)
    ax2.plot(iterations2, square2, 'X', label=r'$0.75$', c='red', linewidth=1)
    ax2.plot(iterations3, square3, 'X', label=r'$0.9$', c='blue', linewidth=1)

    fig3, ax3 = plt.subplots()
    ax3.plot(iterations1, volume1, '*', label=r'$0.6$', c='black', linewidth=1)
    ax3.plot(iterations2, volume2, '*', label=r'$0.75$', c='red', linewidth=1)
    ax3.plot(iterations3, volume3, '*', label=r'$0.9$', c='blue', linewidth=1)

    fig4, ax4 = plt.subplots()
    ax4.plot(iterations1, s_l1, '*', label=r'$0.6$', c='black', linewidth=1)
    ax4.plot(iterations2, s_l2, '*', label=r'$0.75$', c='red', linewidth=1)
    ax4.plot(iterations3, s_l3, '*', label=r'$0.9$', c='blue', linewidth=1)

    fig5, ax5 = plt.subplots()
    ax5.plot(iterations1, v_s1, '*', label=r'$0.6$', c='black', linewidth=1)
    ax5.plot(iterations2, v_s2, '*', label=r'$0.75$', c='red', linewidth=1)
    ax5.plot(iterations3, v_s3, '*', label=r'$0.9$', c='blue', linewidth=1)

    fig6, ax6 = plt.subplots()
    ax6.plot(iterations1, v_l1, '*', label=r'$0.6$', c='black', linewidth=1)
    ax6.plot(iterations2, v_l2, '*', label=r'$0.75$', c='red', linewidth=1)
    ax6.plot(iterations3, v_l3, '*', label=r'$0.9$', c='blue', linewidth=1)

    fig7, ax7 = plt.subplots()
    ax7.plot(iterations1, v_v_base1, '*', label=r'$0.6$', c='black', linewidth=1)
    ax7.plot(iterations2, v_v_base2, '*', label=r'$0.75$', c='red', linewidth=1)
    ax7.plot(iterations3, v_v_base3, '*', label=r'$0.9$', c='blue', linewidth=1)

    fig8, ax8 = plt.subplots()
    ax8.plot(iterations1, fractal_span1, '*', label=r'$0.6$', c='black', linewidth=1)
    ax8.plot(iterations2, fractal_span2, '*', label=r'$0.75$', c='red', linewidth=1)
    ax8.plot(iterations3, fractal_span3, '*', label=r'$0.9$', c='blue', linewidth=1)

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    ax5.grid(True)
    ax6.grid(True)
    ax7.grid(True)
    ax8.grid(True)

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

    ax8.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax8.set(xlabel='Число циклов роста, ед.', ylabel='Размах фрактала, ед.')

    fig1.savefig(f'../metrics/graphics/stochasticity/length.png')
    fig2.savefig(f'../metrics/graphics/stochasticity/square.png')
    fig3.savefig(f'../metrics/graphics/stochasticity/value.png')
    fig4.savefig(f'../metrics/graphics/stochasticity/s_l.png')
    fig5.savefig(f'../metrics/graphics/stochasticity/v_s.png')
    fig6.savefig(f'../metrics/graphics/stochasticity/v_l.png')
    fig7.savefig(f'../metrics/graphics/stochasticity/4v1_v0.png')
    fig8.savefig(f'../metrics/graphics/stochasticity/fractal_span.png')

    plt.show()


if __name__ == '__main__':
    build_classic_one_phase(1000, 7)
    build_one_phase(1000, 7)
    build_several_phases(1000, 7, [1, 200, 400])
    build_stochastic(1000, 7, [0.6, 0.75, 0.9])
