import numpy as np
import pickle
from typing import List

import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


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

    # with open(f'../metrics/datasets/classic/v_s_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
    #     v_s = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/classic/v_l_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
    #     v_l = pickle.load(fp)
    #
    with open(f'../metrics/datasets/classic/v_v_base_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        v_v_base = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/classic/fractal_span_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
    #     fractal_span = pickle.load(fp)

    # # TODO: разкомментировать по необходиомости
    # # Производим интерполяцию по найденным метрикам
    # y_length = make_interpolation(iterations, line_length)
    # y_square = make_interpolation(iterations, square)
    # y_volume = make_interpolation(iterations, volume)

    # Строим графики для найденных и апроксимируемыъ метрик.
    fig1, ax1 = plt.subplots()
    ax1.plot(iterations, line_length, 'o', c='black', linewidth=1)
    fig2, ax2 = plt.subplots()
    ax2.plot(iterations, square, 'o', c='black', linewidth=1)
    fig3, ax3 = plt.subplots()
    ax3.plot(iterations, volume, 'o', c='black', linewidth=1)
    fig4, ax4 = plt.subplots()
    ax4.plot(iterations, s_l, 'o', c='black', linewidth=1)
    # fig5, ax5 = plt.subplots()
    # ax5.plot(iterations, v_s, '*', label=r'$a$', c='black', linewidth=1)
    # fig6, ax6 = plt.subplots()
    # ax6.plot(iterations, v_l, '*', label=r'$a$', c='black', linewidth=1)
    fig7, ax7 = plt.subplots()
    ax7.plot(iterations, v_v_base, 'o', c='black', linewidth=1)
    # fig8, ax8 = plt.subplots()
    # ax8.plot(iterations, fractal_span, '*', label=r'$a$', c='black', linewidth=1)

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    # ax5.grid(True)
    # ax6.grid(True)
    ax7.grid(True)
    # ax8.grid(True)

    def format_fn(tick_val, tick_pos):
        if tick_val == 0:
            return 0

        integer = int(tick_val)
        number_of_digits = len(str(integer))
        if number_of_digits > 2:
            return f'${(tick_val / (10**(number_of_digits - 1))):.1f}*10^{number_of_digits-1}$'

        return tick_val

    # ax1.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax1.set(xlabel='Число циклов роста', ylabel='Длина фрактальной линии, ед.')
    ax1.set_xlabel('Число циклов роста', fontsize=20)
    ax1.set_ylabel('Длина фрактальной линии, ед.', fontsize=20)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.xaxis.set_major_formatter(format_fn)
    ax1.yaxis.set_major_formatter(format_fn)

    # ax2.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax2.set_xlabel('Число циклов роста', fontsize=20)
    ax2.set_ylabel('Площадь фрактала, $ед^2$.', fontsize=20)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.xaxis.set_major_formatter(format_fn)
    # ax2.yaxis.set_major_formatter(format_fn)

    # ax3.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax3.set_xlabel('Число циклов роста', fontsize=20)
    ax3.set_ylabel('Объем фрактала, $ед^3$.', fontsize=20)
    ax3.tick_params(axis='x', labelsize=15)
    ax3.tick_params(axis='y', labelsize=15)
    ax3.xaxis.set_major_formatter(format_fn)
    # ax3.yaxis.set_major_formatter(format_fn)

    # ax4.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax4.set_xlabel('Число циклов роста', fontsize=20)
    ax4.set_ylabel('Отношение площади к длине фрактала, ед.', fontsize=20)
    ax4.tick_params(axis='x', labelsize=15)
    ax4.tick_params(axis='y', labelsize=15)
    ax4.xaxis.set_major_formatter(format_fn)
    # ax4.yaxis.set_major_formatter(format_fn)

    # ax5.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax5.set(xlabel='Число циклов роста, ед.', ylabel='Отношение V/S, ед.')
    #
    # ax6.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax6.set(xlabel='Число циклов роста, ед.', ylabel='Отношение V/L, ед.')
    #
    # ax7.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax7.set(xlabel='Число циклов роста, ед.', ylabel='Отношение 4*V1/V0, ед.')
    ax7.set_xlabel('Число циклов роста', fontsize=20)
    ax7.set_ylabel('Отношение объемов фрактальных поверхностей\nи ограниченного ими тетраэдра, ед.', fontsize=20)
    ax7.tick_params(axis='x', labelsize=15)
    ax7.tick_params(axis='y', labelsize=15)
    ax7.xaxis.set_major_formatter(format_fn)
    #
    # ax8.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax8.set(xlabel='Число циклов роста, ед.', ylabel='Размах фрактала, ед.')

    fig1.savefig(f'../metrics/graphics/classic/length.png')
    fig2.savefig(f'../metrics/graphics/classic/square.png')
    fig3.savefig(f'../metrics/graphics/classic/value.png')
    fig4.savefig(f'../metrics/graphics/classic/s_l.png')
    # fig5.savefig(f'../metrics/graphics/classic/v_s.png')
    # fig6.savefig(f'../metrics/graphics/classic/v_l.png')
    fig7.savefig(f'../metrics/graphics/classic/4v1_v0.png')
    # fig8.savefig(f'../metrics/graphics/classic/fractal_span.png')

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

    # with open(f'../metrics/datasets/one_phase/v_s_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
    #     v_s = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/one_phase/v_l_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
    #     v_l = pickle.load(fp)
    #
    with open(f'../metrics/datasets/one_phase/v_v_base_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
        v_v_base = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/one_phase/fractal_span_iter_count_{iter_count}_depth_{depth}.txt', 'rb') as fp:
    #     fractal_span = pickle.load(fp)

    # Строим графики для найденных и апроксимируемыъ метрик.
    fig1, ax1 = plt.subplots()
    ax1.plot(iterations, line_length, 'o', c='black', linewidth=1)
    fig2, ax2 = plt.subplots()
    ax2.plot(iterations, square, 'o', c='black', linewidth=1)
    fig3, ax3 = plt.subplots()
    ax3.plot(iterations, volume, 'o', c='black', linewidth=1)
    fig4, ax4 = plt.subplots()
    ax4.plot(iterations, s_l, 'o', c='black', linewidth=1)
    # fig5, ax5 = plt.subplots()
    # ax5.plot(iterations, v_s, '*', label=r'$a$', c='black', linewidth=1)
    # fig6, ax6 = plt.subplots()
    # ax6.plot(iterations, v_l, '*', label=r'$a$', c='black', linewidth=1)
    fig7, ax7 = plt.subplots()
    ax7.plot(iterations, v_v_base, 'o', c='black', linewidth=1)
    # fig8, ax8 = plt.subplots()
    # ax8.plot(iterations, fractal_span, '*', label=r'$a$', c='black', linewidth=1)

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    # ax5.grid(True)
    # ax6.grid(True)
    ax7.grid(True)
    # ax8.grid(True)

    def format_fn(tick_val, tick_pos):
        if tick_val == 0:
            return 0

        integer = int(tick_val)
        number_of_digits = len(str(integer))
        if number_of_digits > 2:
            return f'${(tick_val / (10**(number_of_digits - 1))):.1f}*10^{number_of_digits-1}$'

        return tick_val

    # ax1.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax1.set(xlabel='Число циклов роста, ед.', ylabel='Длина фрактальной линии, ед.')
    ax1.set_xlabel('Число циклов роста', fontsize=20)
    ax1.set_ylabel('Длина фрактальной линии, ед.', fontsize=20)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.xaxis.set_major_formatter(format_fn)
    ax1.yaxis.set_major_formatter(format_fn)

    # ax2.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax2.set(xlabel='Число циклов роста, ед.', ylabel='Площадь фрактала, ед.')
    ax2.set_xlabel('Число циклов роста', fontsize=20)
    ax2.set_ylabel('Площадь фрактала, $ед^2$.', fontsize=20)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.xaxis.set_major_formatter(format_fn)
    ax2.yaxis.set_major_formatter(format_fn)

    # ax3.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax3.set(xlabel='Число циклов роста, ед.', ylabel='Объем фрактала, ед.')
    ax3.set_xlabel('Число циклов роста', fontsize=20)
    ax3.set_ylabel('Объем фрактала, $ед^3$.', fontsize=20)
    ax3.tick_params(axis='x', labelsize=15)
    ax3.tick_params(axis='y', labelsize=15)
    ax3.xaxis.set_major_formatter(format_fn)
    ax3.yaxis.set_major_formatter(format_fn)

    # ax4.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax4.set(xlabel='Число циклов роста, ед.', ylabel='Отношение S/L, ед.')
    ax4.set_xlabel('Число циклов роста', fontsize=20)
    ax4.set_ylabel('Отношение площади к длине фрактала, ед.', fontsize=20)
    ax4.tick_params(axis='x', labelsize=15)
    ax4.tick_params(axis='y', labelsize=15)
    ax4.xaxis.set_major_formatter(format_fn)
    ax4.yaxis.set_major_formatter(format_fn)

    # ax5.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax5.set(xlabel='Число циклов роста, ед.', ylabel='Отношение V/S, ед.')
    #
    # ax6.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax6.set(xlabel='Число циклов роста, ед.', ylabel='Отношение V/L, ед.')
    #
    # ax7.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax7.set(xlabel='Число циклов роста, ед.', ylabel='Отношение 4*V1/V0, ед.')
    ax7.set_xlabel('Число циклов роста', fontsize=20)
    ax7.set_ylabel('Отношение объемов фрактальных поверхностей\nи ограниченного ими тетраэдра, ед.', fontsize=20)
    ax7.tick_params(axis='x', labelsize=15)
    ax7.tick_params(axis='y', labelsize=15)
    ax7.xaxis.set_major_formatter(format_fn)
    #
    # ax8.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax8.set(xlabel='Число циклов роста, ед.', ylabel='Размах фрактала, ед.')

    fig1.savefig(f'../metrics/graphics/one_phase/length.png')
    fig2.savefig(f'../metrics/graphics/one_phase/square.png')
    fig3.savefig(f'../metrics/graphics/one_phase/value.png')
    fig4.savefig(f'../metrics/graphics/one_phase/s_l.png')
    # fig5.savefig(f'../metrics/graphics/one_phase/v_s.png')
    # fig6.savefig(f'../metrics/graphics/one_phase/v_l.png')
    fig7.savefig(f'../metrics/graphics/one_phase/4v1_v0.png')
    # fig8.savefig(f'../metrics/graphics/one_phase/fractal_span.png')

    plt.show()


def build_several_phases(iter_count: int, depth: int, deltas: List[int]):
    # with open(f'../metrics/datasets/several_phases/iterations_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}.txt', 'rb') as fp:
    #     iterations1 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/several_phases/length_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}.txt', 'rb') as fp:
    #     line_length1 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/several_phases/square_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}.txt', 'rb') as fp:
    #     square1 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/several_phases/volume_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}.txt', 'rb') as fp:
    #     volume1 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/several_phases/s_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}.txt', 'rb') as fp:
    #     s_l1 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/several_phases/v_s_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}.txt', 'rb') as fp:
    #     v_s1 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/several_phases/v_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}.txt', 'rb') as fp:
    #     v_l1 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/several_phases/v_v_base_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}.txt', 'rb') as fp:
    #     v_v_base1 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/several_phases/fractal_span_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}.txt', 'rb') as fp:
    #     fractal_span1 = pickle.load(fp)


    with open(f'../metrics/datasets/several_phases/iterations_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}.txt', 'rb') as fp:
        iterations2 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/length_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}.txt', 'rb') as fp:
        line_length2 = pickle.load(fp)

    # with open(f'../metrics/datasets/several_phases/square_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}.txt', 'rb') as fp:
    #     square2 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/several_phases/volume_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}.txt', 'rb') as fp:
    #     volume2 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/s_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}.txt', 'rb') as fp:
        s_l2 = pickle.load(fp)

    # with open(f'../metrics/datasets/several_phases/v_s_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}.txt', 'rb') as fp:
    #     v_s2 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/several_phases/v_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}.txt', 'rb') as fp:
    #     v_l2 = pickle.load(fp)
    #
    with open(f'../metrics/datasets/several_phases/v_v_base_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}.txt', 'rb') as fp:
        v_v_base2 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/several_phases/fractal_span_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}.txt', 'rb') as fp:
    #     fractal_span2 = pickle.load(fp)


    with open(f'../metrics/datasets/several_phases/iterations_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}.txt', 'rb') as fp:
        iterations3 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/length_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}.txt', 'rb') as fp:
        line_length3 = pickle.load(fp)

    # with open(f'../metrics/datasets/several_phases/square_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}.txt', 'rb') as fp:
    #     square3 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/several_phases/volume_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}.txt', 'rb') as fp:
    #     volume3 = pickle.load(fp)

    with open(f'../metrics/datasets/several_phases/s_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}.txt', 'rb') as fp:
        s_l3 = pickle.load(fp)

    # with open(f'../metrics/datasets/several_phases/v_s_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}.txt', 'rb') as fp:
    #     v_s3 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/several_phases/v_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}.txt', 'rb') as fp:
    #     v_l3 = pickle.load(fp)
    #
    with open(f'../metrics/datasets/several_phases/v_v_base_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}.txt', 'rb') as fp:
        v_v_base3 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/several_phases/fractal_span_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}.txt', 'rb') as fp:
    #     fractal_span3 = pickle.load(fp)

    # Строим графики для найденных и апроксимируемыъ метрик.
    fig1, ax1 = plt.subplots()
    # ax1.plot(iterations1, line_length1, 'o', label=r'$1$', c='black', linewidth=1)
    ax1.plot(iterations2, line_length2, 'o', label=r'$a$', c='black', linewidth=1)
    ax1.plot(iterations3, line_length3, 'o', label=r'$b$', c='black', linewidth=1)

    # fig2, ax2 = plt.subplots()
    # ax2.plot(iterations1, square1, 'X', label=r'$1$', c='black', linewidth=1)
    # ax2.plot(iterations2, square2, 'X', label=r'$200$', c='red', linewidth=1)
    # ax2.plot(iterations3, square3, 'X', label=r'$400$', c='blue', linewidth=1)
    #
    # fig3, ax3 = plt.subplots()
    # ax3.plot(iterations1, volume1, '*', label=r'$1$', c='black', linewidth=1)
    # ax3.plot(iterations2, volume2, '*', label=r'$200$', c='red', linewidth=1)
    # ax3.plot(iterations3, volume3, '*', label=r'$400$', c='blue', linewidth=1)

    fig4, ax4 = plt.subplots()
    # ax4.plot(iterations1, s_l1, '*',  label=r'$1$', c='black', linewidth=1)
    ax4.plot(iterations2, s_l2, 'o', label=r'$a$', c='black', linewidth=1)
    ax4.plot(iterations3, s_l3, 'o', label=r'$b$', c='black', linewidth=1)

    # fig5, ax5 = plt.subplots()
    # ax5.plot(iterations1, v_s1, '*', label=r'$1$', c='black', linewidth=1)
    # ax5.plot(iterations2, v_s2, '*', label=r'$200$', c='red', linewidth=1)
    # ax5.plot(iterations3, v_s3, '*', label=r'$400$', c='blue', linewidth=1)
    #
    # fig6, ax6 = plt.subplots()
    # ax6.plot(iterations1, v_l1, '*', label=r'$1$', c='black', linewidth=1)
    # ax6.plot(iterations2, v_l2, '*', label=r'$200$', c='red', linewidth=1)
    # ax6.plot(iterations3, v_l3, '*', label=r'$400$', c='blue', linewidth=1)
    #
    fig7, ax7 = plt.subplots()
    # ax7.plot(iterations1, v_v_base1, '*', label=r'$a$', c='black', linewidth=1)
    ax7.plot(iterations2, v_v_base2, 'o', label=r'$a$', c='black', linewidth=1)
    ax7.plot(iterations3, v_v_base3, 'o', label=r'$a$', c='black', linewidth=1)
    #
    # fig8, ax8 = plt.subplots()
    # ax8.plot(iterations1, fractal_span1, '*', label=r'$1$', c='black', linewidth=1)
    # ax8.plot(iterations2, fractal_span2, '*', label=r'$200$', c='red', linewidth=1)
    # ax8.plot(iterations3, fractal_span3, '*', label=r'$400$', c='blue', linewidth=1)

    ax1.grid(True)
    # ax2.grid(True)
    # ax3.grid(True)
    ax4.grid(True)
    # ax5.grid(True)
    # ax6.grid(True)
    ax7.grid(True)
    # ax8.grid(True)

    def format_fn(tick_val, tick_pos):
        if tick_val == 0:
            return 0

        integer = int(tick_val)
        number_of_digits = len(str(integer))
        if number_of_digits > 2:
            return f'${(tick_val / (10**(number_of_digits - 1))):.1f}*10^{number_of_digits-1}$'

        return tick_val

    # ax1.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax1.set(xlabel='Число циклов роста, ед.', ylabel='Длина фрактальной линии, ед.')
    ax1.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax1.set_xlabel('Число циклов роста', fontsize=20)
    ax1.set_ylabel('Длина фрактальной линии, ед.', fontsize=20)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.xaxis.set_major_formatter(format_fn)
    ax1.yaxis.set_major_formatter(format_fn)

    # ax2.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax2.set(xlabel='Число циклов роста, ед.', ylabel='Площадь фрактала, ед.')
    #
    # ax3.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax3.set(xlabel='Число циклов роста, ед.', ylabel='Объем фрактала, ед.')

    # ax4.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax4.set(xlabel='Число циклов роста, ед.', ylabel='Отношение S/L, ед.')
    ax4.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax4.set_xlabel('Число циклов роста', fontsize=20)
    ax4.set_ylabel('Отношение площади к длине фрактала, ед.', fontsize=20)
    ax4.tick_params(axis='x', labelsize=15)
    ax4.tick_params(axis='y', labelsize=15)
    ax4.xaxis.set_major_formatter(format_fn)

    #
    # ax5.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax5.set(xlabel='Число циклов роста, ед.', ylabel='Отношение V/S, ед.')
    #
    # ax6.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax6.set(xlabel='Число циклов роста, ед.', ylabel='Отношение V/L, ед.')
    #
    # ax7.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax7.set(xlabel='Число циклов роста, ед.', ylabel='Отношение 4*V1/V0, ед.')
    ax7.set_xlabel('Число циклов роста', fontsize=20)
    ax7.set_ylabel('Отношение объемов фрактальных поверхностей\nи ограниченного ими тетраэдра, ед.', fontsize=20)
    ax7.tick_params(axis='x', labelsize=15)
    ax7.tick_params(axis='y', labelsize=15)
    ax7.xaxis.set_major_formatter(format_fn)
    #
    # ax8.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax8.set(xlabel='Число циклов роста, ед.', ylabel='Размах фрактала, ед.')

    fig1.savefig(f'../metrics/graphics/several_phases/length.png')
    # fig2.savefig(f'../metrics/graphics/several_phases/square.png')
    # fig3.savefig(f'../metrics/graphics/several_phases/value.png')
    fig4.savefig(f'../metrics/graphics/several_phases/s_l.png')
    # fig5.savefig(f'../metrics/graphics/several_phases/v_s.png')
    # fig6.savefig(f'../metrics/graphics/several_phases/v_l.png')
    fig7.savefig(f'../metrics/graphics/several_phases/4v1_v0.png')
    # fig8.savefig(f'../metrics/graphics/several_phases/fractal_span.png')

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


def build_combined(iter_count: int, depth: int, deltas: List[int], l_rndms: List[float]):
    # # 0--------------------------------------------------------------------------------------------------------------------------------------------------
    # with open(f'../metrics/datasets/combined/iterations_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     iterations00 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/length_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     line_length00 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/square_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     square00 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/volume_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     volume00 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/s_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     s_l00 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_s_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     v_s00 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     v_l00 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_v_base_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     v_v_base00 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/fractal_span_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     fractal_span00 = pickle.load(fp)


    # with open(f'../metrics/datasets/combined/iterations_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     iterations01 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/length_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     line_length01 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/square_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     square01 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/volume_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     volume01 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/s_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     s_l01 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_s_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     v_s01 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     v_l01 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_v_base_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     v_v_base01 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/fractal_span_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     fractal_span01 = pickle.load(fp)


    # with open(f'../metrics/datasets/combined/iterations_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     iterations02 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/length_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     line_length02 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/square_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     square02 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/volume_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     volume02 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/s_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     s_l02 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_s_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     v_s02 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     v_l02 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_v_base_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     v_v_base02 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/fractal_span_iter_count_{iter_count}_depth_{depth}_delta_{deltas[0]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     fractal_span02 = pickle.load(fp)
    # # 0---------------------------------------------------------------------------------------------------------------------------------------------------

    # # 1---------------------------------------------------------------------------------------------------------------------------------------------------
    # with open(f'../metrics/datasets/combined/iterations_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     iterations10 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/length_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     line_length10 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/square_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     square10 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/volume_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     volume10 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/s_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     s_l10 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_s_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     v_s10 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     v_l10 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_v_base_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     v_v_base10 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/fractal_span_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     fractal_span10 = pickle.load(fp)
    #
    #
    # with open(f'../metrics/datasets/combined/iterations_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     iterations11 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/length_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     line_length11 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/square_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     square11 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/volume_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     volume11 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/s_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     s_l11 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_s_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     v_s11 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     v_l11 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_v_base_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     v_v_base11 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/fractal_span_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     fractal_span11 = pickle.load(fp)
    #
    #
    # with open(f'../metrics/datasets/combined/iterations_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     iterations12 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/length_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     line_length12 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/square_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     square12 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/volume_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     volume12 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/s_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     s_l12 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_s_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     v_s12 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     v_l12 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_v_base_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     v_v_base12 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/fractal_span_iter_count_{iter_count}_depth_{depth}_delta_{deltas[1]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     fractal_span12 = pickle.load(fp)
    # # 1-------------------------------------------------------------------------------------------------------------------------------------------------------


    # 2-------------------------------------------------------------------------------------------------------------------------------------------------------
    with open(f'../metrics/datasets/combined/iterations_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
        iterations20 = pickle.load(fp)

    with open(f'../metrics/datasets/combined/length_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
        line_length20 = pickle.load(fp)

    # with open(f'../metrics/datasets/combined/square_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     square20 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/volume_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     volume20 = pickle.load(fp)

    with open(f'../metrics/datasets/combined/s_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
        s_l20 = pickle.load(fp)

    # with open(f'../metrics/datasets/combined/v_s_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     v_s20 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     v_l20 = pickle.load(fp)

    with open(f'../metrics/datasets/combined/v_v_base_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
        v_v_base20 = pickle.load(fp)

    # with open(f'../metrics/datasets/combined/fractal_span_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[0]}.txt', 'rb') as fp:
    #     fractal_span20 = pickle.load(fp)


    with open(f'../metrics/datasets/combined/iterations_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
        iterations21 = pickle.load(fp)

    with open(f'../metrics/datasets/combined/length_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
        line_length21 = pickle.load(fp)

    # with open(f'../metrics/datasets/combined/square_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     square21 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/volume_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     volume21 = pickle.load(fp)

    with open(f'../metrics/datasets/combined/s_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
        s_l21 = pickle.load(fp)

    # with open(f'../metrics/datasets/combined/v_s_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     v_s21 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     v_l21 = pickle.load(fp)

    with open(f'../metrics/datasets/combined/v_v_base_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
        v_v_base21 = pickle.load(fp)

    # with open(f'../metrics/datasets/combined/fractal_span_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[1]}.txt', 'rb') as fp:
    #     fractal_span21 = pickle.load(fp)


    with open(f'../metrics/datasets/combined/iterations_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
        iterations22 = pickle.load(fp)

    with open(f'../metrics/datasets/combined/length_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
        line_length22 = pickle.load(fp)

    # with open(f'../metrics/datasets/combined/square_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     square22 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/volume_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     volume22 = pickle.load(fp)

    with open(f'../metrics/datasets/combined/s_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
        s_l22 = pickle.load(fp)

    # with open(f'../metrics/datasets/combined/v_s_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     v_s22 = pickle.load(fp)
    #
    # with open(f'../metrics/datasets/combined/v_l_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     v_l22 = pickle.load(fp)

    with open(f'../metrics/datasets/combined/v_v_base_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
        v_v_base22 = pickle.load(fp)

    # with open(f'../metrics/datasets/combined/fractal_span_iter_count_{iter_count}_depth_{depth}_delta_{deltas[2]}_l_rnd_{l_rndms[2]}.txt', 'rb') as fp:
    #     fractal_span22 = pickle.load(fp)
    # 2-------------------------------------------------------------------------------------------------------------------------------------------------------

    # Строим графики для найденных и апроксимируемыъ метрик.
    fig1, ax1 = plt.subplots()
    # ax1.plot(iterations00, line_length00, 'o', label=r'$1;0.6$', c='black', linewidth=1)
    # ax1.plot(iterations01, line_length01, 'o', label=r'$1;0.75$', c='red', linewidth=1)
    # ax1.plot(iterations02, line_length02, 'o', label=r'$1;0.9$', c='blue', linewidth=1)
    # ax1.plot(iterations10, line_length10, 'o', label=r'$200;0.6$', c='green', linewidth=1)
    # ax1.plot(iterations11, line_length11, 'o', label=r'$200;0.75$', c='violet', linewidth=1)
    # ax1.plot(iterations12, line_length12, 'o', label=r'$200;0.9$', c='yellow', linewidth=1)
    ax1.plot(iterations20, line_length20, 'o', label=r'$a$', c='black', linewidth=1)
    ax1.plot(iterations21, line_length21, 'o', label=r'$b$', c='black', linewidth=1)
    ax1.plot(iterations22, line_length22, 'o', label=r'$c$', c='black', linewidth=1)

    # fig2, ax2 = plt.subplots()
    # ax2.plot(iterations00, square00, 'X', label=r'$1;0.6$', c='black', linewidth=1)
    # ax2.plot(iterations01, square01, 'X', label=r'$1;0.75$', c='red', linewidth=1)
    # ax2.plot(iterations02, square02, 'X', label=r'$1;0.9$', c='blue', linewidth=1)
    # ax2.plot(iterations10, square10, 'X', label=r'$200;0.9$', c='green', linewidth=1)
    # ax2.plot(iterations11, square11, 'X', label=r'$200;0.9$', c='violet', linewidth=1)
    # ax2.plot(iterations12, square12, 'X', label=r'$200;0.9$', c='yellow', linewidth=1)
    # ax2.plot(iterations20, square20, 'X', label=r'$400;0.9$', c='orange', linewidth=1)
    # ax2.plot(iterations21, square21, 'X', label=r'$400;0.9$', c='pink', linewidth=1)
    # ax2.plot(iterations22, square22, 'X', label=r'$400;0.9$', c='brown', linewidth=1)
    #
    # fig3, ax3 = plt.subplots()
    # ax3.plot(iterations00, volume00, '*', label=r'$1;0.6$', c='black', linewidth=1)
    # ax3.plot(iterations01, volume01, '*', label=r'$1;0.75$', c='red', linewidth=1)
    # ax3.plot(iterations02, volume02, '*', label=r'$1;0.9$', c='blue', linewidth=1)
    # ax3.plot(iterations10, volume10, '*', label=r'$200;0.6$', c='green', linewidth=1)
    # ax3.plot(iterations11, volume11, '*', label=r'$200;0.75$', c='violet', linewidth=1)
    # ax3.plot(iterations12, volume12, '*', label=r'$200;0.9$', c='yellow', linewidth=1)
    # ax3.plot(iterations20, volume20, '*', label=r'$400;0.6$', c='orange', linewidth=1)
    # ax3.plot(iterations21, volume21, '*', label=r'$400;0.75$', c='pink', linewidth=1)
    # ax3.plot(iterations22, volume22, '*', label=r'$400;0.9$', c='brown', linewidth=1)

    fig4, ax4 = plt.subplots()
    # ax4.plot(iterations00, s_l00, '*', label=r'$1;0.6$', c='black', linewidth=1)
    # ax4.plot(iterations01, s_l01, '*', label=r'$1;0.75$', c='red', linewidth=1)
    # ax4.plot(iterations02, s_l02, '*', label=r'$1;0.9$', c='blue', linewidth=1)
    # ax4.plot(iterations10, s_l10, '*', label=r'$200;0.6$', c='green', linewidth=1)
    # ax4.plot(iterations11, s_l11, '*', label=r'$200;0.75$', c='violet', linewidth=1)
    # ax4.plot(iterations12, s_l12, '*', label=r'$200;0.9$', c='yellow', linewidth=1)
    ax4.plot(iterations20, s_l20, 'o', label=r'$a$', c='black', linewidth=1)
    ax4.plot(iterations21, s_l21, 'o', label=r'$b$', c='black', linewidth=1)
    ax4.plot(iterations22, s_l22, 'o', label=r'$c$', c='black', linewidth=1)

    # fig5, ax5 = plt.subplots()
    # ax5.plot(iterations00, v_s00, '*', label=r'$1;0.6$', c='black', linewidth=1)
    # ax5.plot(iterations01, v_s01, '*', label=r'$1;0.75$', c='red', linewidth=1)
    # ax5.plot(iterations02, v_s02, '*', label=r'$1;0.9$', c='blue', linewidth=1)
    # ax5.plot(iterations10, v_s10, '*', label=r'$200;0.6$', c='green', linewidth=1)
    # ax5.plot(iterations11, v_s11, '*', label=r'$200;0.75$', c='violet', linewidth=1)
    # ax5.plot(iterations12, v_s12, '*', label=r'$200;0.9$', c='yellow', linewidth=1)
    # ax5.plot(iterations20, v_s20, '*', label=r'$400;0.6$', c='orange', linewidth=1)
    # ax5.plot(iterations21, v_s21, '*', label=r'$400;0.75$', c='pink', linewidth=1)
    # ax5.plot(iterations22, v_s22, '*', label=r'$400;0.9$', c='brown', linewidth=1)

    # fig6, ax6 = plt.subplots()
    # ax6.plot(iterations00, v_l00, '*', label=r'$1;0.6$', c='black', linewidth=1)
    # ax6.plot(iterations01, v_l01, '*', label=r'$1;0.75$', c='red', linewidth=1)
    # ax6.plot(iterations02, v_l02, '*', label=r'$1;0.9$', c='blue', linewidth=1)
    # ax6.plot(iterations10, v_l10, '*', label=r'$200;0.6$', c='green', linewidth=1)
    # ax6.plot(iterations11, v_l11, '*', label=r'$200;0.75$', c='violet', linewidth=1)
    # ax6.plot(iterations12, v_l12, '*', label=r'$200;0.9$', c='yellow', linewidth=1)
    # ax6.plot(iterations20, v_l20, '*', label=r'$400;0.6$', c='orange', linewidth=1)
    # ax6.plot(iterations21, v_l21, '*', label=r'$400;0.75$', c='pink', linewidth=1)
    # ax6.plot(iterations22, v_l22, '*', label=r'$400;0.9$', c='brown', linewidth=1)

    fig7, ax7 = plt.subplots()
    # ax7.plot(iterations00, v_v_base00, '*', label=r'$1;0.6$', c='black', linewidth=1)
    # ax7.plot(iterations01, v_v_base01, '*', label=r'$1;0.75$', c='red', linewidth=1)
    # ax7.plot(iterations02, v_v_base02, '*', label=r'$1;0.9$', c='blue', linewidth=1)
    # ax7.plot(iterations10, v_v_base10, '*', label=r'$200;0.6$', c='green', linewidth=1)
    # ax7.plot(iterations11, v_v_base11, '*', label=r'$200;0.75$', c='violet', linewidth=1)
    # ax7.plot(iterations12, v_v_base12, '*', label=r'$200;0.9$', c='yellow', linewidth=1)
    ax7.plot(iterations20, v_v_base20, 'o', label=r'$a$', c='black', linewidth=1)
    ax7.plot(iterations21, v_v_base21, 'o', label=r'$b$', c='black', linewidth=1)
    ax7.plot(iterations22, v_v_base22, 'o', label=r'$c$', c='black', linewidth=1)

    # fig8, ax8 = plt.subplots()
    # ax8.plot(iterations00, fractal_span00, '*', label=r'$1;0.6$', c='black', linewidth=1)
    # ax8.plot(iterations01, fractal_span01, '*', label=r'$1;0.75$', c='red', linewidth=1)
    # ax8.plot(iterations02, fractal_span02, '*', label=r'$1;0.9$', c='blue', linewidth=1)
    # ax8.plot(iterations10, fractal_span10, '*', label=r'$200;0.6$', c='green', linewidth=1)
    # ax8.plot(iterations11, fractal_span11, '*', label=r'$200;0.75$', c='violet', linewidth=1)
    # ax8.plot(iterations12, fractal_span12, '*', label=r'$200;0.9$', c='yellow', linewidth=1)
    # ax8.plot(iterations20, fractal_span20, '*', label=r'$400;0.6$', c='orange', linewidth=1)
    # ax8.plot(iterations21, fractal_span21, '*', label=r'$400;0.75$', c='pink', linewidth=1)
    # ax8.plot(iterations22, fractal_span22, '*', label=r'$400;0.9$', c='brown', linewidth=1)

    ax1.grid(True)
    # ax2.grid(True)
    # ax3.grid(True)
    ax4.grid(True)
    # ax5.grid(True)
    # ax6.grid(True)
    ax7.grid(True)
    # ax8.grid(True)

    def format_fn(tick_val, tick_pos):
        if tick_val == 0:
            return 0

        integer = int(tick_val)
        number_of_digits = len(str(integer))
        if number_of_digits > 2:
            return f'${(tick_val / (10**(number_of_digits - 1))):.1f}*10^{number_of_digits-1}$'

        return tick_val

    # ax1.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax1.set(xlabel='Число циклов роста, ед.', ylabel='Длина фрактальной линии, ед.')
    ax1.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax1.set_xlabel('Число циклов роста', fontsize=20)
    ax1.set_ylabel('Длина фрактальной линии, ед.', fontsize=20)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.xaxis.set_major_formatter(format_fn)
    ax1.yaxis.set_major_formatter(format_fn)

    # ax2.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax2.set(xlabel='Число циклов роста, ед.', ylabel='Площадь фрактала, ед.')
    #
    # ax3.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax3.set(xlabel='Число циклов роста, ед.', ylabel='Объем фрактала, ед.')

    # ax4.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax4.set(xlabel='Число циклов роста, ед.', ylabel='Отношение S/L, ед.')
    ax4.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax4.set_xlabel('Число циклов роста', fontsize=20)
    ax4.set_ylabel('Отношение площади к длине фрактала, ед.', fontsize=20)
    ax4.tick_params(axis='x', labelsize=15)
    ax4.tick_params(axis='y', labelsize=15)
    ax4.xaxis.set_major_formatter(format_fn)

    # ax5.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax5.set(xlabel='Число циклов роста, ед.', ylabel='Отношение V/S, ед.')
    #
    # ax6.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax6.set(xlabel='Число циклов роста, ед.', ylabel='Отношение V/L, ед.')

    # ax7.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax7.set(xlabel='Число циклов роста, ед.', ylabel='Отношение $((4*V_1)/V_0)$, ед.')
    ax7.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax7.set_xlabel('Число циклов роста', fontsize=20)
    ax7.set_ylabel('Отношение объемов фрактальных поверхностей\nи ограниченного ими тетраэдра, ед.', fontsize=20)
    ax7.tick_params(axis='x', labelsize=15)
    ax7.tick_params(axis='y', labelsize=15)
    ax7.xaxis.set_major_formatter(format_fn)

    # ax8.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # ax8.set(xlabel='Число циклов роста, ед.', ylabel='Размах фрактала, ед.')

    fig1.savefig(f'../metrics/graphics/combined/length.png')
    # fig2.savefig(f'../metrics/graphics/combined/square.png')
    # fig3.savefig(f'../metrics/graphics/combined/value.png')
    fig4.savefig(f'../metrics/graphics/combined/s_l.png')
    # fig5.savefig(f'../metrics/graphics/combined/v_s.png')
    # fig6.savefig(f'../metrics/graphics/combined/v_l.png')
    fig7.savefig(f'../metrics/graphics/combined/4v1_v0.png')
    # fig8.savefig(f'../metrics/graphics/combined/fractal_span.png')

    plt.show()


if __name__ == '__main__':
    # build_classic_one_phase(1000, 7)
    # build_one_phase(1000, 7)
    build_several_phases(1000, 7, [1, 200, 400])
    # build_stochastic(1000, 7, [0.6, 0.75, 0.9])
    # build_combined(1000, 7, [1, 200, 400], [0.6, 0.75, 0.9])
