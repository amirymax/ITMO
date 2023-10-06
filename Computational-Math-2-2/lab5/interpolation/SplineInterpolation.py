import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def solve(f, precision:float, a:float,b:float,y0):
    # Создание массива узлов
    x_nodes = np.linspace(a, b, int((b - a) / precision) + 1)
    y_nodes = [y0]

    # Расчет значений функции на узлах
    for i in range(1, len(x_nodes)):
        h = x_nodes[i] - x_nodes[i - 1]
        y_next = y_nodes[-1] + h * f.solve(x_nodes[i - 1], y_nodes[-1])
        y_nodes.append(y_next)

    # Построение кубического сплайна
    spline = CubicSpline(x_nodes, y_nodes, bc_type='natural')

    # Генерация значений для построения графика
    x_plot = np.linspace(a, b, 1000)
    y_plot = spline(x_plot)

    return x_plot, y_plot

