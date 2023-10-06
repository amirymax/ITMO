import numpy as np
import matplotlib.pyplot as plt
from methods import *

def f1(x):
    return np.sin(np.exp(3 * x))
def f2(x):
    return 1 / (1 + x**2)
def f3(x):
    return np.sin(x)

def get_func_num():
    try:
        n = int(input("Номер функции, где: \n1.sin(e^(3*x))\n2.1/(1+x^2)\n3. sin(x)\n"))
        if n == 1 or n == 2 or n == 3:
            return n
        raise ValueError
    except ValueError:
        print("Введите с=число от 1 до 3")
        return get_func_num()

functions = {1: f1, 2: f2, 3: f3}

if __name__ == "__main__":
    func_num = get_func_num()
    n = int(input("Количество узлов: "))
    a = float(input("a: "))
    b = float(input("b: "))
    f=functions[func_num]
    x_axis, y_axis = Chebyshev_nodes(f,n, a, b)

    # if func_num==3:
    #         n=1000
    x_values = np.linspace(a, b, 1000)
    y_true = [f(x_i) for x_i in x_values]
    y_interp = Lagrange(x_axis,y_axis,x_values)
    error_i = np.argmax(np.abs(np.array(y_interp) - np.array(y_true)))
    error_p = (x_values[error_i], y_interp[error_i])
    plt.plot(x_axis, y_axis, 'o', label='Заданные точки')
    plt.plot(x_values, y_true, label='Исходная функция')
    plt.plot(x_values, y_interp, label='Интерполированная функция')
    plt.plot(error_p[0], error_p[1], 'ro', label='Максимальное отклонение')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
