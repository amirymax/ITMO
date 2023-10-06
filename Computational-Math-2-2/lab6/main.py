import numpy as np
import matplotlib.pyplot as plt
from methods import Chebyshev_nodes, Lagrange

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
        print("Введите число от 1 до 3")
        return get_func_num()

functions = {1: f1, 2: f2, 3: f3}

if __name__ == "__main__":
    func_num = get_func_num()
    n = int(input("Количество узлов: "))
    a = float(input("a: "))
    b = float(input("b: "))
    f = functions[func_num]
    
    # Используем равномерное разбиение для x_axis
    x_axis = np.linspace(a, b, n);y_axis = [f(x_i) for x_i in x_axis]; xx_values=x_axis;yy_axis=y_axis
    
    if func_num==3:
        n=1000
    # Увеличиваем количество точек для x_values
    x_values = np.linspace(a, b, n)
    y_true = [f(x_i) for x_i in x_values]
    y_interp = Lagrange(x_axis, y_axis, x_values)
    
    error_i = np.max(np.abs(np.array(y_interp) - np.array(y_true)))
    error_i_index = np.argmax(np.abs(np.array(y_interp) - np.array(y_true)))
    error_p = (x_values[error_i_index], y_interp[error_i_index])
    
    plt.plot(x_axis, y_axis, 'o', label='Заданные точки')
    plt.plot(xx_values, yy_axis, label='Интерполированная функция')
    plt.plot(x_values, y_true, label='Исходная функция')
    
    plt.plot(error_p[0], error_p[1], 'ro', label=f'Максимальное отклонение: {error_i:.4f}')
    
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
