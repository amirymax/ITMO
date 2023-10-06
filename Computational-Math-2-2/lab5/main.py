import matplotlib.pyplot as plt
from functions import FirstFunction, SecondFunction
from method import Methods
from interpolation import SplineInterpolation

# def plot(x_values: list, y_values: list, label):
#     # plt.figure(figsize=(8, 8))
#     plt.plot(x_values, y_values, label=label)
    


def get_func_num():
    try:
        n = int(
            input(
                "Выберите вариант:\n1: y' = e^(-sin(x)) - y*cos(x)\n2: y' = 3*x^2 - 2*x^4 + 2*x*y\n"
            )
        )
        if n != 1 and n != 2:
            raise ValueError
        return n
    except ValueError:
        print("Нужно ввести 1 или 2")
        return get_func_num()


functions = {1: FirstFunction, 2: SecondFunction}

if __name__ == "__main__":
    # Определени номера функции
    func_num = get_func_num()
    # Начальные условия
    x0 = float(input("Введите начало отрезка: "))
    xn = float(input("Введите конец отрезка: "))
    y0 = float(input("Введите y0: "))
    eps = float(input("Введите точность: "))

    # Решение методом Эйлера
    x_values, y_values = Methods.Euler(functions[func_num], eps, x0, xn, y0)

    # Аппроксимация
    xs_values, ys_values = SplineInterpolation.solve(functions[func_num], eps, x0,xn,y0)

    # Аналитическое решение
    # xa_values, ya_values = Methods.Analytical(functions[func_num], eps, x0,xn)
    
    print(f"X: {x_values[-1]}  Y: {y_values[-1]}")
    # print(f"XS: {xs_values[-1]}  YS: {ys_values[-1]}")

    plt.plot(x_values, y_values,'o', label='Решение методом Эйлера')
    plt.plot(xs_values, ys_values, label='Аппроксимация методом сплайнов')
    # plt.plot(xa_values, ya_values, label='Аналитическое решение')
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Решение дифференциального уравнения методом Эйлера")
    plt.legend()
    plt.grid(True)
    plt.show()