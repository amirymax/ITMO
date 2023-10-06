#Laba 3

import math
import sys


class Result:
    error_message = "Выбранная функция имеет неустранимый разрыв на заданном интервале"
    has_discontinuity = False

    def first_function(x: float):
        return 1 / x if x != 0 else float('inf')

    def second_function(x: float):
        if x == 0:
            return (math.sin(Result.eps) / Result.eps + math.sin(-Result.eps) / -Result.eps) / 2
        return math.sin(x) / x

    def third_function(x: float):
        return x * x + 2

    def fourth_function(x: float):
        return 2 * x + 2

    def five_function(x: float):
        return math.log(x) if x > 0 else float('inf')


    def six_function(x: float):
        return (x**2-3*x-10)/(x-5) if x!=5 else 7

    def get_function(n: int):
        if n == 1:
            return Result.first_function
        elif n == 2:
            return Result.second_function
        elif n == 3:
            return Result.third_function
        elif n == 4:
            return Result.fourth_function
        elif n == 5:
            return Result.five_function
        elif n == 6:
            return Result.five_function
        else:
            print("Такой функции нет, выберите функцию из списка")
            sys.exit(0)

    def calculate_integral(a, b, f, epsilon):
        func = Result.get_function(f)
        n = 1000
        width = (b - a) / n
        integral = 0
        for i in range(0, n):
            x1 = a + i*width
            x2 = a + (i+1)*width
            if func(x1)==float('inf') or func(x2)==float('inf') or func(0.5*(x1+x2))==float('inf') :
              Result.has_discontinuity=True
              return
            integral += (x2-x1)/6.0*(func(x2) + 4.0*func(0.5*(x1+x2)) + func(x1));

        return integral

def solve():
    id = int(input("Выберите функцию для интегрирования:\n"
                   "1)1/x\n"
                   "2)sin(x)/x\n"
                   "3)x²+2\n"
                   "4)2x+2\n"
                   "5)ln(x)\n"
                   "6)(x²-3x-10)/(x-5)\n"
                   ))
    a=float(input("Введите левую границу интегрирования: "))
    b=float(input("Введите правую границу интегрирования: "))
    result = Result.calculate_integral(a, b, id, 0.000001)
    if Result.has_discontinuity:
        print(Result.error_message + '\n')
    else:
        print("Результат численного интегрирования методом Симпсона: ",result)




solve()