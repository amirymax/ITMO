# Лаба 2
import math


def first_equation(x) -> float:
    return math.sin(x) - 0.5


def second_equation(x) -> float:
    return x ** 2 - 6


def third_equation(x) -> float:
    return 2 * math.log(x / 4) / (x ** 2)


def fourth_equation(x) -> float:
    return x ** 3 - x + 4


def first_det(x) -> float:
    return math.cos(x)


def second_det(x) -> float:
    return 3 * x ** 2 - 5.84 * x + 4.435


def third_det(x) -> float:
    if (x == 0):
        x += 0.0001
    return (2 * (x - 2 * x * math.log(x))) / (x ** 4)


def fourth_det(x) -> float:
    return 3 * x ** 2 - 1


def get_equation(n):
    if n == 1:
        return first_equation
    elif n == 2:
        return second_equation
    elif n == 3:
        return third_equation
    elif n == 4:
        return fourth_equation


def bisection_method(equation, interval):
    accuracy = 0.001
    left = interval[0]
    right = interval[1]
    mid = (left + right) / 2
    lenth = right - left
    res = equation(left) * equation(mid)

    while lenth >= accuracy:
        if res < 0:
            right = mid
        else:
            left = mid
        mid = (left + right) / 2
        lenth = right - left
        res = equation(left) * equation(mid)

    return mid


def iter_method(equation, interval):
    left, right = interval

    def g(x):
        return x - equation(x)

    x = (left + right) / 2
    max_iterations = 100
    epsilon = 1e-6

    for _ in range(max_iterations):
        try:
            x_next = g(x)
        except:
            return 'Numerical result out of range'
        if abs(x_next - x) < epsilon:
            return x_next
        x = x_next
    return x_next


def first_function(args: []) -> float:
    return math.sin(args[0])


def second_function(args: []) -> float:
    return (args[0] * args[1]) / 2


def third_function(args: []) -> float:
    return pow(args[0], 2) * pow(args[1], 2) - 3 * pow(args[0], 3) - 6 * pow(args[1], 3) + 8


def fourth_function(args: []) -> float:
    return pow(args[0], 4) - 9 * args[1] + 2


def fifth_function(args: []) -> float:
    return args[0] + pow(args[0], 2) - 2 * args[1] * args[2] - 0.1


def six_function(args: []) -> float:
    return args[1] + pow(args[1], 2) + 3 * args[0] * args[2] + 0.2


def seven_function(args: []) -> float:
    return args[2] + pow(args[2], 2) + 2 * args[0] * args[1] - 0.3


def default_function(args: []) -> float:
    return 0.0


# How to use this function:
# funcs = Result.get_functions(4)
# funcs[0](0.01)
def get_functions(n: int):
    if n == 1:
        return [first_function, second_function]
    elif n == 2:
        return [third_function, fourth_function]
    elif n == 3:
        return [fifth_function, six_function, seven_function]
    else:
        return [default_function]


#
# Complete the 'solve_by_fixed_point_iterations' function below.
#
# The function is expected to return a DOUBLE_ARRAY.
# The function accepts following parameters:
#  1. INTEGER system_id
#  2. INTEGER number_of_unknowns
#  3. DOUBLE_ARRAY initial_approximations
#


def reversion_of_matrix(matrix):
    n = len(matrix)
    a = [[0.0] * n for _ in range(n)]
    b = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            a[i][j] = matrix[i][j]
            if i == j:
                b[i][j] = 1.0

    for k in range(n):
        if a[k][k] == 0.0:
            break

        for j in range(n):
            a[k][j] /= a[k][k]
            b[k][j] /= a[k][k]

        for i in range(n):
            if i == k:
                continue
            for j in range(n):
                a[i][j] -= a[i][k] * a[k][j]
                b[i][j] -= a[i][k] * b[k][j]

    return b


def vector(m, v):
    rows = len(m)
    cols = len(m[0])
    result = [0] * rows
    for i in range(rows):
        row_sum = 0
        for j in range(cols):
            row_sum += m[i][j] * v[j]
        result[i] = row_sum
    return result


def det(a, f, i):
    x = a.copy()
    h = 0.000001
    x[i] += h
    return (f(x) - f(a)) / h


def solve_by_fixed_point_iterations(system_id, number_of_unknowns, initial_approximations):
    funcs = get_functions(system_id)
    result = initial_approximations
    for i in range(100):

        jacobian_matrix = [[det(result, funcs[findx], j) for j in range(number_of_unknowns)] for findx in
                           range(number_of_unknowns)]
        new_f = [fx(result) for fx in funcs]
        for jac in range(number_of_unknowns):
            jacobian_matrix[jac][jac] += 1e-5
        matrix_T = reversion_of_matrix(jacobian_matrix)
        l = vector(matrix_T, new_f)
        result = [result[i] - l[i] for i in range(number_of_unknowns)]
        if all(1e-3 > abs(l[i]) for i in range(number_of_unknowns)):
            return result
    return None


system_id = int(input("1)sin(x)-0,5=0\n"
                      "2)x²-6\n"
                      "3)2ln(x/3)/x²\n"
                      "4)x³-x+4\n"
                      "5)\n  sin(x)\n  xy/2\n"
                      "6)\n  x²y²-3x³-6y³+8\n  x⁴-9y+1\n"
                      "7)\n  x+x²-2yz-0.1\n  y + y²+3*x*z+0.2\n  z+z²+2xy-0.3\n"
                      "Введите номер системы: ").strip())

if (system_id > 0 and system_id < 5):

    interval = list(map(float, input("Введите правую и левую границу: ").split()))
    print("Результат методом деления пополам: " + str(bisection_method(get_equation(system_id), interval)))
    print("Результат методом простой итерации: " + str(iter_method(get_equation(system_id), interval)))

elif system_id > 4 and system_id < 8:
    number_of_unknowns = int(input("Введите количество неизвестных:").strip())
    initial_approximations = []
    for _ in range(number_of_unknowns):
        initial_approximations_item = float(
            input("Введите апроксимацию для неизвестноей под номером " + str(_ + 1) + " : ").strip())
        initial_approximations.append(initial_approximations_item)
    result = solve_by_fixed_point_iterations(system_id - 4, number_of_unknowns, initial_approximations)
    print('  \n'.join(map(str, result)))
