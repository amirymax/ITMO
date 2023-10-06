import numpy as np
import matplotlib.pyplot as plt
import math


def interpolate_by_spline(x_axis, y_axis, x):
    num_values = len(x_axis)
    intervals = [x_axis[i+1] - x_axis[i] for i in range(num_values-1)]

    slope_values = [0.0] + [3.0/intervals[i] * (y_axis[i+1]-y_axis[i]) - 3.0/intervals[i-1] * (y_axis[i]-y_axis[i-1]) for i in range(1, num_values-1)]
    first_coefficient = [1] + [0]*(num_values-2)
    second_coefficient = [0] + [0]*(num_values-2)
    third_coefficient = [0] + [0]*(num_values-2)

    for i in range(1, num_values-1):
        first_coefficient[i] = 2.0*(x_axis[i+1]-x_axis[i-1]) - intervals[i-1]*second_coefficient[i-1]
        second_coefficient[i] = intervals[i]/first_coefficient[i]
        third_coefficient[i] = (slope_values[i]-intervals[i-1]*third_coefficient[i-1])/first_coefficient[i]

    first_coefficient.append(1)
    third_coefficient.append(0)
    coefficients = [0]*num_values
    for j in range(num_values-2, -1, -1):
        coefficients[j] = third_coefficient[j] - second_coefficient[j]*coefficients[j+1]
    a_values = [(y_axis[i+1]-y_axis[i])/intervals[i] - intervals[i]*(coefficients[i+1]+2*coefficients[i])/3 for i in range(num_values-1)]
    b_values = [(coefficients[i+1]-coefficients[i])/(3*intervals[i]) for i in range(num_values-1)]

    i = 0
    while i < num_values-1 and x > x_axis[i+1]:
        i += 1

    interpolated_value = y_axis[i] + a_values[i]*(x-x_axis[i]) + coefficients[i]*(x-x_axis[i])**2 + b_values[i]*(x-x_axis[i])**3

    return interpolated_value


function_choice = int(input("Выберите функцию для интерполяции:\n1. x * sin(x)\n2. sin(x)*x+sin(x)/3\n3. sin(x)\n"))

if function_choice == 1:
    print("Введите список данных:")
    x_axis = list(map(float, input().rstrip().split()))
    y_axis = x_axis * (np.sin(x_axis))
    x = float(input("Введите значение x, для которого требуется найти значение интерполяционного полинома: "))
elif function_choice == 2:
    print("Введите список данных:")
    x_axis = list(map(float, input().rstrip().split()))
    y_axis = np.sin(x_axis)* x_axis + np.sin(x_axis)/3
    x = float(input("Введите значение x, для которого требуется найти значение интерполяционного полинома: "))
elif function_choice == 3:
    print("Введите список данных:")
    x_axis = list(map(float, input().rstrip().split()))
    y_axis = np.sin(x_axis)
    x = float(input("Введите значение x, для которого требуется найти значение интерполяционного полинома: "))
else:
    print("Ошибка ввода. Выберите 1, 2 или 3.")
    exit()

interpolated_value = interpolate_by_spline(x_axis, y_axis, x)

print("Значением интерполяционной функции в точке x = ", interpolated_value)


x_values = np.linspace(min(x_axis), max(x_axis), 100)
y_true = [math.sin(x) for x in x_values]
y_interp = [interpolate_by_spline(x_axis, y_axis, x) for x in x_values]

max_error_index = np.argmax(np.abs(np.array(y_interp) - np.array(y_true)))
max_error_point = (x_values[max_error_index], y_interp[max_error_index])


plt.plot(x_axis, y_axis, 'o', label='Заданные точки')
plt.plot(x_values, y_true, label='Исходная функция')
plt.plot(x_values, y_interp, label='Интерполированная функция')
plt.plot(max_error_point[0], max_error_point[1], 'ro', label='Максимальное отклонение')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()




