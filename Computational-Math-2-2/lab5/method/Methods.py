from numpy import sqrt


# Метод Эйлера для численного решения
def Euler(f, eps: float, x0: float, xn: float, y0: float):
    h = sqrt(eps)
    y = y0
    x = x0
    x_values = [x]
    y_values = [y]

    while x <= xn:
        y += h * f.solve(x,y)
        x += h
        x=round(x,7)

        x_values.append(x)
        y_values.append(y)

    return x_values, y_values

def Analytical(f, eps: float, x0:float,xn:float):
    h = sqrt(eps)
    x = x0
    x_values = [x]
    y_values = [f.solve2(x)]

    while x <= xn:
        x += h
        y = f.solve2(x)
        
        x=round(x,7)

        x_values.append(x)
        y_values.append(y)

    return x_values, y_values