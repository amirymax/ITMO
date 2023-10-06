from math import e, sin, cos


def solve(x: float, y: float) -> float:
    return e ** (-sin(x)) - y * cos(x)


def solve2(x: float) -> float:
    return x * e ** (-sin(x))
