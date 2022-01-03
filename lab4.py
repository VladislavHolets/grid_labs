import time
import sys
from math import sin, cos
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../')
# fig, ax = plt.subplots()
H = 0.01
m1 = 0.2
m2 = 0.1
l1 = 1.
l2 = 1.
d = 0.2
k = 5
t0 = 0
tmax = 30
deltaT = 0.01
dt1 = 0
dt2 = 0
theta1 = -0.20
theta2 = -0.20
g = 9.81


# In functions below t and u params are unused but was added for same interface
# to pass these functions as functors
def f1(t, x1, x2, u):
    return -g / l1 * sin(x1) + k * d * d / (m1 * l1 * l1) * (sin(x2) - sin(x1)) * cos(x1)


def f2(t, x1, x2, u):
    return -g / l2 * sin(x2) - k * d * d / (m2 * l2 ** 2) * (sin(x2) - sin(x1)) * cos(x2)



# вводимо другу функцію y1 = y' = u
# We have the Dif. equation, kind of y'' = f(x, y, y')
#
# y' = u            (f - func)
# u' = f(x, y, y)   (g - func)
def runge_kutta_4_2(ti, xi, ui, func, h=H):
    # calculations of additional coeficients
    # all Ki are depending on f (only u(y') actually)
    # all Li are depending on g (our y'' function)

    # helper function, depending on u only
    def f(ti, xi, ui):
        return ui

    k1 = h * f(ti, xi, ui)
    l1 = h * func(ti, xi, ui)

    k2 = h * f(ti + h / 2, xi + k1 / 2, ui + l1 / 2)
    l2 = h * func(ti + h / 2, xi + k1 / 2, ui + l1 / 2)

    k3 = h * f(ti + h / 2, xi + k2 / 2, ui + l2 / 2)
    l3 = h * func(ti + h / 2, xi + k2 / 2, ui + l2 / 2)

    k4 = h * f(ti + h, xi + k3, ui + l3)
    l4 = h * func(ti + h, xi + k3, ui + l3)

    dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    du = (l1 + 2 * l2 + 2 * l3 + l4) / 6

    x_ = xi + dx
    u_ = ui + du

    return x_, u_


class Calculations:
    X1 = 0
    X2 = 1


def runge_kutta_4_2_local(calculation, ti, x1i, x2i, ui, func, h=H):
    # calculations of additional coeficients
    # all Ki are depending on f (only u(y') actually)
    # all Li are depending on g (our y'' function)

    # helper function, depending on u only
    def f(ti, x1i, x2i, ui):
        return ui

    if calculation == Calculations.X1:
        k1 = h * f(ti, x1i, x2i, ui)
        l1 = h * func(ti, x1i, x2i, ui)

        k2 = h * f(ti + h / 2, x1i + k1 / 2, x2i, ui + l1 / 2)
        l2 = h * func(ti + h / 2, x1i + k1 / 2, x2i, ui + l1 / 2)

        k3 = h * f(ti + h / 2, x1i + k2 / 2, x2i, ui + l2 / 2)
        l3 = h * func(ti + h / 2, x1i + k2 / 2, x2i, ui + l2 / 2)

        k4 = h * f(ti + h, x1i + k3, x2i, ui + l3)
        l4 = h * func(ti + h, x1i + k3, x2i, ui + l3)

        previous_value = x1i

    elif calculation == Calculations.X2:
        k1 = h * f(ti, x1i, x2i, ui)
        l1 = h * func(ti, x1i, x2i, ui)

        k2 = h * f(ti + h / 2, x1i, x2i + k1 / 2, ui + l1 / 2)
        l2 = h * func(ti + h / 2, x1i, x2i + k1 / 2, ui + l1 / 2)

        k3 = h * f(ti + h / 2, x1i, x2i + k2 / 2, ui + l2 / 2)
        l3 = h * func(ti + h / 2, x1i, x2i + k2 / 2, ui + l2 / 2)

        k4 = h * f(ti + h, x1i, x2i + k3, ui + l3)
        l4 = h * func(ti + h, x1i, x2i + k3, ui + l3)

        previous_value = x2i

    dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    du = (l1 + 2 * l2 + 2 * l3 + l4) / 6

    x_ = previous_value + dx
    u_ = ui + du

    return x_, u_


def main():
    # Setting of started values
    t0 = 0

    x10 = 0.05
    x20 = 0.15

    # u = y' = dy/dx
    u01 = 0
    u02 = 0

    u1_list = []
    u1_list.append(u01)  # first value
    u2_list = []
    u2_list.append(u02)  # first value

    x1_list = []
    x1_list.append(theta1 + 0.05)  # 0.1 + 0.05
    x2_list = []
    x2_list.append(theta2 + 0.05)  # 0.1 + 0.1 + 0.05

    t_list = []
    t_list.append(t0)

    h = H
    i = 0
    t = t0
    t_end = 30
    while t < t_end:
        x1_new, u1_new = runge_kutta_4_2_local(Calculations.X1, t, x1_list[i], x2_list[i], u1_list[i], f1)
        x1_list.append(x1_new)
        u1_list.append(u1_new)

        x2_new, u2_new = runge_kutta_4_2_local(Calculations.X2, t, x1_new, x2_list[i], u2_list[i], f2)
        x2_list.append(x2_new)
        u2_list.append(u2_new)

        t += h
        t_list.append(t)

        i += 1

    count = i + 1

    for i in range(count):
        print("{x1};{x2};{v1};{v2};".format(x1=x1_list[i], x2=x2_list[i], v1=u1_list[i],
                                                               v2=u2_list[i]))

    # t_list, x1_list, t_list, x2_list NOTE Done
    plt.plot(t_list, x1_list)
    plt.show()

    plt.plot(t_list, x2_list)
    plt.show()
if __name__ == "__main__":
    main()
