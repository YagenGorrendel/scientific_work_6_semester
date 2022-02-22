import math
import numpy
import scipy
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

a = 0.7
b = 0.8
tao1 = 0.01
tao2 = 3.1
tao3 = 1.15
v_inh = -1.5
v_ex = 1.5

G = 1000

g_ex = []
g_inh = []

#Функция 
def f_naguma_chem_(t, r):
    global tao1, tao2, tao3, a, b, v_ex, v_inh, G

    r = r.tolist()

    x = r[: len(r) // 4]
    y = r[len(r) // 4: len(r) // 2]
    z1 = r[len(r) // 2: 3 * len(r) // 4]
    z2 = r[3 * len(r) // 4:]

    list_fx = []
    list_fy = []
    list_fz1 = []
    list_fz2 = []

    for i in range(len(x)):
        list_fx.append((x[i] - (x[i] ** 3) / 3 - y[i] - z1[i] * (x[i] - v_inh) - z2[i] * (x[i] - v_ex)) / tao1)
        list_fy.append(x[i] - b * y[i] + a)
        list_fz1.append((sum([g_inh[i][n] * numpy.heaviside(x[n], 0.5) for n in range(len(g_inh[i]))]) - z1[i]) / tao2)
        list_fz2.append((sum([g_ex[i][n] * numpy.heaviside(x[n], 0.5) for n in range(len(g_ex[i]))]) - z1[i]) / tao3)

    res = []

    res.extend(list_fx)
    res.extend(list_fy)
    res.extend(list_fz1)
    res.extend(list_fz2)

    return res


def graphs_chem(func, start_x, start_y, start_z1, start_z2):
    start = []
    start.extend(start_x)
    start.extend(start_y)
    start.extend(start_z1)
    start.extend(start_z2)

    sol = solve_ivp(func, [0, 500], start, method='RK45', dense_output=True)

    y = sol.y

    t = sol.t

    return y, t


# Pаботаем с химической синаптической моделью связи:

# Сомневаюсь в правильности фазового портрета
# Построим фазовый портрет:

"""G = 1
g_ex = [[0, 0, 0], [0.5, 0, 0], [0.5, 0, 0]]
g_inh = [[0, 0, 0], [0, 0, G], [0, G, 0]]

for coeff in range(30):
    c = coeff * 0.01
    print(c)
    res, t = graphs_chem(f_naguma_chem_, [c, c, c], [c, c, c], [c, c, c], [c, c, c])

    res = res.tolist()

    x = res[: len(res) // 4]
    y = res[len(res) // 4: len(res) // 2]

    plt.figure()
    plt.plot(x[0], y[0], label='y')
    plt.xlabel('G')
    plt.ylabel('x')
    plt.legend()
    plt.grid(True)"""

# Строим график зависимости G от max (зачем написал - не понятно) (скорее всего работает не до конца корректно)

"""list_x = []
list_g = []

for coeff in range(11):
    G = coeff * 0.05
    g_ex = [[0, 0, 0], [0.5, 0, 0], [0.5, 0, 0]]
    g_inh = [[0, 0, 0], [0, 0, G], [0, G, 0]]
    print(G)
    g_inh[1][2] = G
    g_inh[2][1] = G
    list_g.append(G)

    c = -1

    res, t = graphs_chem(f_naguma_chem_, [c, c, c], [c, c, c], [c, c, c], [c, c, c])

    res = res.tolist()

    print(len(res))

    x = res[: len(res) // 4]
    y = res[len(res) // 4: len(res) // 2]
    z1 = res[len(res) // 2: 3 * len(res) // 4]
    z2 = res[3 * len(res) // 4:]

    print(min(x[0]), max(x[0][100:]))

    list_x.append(0.5 * (max(x[1]) + max(x[2])))

plt.figure()
plt.plot(list_g, list_x, label='G')
plt.xlabel('G')
plt.ylabel('x')
plt.legend()
plt.grid(True)"""

# Строим графики изоклин в зависимости от коэффициента S для одного элемента ()

"""g_ex = [[0]]
g_inh = [[0]]

for k in range(21):
    s = k / 10
    x = [(n - 100) / 50 for n in range(201)]
    y1 = []
    y2 = []

    for count in range(201):
        y1.append(x[count] - x[count] ** 3 / 3 + s)
        y2.append(1 / b * (x[count] + a))

    plt.figure()
    plt.plot(x, y1, label='y1')
    plt.plot(x, y2, label='y2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)"""

# Строим графики зависимости коэффициентов связи клеток при тормозящей связи

d1 = 0
d2 = 0

list_x = []
list_g = []

size = 3

list_d = []

for coeff in range(size):
    d1 = coeff * 0.01
    print(d1, '*')
    append_list = []
    for coeff2 in range(size):
        d2 = coeff2 * 0.05
        g_ex = [[0, 0], [0, 0]]
        g_inh = [[0, d1], [d2, 0]]
        print(d2)

        c = -0.5

        res, t = graphs_chem(f_naguma_chem_, [c, c], [c, c], [c, c], [c, c])

        res = res.tolist()

        print(len(res))

        x = res[: len(res) // 4]
        y = res[len(res) // 4: len(res) // 2]
        z1 = res[len(res) // 2: 3 * len(res) // 4]
        z2 = res[3 * len(res) // 4:]

        # append_list.append()

        print(x[0][-25], x[0][-1])
        print(x[1][-25], x[1][-1])

        if abs(x[0][-5] - x[0][-1]) < 4 * abs(x[1][-5] - x[1][-1]):
            list_d.append([[d1], [d2], 'ro'])
        elif abs(x[0][-5] - x[0][-1]) > 4 * abs(x[1][-5] - x[1][-1]):
            list_d.append([[d1], [d2], 'bo'])
        else:
            list_d.append([[d1], [d2], 'o'])

        plt.figure()
        plt.plot(t, x[0], label='x1')
        plt.plot(t, x[1], label='x2')
        plt.xlabel('t')
        plt.ylabel('x/y')
        plt.legend()
        plt.grid(True)

plt.figure()
for element in list_d:
    plt.plot(element[0], element[1], element[2])
plt.xlabel('t')
plt.ylabel('x/y')
plt.legend()
plt.grid(True)

plt.show()