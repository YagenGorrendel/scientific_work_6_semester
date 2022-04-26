import math
import numpy
import scipy
import sys
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

a = 0.7
b = 0.8
tao1 = 0.08
tao2 = 3.1
tao3 = 1.15
v_inh = -1.5
v_ex = 1.5
S = [0.375, 0, 0]

g_ex = []
g_inh = []


def f_naguma_chem_(t, r):
    global tao1, tao2, tao3, a, b, v_ex, v_inh, S

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
        list_fx.append((x[i] - (x[i] ** 3) / 3 - y[i] - z1[i] * (x[i] - v_inh) - z2[i] * (x[i] - v_ex) + S[i]) / tao1)
        list_fy.append(x[i] - b * y[i] + a)
        list_fz1.append((sum([g_inh[i][n] * numpy.heaviside(x[n], 0) for n in range(len(g_inh[i]))]) - z1[i]) / tao2)
        list_fz2.append((sum([g_ex[i][n] * numpy.heaviside(x[n], 0) for n in range(len(g_ex[i]))]) - z2[i]) / tao3)

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

    sol = solve_ivp(func, [0, 300], start, method='RK45', dense_output=True)

    y = sol.y

    t = sol.t

    return y, t


def horizontal_ways_finding(n, step, num_elem):
    global g_inh, g_ex

    list_x = []
    list_y = [0.0]
    H = 0.0

    finding_hor = True
    G_max = 0.5
    G_min = 0

    while finding_hor:
        G = (G_max + G_min) / 2
        sys.stdout.write(str(G))

        g_ex = [[0, 0, 0], [G, 0, 0], [H, 0, 0]]
        g_inh = [[0, 0, 0], [0, 0, 2.5], [0, 2.5, 0]]

        if num_elem == 2:
            res, t = graphs_chem(f_naguma_chem_, [-10, -1, -15], [0, 0, 0.6], [0, 0.05, 0.05],
                                 [0, 0.5, 0.5])  # 2 elem
        else:
            res, t = graphs_chem(f_naguma_chem_, [10, 1, 1], [2, 1, 0], [0, 0.4, 0.4], [0, 0.4, 0.4])  # 3 elem

        res = res.tolist()
        x = res[: len(res) // 4]

        size = max(x[num_elem - 1][300:]) - min(x[num_elem - 1][300:])

        if size < 1.5:
            if (G_max - G_min) > (1 / n):
                G_min = G
            else:
                print("afa", G_max, G_min)
                list_x.append(G)
                finding_hor = False
        else:
            G_max = G

    while H < list_x[0]:
        H += step
        list_y.append(H)
        list_x.append(list_x[0])


# Build phase portret:

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

# Build graph G / max (may work not full correct)

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

# Build graphs isocline depending on S for one element

"""g_ex = [[0]]
g_inh = [[0]]

for k in range(21):
    s = k / 10
    x = [(n - 100) * 0.02 for n in range(201)]
    y1 = []
    y2 = []

    for count in range(201):
        y1.append(x[count] - (x[count] ** 3) / 3 + s)
        y2.append((x[count] + a) / b)

    plt.figure()
    plt.plot(x, y1, label='y1')
    plt.plot(x, y2, label='y2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)"""

# Build graphs depending on the coefficient of inghibitor intercellular communication

"""d1 = 0
d2 = 0

list_x = []
list_g = []

size = 3

figures = [None] * size ** 2
axes = [None] * size ** 2

list_d = []

for coeff in range(size):
    d1 = coeff * 0.5
    print(d1, '*')
    append_list = []
    for coeff2 in range(size):
        d2 = coeff2 * 0.5
        g_ex = [[0, 0], [0, 0]]
        g_inh = [[0, d1], [d2, 0]]
        print(d2)

        c = -0.5

        res, t = graphs_chem(f_naguma_chem_, [c, c], [c, c], [c, c], [c, c])

        res = res.tolist()

        x = res[: len(res) // 4]
        y = res[len(res) // 4: len(res) // 2]
        z1 = res[len(res) // 2: 3 * len(res) // 4]
        z2 = res[3 * len(res) // 4:]

        # print(x[0][-5], x[0][-1])
        # print(x[1][-5], x[1][-1])

        if abs(x[0][-5] - x[0][-1]) < 4 * abs(x[1][-5] - x[1][-1]):
            list_d.append([[d1], [d2], 'ro'])
        elif abs(x[0][-5] - x[0][-1]) > 4 * abs(x[1][-5] - x[1][-1]):
            list_d.append([[d1], [d2], 'bo'])
        else:
            list_d.append([[d1], [d2], 'o'])

        figures[coeff * size + coeff2], axes[coeff * size + coeff2] = plt.subplots(2, 2)

        axes[coeff * size + coeff2][0][0].plot(t, x[0], label='x1')
        axes[coeff * size + coeff2][0][0].plot(t, x[1], label='x2')
        axes[coeff * size + coeff2][0][0].set_xlabel('t')
        axes[coeff * size + coeff2][0][0].set_ylabel('x/y')
        axes[coeff * size + coeff2][0][0].legend()
        axes[coeff * size + coeff2][0][0].grid(True)

        axes[coeff * size + coeff2][0][1].plot(t, y[0], label='y1')
        axes[coeff * size + coeff2][0][1].plot(t, y[1], label='y2')
        axes[coeff * size + coeff2][0][1].set_xlabel('t')
        axes[coeff * size + coeff2][0][1].set_ylabel('x/y')
        axes[coeff * size + coeff2][0][1].legend()
        axes[coeff * size + coeff2][0][1].grid(True)

        axes[coeff * size + coeff2][1][0].plot(x[0], y[0])
        axes[coeff * size + coeff2][1][0].set_xlabel('x')
        axes[coeff * size + coeff2][1][0].set_ylabel('y')
        axes[coeff * size + coeff2][1][0].grid(True)

        axes[coeff * size + coeff2][1][1].plot(x[1], y[1])
        axes[coeff * size + coeff2][1][1].set_xlabel('x')
        axes[coeff * size + coeff2][1][1].set_ylabel('y')
        axes[coeff * size + coeff2][1][1].grid(True)

plt.figure()
for element in list_d:
    plt.plot(element[0], element[1], element[2])
plt.xlabel('t')
plt.ylabel('x/y')
plt.legend()
plt.grid(True)"""

# Work with 3 elements:

"""list_d1 = []
list_d2 = []
list_he = []
list_ge = []
list_ge_ex = []

for count1 in range(11):
    He = count1 * 0.05
    print(He)

    for count2 in range(11):
        Ge = count2 * 0.05
        sys.stdout.write(str(Ge))

        list_he.append(He)
        list_ge.append(Ge)

        g_ex = [[0, 0, 0], [Ge, 0, 0], [He, 0, 0]]
        g_inh = [[0, 0, 0], [0, 0, 2.5], [0, 2.5, 0]]

        res, t = graphs_chem(f_naguma_chem_, [-10, -1, -15], [0, 0, 0.6], [0, 0.05, 0.05], [0, 0.5, 0.5])  # 2 elem

        res = res.tolist()

        x = res[: len(res) // 4]

        size_l2_1 = max(x[1][100:]) - min(x[1][100:])
        size_l3_1 = max(x[2][100:]) - min(x[2][100:])

        res, t = graphs_chem(f_naguma_chem_, [10, 1, 1], [2, 1, 0], [0, 0.4, 0.4], [0, 0.4, 0.4])  # 3 elem

        res = res.tolist()

        x = res[: len(res) // 4]

        size_l2_2 = max(x[1][100:]) - min(x[1][100:])
        size_l3_2 = max(x[2][100:]) - min(x[2][100:])

        if size_l2_1 > 1.5:
            if size_l3_2 > 1.5:
                list_ge_ex.append('r')
            else:
                list_ge_ex.append('g')
        elif size_l3_1 > 1.5:
            if size_l2_2 > 1.5:
                list_ge_ex.append('r')
            else:
                list_ge_ex.append('b')
        else:
            if size_l2_2 > 1.5:
                list_ge_ex.append('g')
            elif size_l3_2 > 1.5:
                list_ge_ex.append('b')
            else:
                list_ge_ex.append('k')


plt.figure()
for s in range(len(list_ge_ex)):
    plt.plot(list_ge[s], list_he[s], c=list_ge_ex[s], marker='o')
plt.xlabel('G')
plt.ylabel('M')
plt.grid(True)"""

# Print right graph, addiction between Gex

list_he = []
list_ge = []
noise = 100.0

# First way:
list_ge, list_he = horizontal_ways_finding(noise)

# Second way:
while He < 0.5:
    list_he.append(He)
    print(He)

    finding_hor = True
    Ge_max = 0.5
    Ge_min = 0

    while finding_hor:
        Ge = (Ge_max + Ge_min) / 2
        sys.stdout.write(str(Ge))

        g_ex = [[0, 0, 0], [Ge, 0, 0], [He, 0, 0]]
        g_inh = [[0, 0, 0], [0, 0, 2.5], [0, 2.5, 0]]

        res, t = graphs_chem(f_naguma_chem_, [-10, -1, -15], [0, 0, 0.6], [0, 0.05, 0.05], [0, 0.5, 0.5])  # 2 elem

        res = res.tolist()
        x = res[: len(res) // 4]

        size_l2 = max(x[1][300:]) - min(x[1][300:])

        if size_l2 > 1.5:
            if (Ge_max - Ge_min) > (1 / noise):
                Ge_max = Ge
            else:
                list_ge.append(Ge)
                finding_hor = False
        else:
            Ge_min = Ge

    He += 0.01

list_he_1 = []
list_ge_1 = []
Ge = 0.0
Ge_max = 10

# Third way:
while Ge < Ge_max:
    list_ge_1.append(Ge)
    print(Ge)

    if Ge_max > 1:

        finding_hor = True
        He_max = 0.5
        He_min = 0

        while finding_hor:
            He = (He_max + He_min) / 2
            sys.stdout.write(str(He))

            g_ex = [[0, 0, 0], [Ge, 0, 0], [He, 0, 0]]
            g_inh = [[0, 0, 0], [0, 0, 2.5], [0, 2.5, 0]]

            res, t = graphs_chem(f_naguma_chem_, [10, 1, 1], [2, 1, 0], [0, 0.4, 0.4], [0, 0.4, 0.4])  # 3 elem

            res = res.tolist()
            x = res[: len(res) // 4]

            size_l2 = max(x[2][300:]) - min(x[2][300:])

            if size_l2 < 1.5:
                if (He_max - He_min) > (1 / noise):
                    He_min = He
                else:
                    if len(list_he_1) == 0:
                        Ge_max = He
                    list_he_1.append(Ge_max)
                    finding_hor = False
            else:
                He_max = He
    else:
        list_he_1.append(Ge_max)

    Ge += 0.01

# Fourth way:
while Ge < 0.5:
    list_ge_1.append(Ge)
    print(Ge)

    finding_hor = True
    He_max = 0.5
    He_min = 0

    while finding_hor:
        He = (He_max + He_min) / 2
        sys.stdout.write(str(He))

        g_ex = [[0, 0, 0], [Ge, 0, 0], [He, 0, 0]]
        g_inh = [[0, 0, 0], [0, 0, 2.5], [0, 2.5, 0]]

        res, t = graphs_chem(f_naguma_chem_, [10, 1, 1], [2, 1, 0], [0, 0.4, 0.4], [0, 0.4, 0.4])  # 3 elem

        res = res.tolist()
        x = res[: len(res) // 4]

        size_l2 = max(x[2][300:]) - min(x[2][300:])

        if size_l2 > 1.5:
            if (He_max - He_min) > (1 / noise):
                He_max = He
            else:
                list_he_1.append(He)
                finding_hor = False
        else:
            He_min = He

    Ge += 0.01

plt.figure()
plt.plot(list_ge, list_he)
plt.plot(list_ge_1, list_he_1)
plt.xlabel('Ge')
plt.ylabel('He')
plt.grid(True)

plt.show()
