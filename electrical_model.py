import math
import numpy
import scipy
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

E = 0.02
a1 = 0.995
a2 = 0.985
tMax = 4000
tMin = 1000
start_u1 = 1
start_u2 = 1
start_v1 = 1
start_v2 = 1
d1 = d2 = 0.005


def f_naguma_2_elem(t, r):
    global d1
    u1, v1, u2, v2 = r
    fu1 = u1 - (u1 ** 3) / 3 - v1 + d1 * (u2 - u1)
    fv1 = E * (u1 - a1)
    fu2 = u2 - (u2 ** 3) / 3 - v2 + d1 * (u1 - u2)
    fv2 = E * (u2 - a2)

    return fu1, fv1, fu2, fv2


def graphs_2_elem(func, Start_u1, Start_v1, Start_u2, Start_v2):
    sol = solve_ivp(func, [0, tMax], [Start_u1, Start_v1, Start_u2, Start_v2], method='RK45', rtol=1e-11, atol=1e-11,
                    dense_output=True)
    sol
    us1, vs1, us2, vs2 = sol.y
    ts = sol.t

    return ts, us1, vs1, us2, vs2


def finding_frequency(t, x):
    time_list = []
    border = max(x) - (max(x) - min(x)) / 20
    last_point = 0
    per_begin = 0
    periods = []

    # Поиск максимума

    for count in range(len(t) - 3):
        if t[count] > tMin:
            if (t[count] - last_point > 10) and (x[count] >= border) and (x[count] >= max(x[count + 1: count + 3])) and\
                    (x[count] >= max(x[count - 3: count - 1])):
                time_list.append(t[count])
                last_point = t[count]

    time_sum = 0

    q = 10000
    su = 0

    for count in range(len(time_list) - 1):
        s = time_list[count + 1] - time_list[count]
        print(s)
        if s < q * 1.1 > q * 0.9:
            time_sum += s
            q = s
        else:
            su -= 1

    period = time_sum / (len(time_list) - 1 + su)

    # print(time_list)
    print(len(time_list) - 1 + su)
    print(period)

    frequency = 2 * math.pi / period

    return frequency


def graphs_coeffs():
    d_list = []
    frequency_list1 = []
    frequency_list2 = []

    for i in range(1, 30):
        global d1
        d1 = 0.001 * i
        print(d1)
        d_list.append(d1)
        t, x, x1, y, y1 = graphs_2_elem(f_naguma_2_elem, start_u1, start_v1, start_u2, start_v2)
        frequency_list1.append(finding_frequency(t, x))
        frequency_list2.append(finding_frequency(t, y))

        # plt.figure()
        # plt.subplot(211)
        # plt.plot(t, x, label='U1')
        # plt.legend()
        # plt.grid(True)

        # plt.subplot(212)
        # plt.plot(t, y, label='U2')
        # plt.legend()
        # plt.grid(True)

    return d_list, frequency_list1, frequency_list2


# Работаем с двумя элементами электрической модели

t1, u1, v1, u2, v2 = graphs_2_elem(f_naguma_2_elem, start_u1, start_v1, start_u2, start_v2)

# Для начала построим два фазовых портрета:

fig1, ax1 = plt.subplots(1, 2)

ax1[0].plot(start_u1, start_v1, marker='o', color='purple')
ax1[0].plot(u1, v1, label='U1 от V1')
ax1[0].set_title('Фазовый портрет:')
ax1[0].set_xlabel('U1')
ax1[0].set_ylabel('V1')
ax1[0].legend()
ax1[0].grid(True)

ax1[1].plot(start_u2, start_v2, marker='o', color='purple')
ax1[1].plot(u2, v2, label='U2 от V2')
ax1[1].set_title('Фазовый портрет:')
ax1[1].set_xlabel('U2')
ax1[1].set_ylabel('V2')
ax1[1].legend()
ax1[1].grid(True)

# Далее построим два фазовых портрета противофазы:

# start_u2 = 0.5
# start_v1 = 0.5

# t1, u1, v1, u2, v2 = graphs_2_elem(f_naguma_2_elem, start_u1, start_v1, start_u2, start_v2)

# fig3, ax1 = plt.subplots(1, 2)

# ax1[0].plot(start_u1, start_v1, marker='o', color='purple')
# ax1[0].plot(u1, v1, label='U1 от V1')
# ax1[0].set_title('Фазовый портрет:')
# ax1[0].set_xlabel('U1')
# ax1[0].set_ylabel('V1')
# ax1[0].legend()
# ax1[0].grid(True)

# ax1[1].plot(start_u2, start_v2, marker='o', color='purple')
# ax1[1].plot(u2, v2, label='U2 от V2')
# ax1[1].set_title('Фазовый портрет:')
# ax1[1].set_xlabel('U2')
# ax1[1].set_ylabel('V2')
# ax1[1].legend()
# ax1[1].grid(True)

# Далее построим зависимости U и V от времени

# fig2, ax2 = plt.subplots(1, 2)

# ax2[0].plot(t1, u1, label='U1')
# ax2[0].legend()
# ax2[0].grid(True)

# ax2[1].legend()
# ax2[1].plot(t1, u2, label='U2')
# ax2[1].grid(True)

# ax2[1, 0].plot(t1, v1, label='V1')
# ax2[1, 0].legend()
# ax2[1, 0].grid(True)

# ax2[1, 1].plot(t1, v2, label='V2')
# ax2[1, 1].legend()
# ax2[1, 1].grid(True)

# Далее построим графики зависимости W от d

# D0, W1, W2 = graphs_coeffs()

# start_u1 = 2
# start_u2 = -2
# start_v1 = 0
# start_v2 = 0

# D1, W3, W4 = graphs_coeffs()

# plt.figure()
# plt.plot(D0, W1, label='w11')
# plt.plot(D0, W2, label='w21')
# plt.plot(D1, W3, label='w11')
# plt.plot(D1, W4, label='w21')
# plt.xlabel('d')
# plt.ylabel('W')

# ax4[1].set_xlabel('d')
# ax4[1].set_ylabel('W')
# ax4[1].legend()
# ax4[1].grid(True)

plt.show()
