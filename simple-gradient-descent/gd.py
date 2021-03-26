# Vinicius Pimenta Bernardo (202002447)
# Simple Gradient Descent Algorithm

import numpy as np
import matplotlib.pyplot as plt
from typing import List


def f_true(x):
    return 2 + 0.8 * x + 0.3 * (x**2) - 0.5 * (x**3)


# Conjunto de dados {(x,y)}
xs = np.linspace(-3, 3, 100)
ys = np.array([f_true(x) + np.random.randn() * 0.5 for x in xs])
m = float(len(xs))


def h(x, theta):
    """ Hipotese
    """
    return sum(np.array([(theta[i] * (x ** i)) for i in range(len(theta))]))


def cost(theta, xs, ys):
    """ Custo com respeito a theta
    """
    yh = np.array([h(x, theta) for x in xs], dtype='double')
    return sum((yh - ys) ** 2) / (2 * m)


def gradient(i, theta, xs, ys):
    """ Derivada parcial com respeito a theta[i]
    """
    yh = np.array([h(x, theta) for x in xs], dtype='double')
    xi = np.array([x**i for x in xs], dtype='double')
    return sum((yh - ys) * xi) / m


def print_modelo(theta, xs, ys):
    """ Plota no mesmo grafico: - o modelo/hipotese (reta)
        - a reta original (true function)
        - e os dados com ruido (xs, ys)
    """
    plt.figure()
    plt.plot(xs, [f_true(x) for x in xs], label='Reta original')
    plt.scatter(xs, ys, label='Dados')
    plt.plot(xs, [h(x, theta) for x in xs], label='Hipotese')
    plt.legend()
    plt.xlabel(xlabel='x')
    plt.ylabel(ylabel='y')
    plt.title(label='Modelo/Hipotese comparado a Reta original')


# Run polinomial gradient descent
epocas = 5000
L = 0.001

theta = [0, 0, 0, 0]
J = []

# Modelo e custo inicial
print_modelo(theta, xs, ys)
J.append(cost(theta, xs, ys))

# Loop principal
for t in range(1, epocas + 1):

    dtheta = []

    # Calcula as derivadas parciais
    for i in range(len(theta)):
        dtheta.append(gradient(i, theta, xs, ys))
    # Atualiza o valor de theta
    for i in range(len(theta)):
        theta[i] = theta[i] - (L * dtheta[i])
    # Salva o custo da epoca
    J.append(cost(theta, xs, ys))
    # Confere o modelo a cada mil epocas
    if t % 1000 == 0:
        print_modelo(theta, xs, ys)

# Mostra o custo ao longo das epocas
plt.figure()
plt.plot(range(len(J)), J, 'o-')

# Mostra coeficientes finais do modelo
for i in range(len(theta)):
    print(theta[i])

plt.show()

###################################################################################################

# Linear gradient descent only (comment previous part and use theta with only two members to get 3D plot)
# epocas = 5000
# # L = 0.6
# # L = 0.1
# L = 0.001
# # L = 0.0001

# theta = [-5, 4]
# J = []
# Jx = []
# Jy = []

# # Modelo e custo inicial
# print_modelo(theta, xs, ys)
# J.append(cost(theta, xs, ys))
# # Auxiliar para o plot 3D do custo, usar apenas para theta de 2 termos
# Jx.append(theta[0])
# Jy.append(theta[1])

# # Loop principal
# for t in range(1, epocas + 1):

#     dtheta = []

#     # Calcula as derivadas parciais
#     for i in range(len(theta)):
#         dtheta.append(gradient(i, theta, xs, ys))
#     # Atualiza o valor de theta
#     for i in range(len(theta)):
#         theta[i] = theta[i] - (L * dtheta[i])
#     # Salva o custo da epoca
#     J.append(cost(theta, xs, ys))
#     # Auxiliar para o plot 3D do custo, usar apenas para theta de 2 termos
#     Jx.append(theta[0])
#     Jy.append(theta[1])
#     # Confere o modelo a cada mil epocas
#     if t % 1000 == 0:
#         print_modelo(theta, xs, ys)

# # Mostra o custo ao longo das epocas
# plt.figure()
# plt.plot(range(len(J)), J, 'o-')


# theta0 = np.linspace(-5, 9, 30)
# theta1 = np.linspace(-3, 5, 30)

# T0, T1 = np.meshgrid(theta0, theta1)


# def cost_f(b, a):
#     yh = b + a * xs
#     return sum((yh - ys) ** 2) / (2 * m)


# J3d = np.array([], dtype=np.float64)
# for i in range(len(theta0)):
#     J3d_line = np.array([], dtype=np.float64)
#     for j in range(len(theta1)):
#         J3d_line = np.append(J3d_line, cost_f(theta0[j], theta1[i]))
#     if i == 0:
#         J3d = np.append(J3d, J3d_line)
#     elif i == 1:
#         J3d = np.append([J3d], [J3d_line], axis=0)
#     else:
#         J3d = np.append(J3d, [J3d_line], axis=0)

# plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(T0, T1, J3d, 50, cmap='binary')
# ax.plot3D(Jx, Jy, J)
# ax.set_xlabel('theta0')
# ax.set_ylabel('theta1')
# ax.set_zlabel('J(theta0, theta1)')


# # Mostra coeficientes finais do modelo
# for i in range(len(theta)):
#     print(theta[i])

# plt.show()
