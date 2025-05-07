import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

import matplotlib
matplotlib.use('TkAgg')


def simulation(e):
    dt = 1E-7
    dx = 0.001
    nx = int(1 / dx) * 2
    nt = 90000
    nd = int(nt / 1000) + 1
    s = dt / (dx ** 2)
    xc = 0.6
    sigma = 0.05
    A = 1 / (math.sqrt(sigma * math.sqrt(math.pi)))
    v0 = -4000
    E = e * v0
    k = math.sqrt(2 * abs(E))

    o = np.linspace(0, (nx - 1) * dx, nx)
    V = np.zeros(nx)
    V[(o >= 0.8) & (o <= 0.9)] = v0

    cpt = A * np.exp(1j * k * o - ((o - xc) ** 2) / (2 * (sigma ** 2)))
    densite = np.zeros((nt, nx))
    densite[0, :] = np.absolute(cpt[:]) ** 2
    re = np.real(cpt[:])
    im = np.imag(cpt[:])
    b = np.zeros(nx)

    for i in range(1, nt):
        if i % 2 != 0:
            b[1:-1] = im[1:-1]
            im[1:-1] = im[1:-1] + s * (re[2:] + re[:-2]) - 2 * re[1:-1] * (s + V[1:-1] * dt)
            densite[i, 1:-1] = re[1:-1]**2 + im[1:-1]*b[1:-1]
        else:
            re[1:-1] = re[1:-1] - s * (im[2:] + im[:-2]) + 2 * im[1:-1] * (s + V[1:-1] * dt)

    # Transmission : densité à droite du puits / densité totale
    zone_transmise = o > 1.1
    T = np.sum(densite[-1, zone_transmise]) / np.sum(densite[-1, :])

    return T

# Animation
def init():
    line.set_data([], [])
    return line,

def animate(j):
    line.set_data(o, final_densite[j,:])  # Affiche la densité à chaque étape de l'animation
    return line,

dt = 1E-7
dx = 0.001
nx = int(1 / dx) * 2
nt = 90000
nd = int(nt / 1000) + 1
n_frame = nd
s = dt / (dx ** 2)
xc = 0.6
sigma = 0.05
A = 1 / (math.sqrt(sigma * math.sqrt(math.pi)))
v0 = -4000
e = 5
E = e * v0
k = math.sqrt(2 * abs(E))

o = np.zeros(nx)
V = np.zeros(nx)
o = np.linspace(0, (nx - 1) * dx, nx)
V[(o >= 0.8) & (o <= 0.9)] = v0

cpt = A * np.exp(1j * k * o - ((o - xc) ** 2) / (2 * (sigma ** 2)))
densite = np.zeros((nt, nx))
densite[0, :] = np.absolute(cpt[:]) ** 2
final_densite = np.zeros((n_frame, nx))
re = np.zeros(nx)
re[:] = np.real(cpt[:])

b = np.zeros(nx)
im = np.zeros(nx)
im[:] = np.imag(cpt[:])

it = 0


for i in range(1, nt):
    if i % 2 != 0:
        b[1:-1] = im[1:-1]
        im[1:-1] = im[1:-1] + s * (re[2:] + re[:-2]) - 2 * re[1:-1] * (s + V[1:-1] * dt)
        densite[i, 1:-1] = re[1:-1]**2 + im[1:-1] * b[1:-1]
    else:
        re[1:-1] = re[1:-1] - s * (im[2:] + im[:-2]) + 2 * im[1:-1] * (s + V[1:-1] * dt)

for i in range(1, nt):
    if (i - 1) % 1000 == 0:
        it += 1
        final_densite[it][:] = densite[i][:]

# Tracé de l'animation
plot_title = "Marche Ascendante avec E/Vo=" + str(e)
fig = plt.figure()  # Initialise la figure principale
line, = plt.plot([], [])
plt.ylim(0, 13)
plt.xlim(0, 2)
plt.plot(o, V, label="Potentiel")
plt.title(plot_title)
plt.xlabel("x")
plt.ylabel("Densité de probabilité de présence")
plt.legend()

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=nd, blit=False, interval=100, repeat=False)

# Calcul de la transmission T(e)
e_list = np.linspace(0.1, 5, 50)
T_list = []

for e in e_list:
    print(f"Calcul pour E/V0 = {e}")
    T = simulation(e)
    T_list.append(T)

# Tracer T(E)
plt.plot(e_list, T_list)
plt.xlabel("E / V0")
plt.ylabel("Transmission T(E)")
plt.title("Transmission en fonction de l’énergie (effet Ramsauer)")
plt.grid(True)
plt.show()



#il faut prendre E > V0