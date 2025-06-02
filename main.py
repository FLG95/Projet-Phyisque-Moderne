import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import matplotlib
import os
from scipy.linalg import eigh
matplotlib.use('TkAgg')

start_time = time.time()

data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

def etats_stationnaires(dx, nx, V, n_states=5):
    x = np.linspace(0, (nx - 1) * dx, nx)

    diag = np.full(nx, -2.0)
    offdiag = np.full(nx - 1, 1.0)
    T = (-1 / dx**2) * (np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1))
    H = T + np.diag(V)

    energies, states = eigh(H, subset_by_index=(0, n_states - 1))

    plt.figure(figsize=(10,6))
    for i in range(n_states):
        psi = states[:, i]
        psi = psi / np.sqrt(np.sum(psi**2) * dx)
        plt.plot(x, psi**2 + energies[i], label=f"État {i} (E = {energies[i]:.2f})")

    plt.plot(x, V, 'k--', label='Potentiel V(x)')
    plt.title("États stationnaires")
    plt.xlabel("x")
    plt.ylabel("Énergie / Densité de probabilité")
    plt.legend()
    plt.grid()
    filepath = os.path.join(data_dir, "etats_stationnaires.png")
    plt.savefig(filepath)
    print(f"Graphique des états stationnaires exporté dans '{filepath}'")
    plt.close()

def init():
    line.set_data([], [])
    return line,

def animate(j):
    line.set_data(o, final_densite[j,:])
    return line,

if input("Voulez-vous utiliser des valeurs personnalisées ? oui - non : ") == 'oui':
    dt = float(input("Valeur de dt (ref 1E-7) : "))
    dx = float(input("Valeur de dx (ref 0.001) : "))
    nt = int(input("Valeur de nt (ref 90000) : "))
    nd = int(nt/1000)+1
    xc = float(input("Valeur de xc (ref 0.6) : "))
    sigma = float(input("Valeur de sigma (ref 0.05) : "))
    v0 = float(input("Valeur de v0 (ref -4000) : "))
    e = float(input("Valeur de e (ref 5) : "))
else:
    dt = 1E-7
    dx = 0.001
    nt = 90000
    nd = int(nt/1000)+1
    xc = 0.6
    sigma = 0.05
    v0 = -4000
    e = 5

nx = int(1/dx)*2
n_frame = nd
s = dt/(dx**2)
A = 1/(math.sqrt(sigma*math.sqrt(math.pi)))
E = e*v0
k = math.sqrt(2*abs(E))

o = np.linspace(0, (nx - 1) * dx, nx)
V = np.zeros(nx)
V[(o >= 0.8) & (o<=0.9)] = v0

cpt = A * np.exp(1j * k * o - ((o - xc) ** 2) / (2 * (sigma ** 2)))
densite = np.zeros((nt,nx))
densite[0,:] = np.abs(cpt[:]) ** 2
final_densite = np.zeros((n_frame,nx))
re = np.real(cpt[:])
b = np.zeros(nx)
im = np.imag(cpt[:])

it = 0
for i in range(1, nt):
    if i % 2 != 0:
        b[1:-1] = im[1:-1]
        im[1:-1] = im[1:-1] + s * (re[2:] + re[:-2]) - 2 * re[1:-1] * (s + V[1:-1] * dt)
        densite[i,1:-1] = re[1:-1]*re[1:-1] + im[1:-1]*b[1:-1]
    else:
        re[1:-1] = re[1:-1] - s * (im[2:] + im[:-2]) + 2 * im[1:-1] * (s + V[1:-1] * dt)

for i in range(1, nt):
    if (i-1) % 1000 == 0:
        it += 1
        final_densite[it][:] = densite[i][:]

plot_title = "Marche Ascendante avec E/Vo=" + str(e)

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_ylim(0, 13)
ax.set_xlim(0, 2)
ax.plot(o, V, label="Potentiel")
ax.set_title(plot_title)
ax.set_xlabel("x")
ax.set_ylabel("Densité de probabilité de présence")
ax.legend()

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=nd, blit=False, interval=100, repeat=False)
ani.save(os.path.join(data_dir, "animation.gif"), fps=10)


print(f"Animation exportée dans data")

plt.show()

if input("Voulez-vous afficher les états stationnaires ? oui - non : ") == 'oui':
    etats_stationnaires(dx, nx, V)
