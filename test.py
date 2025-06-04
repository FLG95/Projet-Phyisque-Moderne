import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# Paramètres physiques
hbar = 1.0
m = 1.0
V0 = 50.0    # Profondeur du puits (valeur positive car le puits est à -V0)
a = 1.0      # Demi-largeur du puits
L = 5.0      # Domaine spatial total
N = 1000     # Nombre de points spatiaux
x = np.linspace(-L, L, N)
dx = x[1] - x[0]

# Construction du potentiel
V = np.zeros(N)
V[np.abs(x) <= a] = -V0

# Construction du Hamiltonien tridiagonal (méthode des différences finies)
diagonal = (hbar**2) / (m * dx**2) + V
off_diagonal = np.full(N-1, - (hbar**2) / (2 * m * dx**2))

# Résolution des valeurs et vecteurs propres
eigenvalues, eigenvectors = eigh_tridiagonal(diagonal, off_diagonal)

# Sélection des états liés (énergies < 0)
bound_states_indices = np.where(eigenvalues < 0)[0]
bound_energies = eigenvalues[bound_states_indices]
bound_wavefuncs = eigenvectors[:, bound_states_indices]

# Normalisation des fonctions d'onde
for i in range(bound_wavefuncs.shape[1]):
    bound_wavefuncs[:, i] /= np.sqrt(np.trapz(bound_wavefuncs[:, i]**2, x))

# Affichage
plt.figure(figsize=(10, 6))

# Tracé du potentiel
plt.plot(x, V, 'k', label='Potentiel')

# Tracé des états liés
colors = ['b', 'g', 'r', 'm']
for i, idx in enumerate(bound_states_indices):
    plt.plot(x, bound_wavefuncs[:, i] + bound_energies[i], colors[i % len(colors)],
             label=f'État n={i+1}, E={bound_energies[i]:.2f}')

plt.title("États stationnaires dans un puits de potentiel fini")
plt.xlabel("x")
plt.ylabel("Énergie et fonctions d'onde")
plt.legend()
plt.grid(True)
plt.show()
