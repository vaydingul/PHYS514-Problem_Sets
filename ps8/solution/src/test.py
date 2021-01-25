import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import




Nx = int((x[-1] - x[0]) / dx + 1)
Ny = int((y[-1] - y[0]) / dy + 1)

x = np.linspace(x[0], x[1], Nx)
y = np.linspace(y[0], y[1], Ny)

A = np.zeros((Nx, Nx))  # Preallocation of the A matrix
A1 = np.zeros((Nx, Nx))  # Preallocation of the A matrix

for k in range(1, Nx-1):
    A[k, (k-1):(k+2)] = [1, -4, 1]  # Intermediate rows
    A1[k, (k-1):(k+2)] = [1, -3, 1]  # Intermediate rows


A[0, 0:2] = [-3, 1]  # First row
A[-1, -2:] = [1, -3]  # Last row

A1[0, 0:2] = [-2, 1]  # First row
A1[-1, -2:] = [1, -2]  # Last row


n = Nx * Ny

A_ = np.zeros((n, n))

for j in range(Nx, n - Nx, Nx):
    A_[j:j+Nx, j-Nx:j+2*Nx] = np.block([np.eye(Nx), A, np.eye(Nx)])


A_[:Nx, :2*Nx] = np.block([A1, np.eye(Nx)])
A_[-Nx:, -2*Nx:] = np.block([np.eye(Nx), A1])

f = np.zeros(n)
a = 0.5
for j in range(Nx):
    for l in range(Ny):
        i = j * (Nx - 2) + l
        x_ = x[l]
        y_ = y[j]

        func = 1-((x_**2 + y_**2)/(a**2))
        if (x_**2 + y_**2) <= a**2:
            f[i] = func

u = solve(A_, f * (h**2), assume_a = "sym")
u = np.reshape(u, (Nx, Ny))


fig = plt.figure()
ax = fig.gca(projection='3d')

x, y = np.meshgrid(x, y)

# Plot the surface.
surf = ax.plot_surface(x, y, u, cmap = "hot")
plt.show()


