import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import



def mytests():
    """
    mytests

    This function executes the following processes:
        - It visualizes the solution of the wave equation, numerically and analytically. 
        - It compares the solution of different boundary conditions.
        - It also compares the speed of different eigenvalue solver for Dirichlet BC.

    Input:
        []

    Output:
        []

    Usage:
    mytests()
    """
    
    
    x = [-1, 1] # x interval
    y = [-1, 1] # y interval

    h = 0.02 # step-size
    dx = h
    dy = h

    a = 0.05 # forcing factor

    # Solution of Poisson equation
    x, y, u = solve_poisson(x, y, dx, dy, a)

    ### PLOTTING
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y = np.meshgrid(x, y)
    surf = ax.plot_surface(x, y, u, cmap = "hot")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Solution of Poisson Equation for\n$\Delta x = ${0}  $\Delta y = ${1}  $a = ${2}".format(dx, dy, a))
    plt.tight_layout()
    #plt.savefig("report/figures/{0}.png".format(str(a).replace(".", "")))
    plt.show()



    x = [-1, 1] # x interval
    y = [-1, 1] # y interval

    h = 0.02 # step-size
    dx = h
    dy = h

    a = 0.5 # forcing factor

    # Solution of Poisson equation
    x, y, u = solve_poisson(x, y, dx, dy, a)

    ### PLOTTING
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y = np.meshgrid(x, y)
    surf = ax.plot_surface(x, y, u, cmap = "hot")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Solution of Poisson Equation for\n$\Delta x = ${0}  $\Delta y = ${1}  $a = ${2}".format(dx, dy, a))
    plt.tight_layout()
    #plt.savefig("report/figures/{0}.png".format(str(a).replace(".", "")))
    plt.show()


def solve_poisson(x, y, dx, dy, a):
    """
    wave_solver

    This function executes the following processes:
        - It solves Poisson equation, which is imposed to the Neumann BC.

    Input:
        x = x interval
        y = y interval
        dx = x spacing
        dy = y spacing
        a = Forcing boundary
    Output:
        x = Sampling points of x
        y = Sampling points of y
        u = Computed u value from Poisson equation

    Usage:
        x = [-1, 1] # x interval
        y = [-1, 1] # y interval
    
        h = 0.02 # step-size
        dx = h
        dy = h
    
        a = 0.05 # forcing factor
    
        # Solution of Poisson equation
        x, y, u = solve_poisson(x, y, dx, dy, a)
    """
    # Calculation of number of sampling points
    Nx = int((x[-1] - x[0]) / dx + 1)
    Ny = int((y[-1] - y[0]) / dy + 1)

    # Calculation of coordinates
    x = np.linspace(x[0], x[1], Nx)
    y = np.linspace(y[0], y[1], Ny)


    n = Nx * Ny

    # Preallocation of forcing vector
    f = np.zeros(n)

    # Caluclation of force term
    for j in range(Nx):
        for l in range(Ny):
            i = j * (Nx - 2) + l
            x_ = x[l]
            y_ = y[j]

            func = 1-((x_**2 + y_**2)/(a**2))
            if (x_**2 + y_**2) <= a**2:
                f[i] = func

    # A matrix for Poisson eqn. with Neumann BC
    A = get_A_matrix(Nx, Ny)

    # Solution of systems of equation
    u = solve(A, f * (dx * dy), assume_a = "sym")

    # Reshape from lexicographical to matrix version
    u = np.reshape(u, (Nx, Ny))

    return x, y, u


def get_A_matrix(Nx, Ny):
    """
    It calculates the A matrix which is required for the calculation of the 2D Poisson equation,
    which is imposed to the Neumann BC
    """
    
    A = np.zeros((Nx, Nx))  # Preallocation of the A matrix
    A1 = np.zeros((Nx, Nx))  # Preallocation of the A matrix

    for k in range(1, Nx-1):
        # Intermediate blocks
        A[k, (k-1):(k+2)] = [1, -4, 1]  # Intermediate rows
        # First and last block
        A1[k, (k-1):(k+2)] = [1, -4, 1]  # Intermediate rows


    A[0, 0:2] = [-3, 1]  # First row
    A[-1, -2:] = [1, -3]  # Last row

    A1[0, 0:2] = [-2, 1]  # First row
    A1[-1, -2:] = [1, -2]  # Last row


    n = Nx * Ny

    A_ = np.zeros((n, n))

    # Localization of the blocks
    for j in range(Nx, n - Nx, Nx):
        A_[j:j+Nx, j-Nx:j+2*Nx] = np.block([np.eye(Nx), A, np.eye(Nx)])


    A_[:Nx, :2*Nx] = np.block([A1, np.eye(Nx)])
    A_[-Nx:, -2*Nx:] = np.block([np.eye(Nx), A1])

    return A_



if __name__ == "__main__":
    mytests()