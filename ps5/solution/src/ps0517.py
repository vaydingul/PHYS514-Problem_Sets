import numpy as np
from numpy.polynomial import legendre as L
from scipy.interpolate import CubicSpline
from scipy.integrate import simps
import matplotlib.pyplot as plt
from re import error
import random
from timeit import timeit


def mytests():
    """
    mytests

    This function executes the following processes:
        - 
    Input:
    []

    Output:
    []

    Usage:
    mytests()
    """

    def fx(x): return np.sin(np.pi * x + np.pi / 4) + 1
    a = -1.0 # Start point
    b = 1.0 # End point

    # Analytical integration of fx
    def I_analytical(a, b): return (-np.cos(np.pi * b + np.pi / 4) *
                                    (1/np.pi) + b) - (-np.cos(np.pi * a + np.pi / 4) * (1/np.pi) + a)


    x = np.arange(a, b, 0.001) # x vector
    I_fx_quad = np.empty((6, x[1:].shape[0])) # Integration calculation using Gaussian quadrature
    I_fx_analytical = np.empty(x[1:].shape[0]) # Analytical integration calculation

    N = [10, 12, 14, 16, 18, 20] # Trial n for quad method

    for (ix1, n) in enumerate(N):
        for (ix2, k) in enumerate(x[1:]):   

            # Loop over different intervals and n values
            I_fx_quad[ix1, ix2] = quad_custom(fx, a=a, b=k, n=n) 
            I_fx_analytical[ix2] = I_analytical(a=a, b=k)

    ################ PLOTTING ###############################################
    plt.figure()
    plt.subplot(2, 1, 1)
    for k in range(0, 6):
        plt.plot(x[1:], I_fx_quad[k, :], label="n = {0}".format(N[k]), marker="o",
                 markevery=int(I_fx_analytical.shape[0] / random.randint(10, 20)))
    plt.plot(x[1:], I_fx_analytical, label="Original")
    plt.legend(loc="best", ncol = 2)
    plt.xlabel("$x$")
    plt.ylabel("$\int_{-1}^x f(x)$")
    plt.title(
        "Integral results of the function\n$f(x) = 1+\sin(\pi x + \dfrac{\pi}{4})$")

    plt.subplot(2, 1, 2)
    for k in range(0, 6):
        plt.plot(x[1:], I_fx_analytical - I_fx_quad[k, :], label="n = {0}".format(N[k]), marker="o",
                 markevery=int(I_fx_analytical.shape[0] / random.randint(10, 20)))
    plt.legend(loc="best", ncol = 2)
    plt.xlabel("$x$")
    plt.ylabel(
        "${\int_{-1}^x f(x)}_{original} - {\int_{-1}^x f(x)}_{quadrature}$")
    plt.title("Error in integral")
    plt.tight_layout()
    plt.savefig("./report/figures/17-1.png")
    ###############################################################################


    #plt.show()

def quad_custom(fun, a, b, n):
    """
    quad_custom

    This function executes the following processes:
        - It calculates the numerical integral of a function 
        - It uses Gaussian Quadrature

    Input:

        fun = Function to be integrated over given integral
        n = Number of points to be used
    Output:

        I_fx = Calculated numerical integral of the function vector fx

    Usage:

        fx = lambda x: np.sin(np.pi * x + np.pi / 4) + 1
        a = 0.0
        b = 1.0
        I_fx_quad = quad_custom(fx, a = a, b = b, n = 10)


    """
    def fun_extended(x): return fun(((b+a) + (b-a) * x) *
                                    0.5) * (b-a) * 0.5  # Change of variables

    C = np.zeros(n)  # Coefficient vector initialization
    C[n-1] = 1  # We are only dealing with n_th Legendre polynomial

    root = L.legroots(C)  # Get roots of the n_th Legendre polynomial
    d_C = L.legder(C)  # Get coefficients of the n_th LEgendre polynomial
    # Calculate weights
    weight = 2 / ((1 - root ** 2) * ((L.legval(root, d_C)) ** 2))

    # Gauss-Legendre Quadrature calculation
    I_fx = sum(fun_extended(root) * weight)

    return I_fx


if __name__ == "__main__":
    mytests()
