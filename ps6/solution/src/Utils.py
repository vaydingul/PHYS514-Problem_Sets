import numpy as np
import matplotlib.pyplot as plt
from re import error
import random
from scipy.optimize import newton
from scipy.integrate import quad



def numerical_derivative(fx, a, b, n):
    """
    numerical_derivative

    This function executes the following processes:
        - It calculates the numerical derivative of a function for a given interval [a, b]
        - It uses 3 point stencil at center, forward and backward

    Input:

        fx = Evaluated function vector whose derivative will be calculated
        a = Start point of interval
        b = End point of interval
        n = Number of spacing in the whole interval

    Output:

        d_fx = Calculated numerical derivative of the function vector fx

    Usage:

        a = -1 # Start point of the interval
        b = 1 # End point of the interval
        n = 2**15 # Number of "spacing", not length/size 
        x = np.linspace(a, b, n + 1) # Sampling between a and b
        fx = np.sin(np.pi * x + np.pi / 4 ) + 1 # Desired function evaluated at this interval
        d_fx_numerical = numerical_derivative(fx, a, b, n)

    """
    d_fx = np.empty(n+1)
    h = (b - a) / n  # Individual spacing length / step size

    for k in range(1, n):  # Terminal point will be processed differently

        d_fx[k] = (-0.5 * fx[k - 1] + 0.5 * fx[k + 1]) / \
            h  # Central difference with 3 point stencil

    d_fx[0] = (-1.5 * fx[0] + 2 * fx[1] - 0.5 * fx[2]) / \
        h  # Forward difference with 3 point stencil
    d_fx[n] = (1.5 * fx[n] - 2 * fx[n-1] + 0.5 * fx[n - 2]) / \
        h  # Backward difference with 3 point stencil

    return d_fx


def numerical_integral(fx, a, b, n, method="simpson"):
    """
    numerical_integral

    This function executes the following processes:
        - It calculates the numerical integral of a function for a given interval [a, b]
        - It uses Simpson method

    Input:

        fx = Evaluated function vector whose derivative will be calculated
        a = Start point of interval
        b = End point of interval
        n = Number of spacing in the whole interval
        method = The method that will be used in integration

    Output:

        I_fx = Calculated numerical integral of the function vector fx

    Usage:

        a = -1 # Start point of the interval
        b = 1 # End point of the interval
        n = 2**15 # Number of "spacing", not length/size 
        x = np.linspace(a, b, n + 1) # Sampling between a and b
        fx = np.sin(np.pi * x + np.pi / 4 ) + 1 # Desired function evaluated at this interval
        I_fx_analytical = -np.cos(np.pi * x + np.pi / 4 ) * (1/np.pi) + x  # Analytical integral calculation
        I_fx_numerical = numerical_integral(fx, a, b, n, method = "trapezoidal") # Numerical integral calculation

    """
    I_fx = np.empty(n)
    h = (b - a) / n

    if method == "simpson":

        for k in range(1, n):

            I_fx[k] = (h/3) * (fx[0] + 2 * np.sum(fx[2: k: 2]) +
                               4 * np.sum(fx[1: k-1: 2]) + fx[k])

    elif method == "trapezoidal":

        for k in range(1, n):

            I_fx[k] = (fx[0] + 2 * np.sum(fx[1:k]) + fx[k]) * h * 0.5

    return I_fx
    


def cycloid(x2, y2, N=100):
    """Return the path of Brachistochrone curve from (0,0) to (x2, y2).

    The Brachistochrone curve is the path down which a bead will fall without
    friction between two points in the least time (an arc of a cycloid).
    It is returned as an array of N values of (x,y) between (0,0) and (x2,y2).

    """
    g = 9.81

    # First find theta2 from (x2, y2) numerically (by Newton-Raphson).
    def f(theta):
        return y2/x2 - (1-np.cos(theta))/(theta-np.sin(theta))
    theta2 = newton(f, np.pi/2)

    # The radius of the circle generating the cycloid.
    R = y2 / (1 - np.cos(theta2))

    theta = np.linspace(0, theta2, N)
    x = R * (theta - np.sin(theta))
    y = R * (1 - np.cos(theta))

    # The time of travel
    T = theta2 * np.sqrt(R / g)
    return x, y, T