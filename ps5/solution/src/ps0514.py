import numpy as np
import matplotlib.pyplot as plt
from re import error
import random


def mytests():
    """
    mytests

    This function executes the following processes:
        - It visually compares the numerical derivative and analytical derivative
        - It visually compares the numerical integral and analytical integral
        - It visually shows the effect of step size in the 
            "relative" error for both derivative and integral
    Input:
        []

    Output:
        []

    Usage:
    mytests()

    """
    ############################ NUMERICAL CALCULATION VS ANALYTICAL DERIVATIVE COMPARISON ######################
    a = -1  # Start point of the interval
    b = 1  # End point of the interval
    n = 2**15  # Number of "spacing", not length/size
    x = np.linspace(a, b, n + 1)  # Sampling between a and b
    # Desired function evaluated at this interval
    fx = np.sin(np.pi * x + np.pi / 4) + 1
    # Analytical derivative calculation
    d_fx_analytical = np.cos(np.pi * x + np.pi / 4) * np.pi
    d_fx_numerical = numerical_derivative(
        fx, a, b, n)  # Numerical derivative calculation

    ############################ PLOTTING OF DERIVATIVE CALCULATIONS #############################################
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x, d_fx_analytical, label="Analytical Derivative", marker="o",
             markevery=int(d_fx_analytical.shape[0] / random.randint(10, 20)))
    plt.plot(x, d_fx_numerical, label="Numerical Derivative")
    plt.xlabel("$x$")
    plt.ylabel("$f^{\;\prime}(x)$")
    plt.title(
        "Investigation of the Function\n$f(x) = 1+\sin(\pi x + \dfrac{\pi}{4})$")
    plt.legend(loc="best")
    plt.subplot(2, 1, 2)
    plt.plot(x, d_fx_numerical - d_fx_analytical, marker="o",
             markevery=int(d_fx_analytical.shape[0] / random.randint(10, 20)))
    plt.xlabel("$x$")
    plt.ylabel("$f_h^{\;\prime}(x) - f^{\;\prime}(x)$")
    plt.title("Difference between derivative calculations")
    plt.tight_layout()
    plt.savefig("./report/figures/14-1.png")
    ############################################################################################
    ############################################################################################

    ###################### STEP SIZE COMPARISON FOR NUMERICAL DERIVATIVE CALCULATION #########

    a = -1  # Start point of the interval
    b = 1  # End point of the interval
    start_power = 15
    scale_factor = 4
    ###################### PLOTTING ##########################################################
    plt.figure()
    plt.xlabel("$x$")
    plt.ylabel("$f_h^{\;\prime}(x) - f_{\dfrac{h}{2}}^{\;\prime}(x)$")
    plt.title("Differences between numerical derivatives\nScale Factor = {0}".format(scale_factor))

    for k in [3, 2, 1, 0]:  # Iteration is flipped to make the figure clearer

        n1 = 2**(start_power + k)  # Number of spacing for first step size
        x = np.linspace(a, b, n1 + 1)  # Sampling between a and b
        # Desired function evaluated at this interval
        fx = np.sin(np.pi * x + np.pi / 4) + 1
        # Numerical derivative calculation for first step size
        d_fx_1 = numerical_derivative(fx, a, b, n1)

        n2 = 2**(start_power + k + 1)  # Number of spacing for second step size
        x = np.linspace(a, b, n2 + 1)  # Sampling between a and b
        # Desired function evaluated at this interval
        fx = np.sin(np.pi * x + np.pi / 4) + 1
        # Numerical derivative calculation for second step size
        d_fx_2 = numerical_derivative(fx, a, b, n2)

        plt.plot(x[::2], (d_fx_1 - d_fx_2[::2]) * (scale_factor ** k), label="h = {0}".format(str(
            (b-a)/n1)), ls="-.", marker="o", markevery=int(d_fx_1.shape[0] / random.randint(10, 20)))

    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("./report/figures/14-2.png")

    #########################################################################################
    #########################################################################################

    #################### NUMERICAL INTEGRATION #############################################

    a = -1  # Start point of the interval
    b = 1  # End point of the interval
    n = 2**15  # Number of "spacing", not length/size
    x = np.linspace(a, b, n + 1)  # Sampling between a and b
    # Desired function evaluated at this interval
    fx = np.sin(np.pi * x + np.pi / 4) + 1
    # Analytical integral calculation
    I_fx_analytical_cum = -np.cos(np.pi * x + np.pi / 4) * (1/np.pi) + x
    I_fx_analytical = I_fx_analytical_cum[1:] - I_fx_analytical_cum[0]
    I_fx_numerical_simpson = numerical_integral(
        fx, a, b, n, method="simpson")  # Numerical integral calculation
    I_fx_numerical_trapz = numerical_integral(
        fx, a, b, n, method="trapezoidal")  # Numerical integral calculation
    ####################### PLOTTING ##############################################

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x[1:], I_fx_analytical, label="Analytical Integral", marker="o",
             markevery=int(I_fx_analytical.shape[0] / random.randint(10, 20)))
    plt.plot(x[1:], I_fx_numerical_simpson, label="Numerical Integral (Simpson's Method)", marker="o",
             markevery=int(I_fx_analytical.shape[0] / random.randint(10, 20)))
    plt.plot(x[1:], I_fx_numerical_trapz, label="Numerical Integral (Trapezoidal Rule)", marker="o",
             markevery=int(I_fx_analytical.shape[0] / random.randint(10, 20)))
    plt.xlabel("$x$")
    plt.ylabel("$\int f(x)$")
    plt.title(
        "Investigation of the Function\n$f(x) = 1+\sin(\pi x + \dfrac{\pi}{4})$")
    plt.legend(loc="best")
    plt.subplot(2, 1, 2)
    plt.plot(x[1:], I_fx_numerical_trapz - I_fx_analytical, label  = "Error of Trapezoidal Rule", marker="o",
             markevery=int(I_fx_analytical.shape[0] / random.randint(10, 20)))
    plt.plot(x[1:], I_fx_numerical_simpson - I_fx_analytical, label  = "Error of Simpson's Method", marker="o",
             markevery=int(I_fx_analytical.shape[0] / random.randint(10, 20)))
    plt.xlabel("$x$")
    plt.ylabel("$\int f_h(x) - \int f(x)$")
    plt.title("Difference between integral calculations")
    plt.legend(loc = "best")
    plt.tight_layout()
    plt.savefig("./report/figures/14-3.png")
    ##############################################################################
    ##############################################################################


    ###################### STEP SIZE COMPARISON FOR NUMERICAL INTEGRAL CALCULATION #########

    a = -1  # Start point of the interval
    b = 1  # End point of the interval
    ###################### PLOTTING ##########################################################

    methods = ["trapezoidal", "simpson"]
    for (ix, method) in enumerate(methods):

        plt.figure()
        plt.xlabel("$x$")
        plt.ylabel("$\int f_h(x) - \int f_{\dfrac{h}{2}}(x)$")

        if method == "trapezoidal":
            scale_factor = 4
            start_power = 15
        else:
            scale_factor = 16
            start_power = 12

        for k in [2, 1, 0]:  # Iteration is flipped to make the figure clearer

            n1 = 2**(start_power + k)  # Number of spacing for first step size
            x = np.linspace(a, b, n1 + 1)  # Sampling between a and b
            # Desired function evaluated at this interval
            fx = np.sin(np.pi * x + np.pi / 4) + 1
            # Numerical integral calculation for first step size
            I_fx_1 = numerical_integral(fx, a, b, n1, method=method)

            # Number of spacing for second step size
            n2 = 2**(start_power + k + 1)
            x = np.linspace(a, b, n2 + 1)  # Sampling between a and b
            # Desired function evaluated at this interval
            fx = np.sin(np.pi * x + np.pi / 4) + 1
            # Numerical integral calculation for second step size
            I_fx_2 = numerical_integral(fx, a, b, n2, method=method)

            plt.plot(x[1::2], (I_fx_1 - I_fx_2[::2]) * (scale_factor ** k), label="h = {0}".format(
                str((b-a)/n1)), ls='-.', marker="o", markevery=int(I_fx_1.shape[0] / random.randint(10, 20)))

            plt.title("Differences between numerical integrals\nusing {0} method\nScale Factor = {1}".format(
                method.capitalize(), scale_factor))

        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig("./report/figures/14-4-{0}.png".format(ix))
    #########################################################################################
    #########################################################################################

    #plt.show()


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


if __name__ == "__main__":
    mytests()
