import numpy as np
import matplotlib.pyplot as plt
from re import error
import random
from ps0514 import numerical_derivative, numerical_integral

def mytests():
    """
    mytests

    This function executes the following processes:
        - It visually compares the numerical and analytical second derivative
        - It visually compares the numerical and analytical second derivative,
            where the numerical second derivative is obtained by two successive
            first derivative
        - It visually compares the original function and the function obtained
            after successive numerical integration and derivation
        - It visually shows the effect of step size in the 
            "relative" error for all above operations
    Input:
    []

    Output:
    []

    Usage:
    mytests()

    """

    ############################ NUMERICAL CALCULATION VS ANALYTICAL SECOND DERIVATIVE COMPARISON ######################
    a = -1  # Start point of the interval
    b = 1  # End point of the interval
    n = 2**15  # Number of "spacing", not length/size
    x = np.linspace(a, b, n + 1)  # Sampling between a and b
    # Desired function evaluated at this interval
    fx = np.sin(np.pi * x + np.pi / 4) + 1
    # Analytical second derivative calculation
    d2_fx_analytical = -np.sin(np.pi * x + np.pi / 4) * (np.pi ** 2)
    # Numerical second derivative calculation
    d2_fx_numerical = numerical_2derivative(fx, a, b, n)

    ############################ PLOTTING OF SECOND DERIVATIVE CALCULATIONS #############################################
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x, d2_fx_analytical, label="Analytical $2^{nd}$ Derivative", marker="o", markevery=int(
        d2_fx_analytical.shape[0] / random.randint(10, 20)))
    plt.plot(x, d2_fx_numerical, label="Numerical $2^{nd}$ Derivative")
    plt.xlabel("$x$")
    plt.ylabel("$f^{\;\prime\prime}(x)$")
    plt.title(
        "Investigation of the Function\n$f(x) = 1+\sin(\pi x + \dfrac{\pi}{4})$")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.subplot(2, 1, 2)
    plt.plot(x, d2_fx_numerical - d2_fx_analytical, marker="o",
             markevery=int(d2_fx_analytical.shape[0] / random.randint(10, 20)))
    plt.xlabel("$x$")
    plt.ylabel("$f_h^{\;\prime\prime}(x) - f^{\;\prime\prime}(x)$")
    plt.title("Difference between $2^{nd}$ derivative calculations")
    plt.tight_layout()
    plt.savefig("./report/figures/15-1.png")

    ############################################################################################
    ############################################################################################

    ###################### STEP SIZE COMPARISON FOR NUMERICAL SECOND DERIVATIVE CALCULATION #########

    a = -1  # Start point of the interval
    b = 1  # End point of the interval
    start_power = 12
    scale_factor = 4
    ###################### PLOTTING ##########################################################
    plt.figure()
    plt.xlabel("$x$")
    plt.ylabel(
        "$f_h^{\;\prime\prime}(x) - f_{\dfrac{h}{2}}^{\;\prime\prime}(x)$")
    plt.title("Differences between numerical second derivatives\nScale Factor = {0}".format(scale_factor))

    for k in [2, 1, 0]:  # Iteration is flipped to make the figure clearer

        n1 = 2**(start_power + k)  # Number of spacing for first step size
        x = np.linspace(a, b, n1 + 1)  # Sampling between a and b
        # Desired function evaluated at this interval
        fx = np.sin(np.pi * x + np.pi / 4) + 1
        # Numerical second derivative calculation for first step size
        d2_fx_1 = numerical_2derivative(fx, a, b, n1)

        n2 = 2**(start_power + k + 1)  # Number of spacing for second step size
        x = np.linspace(a, b, n2 + 1)  # Sampling between a and b
        # Desired function evaluated at this interval
        fx = np.sin(np.pi * x + np.pi / 4) + 1
        # Numerical second derivative calculation for second step size
        d2_fx_2 = numerical_2derivative(fx, a, b, n2)

        plt.plot(x[::2], (d2_fx_1 - d2_fx_2[::2]) * (scale_factor ** k), label="h = {0}".format(str(
            (b-a)/n1)), ls="-.", marker="o", markevery=int(d2_fx_1.shape[0] / random.randint(10, 20)))

    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("./report/figures/15-2.png")
    #########################################################################################
    #########################################################################################

    ############################ NUMERICAL CALCULATION VS ANALYTICAL SECOND DERIVATIVE COMPARISON ######################
    ############################ SECOND DERIVATIVE AS SUCCESSIVE TWO FIRST DERIVATIVES #################################

    a = -1  # Start point of the interval
    b = 1  # End point of the interval
    n = 2**15  # Number of "spacing", not length/size
    x = np.linspace(a, b, n + 1)  # Sampling between a and b
    # Desired function evaluated at this interval
    fx = np.sin(np.pi * x + np.pi / 4) + 1
    # Analytical second derivative calculation
    d2_fx_analytical = -np.sin(np.pi * x + np.pi / 4) * (np.pi ** 2)
    # Numerical first derivative calculation
    d_fx_numerical = numerical_derivative(fx, a, b, n)
    # Numerical first derivative calculation
    d2_fx_numerical = numerical_derivative(d_fx_numerical, a, b, n)

    ############################ PLOTTING OF SECOND DERIVATIVE CALCULATIONS #############################################
    plt.figure(figsize = [6.4, 7])
    plt.subplot(3, 1, 1)
    plt.plot(x, d2_fx_analytical, label="Analytical $2^{nd}$ Derivative", marker="o", markevery=int(
        d2_fx_analytical.shape[0] / random.randint(10, 20)))
    plt.plot(x, d2_fx_numerical, label="Numerical $2^{nd}$ Derivative")
    plt.xlabel("$x$")
    plt.ylabel("$f^{\;\prime\prime}(x)$")
    plt.title(
        "Investigation of the Function\n$f(x) = 1+\sin(\pi x + \dfrac{\pi}{4})$\n$2^{nd}$ Derivative as Successive $1^{st}$ Derivatives")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.subplot(3, 1, 2)
    plt.plot(x, (d2_fx_numerical - d2_fx_analytical), marker="o",
             markevery=int(d2_fx_analytical.shape[0] / random.randint(10, 20)))
    plt.xlabel("$x$")
    plt.ylabel("$f_h^{\;\prime\prime}(x) - f^{\;\prime\prime}(x)$")
    plt.title("Difference between $2^{nd}$ derivative calculations")
    plt.tight_layout()
    plt.subplot(3, 1, 3)
    plt.plot(x[10:-10], (d2_fx_numerical - d2_fx_analytical)[10:-10], marker="o",
             markevery=int(d2_fx_analytical.shape[0] / random.randint(10, 20)))
    plt.xlabel("$x$")
    plt.ylabel("$f_h^{\;\prime\prime}(x) - f^{\;\prime\prime}(x)$")
    plt.title("Difference between $2^{nd}$ derivative calculations\n(Clipped from tips)")
    plt.tight_layout()
    plt.savefig("./report/figures/15-3.png")
    ############################################################################################
    ############################################################################################

    ######################## DERIVATIVE INTEGRAL INVERSION ####################################
    a = -1  # Start point of the interval
    b = 1  # End point of the interval
    n = 2**15  # Number of "spacing", not length/size
    x = np.linspace(a, b, n + 1)  # Sampling between a and b
    # Desired function evaluated at this interval
    fx = np.sin(np.pi * x + np.pi / 4) + 1
    I_fx_numerical = numerical_integral(
        fx, a, b, n, method="trapezoidal")  # Numerical integral calculation
    # Adding to the trivial integral of the function its starting point to match size
    I_fx_numerical_cum = np.block([0.0, I_fx_numerical])
    # Numerical derivative calculation of the numerical integral
    fx_numerical = numerical_derivative(I_fx_numerical_cum, a, b, n)

    ##################### PLOTTING ##############################################
    plt.figure(figsize = [6.4, 7])
    plt.subplot(3, 1, 1)
    plt.plot(x, fx, label="Original Function", marker="o", markevery=int(
        fx.shape[0] / random.randint(10, 20)))
    plt.plot(x, fx_numerical,
             label="Function Obtained After\nNumerical Integration and Derivation")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.title(
        "Investigation of the Function\n$f(x) = 1+\sin(\pi x + \dfrac{\pi}{4})$")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.subplot(3, 1, 2)
    plt.plot(x, fx - fx_numerical, marker="o",
             markevery=int(fx.shape[0] / random.randint(10, 20)))
    plt.xlabel("$x$")
    plt.ylabel("$f_h(x) - f(x)$")
    plt.title("Difference between original and numerically calculated function")
    plt.tight_layout()    
    plt.subplot(3, 1, 3)
    plt.plot(x[10:-10], (fx - fx_numerical)[10:-10], marker="o",
             markevery=int(fx.shape[0] / random.randint(10, 20)))
    plt.xlabel("$x$")
    plt.ylabel("$f_h(x) - f(x)$")
    plt.title("Difference between original and numerically calculated function\n(Clipped from tips)")
    plt.tight_layout()
    plt.savefig("./report/figures/15-4.png")
    ############################################################################################
    ############################################################################################

    ###################### STEP SIZE COMPARISON FOR INTEGRAL DERIVATIVE INVERSION #########

    a = -1  # Start point of the interval
    b = 1  # End point of the interval

    ###################### PLOTTING ##########################################################
    methods = ["trapezoidal", "simpson"]
    for (ix, method) in enumerate(methods):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_xlabel("$x$")
        ax1.set_ylabel(
            "$f_h(x) - f_{\dfrac{h}{2}}(x)$")
        ax2.set_xlabel("$x$")
        ax2.set_ylabel(
            "$f_h(x) - f_{\dfrac{h}{2}}(x)$")
        
        if method == "trapezoidal":
            scale_factor = 2
            start_power = 15
        else:
            scale_factor = 2
            start_power = 13

        for k in [2, 1, 0]:  # Iteration is flipped to make the figure clearer

            n1 = 2**(start_power + k)  # Number of spacing for first step size
            x = np.linspace(a, b, n1 + 1)  # Sampling between a and b
            # Desired function evaluated at this interval
            fx = np.sin(np.pi * x + np.pi / 4) + 1
            I_fx_1 = numerical_integral(
                fx, a, b, n1, method=method)  # Numerical integral calculation
            # Adding to the trivial integral of the function its starting point to match size
            I_fx_1_cum = np.block([0.0, I_fx_1])
            # Numerical derivative calculation of the numerical integral
            fx_1 = numerical_derivative(I_fx_1_cum, a, b, n1)

            # Number of spacing for second step size
            n2 = 2**(start_power + k + 1)
            x = np.linspace(a, b, n2 + 1)  # Sampling between a and b
            # Desired function evaluated at this interval
            fx = np.sin(np.pi * x + np.pi / 4) + 1
            I_fx_2 = numerical_integral(
                fx, a, b, n2, method=method)  # Numerical integral calculation
            # Adding to the trivial integral of the function its starting point to match size
            I_fx_2_cum = np.block([0.0, I_fx_2])
            # Numerical derivative calculation of the numerical integral
            fx_2 = numerical_derivative(I_fx_2_cum, a, b, n2)

            
            ax1.plot(x[::2], (fx_1[:] - fx_2[::2]) * (scale_factor ** k), label="h = {0}".format(str(
                (b-a)/n1)), ls="-.", marker="o", markevery=int(fx_1.shape[0] / random.randint(10, 20)))

            ax2.plot(x[10:-10:2], (fx_1[5:-5] - fx_2[10:-10:2]) * (scale_factor ** k), label="h = {0}".format(str(
                (b-a)/n1)), ls='-.', marker="o", markevery=int(fx_1.shape[0] / random.randint(10, 20)))

        ax1.set_title("Differences between numerically calculated function\nusing {0} method\nScale Factor = {1}".format(method.capitalize(), scale_factor))
        ax2.set_title("Clipped from tips")
        ax1.legend(loc = "best")
        ax2.legend(loc = "best")
        fig.tight_layout()
        fig.savefig("./report/figures/15-5-{0}.png".format(ix))
    #########################################################################################
    #########################################################################################

    #plt.show()


def numerical_2derivative(fx, a, b, n):
    """
    numerical_2derivative

    This function executes the following processes:
        - It calculates the numerical second derivative of a function for a given interval [a, b]
        - It uses 3 point stencil at center, forward and backward

    Input:

        fx = Evaluated function vector whose derivative will be calculated
        a = Start point of interval
        b = End point of interval
        n = Number of spacing in the whole interval

    Output:

        d2_fx = Calculated numerical derivative of the function vector fx

    Usage:

        a = -1  # Start point of the interval
        b = 1  # End point of the interval
        n = 2**15  # Number of "spacing", not length/size
        x = np.linspace(a, b, n + 1)  # Sampling between a and b
        # Desired function evaluated at this interval
        fx = np.sin(np.pi * x + np.pi / 4) + 1
        # Analytical second derivative calculation
        d2_fx_analytical = -np.sin(np.pi * x + np.pi / 4) * (np.pi ** 2)
        # Numerical second derivative calculation
        d2_fx_numerical = numerical_2derivative(fx, a, b, n)

    """

    d2_fx = np.empty(n+1)
    h = (b - a) / n  # Individual spacing length / step size

    for k in range(1, n):  # Terminal point will be processed differently

        d2_fx[k] = (fx[k - 1] - 2 * fx[k] + fx[k + 1]) / \
            (h ** 2)  # Central difference with 3 point stencil

    d2_fx[0] = (2 * fx[0] - 5 * fx[1] + 4 * fx[2] - fx[3]) / \
        (h ** 2)  # Forward difference with 4 point stencil
    d2_fx[n] = (2 * fx[n] - 5 * fx[n-1] + 4 * fx[n-2] - fx[n-3]) / \
        (h ** 2)  # Backward difference with 4 point stencil

    return d2_fx


if __name__ == "__main__":
    mytests()
