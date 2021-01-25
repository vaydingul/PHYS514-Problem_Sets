import numpy as np
from timeit import timeit
import matplotlib.pyplot as plt
from numpy.core.multiarray import concatenate
import scipy.special as spspec
import scipy.sparse as spspar


def mytests():
    """
    mytests 

    Syntax: mytests()

    This function do the following processes:
        - It calculates the average elapsed time for root finding of Lambert W function for a given input vector using different
          iterative methods.
        - It plots the results of the average and total elapsed time for different methods.
        - It plots the function evaluations and differences of different iterative methods and Numpy implementation of 
          Lambert W function
        - It calculates the average elapsed time for root finding of Koc W function for a given input vector using different
          iterative methods.
        - It plots the results of the average and total elapsed time for different methods.
        - It plots the function evaluations and differences of different iterative methods and Numpy implementation of 
          Koc W function


    """
    num_iter = 100  # Number of iteration to calculate average of elapsed time

    ###### Lambert W Function ################################################
    time_results = np.empty((5, ))

    x_lambert = np.logspace(-np.log10(1/np.e), 20, num=100)  # Input vector
    # Vectorized form of Lambert W function
    vectorized_lambert = np.vectorize(lambertw_custom)

    # Newton-Raphson solution for Lambert W function
    y_lambertw_custom_newton = lambertw_custom(x_lambert)
    # Halley's solution for Lambert W function
    y_lambertw_custom_halley = lambertw_custom(x_lambert, method="halley")
    y_lambertw_numpy = spspec.lambertw(x_lambert)  # NumPy solution

    # Timing of Newton-Raphson method for Lambert W function
    time_results[0] = timeit(
        lambda: lambertw_custom(x_lambert), number=num_iter)

    # Timing of Halley's method for Lambert W function
    time_results[1] = timeit(lambda: lambertw_custom(
        x_lambert, method="halley"), number=num_iter)

    # Timing of Newton-Raphson method for Lambert W function [vectorized]
    time_results[2] = timeit(
        lambda: vectorized_lambert(x_lambert), number=num_iter)

    # Timing of Halley's method for Lambert W function [vectorized]
    time_results[3] = timeit(lambda: vectorized_lambert(
        x_lambert, method="halley"), number=num_iter)

    # Timing of NumPy built-in for Lambert W function
    time_results[4] = timeit(
        lambda: spspec.lambertw(x_lambert), number=num_iter)

    ##################### PLOTTING OF LAMBERT W RESULTS ##################################
    plt.figure()
    labels = ["Newton-Raphson \nMethod", "Halley's \nMethod",
              "Vectorized \nNewton-Raphson \nMethod", "Vectorized \nHalley's \nMethod", "SciPy \nBuilt-in"]
    plt.title("Time Comparison of LambertW Function Calculation")
    plt.ylabel("Average Elapsed Time [sec]")
    plt.bar(labels, time_results/num_iter)

    plt.figure()
    labels = ["Newton-Raphson \nMethod", "Halley's \nMethod",
              "SciPy \nBuilt-in"]
    plt.title(
        "Time Comparison of LambertW Function \nCalculation without Vectorized Functions")
    plt.ylabel("Average Elapsed Time [sec]")
    plt.bar(labels, time_results[[0, 1, 4]]/num_iter)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x_lambert, y_lambertw_custom_newton, "ro")
    plt.plot(x_lambert, y_lambertw_custom_halley, "b.")
    plt.plot(x_lambert, y_lambertw_numpy)
    # plt.xlabel("x")
    plt.ylabel("y")
    plt.title("LambertW Function Calculation")
    plt.legend(["Newton-Raphson Method", "Halley's Method",
                "SciPy Built-in"], loc="best")

    plt.subplot(2, 1, 2)
    plt.plot(x_lambert, y_lambertw_numpy - y_lambertw_custom_newton, "r-")
    plt.plot(x_lambert, y_lambertw_numpy - y_lambertw_custom_halley, "bo")
    plt.xlabel("x")
    plt.ylabel("Difference in y")
    plt.title("Difference in LambertW Function Calculation")
    plt.legend(["SciPy Built-in - Newton-Raphson Method",
                "SciPy Built-in - Halley's Method"], loc="best")
    #########################################################################
    #########################################################################

###### Koc W Function ################################################
    time_results = np.empty((5, ))

    x_kocw = np.logspace(0, 20, num=100)  # Input vector
    vectorized_kocw = np.vectorize(kocw)  # Vectorized form of Koc W function

    # Newton-Raphson solution for Koc W function
    y_kocw_custom_newton = kocw(x_kocw)
    # Halley's solution for Koc W function
    y_kocw_custom_halley = kocw(x_kocw, method="halley")
    # NumPy solution for Koc W function
    y_kocw_numpy = np.sqrt(spspec.lambertw(x_kocw))

    # Timing of Newton-Raphson method for Koc W function
    time_results[0] = timeit(
        lambda: kocw(x_kocw), number=num_iter)

    # Timing of Halley's method for Koc W function
    time_results[1] = timeit(lambda: kocw(
        x_kocw, method="halley"), number=num_iter)

    # Timing of Newton-Raphson method for Koc W function [vectorized]
    time_results[2] = timeit(
        lambda: vectorized_kocw(x_kocw), number=num_iter)

    # Timing of Halley's method for Koc W function [ vectorized]
    time_results[3] = timeit(lambda: vectorized_kocw(
        x_kocw, method="halley"), number=num_iter)

    # Timing of NumPy built-in
    time_results[4] = timeit(
        lambda: np.sqrt(spspec.lambertw(x_kocw)), number=num_iter)

    ################### PLOTTING OF KOC W FUNCTION #####################################
    plt.figure()
    labels = ["Newton-Raphson \nMethod", "Halley's \nMethod",
              "Vectorized \nNewton-Raphson \nMethod", "Vectorized \nHalley's \nMethod", "SciPy \nBuilt-in"]
    plt.title("Time Comparison of KocW Function Calculation")
    plt.ylabel("Average Elapsed Time [sec]")
    plt.bar(labels, time_results/num_iter)

    plt.figure()
    labels = ["Newton-Raphson \nMethod", "Halley's \nMethod",
              "SciPy \nBuilt-in"]
    plt.title(
        "Time Comparison of KocW Function \nCalculation without Vectorized Functions")
    plt.ylabel("Average Elapsed Time [sec]")
    plt.bar(labels, time_results[[0, 1, 4]]/num_iter)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x_kocw, y_kocw_custom_newton, "ro")
    plt.plot(x_kocw, y_kocw_custom_halley, "b.")
    plt.plot(x_kocw, y_kocw_numpy)
    # plt.xlabel("x")
    plt.ylabel("y")
    plt.title("KocW Function Calculation")
    plt.legend(["Newton-Raphson Method", "Halley's Method",
                "SciPy Built-in"], loc="best")

    plt.subplot(2, 1, 2)
    plt.plot(x_kocw, y_lambertw_numpy - y_lambertw_custom_newton, "r-")
    plt.plot(x_kocw, y_lambertw_numpy - y_lambertw_custom_halley, "bo")
    plt.xlabel("x")
    plt.ylabel("Difference in y")
    plt.title("Difference in KocW Function Calculation")
    plt.legend(["SciPy Built-in - Newton-Raphson Method",
                "SciPy Built-in - Halley's Method"], loc="best")
    ###############################################################
    ###############################################################
    plt.show()


def lambertw_custom(x, method="newton"):
    """
    lambertwCustom
    Syntax: y = lambertwCustom(x, method)

    It calculates the Lambert W fucntion defined as W(xe^x) = y if
    y = xe^x, using the Newton-Raphson method or the Halley's method.

    Inputs:
    x = The vector of points which the Lambert W function should be evaluated
    method = Method selection ==> "h" for Halley's method, "n" for Newton-Raphson Method

    Outputs:
    y = The vector of evaluated function points
    """
    def fun_shifted(z): return z * np.exp(z) - \
        x  # Function whose root is desired

    def fun_deriv(z): return (z+1) * np.exp(z)  # Derivative of fun_shifted
    def fun_deriv2(z): return (z+2) * np.exp(z)  # Derivative of fun_deriv

    x_init = np.where(x > np.e, np.log(np.abs(x)), 1)

    if method == "newton":
        y = newton(fun_shifted, fun_deriv, x_init)
    elif method == "halley":
        y = halley(fun_shifted, fun_deriv, fun_deriv2, x_init)
    else:
        y = newton(fun_shifted, fun_deriv, x_init)

    return y


def kocw(x, method="newton"):
    """
    kocw
    Syntax: y = kocw(x, method)

    It calculates the Koc W fucntion defined as KocW(x^2 e^(x^2)) = y if
    y = x^2 e^(x^2), using the Newton-Raphson method or the Halley's method.

    Inputs:
    x = The vector of points which the Lambert W function should be evaluated
    method = Method selection ==> "h" for Halley's method, "n" for Newton-Raphson Method

    Outputs:
    y = The vector of evaluated function points
    """

    def fun_shifted(z): return (z**2) * np.exp(z**2) - \
        x  # Function whose root is desired

    def fun_deriv(z): return (2*z**3 + 2*z) * \
        np.exp(z**2)  # Derivative of fun_shifted
    def fun_deriv2(z): return (4*z**4 + 10*z**2 + 2) * \
        np.exp(z**2)  # Derivative of fun_deriv

    x_init = np.where(x > np.e, np.sqrt(np.log(np.abs(x))), 1)

    if method == "newton":
        y = newton(fun_shifted, fun_deriv, x_init)
    elif method == "halley":
        y = halley(fun_shifted, fun_deriv, fun_deriv2, x_init)
    else:
        y = newton(fun_shifted, fun_deriv, x_init)

    return y


def newton(fun, dfun, x_init, tolerance=1e-6):
    """
    newton 

    Syntax: root = newton(fun, dfun, x_init)


    Solves fun(x) = 0 using the Newton-Raphson method. It requires an initial guess
    and function handles to fun(x) and its derivative, fun and dfun respectively.

    Inputs:
    fun = Function whose root is to be determined
    dfun = Derivative of fun
    x_init = Initial root estimation
    tolerance = Iteration termination condition

    Outputs:
    root = Estimated root
    """
    x_prev = x_init  # Initial estimation
    x_next = x_prev + 1

    while np.linalg.norm(x_next - x_prev) > tolerance:
        x_prev = x_next  # Storing previous estimate for comparison
        # Newton-Raphson iteration
        x_next = x_prev - fun(x_prev) / dfun(x_prev)

    root = x_next
    return root


def halley(fun, dfun, ddfun, x_init, tolerance=1e-6):
    """
    halley - Description

    Syntax: root = halley(fun, dfun, ddfun, x_init)

    Solves fun(x) = 0 using the Halley's method. It requires an initial guess
    and function handles to fun(x) and its first and second derivative, fun, dfun and ddfun respectively.

    Inputs:
    fun = Function whose root is to be determined
    dfun = Derivative of fun
    ddfun = Derivative of dfun
    x_init = Initial root estimation
    tolerance = Iteration termination condition

    Outputs:
    root = Estimated root
    """
    x_prev = x_init  # Initial estimation allocation
    x_next = x_prev + 1

    while np.linalg.norm(x_next - x_prev) > tolerance:
        x_prev = x_next  # Storing the previous estimation

        x_next = x_prev - (fun(x_prev) / dfun(x_prev)) * (
            1 - ((fun(x_prev) * ddfun(x_prev)) / (2 * dfun(x_prev) * dfun(x_prev))))**(-1)

    root = x_next
    return root


if __name__ == "__main__":
    mytests()
