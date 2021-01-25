import numpy as np
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
        - It visually compares the numerical derivative of the irregularly spaced data with
            the derivative obtained by the fitted cubic spline of the same data
        - It visually compares the numerical integral of the irregularly spaced data with
            the derivative obtained by the fitted cubic spline of the same data
        - It also compares the custom Simpson's method and SciPy's Simpson's method implementation
    Input:
    []

    Output:
    []

    Usage:
    mytests()
    """

    ################### DERIVATIVE CALCULATION OF IRREGULARLY SPACED DATA ############################

    DATA_PATH = "data/hubble.dat"  # Specify data path
    z_raw, dL_raw, _ = read_hubble_data(
        filename=DATA_PATH, array_type="np")  # Read it with custom function

    z, ixs = np.unique(z_raw, return_inverse=True)  # Get unique value of z
    # Calculate mean of the repeate values
    dL = np.bincount(ixs, weights=dL_raw) / np.bincount(ixs)

    # Custom irregularly spaced finite difference calculation
    d_dL = numerical_derivative_irregular(dL, z)

    cs = CubicSpline(z, dL)  # SciPy cubic spline calculation

    ############################ PLOTTING OF DERIVATIVE CALCULATIONS #############################################
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(z, d_dL, label="Numerical Derivative", marker="o",
             markevery=int(z.shape[0] / random.randint(10, 20)))
    plt.plot(z, cs(z, 1), label="Cubic Spline Derivative")
    plt.xlabel("$z$")
    plt.ylabel("$d_L^{\;\prime}$")
    plt.title(
        "Derivative calculations")
    plt.legend(loc="best")
    plt.subplot(2, 1, 2)
    plt.plot(z, d_dL - cs(z, 1), marker="o",
             markevery=int(z.shape[0] / random.randint(10, 20)))
    plt.xlabel("$z$")
    plt.ylabel(
        "${d_L^{\;\prime}}_{numerical} \;-\; {d_L^{\;\prime}}_{spline} $")
    plt.title("Difference between derivative calculations")
    plt.tight_layout()
    plt.savefig("./report/figures/16-1.png")
    ############################################################################################
    ############################################################################################

    ########################## NUMERICAL INTEGRAL FOR UNEVENLY SPACED DATA ##################

    I_dL_numerical = numerical_integral_irregular(
        dL, z)  # Irregular integral calculation
    cs = CubicSpline(z, dL)  # Cubic spline fitting to data
    # Array comprehension is used to iterate over different intervals
    I_dL_spline = [cs.integrate(z[0], z[k]) for k in range(1, z.shape[0])]
    ################## PLOTTING ########################################################
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(z[1:], I_dL_numerical, label="Numerical Integral", marker="o",
             markevery=int(z.shape[0] / random.randint(10, 20)))
    plt.plot(z[1:], I_dL_spline, label="Cubic Spline Integral")
    plt.xlabel("$z$")
    plt.ylabel("$\int_0^z d_L(z)$")
    plt.title(
        "Integral calculations")
    plt.legend(loc="best")
    plt.subplot(2, 1, 2)
    plt.plot(z[1:], I_dL_numerical - I_dL_spline, marker="o",
             markevery=int(z.shape[0] / random.randint(10, 20)))
    plt.xlabel("$x$")
    plt.ylabel(
        "${\int_0^z d_L(z)}_{numerical} \;-\; {\int_0^z d_L(z)}_{spline} $")
    plt.title("Difference between integral calculations")
    plt.tight_layout()
    plt.savefig("./report/figures/16-2.png")
    ############################################################################################
    ############################################################################################

    ############################## TIMING CALCULATIONS #########################################
    time_vec = np.empty(2)  # Preallocation of timing vector
    num_iter = 100  # Number of iterations to calculate average timing
    # Timing of custom Simpson's
    time_vec[0] = timeit(
        lambda: numerical_integral_irregular(dL, z), number=num_iter)
    # Timing of SciPy Simpson's, generator object is used to iterate over different ranges
    time_vec[1] = timeit(lambda: (simps(dL[:k], z[:k])
                                  for k in range(1, z.shape[0])), number=num_iter)

    time_vec /= num_iter  # Take average
    print(time_vec)
    ################ PLOTTING ###################################################
    plt.figure()
    labels = ["Custom\nSimpson's\nMethod", "SciPy\nSimpson's\nMethod"]
    plt.title("Time Comparison of Simpson's Method Execution")
    plt.ylabel("Average Elapsed Time [sec]")
    plt.bar(labels, time_vec)
    plt.savefig("./report/figures/16-3.png")
    #############################################################################
    # Custom Simpson's method calculation
    I_dL_custom = numerical_integral_irregular(dL, z)
    # SciPy Simpson's method calculation, array comprehension is used to iterate over different intervals
    I_dL_scipy = [simps(dL[:k], z[:k]) for k in range(1, z.shape[0])]

    ################### PLOTTING ################################################

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(z[1:], I_dL_custom, label="Custom Simpson's Method Implementation", marker="o",
             markevery=int(z.shape[0] / random.randint(10, 20)))
    plt.plot(z[1:], I_dL_scipy, label="SciPy Simpson's Method Implementation")
    plt.xlabel("$z$")
    plt.ylabel("$\int_0^z d_L(z)$")
    plt.title(
        "Integral calculations")
    plt.legend(loc="best")
    plt.subplot(2, 1, 2)
    plt.plot(z[1:], I_dL_custom - I_dL_scipy, marker="o",
             markevery=int(z.shape[0] / random.randint(10, 20)))
    plt.xlabel("$x$")
    plt.ylabel(
        "${\int_0^z d_L(z)}_{custom} \;-\; {\int_0^z d_L(z)}_{scipy} $")
    plt.title("Difference between integral calculations")
    plt.tight_layout()
    plt.savefig("./report/figures/16-4.png")
    #############################################################################
    #############################################################################

    #plt.show()


def read_hubble_data(filename, array_type="np"):
    """
    Syntax:
    (z, dL, error_dL) = read_hubble_data(filename)

    This function perform the following processes:
        -It reads the desired "Hubble" data file
        -Outputs its three columns as a seperate lists

    Input:
        filename = Filename path to be read 
        array_type = Output array format. It can be normal Python array or NumPy array

    Output:
        z = Redshift parameter
        dL = Luminosity distance of the source in megaparsecs [Mpc]
        error_dL = Error in the measurement of dL 

    Example:
        fpath = "data/hubble.dat" # Please specify the hubble data location
        z, dL, error_dL = read_hubble_data(fpath) # Function outputs columns as a seperate lists
    """

    z = []  # z list preallocation
    dL = []  # dL list preallocation
    error_dL = []  # error_dL list preallocation

    with open(filename, "r") as hubble:  # Open file to read
        for line in hubble:  # Iterate over lines

            z_temp, dL_temp, error_dL_temp = line.split()  # Fetch element-wise values

            z.append(float(z_temp))  # Append z
            dL.append(float(dL_temp))  # Append dL
            error_dL.append(float(error_dL_temp))  # Append error_dL

    if array_type == "np":
        return np.array(z), np.array(dL), np.array(error_dL)
    elif array_type == "pyt":
        return z, dL, error_dL
    else:
        return z, dL, error_dL


def numerical_derivative_irregular(fx, x):
    """
    numerical_derivative_irregular

    This function executes the following processes:
        - It calculates the numerical derivative of a function for a given irregular x
        - It uses 3 point stencil at center, and 2 point stencil at forward and backward

    Input:

        fx = Evaluated function vector whose derivative will be calculated
        x = Irregularly spaced sample points

    Output:

        d_fx = Calculated numerical derivative of the function vector fx

    Usage:

        z, dL, dL_error = read_hubble_data(filename=DATA_PATH, array_type="np") # Read it with custom function
        d_dL = numerical_derivative_irregular(dL, z)
    """

    N = x.shape[0]  # Size of the data
    d_fx = np.empty(N)  # Preallocation of the derivative vector

    n = N-1

    for k in range(1, n):

        # Stepsizes for the stencil calculation
        S = np.array([x[k-1] - x[k], 0.0, x[k+1] - x[k]])
        stencils = get_stencils(S)  # Stencil calculation

        # Derivative calculation with 3 point arbitrary stencil
        d_fx[k] = stencils[0] * fx[k-1] + \
            stencils[1] * fx[k] + stencils[2] * fx[k+1]

    d_fx[0] = (fx[1] - fx[0]) / (x[1] - x[0])  # Forward finite difference
    d_fx[n] = (fx[n] - fx[n-1]) / (x[n] - x[n-1])  # Backward finite difference

    return d_fx


def get_stencils(S, order=1):
    """
    get_stencils

    This function executes the following processes:
        - It calculates stencils according to given sample points

    Input:

        S = Step sizes at which the stencils will be calculated 

    Output:

        stencils = Calculated finite difference stencils

    Usage:

        S = np.array([-0.01, 0.0, 0.01])
        stencils = get_stencils(S)
    """

    n = S.shape[0]  # Number of stencils required
    A = np.empty((n, n))  # Preallocation of the A matrix

    b = np.zeros(n)  # Preallocation of the b matrix
    # Fill (order+1)th row with order! [it is from derivation]
    b[order] = np.math.factorial(order)

    for k in range(n):
        A[k] = S ** k  # Fill the A matrix [it is from derivation]

    stencils = np.linalg.solve(A, b)  # Solve the linear system

    return stencils


def numerical_integral_irregular(fx, x):
    """
    numerical_integral_irregular

    This function executes the following processes:
        - It calculates the numerical integral of a function which is unevenly spaced
        - It uses Simpson method

    Input:

        fx = Evaluated function vector whose derivative will be calculated
        x = Sample points

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
    N = x.shape[0]  # Size of x
    n = N - 1  # Number of interval
    I_fx = np.empty(n)  # Preallocation of output integral vector

    # Calculation scheme taken from Wikipedia
    for k in range(1, n+1):

        I = 0.0

        if k % 2 == 0:

            for i in range(0, k, 2):

                # Intermediate spacings
                h1 = x[i + 1] - x[i]
                h2 = x[i + 2] - x[i + 1]
                h3 = h1 + h2
                h4 = h1 * h2

                alpha = (2 * h2**3 - h1**3 + 3 * h4 * h2) / (6 * h2 * h3)
                beta = (h2**3 + h1**3 + 3 * h3 * h4) / (6 * h4)
                eta = (2 * h1**3 - h2**3 + 3 * h4 * h1) / (6 * h1 * h3)

                I += alpha * fx[i + 2] + beta * fx[i + 1] + eta * fx[i]

            I_fx[k-1] = I

        else:

            for i in range(0, k-1, 2):

                # Intermediate spacings
                h1 = x[i + 1] - x[i]
                h2 = x[i + 2] - x[i + 1]
                h3 = h1 + h2
                h4 = h1 * h2

                alpha = (2 * h2**3 - h1**3 + 3 * h4 * h2) / (6 * h2 * h3)
                beta = (h2**3 + h1**3 + 3 * h3 * h4) / (6 * h4)
                eta = (2 * h1**3 - h2**3 + 3 * h4 * h1) / (6 * h1 * h3)

                I += alpha * fx[i + 2] + beta * fx[i + 1] + eta * fx[i]

            # Intermediate spacings
            h1 = x[k] - x[k-1]
            h2 = x[k-1] - x[k-2]
            h3 = h1 + h2
            h4 = h1 * h2

            alpha = (2 * h1**2 + 3 * h4) / (6 * h3)
            beta = (h1**2 + 3 * h4) / (6 * h2)
            eta = (h1**3) / (6 * h2 * h3)
            I += alpha * fx[k] + beta * fx[k - 1] - eta * fx[k - 2]

            I_fx[k - 1] = I

    return I_fx


if __name__ == "__main__":
    mytests()
