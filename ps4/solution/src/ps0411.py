import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import scipy.optimize as spo
from re import error

def mytests():
    """
    Syntax:
    mytests()

    This function perform the following processes:
        - It plots dL vs. z with error bars
        - It fits a linear line on the (z, dL) data and calculates Hubble Length
        - It executes a  custom nonlinear curve fitting to the (z,dL) data and compares with the actual data
        - It calculates the uncertainities in the solution of nonlinear fit parameters
        - It executes a  built-in nonlinear curve fitting to the (z,dL) data and calculates the accuracy of the
            obtained nonlinear fit parameters

    Input:
        [] 

    Output:
        [] 

    Example:
        mytests()
    """
    DATA_PATH = "data/hubble.dat" # Specify data path
    z, dL, dL_error = read_hubble_data(filename=DATA_PATH, array_type="np") # Read it with custom function

    ################## Plotting of (dL vs. z) with error bars ##########################
    plt.figure()
    plt.errorbar(z, dL, dL_error, ecolor = "red")
    plt.xlabel("$z$")
    plt.ylabel("$d_L$")
    plt.title("$d_L$ vs. $z$ with errorbars")
    plt.tight_layout()
    plt.savefig("report/figures/errorplot.png")

    #######################################################################################

    a, b = hubble_linear_fit(z , dL, xlim=0.1) # Solution of fitting in the form of dL = az + b
    dL_lin_fitted = a*z + b # Generation of results with fitted parameters


    ################## Plotting of (dL vs. z) with Linear Fit ##########################
    plt.figure()
    plt.plot(z, dL, label = "Original data")
    plt.plot(z, dL_lin_fitted, label = "Linearly Fitted data")
    plt.legend(loc = "best")
    plt.xlabel("$z$")
    plt.ylabel("$d_L$")
    plt.title("Linear Fitting Parameters ($d_L = az+b$):\na = {0} \nb = {1}".format(a,b))
    plt.tight_layout()
    plt.savefig("report/figures/linfit.png")

    #######################################################################################


    dH_guess = a[0] # The Hubble length estimation obtained form linear fit
    # is fed as a guess to nonlinear fit problem

    omega_M_guess = 0.5 # The first guess for omega_M

    vec_hubble_integral = np.vectorize(hubble_integral) # Vectorization of hubble_integral function
    # scipy.integrate.quad does not accept arrays as inputs

    # Nonlinear fit
    dH , omega_M = hubble_nonlinear_fit(z, dL, dH_guess=dH_guess, omega_M_guess=omega_M_guess)
    dL_nonlin_fitted = (1 + z) * dH * vec_hubble_integral(z, omega_M)[0] 
    # Generation of results with nonlinearly fitted parameters
    
    ################## Plotting of (dL vs. z) with Nonlinear Fit ##########################
    plt.figure()
    plt.plot(z, dL, label = "Original data")
    plt.plot(z, dL_nonlin_fitted, label = "Nonlinearly Fitted data with Custom Calculation")
    plt.legend(loc = "best")
    plt.xlabel("$z$")
    plt.ylabel("$d_L$")
    plt.title("Nonlinear Fitting Parameters $d_L = (1+z)d_HI(z,\Omega_M)$:\n$d_H$ = {0} \n$\Omega_M$ = {1}".format(dH, omega_M))
    plt.tight_layout()
    plt.savefig("report/figures/nonlinfit.png")

    #######################################################################################

    ################## Plotting of (dH and omega_M) distributions ##########################
    dH_dist, omega_M_dist = error_approximation(z, dL, dL_error, dH_guess, omega_M_guess) # Distribution calculation of dH and omega_M
    plt.figure()
    plt.subplot(2,1,1)
    plt.hist(dH_dist, density = True, bins = 20, color="red", label= "$d_H \pm \Delta d_H $ = {0:6.2f} $\pm$ {1:4.2f}".format(np.mean(dH_dist), np.std(dH_dist)))
    plt.xlabel("$d_H$ Distribution")
    plt.legend(loc = "best")    
    plt.subplot(2,1,2)
    plt.hist(omega_M_dist, density = True, bins = 20, color="green", label= "$\Omega_M \pm \Delta \Omega_M $ = {0:6.2f} $\pm$ {1:4.2f}".format(np.mean(omega_M_dist), np.std(omega_M_dist)))
    plt.xlabel("$\Omega_M$ Distribution")
    plt.legend(loc = "best")    
    plt.tight_layout()
    plt.savefig("report/figures/dhomegadist.png")

    #######################################################################################
    
    ################## Plotting of (dH and omega_M) obtained from SciPy builtin ##########################
    dH, dH_err_std, omega_M, omega_M_err_std = hubble_nonlinear_fit_numpy(z, dL, dH_guess, omega_M_guess) # Nonlinear built-in curve fitting
    plt.figure()
    plt.subplot(1,2,1)
    plt.bar(x = ["NumPy\nCalculation", "Custom\nCalculation"], height = [dH, np.mean(dH_dist)], yerr = [dH_err_std, np.std(dH_dist)])
    plt.ylabel("$dH$ Values")
    plt.title("$d_H$ Distributions")
    plt.subplot(1,2,2)
    plt.bar(x = ["NumPy\nCalculation", "Custom\nCalculation"], height = [omega_M, np.mean(omega_M_dist)], yerr = [omega_M_err_std, np.std(omega_M_dist)])
    plt.ylabel("$\Omega_M$ Values")
    plt.title("$\Omega_M$ Distributions")
    plt.tight_layout()
    plt.savefig("report/figures/dhomegadistcomp.png")

    #######################################################################################


    dL_nonlin_fitted = (1 + z) * dH * vec_hubble_integral(z, omega_M)[0] 
    # Generation of results with nonlinearly fitted parameters
    ################## Plotting of (dL vs. z) with Nonlinear Fit ##########################
    plt.figure()
    plt.plot(z, dL, label = "Original data")
    plt.plot(z, dL_nonlin_fitted, label = "Nonlinearly Fitted data with SciPy")
    plt.legend(loc = "best")
    plt.xlabel("$z$")
    plt.ylabel("$d_L$")
    plt.title("Nonlinear Fitting Parameters $d_L = (1+z)d_HI(z,\Omega_M)$:\n$d_H$ = {0} \n$\Omega_M$ = {1}".format(dH, omega_M))
    plt.tight_layout()
    plt.savefig("report/figures/nonlinfitscipy.png")

    #######################################################################################

    plt.show()


def read_hubble_data(filename, array_type = "np"):
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

    z = [] # z list preallocation
    dL = [] # dL list preallocation
    error_dL = [] # error_dL list preallocation

    with open(filename, "r") as hubble: # Open file to read
        for line in hubble: # Iterate over lines

            z_temp, dL_temp, error_dL_temp = line.split() # Fetch element-wise values

            z.append(float(z_temp)) # Append z
            dL.append(float(dL_temp)) # Append dL
            error_dL.append(float(error_dL_temp)) # Append error_dL

    if array_type == "np":
        return np.array(z), np.array(dL), np.array(error_dL)
    elif array_type == "pyt":
        return z, dL, error_dL
    else: 
        return z, dL, error_dL


def hubble_linear_fit(x, y, xlim = np.Inf):
    """
    Syntax:
    a, b = hubble_linear_fit(x, y, xlim)

    This function perform the following processes:
        - It fits a line to the data consists of (x,y) pairs in the form of [ax + b = y] 
        - It constraints the of which the x should be used acording to the xlim

    Input:
        x = independent variable 
        y = dependent variable
        xlim = constraint on the independent variable

    Output:
        a = independent variable coefficient
        b = constant coefficient

    Example:
        z, dL, error_dL = read_hubble_data(filename=DATA_PATH, array_type="np") # Read it with custom function
        a, b = linear_fit(z, dL, xlim = 0.1)
    """
    # It can be definitely written in a more compact form. However, for the sake of the simplicity of 
    # the problem, this long-form style was chosen. 


    arg_cond = x < xlim # Boolean indices that satisfies x < xlim
    x = x[arg_cond] # Apply constraint on x
    y = y[arg_cond] # Fetch same constrained indices with x

    n = y.shape[0]    # Calculate elements of linear fit calculation matrices
    xk = np.sum(x)    # Calculate elements of linear fit calculation matrices
    xk2 = np.sum(x**2)# Calculate elements of linear fit calculation matrices
    yk = np.sum(y)    # Calculate elements of linear fit calculation matrices
    xkyk = np.sum(x*y)# Calculate elements of linear fit calculation matrices
    
    LHS = np.array([[n, xk],[xk, xk2]]) # Construct LHS of linear fit system
    RHS = np.array([[yk],[xkyk]]) # Construct RHS of linear fit system

    sol = np.linalg.solve(LHS, RHS) # Solve the system to get fit parameters in the form of [b, a]

    a = sol[1] # Due to the convention, second element is the coefficient of independent variable
    b = sol[0] # Due to the convention, first element is the coefficient of constant variable

    return a, b

def hubble_integral(z, omega_M):
    """
    Syntax:
    I = hubble_integral(z, omega_M)

    This function perform the following processes:
        - It calculates the integral expression in cosmological luminosity distance
        - It also computes the derivative of integral w.r.t its variables
        -

    Input:
        z = Redshift parameter
        omega_M = Normalized form of matter density in the universe
    Output:
        I = The value of integral 
        dI_dz = The derivative of the integral expression w.r.t z variable
        dI_domegaM = The derivative of the integral expression w.r.t omega_M variable
    Example:
        vec_hubble_integral = np.vectorize(hubble_integral) # Vectorization of hubble_integral function
        # scipy.integrate.quad does not accept arrays as inputs

        # Nonlinear fit
        dH , omega_M = hubble_nonlinear_fit(z, dL, dH_guess=dH_guess, omega_M_guess=omega_M_guess)
        dL_nonlin_fitted = (1 + z) * dH * vec_hubble_integral(z, omega_M)[0] 
    """
    # Integrand
    fun = lambda z, omega_M: 1 / (np.sqrt(omega_M * (1 + z)**3 + (1 - omega_M))) # The function inside the integral expression
    I = spi.quad(fun, 0.0, z, args = (omega_M, ))[0] # Take integral between the [0, z]
    
    I_z = fun(z, omega_M) # The derivative of the integral with respect to its differential variable is itself

    # Now the derivative operation will not process on the differential variable. Rather,
    # it will take place on another variable in the integrand.
    # Therefore, the Leibniz Integral Rule must be followed
    fun_omegaM = lambda z, omega_M: (-1/2) * ((1 + z)**3 - 1) * ((omega_M * (1 + z)**3 + (1 - omega_M))**(-3/2)) 
    # The derivative of integrand w.r.t omega_M

    I_omegaM = spi.quad(fun_omegaM, 0.0, z, args = (omega_M, ))[0] # Application of Leibniz Rule
    
    # For more detail, please refer to the Solution Set 04

    return I, I_z, I_omegaM




def hubble_nonlinear_fit(z, dL, dH_guess, omega_M_guess):
    """
    Syntax:
    dH, omega_M = hubble_nonlinear_fit(z, dL, dH_guess, omega_M_guess)

    This function perform the following processes:
        - It solves the nonlinear root finding problem to perform a nonlinear fit on the Hubble data
        - In this way, it calculates the Hubble constant and normalized form of matter density in the universe 
        

    Input:
        z = Redshift parameter
        dL = Luminosity distance of the source in megaparsecs [Mpc] 
        dH_guess = Initial guess for dH
        omega_M_guess = Initial guess for omega_M
    Output:
        dH = Hubble constant
        omega_M = Normalized form of matter density in the universe

    Example:
        vec_hubble_integral = np.vectorize(hubble_integral) # Vectorization of hubble_integral function
        # scipy.integrate.quad does not accept arrays as inputs

        # Nonlinear fit
        dH , omega_M = hubble_nonlinear_fit(z, dL, dH_guess=dH_guess, omega_M_guess=omega_M_guess)
        dL_nonlin_fitted = (1 + z) * dH * vec_hubble_integral(z, omega_M)[0] 
    """

    
    vec_hubble_integral = np.vectorize(hubble_integral) # Vectorized hubble_integral function
    # scipy.integrate.quad does not accept arrays as inputs

    # Function to fit
    fun = lambda z, dH, omega_M: ((1 + z) * dH * vec_hubble_integral(z, omega_M)[0])
    # Objective function of the fit, Euclidian norm
    objective_fun = lambda z, dH, omega_M , dL: sum((fun(z, dH, omega_M) - dL)**2)
    # Derivative of function to fit w.r.t. dH, Hubble Length
    fun_dH = lambda z, omega_M:  (1 + z) * vec_hubble_integral(z, omega_M)[0]
    # Derivative of function to fit w.r.t. omega_M, normalized form of matter density in the universe
    fun_omegaM = lambda z, dH, omega_M: (1 + z) * dH * vec_hubble_integral(z, omega_M)[2] # Last term is the derivative of the integral w.r.t omega_M


    def optimize_func(guess_vec):
        """
            The function to be optimized. 
            Each element consists of the derivatives of objective function with respect to fit parameters,
            namely dH and omega_M
        """
        return[np.sum(((fun(z, guess_vec[0], guess_vec[1]) - dL)) * fun_dH(z, guess_vec[1])),
        np.sum(((fun(z, guess_vec[0], guess_vec[1]) - dL)) * fun_omegaM(z, guess_vec[0], guess_vec[1]))]


    #scipy.optimize.root function output the result as OptimizeObject
    optimize_object = spo.root(fun= optimize_func, x0=[dH_guess, omega_M_guess])
    if optimize_object.success:
        solution = optimize_object.x # The "x" field yields the converged solution in the order of first guesses.
        dH = solution[0]
        omega_M = solution[1]
        return dH, omega_M
    else:
        error("Solver could not converge to solution. Please change the initial guesses!")

def hubble_nonlinear_fit_numpy(z, dL, dH_guess, omega_M_guess):
    """
    Syntax:
    dH, omega_M = hubble_nonlinear_fit_numpy(z, dL, dH_guess, omega_M_guess)

    This function perform the following processes:
        - It solves the nonlinear root finding problem to perform a nonlinear fit on the Hubble data
        - It uses built-in SciPy function to execute process 
        

    Input:
        z = Redshift parameter
        dL = Luminosity distance of the source in megaparsecs [Mpc] 
        dH_guess = Initial guess for dH
        omega_M_guess = Initial guess for omega_M

    Output:
        dH = Hubble constant
        omega_M = Normalized form of matter density in the universe

    Example:
        # Nonlinear fit
        dH, dH_err_std, omega_M, omega_M_err_std = hubble_nonlinear_fit_numpy(z, dL, dH_guess, omega_M_guess) # Nonlinear
    """
    def optimize_func(z, dH, omega_M):
        """
        Definition of function to be optimized.
        """
        vec_hubble_integral = np.vectorize(hubble_integral) # Vectorized hubble_integral function
        # scipy.integrate.quad does not accept arrays as inputs
        return ((1 + z) * dH * vec_hubble_integral(z, omega_M)[0]) # Function to fit

    popt, pcov = spo.curve_fit(optimize_func, z, dL, p0 = [dH_guess, omega_M_guess]) # Built-in curve fitting calculation

    dH = popt[0]
    omega_M = popt[1]
    dH_err_std  = np.sqrt(np.diag(pcov))[0]
    omega_M_err_std = np.sqrt(np.diag(pcov))[1]

    return dH, dH_err_std, omega_M, omega_M_err_std


def error_approximation(z, dL, dL_error, dH_guess, omega_M_guess):
    """
    Syntax:
    dH_dist, omega_M_dist = error_approximation(z, dL, dL_error)

    This function perform the following processes:
        - It solves the nonlinear root finding problem to perform a nonlinear fit on the Hubble data
        - Differently, it manages nonlinear fit via using the probability distributions rather than actual(!) measurements
        

    Input:
        z = Redshift parameter
        dL = Luminosity distance of the source in megaparsecs [Mpc] 
        dL_error = Error in luminosity distance of the source

    Output:
        dH = List of Hubble constant calculated at each iteration to represent as a distribution
        omega_M = List of Normalized form of matter density in the universe calculated at each iteration to represent as a distribution 

    Example:
        dH_dist, omega_M_dist = error_approximation(z, dL, dL_error, dH_guess, omega_M_guess) # Distribution calculation of dH and omega_M

    """

    num_iter = 50 # Number of repetitions to calculate dH and omega_M error distributions

    mu_dL = np.mean(dL_error/4) # Mean of measured dL values
    sigma_dL = np.std(dL_error/4) # Standard deviations of errors in the measurement of dL
    dH_dist = np.empty(num_iter) # dH distribution vector preallocation
    omega_M_dist = np.empty(num_iter) # omega_M distribution vector preallocation

    for k in range(num_iter):

        dL_hist = dL +  sigma_dL * np.random.randn(dL.shape[0]) + mu_dL # Selection of random dL vector from its probability distribution
        dH_dist[k], omega_M_dist[k] = hubble_nonlinear_fit(z, dL_hist, dH_guess, omega_M_guess) # Nonlinear fitting according to the random dL
    
    return dH_dist, omega_M_dist


if __name__ == "__main__":
    mytests()
    