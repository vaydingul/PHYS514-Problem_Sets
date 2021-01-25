from re import error
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import scipy.interpolate as spint

def mytests():
    """
    Syntax:
    mytests()

    This function perform the following processes:
        - It operates on the Runge's function
        - It performs various fitting methods
        - It compares the pros and cons of them
          

    Input:
        [] 

    Output:
        [] 

    Example:
        mytests()
    """
    x = np.linspace(-1,1,1000) # Input vector
    y = runges_function(x) # Output vector, calculation of custom Runge's function
    
    #################### Plotting of original_func function and its various polyfit interpolation #################
    plt.figure()
    plt.plot(x, y, ls= "dashed", label = "Actual Function", lw = 4)
    
    N = [4, 8, 16, 32, 64] # Number array which represents the various n values as a number of equidistant points
    E_n = np.empty(len(N)) # Frobenius norm vector preallocation

    for ix,n in enumerate(N):
        coeff, y_n = custom_interpolate(x, runges_function, n) # Computation of interpolation
        E_n[ix] = frobenius_norm(x, runges_function, coeff) # Calculation of Frobenius norm
        plt.plot(x, y_n, label = "Interpolated Function n={0}".format(n))
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Polyfit Interpolation with Equidistant Nodes")
    plt.legend(loc = "best")
    plt.tight_layout()
    plt.savefig("report/figures/poly.png")
    ####################################################################################################

    #################### Plotting of Frobenius Norms #################
    plt.figure()
    plt.subplot(2,1,1)
    plt.loglog(N, E_n)
    plt.ylabel("Frobenius Norm")
    plt.title("Loglog plot of Frobenius Norm vs. Number of Equidistant Nodes")
    plt.subplot(2,1,2)
    plt.plot(N, E_n)
    plt.xlabel("Number of Equidistant Nodes")
    plt.ylabel("Frobenius Norm")
    plt.title("Normal plot of Frobenius Norm vs. Number of Equidistant Nodes")
    plt.tight_layout()
    plt.savefig("report/figures/frobpoly.png")

    ##################################################################

    #################### Plotting of original_func function and its various cubic spline interpolation #################
    plt.figure()
    plt.plot(x, y, ls= "dashed", label = "Actual Function", lw = 4)
    
    N = [4, 8, 16, 32, 64] # Number array which represents the various n values as a number of equidistant points
    E_n = np.empty(len(N)) # Frobenius norm vector preallocation

    for ix,n in enumerate(N):
        func_handle, y_n = custom_interpolate(x, runges_function, n, method = "spline", kind = "cubic") # Computation of interpolation
        E_n[ix] = frobenius_norm(x, runges_function, method = "spline" ,func_handle = func_handle) # Calculation of Frobenius norm
        plt.plot(x, y_n, label = "Interpolated Function n={0}".format(n))
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Cubic Spline Interpolation")
    plt.legend(loc = "best")
    plt.tight_layout()
    plt.savefig("report/figures/cubspline.png")

    ####################################################################################################

    #################### Plotting of Frobenius Norms #################
    plt.figure()
    plt.subplot(2,1,1)
    plt.loglog(N, E_n)
    plt.ylabel("Frobenius Norm")
    plt.title("Loglog plot of Frobenius Norm for Cubic Spline")
    plt.subplot(2,1,2)
    plt.plot(N, E_n)
    plt.xlabel("Number of Equidistant Nodes")
    plt.ylabel("Frobenius Norm")
    plt.title("Normal plot of Frobenius Norm for Cubic Spline")
    plt.tight_layout()
    plt.savefig("report/figures/frobcubspline.png")

    ##################################################################

    #################### Plotting of original_func function and its various linear spline interpolation #################
    plt.figure()
    plt.plot(x, y, ls= "dashed", label = "Actual Function", lw = 4)
    
    N = [4, 8, 16, 32, 64] # Number array which represents the various n values as a number of equidistant points
    E_n = np.empty(len(N)) # Frobenius norm vector preallocation

    for ix,n in enumerate(N):
        func_handle, y_n = custom_interpolate(x, runges_function, n, method = "spline", kind = "linear") # Computation of interpolation
        E_n[ix] = frobenius_norm(x, runges_function, method = "spline" ,func_handle = func_handle) # Calculation of Frobenius norm
        plt.plot(x, y_n, label = "Interpolated Function n={0}".format(n))
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Linear Spline Interpolation")
    plt.legend(loc = "best")
    plt.tight_layout()
    plt.savefig("report/figures/linspline.png")

    ####################################################################################################

    #################### Plotting of Frobenius Norms #################
    plt.figure()
    plt.subplot(2,1,1)
    plt.loglog(N, E_n)
    plt.ylabel("Frobenius Norm")
    plt.title("Loglog plot of Frobenius Norm for Linear Spline")
    plt.subplot(2,1,2)
    plt.plot(N, E_n)
    plt.xlabel("Number of Equidistant Nodes")
    plt.ylabel("Frobenius Norm")
    plt.title("Normal plot of Frobenius Norm for Linear Spline")
    plt.tight_layout()
    plt.savefig("report/figures/froblinspline.png")

    ##################################################################

    #################### Plotting of original_func function and its various polyfit interpolation using Chebyshev nodes #################
    plt.figure()
    plt.plot(x, y, ls= "dashed", label = "Actual Function", lw = 4)
    
    N = [4, 8, 16, 32, 64] # Number array which represents the various n values as a number of equidistant points
    E_n = np.empty(len(N)) # Frobenius norm vector preallocation

    for ix,n in enumerate(N):
        coeff, y_n = custom_interpolate(x, runges_function, n = n, sampling="cheby") # Computation of interpolation
        E_n[ix] = frobenius_norm(x, runges_function, coeff) # Calculation of Frobenius norm
        plt.plot(x, y_n, label = "Interpolated Function n={0}".format(n))
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Polyfit Interpolation using Chebyshev Nodes")
    plt.legend(loc = "best")
    plt.tight_layout()
    plt.savefig("report/figures/polycheby.png")

    ####################################################################################################

    #################### Plotting of Frobenius Norms #################
    plt.figure()
    plt.subplot(2,1,1)
    plt.loglog(N, E_n)
    plt.ylabel("Frobenius Norm")
    plt.title("Loglog plot of Frobenius Norm for Chebyshev Nodes")
    plt.subplot(2,1,2)
    plt.plot(N, E_n)
    plt.xlabel("Number of Chebyshev Nodes")
    plt.ylabel("Frobenius Norm")
    plt.title("Normal plot of Frobenius Norm for Chebyshev Nodes")
    plt.tight_layout()
    plt.savefig("report/figures/frobpolycheby.png")

    ##################################################################


    #### Plotting of original_func function and its various polyfit interpolation while fit degree is not equal to number of nodes #####
    plt.figure()
    plt.plot(x, y, ls= "dashed", label = "Actual Function", lw = 4)
    
    N = [ 16, 64, 256, 1024] # Number array which represents the various n values as a number of equidistant points
    E_n = np.empty(len(N)) # Frobenius norm vector preallocation

    for ix,n in enumerate(N):
        degree = int(np.sqrt(n) - 1)
        coeff, y_n = custom_interpolate(x, runges_function, n = n, degree = degree) # Computation of interpolation
        E_n[ix] = frobenius_norm(x, runges_function, coeff) # Calculation of Frobenius norm
        plt.plot(x, y_n, label = "Interpolated Function n={0}".format(n))
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Polyfit Interpolation\nwhile fit degree is not equal to number of nodes")
    plt.legend(loc = "best")
    plt.tight_layout()
    plt.savefig("report/figures/polynenodes.png")

    ####################################################################################################

    #################### Plotting of Frobenius Norms #################
    plt.figure()
    plt.subplot(2,1,1)
    plt.loglog(N, E_n)
    plt.ylabel("Frobenius Norm")
    plt.title("Loglog plot of Frobenius Norm\nwhile fit degree is not equal to number of nodes")
    plt.subplot(2,1,2)
    plt.plot(N, E_n)
    plt.xlabel("Number of Equidistant Nodes")
    plt.ylabel("Frobenius Norm")
    plt.title("Normal plot of Frobenius Norm\nwhile fit degree is not equal to number of nodes")
    plt.tight_layout()
    plt.savefig("report/figures/frobpolynenodes.png")

    ##################################################################
    plt.show()




def runges_function(x):
    """
    Syntax:
    y = runges_function(x)

    This function perform the following processes:
        - It calculates the Runge's Function in the form of:
            1 / (1 + 25*x^2)
        

    Input:
        x = Input vector

    Output:
        y = Output, result of the Runge's function

    Example:
        x = np.linspace(-1,1,1000)
        y = runges_function(x)
    """
    return 1/((1 + 25*x**2))

def custom_interpolate(x, func_to_fit, n, degree = None, sampling = "normal", method="polyfit", kind = "linear"):
    """
    Syntax:
    y_n = calculate_polyfit(x, func_to_fit, n, degree, sampling = "normal", method="polyfit", kind = "linear")

    This function perform the following processes:
        - It computes coefficient of the exact polynomial fit for a given n number of equidistant point
        
        

    Input:
        x = Input vector to perform integration
        func_to_fit = Function to fit
        n = Number of nodes
        degree = Polyfit interpolation degree
        sampling = Sampling method, normal = linspace equidistant nodes, cheby = Chebyshev Nodes
        method = Metdod type that will be used in interpolation
        kind = Type of spline interpolation, please use when method = "spline" !
    Output:
        y_n = The calculated polynomial fit result

    Example:
        x = np.linspace(-1,1,1000)
        y = runges_function(x)
        y_5 = calculate_polyfit(x, y, 5)
    """
    if sampling == "normal":
        start = 0 # Start index of x
        end = x.shape[0] - 1 # End index of x
        x_nodes = np.linspace(x[start], x[end], n) # Equidistant construction of x for number of n
    elif sampling == "cheby":
        k = np.linspace(0, n-1)
        start = 0 # Start index of x
        end = x.shape[0] - 1 # End index of x
        x_nodes = 0.5 * (x[start] + x[end]) + 0.5*(x[end] - x[start]) * np.cos(((2 * k + 1) / (2 * n)) * np.pi) 
    else:
        error("Only accepted inputs are 'sampling=normal' or 'sampling=cheby'. Your input is not recognized, therefore, 'sampling=normal' is assumed.")
        start = 0 # Start index of x
        end = x.shape[0] - 1 # End index of x
        x_nodes = np.linspace(x[start], x[end], n) # Equidistant construction of x for number of n

    if degree is None and method == "polyfit":
        error("You did not specify the degree of interpolation. It is accepted as an equal number of nodes.")
        degree = n

    if method == "polyfit":
        coeffs = np.polynomial.polynomial.polyfit(x_nodes, func_to_fit(x_nodes), deg = degree )
        y_n = np.polynomial.polynomial.polyval(x, coeffs)
        return coeffs, y_n
    elif method == "spline":
        inter_func = spint.interp1d(x_nodes, func_to_fit(x_nodes), kind = kind)
        y_n = inter_func(x)
        return inter_func, y_n
        


def frobenius_norm(x, original_func, fitted_coeff = None, method = "polyfit", func_handle = None):
    """
    Syntax:
    E_n = frobenius_norm(x, original_func, fitted_coeff)

    This function perform the following processes:
        - It computes the Frobenious Norm between the original_func function and fitted function
        
        

    Input:
        x = Input vector to perform integration
        original_func = original_func function handle
        fitted_coeff = Coefficients of interpolation
        method = It determines the Frobenius Norm calculation basis
        func_handle = Function handle to calculate interpolated function, please use when method = spline

    Output:
        E_n = The calculated Frobenius Norm

    Example:
        coeff, y_n = calculate_polyfit(x, runges_function, n)
        E_n[ix] = frobenius_norm(x, runges_function, coeff)
    """
    len_x = x.shape[0] # Find length to use as a end index
    start = 0 # First value that will be used in integration
    end = len_x - 1 # Last value that will be used in integration
    
    
    if method == "polyfit":
        if fitted_coeff is not None:
            integrand = lambda x: np.abs(original_func(x) - np.polynomial.polynomial.polyval(x, fitted_coeff)) # Function to be integrate
            integral, _ = spi.quad(integrand, x[start], x[end]) # Integral operation
            integral = np.sqrt(integral * 0.5) # Square root applicatzion as in the Frobenius Norm formula
            return integral
        else:
            error("If you interpolated with the 'polyfit' method, please specify 'fitted_coeff'. If you did not use'polyfit' method, please specify 'func_handle' for 'spline' method.")


    elif method == "spline":
        if func_handle is not None:
            integrand = lambda x: np.abs(original_func(x) - func_handle(x)) # Function to be integrate
            integral, _ = spi.quad(integrand, x[start], x[end]) # Integral operation
            integral = np.sqrt(integral * 0.5) # Square root applicatzion as in the Frobenius Norm formula
            return integral
        else:
            error("If you interpolated with the 'spline' method, please specify 'func_handle'. If you did not use'spline' method, please specify 'fitted_coeff' for 'polyfit' method.")





if __name__ == "__main__":
    mytests()
    