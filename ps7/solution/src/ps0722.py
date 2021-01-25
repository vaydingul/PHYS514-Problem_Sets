import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.shape_base import dsplit
from scipy.integrate import solve_ivp 
from scipy.optimize import newton
from scipy.linalg import eig
def mytests():
    """
    mytests

    This function executes the following processes:
        - It visualizes the solution of the HHO
        - It visualizes the solution of the HHO with two different methods

    Input:
        []

    Output:
        []

    Usage:
    mytests()
    """
    zeta_list = [2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5] # The list of upper_bound values for the BVP
    n = 4 # Number of modes
    
    ###################### SHOOTING METHOD #####################################################
    plt.figure(figsize = (10.0, 10.0))

    for (ix, zeta) in enumerate(zeta_list):

        # Half Harmonic Oscillator propagation
        z, psi, epsilon = solve_hho(zeta = zeta, n = n, method = "shooting")

        # Plot settings
        plt.subplot(len(zeta_list)/2, 2, ix+1)
 
        for k in range(n):
            
                 
            #Plotting
            plt.plot(z[k], psi[k][0], label = "n = {0}".format(k+1))             
            plt.title("$\zeta$ = {0}".format(zeta))
            plt.xlabel("$x$")
            plt.ylabel("$\psi$")
            plt.legend(loc = "best", ncol = 2)

    plt.suptitle("Eigenfunctions for different $\zeta$ values")
    plt.tight_layout(pad = 0.1, rect=[0, 0.03, 1, 0.90])
    plt.savefig("report/figures/7_eigenfunctions.png")

    plt.figure()
    for (ix, zeta) in enumerate(zeta_list):

        # Half Harmonic Oscillator propagation
        _, _, epsilon = solve_hho(zeta = zeta, n = n, method = "shooting")

        # Plot settings
                 
        plt.plot(np.arange(1, n + 1), epsilon, marker = "o", label = "$\zeta$ = {0}".format(zeta))

    plt.xlabel("n")
    plt.ylabel("$\epsilon$")
    plt.title("Eigenvalues")
    plt.legend(loc = "best")
    plt.tight_layout()
    plt.savefig("report/figures/7_eigenvalue.png")
    #################################################################################################
    

    ###################### DIRECT METHOD #####################################################
    N = 10 # Number of sampling points
    n = 4 # Number of modes
    plt.figure()
    # Half Harmonic Oscillator propagation
    z, psi, epsilon = solve_hho(n = n, method = "direct", N = N)

    for k in range(n):
        #Plotting
        plt.plot(z[:-1], psi[:-1, k], label = "n = {0}".format(k+1), marker = "o", markevery = np.random.randint(10, 20))             
    plt.xlabel("$x$")
    plt.ylabel("$\psi$")
    plt.legend(loc = "best", ncol = 2)
    plt.title("Eigenfunctions")
    plt.tight_layout()
    plt.savefig("report/figures/8_eigenfunctions.png")

    plt.figure()                
    plt.plot(np.arange(1, n + 1), epsilon, marker = "o")
    plt.xlabel("n")
    plt.ylabel("$\epsilon$")
    plt.title("Eigenvalues")
    plt.tight_layout()
    plt.savefig("report/figures/8_eigenvalue.png")
    #################################################################################################
    plt.show()
    
def solve_hho(zeta = 1.5, n = 4, method = "shooting", N = 100, dt = 1e-2):
    """
    solve_hho

    This function executes the following processes:
        - It solves the Half Harmonic Oscillator
        - It can solve via using different methods

    Input:
        zeta = Characteristic length
        n = Number of eigenvalues and eigenfunctions to be found
        method = Te method will be used in the solution
        N = Number of sampling methods that will be used in Direct Method
        dt = The maximum step-size that will be used in Shooting method
        

    Output:
        z = Sampling points
        psi = Computed eigenfunctions
        epsilon = Computed eigenvalues

    Usage:
        z, psi, epsilon = solve_hho(zeta = 2.0, n = 4, method = "shooting")
    """
    
    span = [0.0, zeta] # Time interval that the ODE will propagate 
    initial_values = [0.0, 1.0] # Initial values of the ODE

    if method == "shooting":
        z, psi, epsilon = _solve_BVP_shooting(fun = ODE_, span = span, initial_values = initial_values, n = n, dt = dt)
    elif method == "direct":
        z, psi, epsilon = _solve_BVP_direct(n = n, N = N)

    return z, psi, epsilon


def _solve_BVP_shooting(fun, span, initial_values, n  = 4, dt = 1e-2):
    """
    Low level function wrap of Shooting method
    """
    # Preallocation of set of different sampling points for different epsilon values
    z = np.empty(n, dtype = np.ndarray)
    # Preallocation of set of different eigenfunctions for different epsilon values
    psi = np.empty(n, dtype = np.ndarray)
    # Preallocation of set of different eigenvalues for different epsilon values
    eps_list = np.empty(n)

    # Parameter initializations
    ix = 0
    epsilon00 = 0.0
    epsilon0 = 0.0

    while (n > 0):
    # This for loop is responsible for the finding of epsilon values iteratively
    #         
        def shoot_them_all(epsilon , fun , span , initial_values):
            """
            Funny function name that will be used in subject function in shooting method :)
            """
            # It solves IVP and outputs last element of the solution
            soln = solve_ivp(fun = fun, t_span = span, y0 = initial_values, args = (epsilon, ), max_step = dt)
            return soln.y[0][-1]

        # These two procedure basically calculates subsequent epsilon values (via root finding)
        # to compare whether they are different or not
        epsilon_, r = newton(func = shoot_them_all, x0 = epsilon0, args = (fun, span, initial_values), full_output = True)
        epsilon__, r = newton(func = shoot_them_all, x0 = epsilon00, args = (fun, span, initial_values), full_output = True)
        
        if not np.allclose(epsilon_, epsilon__) or ix is 0:
            # It these two epsilon values are not same or it is the first iteration,
            # then, it is a valid epsilon value,
            # so, save it as a solution
            if r.converged:
                #### Solution ####
                soln = solve_ivp(fun = fun, t_span = span, y0 = initial_values, args = (epsilon_, ), max_step = dt)
                z[ix] = soln.t
                psi[ix] = soln.y
                eps_list[ix] = epsilon_

                # Propagate parameters
                n -= 1
                ix += 1
                # If the subsequent epsilon values are not same,
                # then, increase them in chained manner.
                epsilon00 = epsilon0
                epsilon0 += 1.5
            else:
                print("It seems that the shooting method could not converge!\nPlease try enother initial value for epsilon!")

        else:
            # If the subsequent epsilon values are same,
            # then, do not propagate both of them,
            # only propagate one of them
            epsilon0 += 1.5



    return z, psi, eps_list

def _solve_BVP_direct(n = 4, N = 100):
    """
    Low level function wrapper for Direct Method
    """
    h = 1 / N

    d2 = np.zeros((N, N)) # Preallocation of the second derivative matrix
    d1 = np.zeros((N, N)) # Preallocation of the first derivative matrix
    
    d2[0, 0:2] = [-2, 1] # First row of second derivative matrix
    d1[0, 0:2] = [0, 1] # First row of first derivative matrix 

    for k in range(1, N-1):
        d2[k, (k-1):(k+2)] = [1, -2, 1]  # Intermediate rows of second derivative matrix
        d1[k, (k-1):(k+2)] = [-1, 0, 1]  # Intermediate rows of first derivative matrix

    d2[-1, -2:] = [1, -2]  # Last row of the second derivative matrix
    d1[-1, -2:] = [-1, 0]  # Last row of the first derivative matrix

    d1 = d1 / (2 * h)
    d2 = d2 / (h ** 2)

    # Preallocation of sampling points
    zc = np.empty((N+2, 1))
    zc[:, 0] = np.linspace(0.0, 1.0, N+2)
    zc_ = zc[1:-1, 0]

    # Calculation of Hermitian
    H = (((-2) / (np.pi ** 2)) * ((1 + (np.tan(np.pi * zc_ / 2 )) ** 2) ** (-2)) * (d2) + (2 / np.pi) * ((1 + (np.tan(np.pi * zc_ / 2 )) ** 2) ** (-1)) * (np.tan(np.pi * zc_ / 2)) * (d1) + (1 / 2) * ((np.tan(np.pi * zc_ / 2 )) ** 2)) 


    w, v = eig(H)
    # Numpy does not output in a ordered manner, we should sort it by ourselves
    ix_sorted = np.argsort(w)
    w = w[ix_sorted]
    v = v[:, ix_sorted]


    # Apply BC
    y = np.vstack((np.zeros((1, N)), v, np.zeros((1, N))))
    # Apply transformation to 
    x = np.tan(np.pi * zc / 2)

    return x, y[:, np.arange(0,2 * n,2)], w[np.arange(0,2 * n,2)]



def ODE_(z, psi, epsilon):
    """
    ODE wrapper that is asked in the problem
    """    
    return np.array([psi[1], (z ** 2) * psi[0] - 2 * epsilon * psi[0]])

    


    

if __name__ == "__main__":
    mytests()