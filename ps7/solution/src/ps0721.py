import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, eigh_tridiagonal, eigh, eig_banded
from timeit import timeit


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
    # Dictionaries for prettifying the figures
    BC_dict = {"d": "Dirichlet BC", "n": "Neumann BC", "p": "Periodical BC"}
    eigen_solver_dict = {eig: "Standart Eigenvalue Solver", eigh: "Symmetric Eigenvalue Solver",
                         eigh_tridiagonal: "Tridiagonal Toeplitz Eigenvalue Solver", eig_banded: "Banded Eigenvalue Solver"}
     # Number of repetitions that will be used in average elapsed time calculations
    num_rep = 100
    


    
    ##################### DIRICHLET BC ########################################################
    ##################### SAMPLING POINT EXPERIMENT ###########################################
    L = 1.0  # Length of the string (normalized)
    num_modes = 8  # Number of modes which we want to investigate
    Ns = [num_modes, 10, 20, 30, 40, 50]  # The list of number of sampling points
    symmetric_solver = True 
    BC_type = "d"  # It stands for ´Dirichlet´ BC
    eigen_solvers =  [eig, eigh, eig_banded, eigh_tridiagonal] # The list of eigenvalue solvers
    eigen_solver = eigen_solvers[0] # Specific eigenvalue solver for this problem

    modes_numerical = np.empty(len(Ns), dtype = np.ndarray)
    for (ix,N) in enumerate(Ns):
        modes_analytical = np.empty((num_modes, 1))
        
        # Wave equation propagation
        x_numerical, y_numerical, modes_numerical[ix] = wave_solver(L=L, num_modes=num_modes, N=N, eigen_solver=eigen_solver,
                               symmetric_solver=symmetric_solver, BC_type=BC_type)
        # Analytical eigenvalue vector preallocation

        # Plot settings
        plt.figure(figsize = (10.0, 10.0))        
        for n_ in range(num_modes):
            n = n_ + 1
            plt.subplot(num_modes /2, 2, n)

            # Analytical calculation of wave equation
            x_analytical = np.linspace(0.0, L, x_numerical.shape[0])
            y_analytical = np.sin((n * np.pi * x_analytical) / L) 
            modes_analytical[n_] = (-(n ** 2) * (np.pi ** 2)) / (L ** 2)

            #Plotting
            plt.plot(x_analytical, y_analytical, label = "Analytical", marker = "^", markevery = 25)
            plt.plot(x_numerical, y_numerical[:, n_], label = "Numerical", marker = "o", markevery = 30)
            plt.xlabel("$x$")
            plt.ylabel("$y$")
            plt.title("n = {0}\nRMSE = {1}".format(n, RMSE(y_analytical, y_numerical[:, n_])))
            plt.legend(loc = "best")

        plt.suptitle("Wave Equation Solution Eigenvector Comparison\nObtained from {0} sampling points\n{1}\nusing {2}".format(
            N, BC_dict[BC_type], eigen_solver_dict[eigen_solver]))
        plt.tight_layout(pad = 0.1, rect=[0, 0.03, 1, 0.90])
        #plt.savefig("report/figures/1_{0}_{1}_{2}_eigenvector.png".format(N, eigen_solver_dict[eigen_solver].replace(" ", "_" ), BC_type))

    plt.figure()
    plt.plot(np.arange(1, num_modes + 1), modes_analytical, label = "Analytical", marker = "^", markevery = 1)
    for k in range(len(Ns)):
        plt.plot(np.arange(1, num_modes + 1), modes_numerical[k], label = "Numerical N = {0}".format(Ns[k]), marker = "o", markevery = 2)
    plt.xlabel("n")
    plt.ylabel("$\lambda$")
    plt.legend(loc = "best")
    plt.title("Wave Equation Solution Eigenvalue Comparison\n{0}\nusing {1}".format(
        BC_dict[BC_type], eigen_solver_dict[eigen_solver]))
    plt.tight_layout()
    #plt.savefig("report/figures/1_{0}_{1}_eigenvalue.png".format(eigen_solver_dict[eigen_solver].replace(" ", "_" ), BC_type))

    #####################################################################################

    ##################### EIGENVALUE SOLVER EXPERIMENT ##################################
    L = 1.0  # Length of the string (normalized)
    num_modes = 8  # Number of modes which we want to investigate
    Ns = [num_modes, 10, 20, 30, 40, 50] # The list of number of sampling points
    symmetric_solver = True
    BC_type = "d"  # It stands for ´Dirichlet´ BC
    eigen_solvers =  [eig, eigh, eig_banded, eigh_tridiagonal]  # The list of eigenvalue solvers

    modes_numerical = np.empty(len(eigen_solvers), dtype = np.ndarray)
    N = Ns[4] # Selected number of sampling points

    for (ix, eigen_solver) in enumerate(eigen_solvers):
        # Wave equation propagation
        x_numerical, y_numerical, modes_numerical[ix] = wave_solver(L=L, num_modes=num_modes, N=N, eigen_solver=eigen_solver,
                               symmetric_solver=symmetric_solver, BC_type=BC_type)

        # Analytical eigenvalue vector preallocation
        modes_analytical = np.empty((num_modes, 1))
        # Plot settings
        plt.figure(figsize = (10.0, 10.0))        
        for n_ in range(num_modes):
            n = n_ + 1
            plt.subplot(num_modes /2, 2, n)

            # Analytical calculation of wave equation
            x_analytical = np.linspace(0.0, L, x_numerical.shape[0])
            y_analytical = np.sin((n * np.pi * x_analytical) / L) 
            modes_analytical[n_] = (-(n ** 2) * (np.pi ** 2)) / (L ** 2)

            #Plotting
            plt.plot(x_analytical, y_analytical, label = "Analytical", marker = "^", markevery = 25)
            plt.plot(x_numerical, y_numerical[:, n_], label = "Numerical", marker = "o", markevery = 30)
            plt.xlabel("$x$")
            plt.ylabel("$y$")
            plt.title("n = {0}\nRMSE = {1}".format(n, RMSE(y_analytical, y_numerical[:, n_])))
            plt.legend(loc = "best")

        plt.suptitle("Wave Equation Solution Eigenvector Comparison\nObtained from {0} sampling points\n{1}\nusing {2}".format(
            N, BC_dict[BC_type], eigen_solver_dict[eigen_solver]))
        plt.tight_layout(pad = 0.1, rect=[0, 0.03, 1, 0.90])
        #plt.savefig("report/figures/2_{0}_{1}_{2}_eigenvector.png".format(N, eigen_solver_dict[eigen_solver].replace(" ", "_" ), BC_type))
        

    plt.figure()
    plt.plot(np.arange(1, num_modes + 1), modes_analytical, label = "Analytical", marker = "^", markevery = 1)
    for k in range(len(eigen_solvers)):
        plt.plot(np.arange(1, num_modes + 1), modes_numerical[k], label = "Numerical Method = {0}".format(eigen_solver_dict[eigen_solvers[k]]), marker = "o", markevery = np.random.randint(2,4))    
    plt.xlabel("n")
    plt.ylabel("$\lambda$")
    plt.legend(loc = "best")
    plt.title("Wave Equation Solution Eigenvalue Comparison\nObtained from {0} sampling points\n{1}".format(
        N, BC_dict[BC_type]))
    #plt.savefig("report/figures/2_{0}_{1}_eigenvalue.png".format(N, BC_type))
    plt.tight_layout()
    
    #####################################################################################


    ##################### EIGENVALUE SOLVER SPEED EXPERIMENT ############################
    L = 1.0  # Length of the string (normalized)
    num_modes = 8  # Number of modes which we want to investigate
    # Number of sampling points
    Ns = [num_modes, 10, 20, 30, 40, 50] # The list of number of sampling points
    symmetric_solver = True
    BC_type = "d"  # It stands for ´Dirichlet´ BC
    eigen_solvers = [eig, eigh, eig_banded, eigh_tridiagonal]  # The list of eigenvalue solvers
    # Preallocation of timing vector
    time_vec = np.empty(len(eigen_solvers))
    # Selected number of sampling points 
    N = Ns[4]

    for (idx, eigen_solver) in enumerate(eigen_solvers):
        # Calculation of elapsed time
        time_vec[idx] = timeit(lambda: wave_solver(L=L, num_modes=num_modes, N=N, eigen_solver=eigen_solver,
                                                   symmetric_solver=symmetric_solver, BC_type=BC_type), number=num_rep)

    labels = [name.replace(" ", "\n") for name in eigen_solver_dict.values()]
    # Plotting
    plt.figure()
    plt.bar(labels, time_vec / num_rep)
    plt.ylabel("Average Elapsed Time [sec]")
    plt.title("Wave Equation Solution Eigenvalue Solver Speed Comparison\nObtained from {0} sampling points\n{1}".format(
        N, BC_dict[BC_type]))
    #plt.savefig("report/figures/3_{0}_{1}_eigen_solver_speed.png".format(N, BC_type))
    
    ###################################################################################

    #################### NEUMANN BC ########################################################
    ##################### SYMMETRIC CASE ###########################################
    L = 1.0  # Length of the string (normalized)
    num_modes = 9  # Number of modes which we want to investigate
    Ns = [num_modes, 10, 20, 30, 40, 50]   # The list of number of sampling points
    symmetric_solver = True
    BC_type = "n"  # It stands for ´Neumann´ BC
    eigen_solvers =  [eig, eigh, eig_banded, eigh_tridiagonal]  # The list of eigenvalue solvers

    N = Ns[4] # Selected number of sampling points
    eigen_solver = eigen_solvers[1] # Selected eigenvalue solver 
    # Wave equation propagation
    x_numerical, y_numerical, modes_symmetric = wave_solver(L=L, num_modes=num_modes, N=N, eigen_solver=eigen_solver,
                            symmetric_solver=symmetric_solver, BC_type=BC_type)
    # Analytical eigenvalue vector preallocation
    modes_analytical = np.empty((num_modes, 1))
    # Plot settings
    plt.figure(figsize = (10.0, 10.0))        
    for n_ in range(num_modes):
        n = n_ + 1
        plt.subplot(np.ceil(num_modes / 2), 2, n)

        # Analytical calculation of wave equation
        x_analytical = np.linspace(0.0, L, x_numerical.shape[0])
        y_analytical = np.cos((n_ * np.pi * x_analytical) / L) 
        modes_analytical[n_] = (-(n_ ** 2) * (np.pi ** 2)) / (L ** 2)

        #Plotting
        plt.plot(x_analytical, y_analytical, label = "Analytical", marker = "^", markevery = 25)
        plt.plot(x_numerical, y_numerical[:, n_], label = "Numerical", marker = "o", markevery = 30)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.title("n = {0}\nRMSE = {1}".format(n, RMSE(y_analytical, y_numerical[:, n_])))
        plt.legend(loc = "best")
#            plt.tight_layout(pad = 1.0)

    plt.suptitle("Wave Equation Solution Eigenvector Comparison\nObtained from {0} sampling points\n{1}\nusing {2}".format(
        N, BC_dict[BC_type], eigen_solver_dict[eigen_solver]))
    plt.tight_layout(pad = 0.1, rect=[0, 0.03, 1, 0.90])
    #plt.savefig("report/figures/4_{0}_{1}_{2}_eigenvector.png".format(N, eigen_solver_dict[eigen_solver].replace(" ", "_" ), BC_type))

    """
    plt.figure()
    plt.plot(np.arange(1, num_modes + 1), modes_analytical, label = "Analytical", marker = "^", markevery = 1)
    plt.plot(np.arange(1, num_modes + 1), modes, label = "Numerical", marker = "o", markevery = 2)
    plt.xlabel("n")
    plt.ylabel("$\lambda$")
    plt.legend(loc = "best")
    plt.title("Wave Equation Solution Eigenvalue Comparison\nObtained from {0} sampling points\n{1}\nusing {2}".format(
        N, BC_dict[BC_type], eigen_solver_dict[eigen_solver]))
    plt.tight_layout()
    #plt.savefig("report/figures/4_{0}_{1}_{2}_eigenvalue.png".format(N, eigen_solver_dict[eigen_solver].replace(" ", "_" ), BC_type))
    """
    #####################################################################################
    
    ##################### NOT SYMMETRIC CASE ###########################################
    L = 1.0  # Length of the string (normalized)
    num_modes = 9  # Number of modes which we want to investigate
    Ns = [num_modes, 10, 20, 30, 40, 50] # The list of number of sampling points
    symmetric_solver = False
    BC_type = "n"  # It stands for ´Neumann´ BC
    eigen_solvers =  [eig, eigh, eig_banded, eigh_tridiagonal]  # The list of eigenvalue solvers

    N = Ns[4] # Selected number of sampling points
    eigen_solver = eigen_solvers[0] # Selected eigenvalue solver 
    # Wave equation propagation
    x_numerical, y_numerical, modes_unsymmetric = wave_solver(L=L, num_modes=num_modes, N=N, eigen_solver=eigen_solver,
                            symmetric_solver=symmetric_solver, BC_type=BC_type)
    # Analytical eigenvalue vector preallocation
    modes_analytical = np.empty((num_modes, 1))
    # Plot settings
    plt.figure(figsize = (10.0, 10.0))        
    for n_ in range(num_modes):
        n = n_ + 1
        plt.subplot(np.ceil(num_modes / 2), 2, n)

        # Analytical calculation of wave equation
        x_analytical = np.linspace(0.0, L, x_numerical.shape[0])
        y_analytical = np.cos((n_ * np.pi * x_analytical) / L) 
        modes_analytical[n_] = (-(n_ ** 2) * (np.pi ** 2)) / (L ** 2)

        #Plotting
        plt.plot(x_analytical, y_analytical, label = "Analytical", marker = "^", markevery = 25)
        plt.plot(x_numerical, y_numerical[:, n_], label = "Numerical", marker = "o", markevery = 30)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.title("n = {0}\nRMSE = {1}".format(n_, RMSE(y_analytical, y_numerical[:, n_])))
        plt.legend(loc = "best")
#            plt.tight_layout(pad = 1.0)

    plt.suptitle("Wave Equation Solution Eigenvector Comparison\nObtained from {0} sampling points\n{1}\nusing {2}".format(
        N, BC_dict[BC_type], eigen_solver_dict[eigen_solver]))
    plt.tight_layout(pad = 0.1, rect=[0, 0.03, 1, 0.90])
    #plt.savefig("report/figures/5_{0}_{1}_{2}_eigenvector.png".format(N, eigen_solver_dict[eigen_solver].replace(" ", "_" ), BC_type))

    plt.figure()
    plt.plot(np.arange(1, num_modes + 1), modes_analytical, label = "Analytical", marker = "^", markevery = 1)
    plt.plot(np.arange(1, num_modes + 1), modes_symmetric, label = "Numerical Symmetric", marker = "o", markevery = 2)
    plt.plot(np.arange(1, num_modes + 1), modes_unsymmetric, label = "Numerical Unsymmetric", marker = "o", markevery = 2)

    plt.xlabel("n")
    plt.ylabel("$\lambda$")
    plt.legend(loc = "best")
    plt.title("Wave Equation Solution Eigenvalue Comparison\nObtained from {0} sampling points\n{1}\nusing {2}".format(
        N, BC_dict[BC_type], eigen_solver_dict[eigen_solver]))
    plt.tight_layout()
    #plt.savefig("report/figures/5_{0}_{1}_{2}_eigenvalue.png".format(N, eigen_solver_dict[eigen_solver].replace(" ", "_" ), BC_type))

    #####################################################################################
    #####################################################################################
   
    #################### PERIODIC BC ########################################################
    ##################### SYMMETRIC CASE ###########################################

    L = 1.0  # Length of the string (normalized)
    num_modes = 8  # Number of modes which we want to investigate
    Ns = [num_modes, 10, 20, 30, 40, 50] # The list of number of sampling points
    symmetric_solver = True
    BC_type = "p"  # It stands for ´Periodic´ BC
    eigen_solvers =  [eig, eigh, eig_banded, eigh_tridiagonal]  # The list of eigenvalue solvers

    N = Ns[4] # Selected number of sampling points
    eigen_solver = eigen_solvers[1] # Selected eigenvalue solver 
    # Wave equation propagation
    x_numerical, y_numerical, modes = wave_solver(L=L, num_modes=num_modes, N=N, eigen_solver=eigen_solver,
                            symmetric_solver=symmetric_solver, BC_type=BC_type)
    # Analytical eigenvalue vector preallocation
    modes_analytical = np.empty((num_modes, 1))
    # Plot settings
    plt.figure(figsize = (10.0, 10.0))        
    for n_ in range(num_modes):
        n = n_ + 1
        plt.subplot(np.ceil(num_modes / 2), 2, n)

        # Analytical calculation of wave equation
        x_analytical = np.linspace(0.0, L, x_numerical.shape[0])
        if n_ % 2 == 0:
            y_analytical = np.sin((n * np.pi * x_analytical) / L)
        else:
            y_analytical = np.cos((n_ * np.pi * x_analytical) / L)
        modes_analytical[n_] = (-(n ** 2) * (np.pi ** 2)) / (L ** 2)

        #Plotting
        plt.plot(x_analytical, y_analytical, label = "Analytical", marker = "^", markevery = 25)
        plt.plot(x_numerical, y_numerical[:, n_], label = "Numerical", marker = "o", markevery = 30)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.title("n = {0}\nRMSE = {1}".format(n, RMSE(y_analytical, y_numerical[:, n_])))
        plt.legend(loc = "best")
#            plt.tight_layout(pad = 1.0)

    plt.suptitle("Wave Equation Solution Eigenvector Comparison\nObtained from {0} sampling points\n{1}\nusing {2}".format(
        N, BC_dict[BC_type], eigen_solver_dict[eigen_solver]))
    plt.tight_layout(pad = 0.1, rect=[0, 0.03, 1, 0.90])
    #plt.savefig("report/figures/6_{0}_{1}_{2}_eigenvector.png".format(N, eigen_solver_dict[eigen_solver].replace(" ", "_" ), BC_type))

    plt.figure()
    plt.plot(np.arange(1, num_modes + 1), modes_analytical, label = "Analytical", marker = "^", markevery = 1)
    plt.plot(np.arange(1, num_modes + 1), modes, label = "Numerical", marker = "o", markevery = 2)
    plt.xlabel("n")
    plt.ylabel("$\lambda$")
    plt.legend(loc = "best")
    plt.title("Wave Equation Solution Eigenvalue Comparison\nObtained from {0} sampling points\n{1}\nusing {2}".format(
        N, BC_dict[BC_type], eigen_solver_dict[eigen_solver]))
    plt.tight_layout()
    #plt.savefig("report/figures/6_{0}_{1}_{2}_eigenvalue.png".format(N, eigen_solver_dict[eigen_solver].replace(" ", "_" ), BC_type))

    #####################################################################################
    #####################################################################################
    
    plt.show()


def wave_solver(L=1.0, num_modes=8, N=8, eigen_solver=eig, symmetric_solver=False, BC_type="d"):
    """
    wave_solver

    This function executes the following processes:
        - It solves the wave equation numerically.
        - It can impose diffeent boundary conditions on the problem.
        - Alsoi it can make use of the different eigenvalue solvers

    Input:
        L = Length of the string
        num_modes = Number of modes that will be calculated
        N = Number of sampling points that will be used for the calculation of the modes
        eigen_solver = The eigenvalue problem solver
        symmetric_solver = The option that specifies the type of the second derivative matrix (Hermitian)
        BC_type = The boundary condition imposed on the problem

    Output:
        x = Sampling points
        y = Computed eigenvectors - spatial solution of the wave equation 
        w = Computed eigenvalues - Spatial modes of the wave equation

    Usage:
        L = 1.0  # Length of the string (normalized)
        num_modes = 8  # Number of modes which we want to investigate 
        Ns = [num_modes, 10, 20, 30, 40, 50] # The list of number of sampling points
        symmetric_solver = True
        BC_type = "p"  # It stands for ´Periodic´ BC
        eigen_solvers =  [eig, eigh, eig_banded, eigh_tridiagonal]  # The list of eigenvalue solvers

        N = Ns[4] # Selected number of sampling points
        eigen_solver = eigen_solvers[1] # Selected eigenvalue solver 
        # Wave equation propagation
        x_numerical, y_numerical, modes = wave_solver(L=L, num_modes=num_modes, N=N, eigen_solver=eigen_solver,
                            symmetric_solver=symmetric_solver, BC_type=BC_type)
    """
    
    h = L / N # Step-size

    # Construction of the A matrix according to the BC type
    if BC_type == "d":

        A = get_A_matrix_d(N=N, is_symmetric=symmetric_solver)

    elif BC_type == "n":

        A = get_A_matrix_n(N=N, is_symmetric=symmetric_solver)

    elif BC_type == "p":

        A = get_A_matrix_p(N=N, is_symmetric=symmetric_solver)

    else:
        print("This BC type is not valid!")

    # Preconditioning on the A matrix
    A = A / (h**2)

    # Solution of eigenvalue problem according to the eigenvalue solver specified in the function
    if eigen_solver == eig:
        w, v = eigen_solver(A)

    elif eigen_solver == eigh_tridiagonal:

        d = np.diag(A)
        e = np.diag(A, k=1)
        w, v = eigen_solver(d, e)

    elif eigen_solver == eigh:
        w, v = eigen_solver(A)

    elif eigen_solver == eig_banded:
        d = np.diag(A)
        e = np.diag(A, k=1)
        e = np.block([e, 0])
        Ab = np.block([[d], [e]])
        w, v = eigen_solver(Ab, lower=True)

    else:
        print("This solver cannot be used!")

    # Wave equation should not result in positive eigenvalue
    w_neg_ixs = w <= 0.0
    w = w[w_neg_ixs]
    v = v[w_neg_ixs]

    # Numpy does not output in a ordered manner, we should sort it by ourselves
    ix_sorted = np.argsort(w)[::-1]
    w = w[ix_sorted]
    v = v[:, ix_sorted]

    # Inclusion of start and end points according to the BC type
    if BC_type == "d":
        scale_factor = np.sqrt(N / 2)
        y = np.vstack((np.zeros((1, N)), v, np.zeros((1, N)))) * scale_factor

    elif BC_type == "n":
        scale_factor = np.sqrt(N / 2)
        if symmetric_solver:
            y = np.vstack((v[0, :], v, v[-1, :])) * scale_factor
        else:
            y = np.vstack(((4/3) * v[0, :] - (1/3) * v[1, :],
                           v, (4/3) * v[-1, :] - (1/3) * v[-2, :])) * scale_factor

    elif BC_type == "p":
        scale_factor = np.sqrt(N / 2)
        y = np.vstack((v, v[0, :])) * scale_factor

    else:
        print("This BC type is not valid!")

    x = np.linspace(0.0, 1.0, y.shape[0])

    # Return the results
    return x,y[:, :num_modes], w[:num_modes]


def get_A_matrix_d(N=8, is_symmetric=False):
    """
    A matrix construction for Dirichlet BC
    """

    A = np.zeros((N, N))  # Preallocation of the A matrix

    A[0, 0:2] = [-2, 1]  # First row

    for k in range(1, N-1):
        A[k, (k-1):(k+2)] = [1, -2, 1]  # Intermediate rows

    A[-1, -2:] = [1, -2]  # Last row

    return A


def get_A_matrix_n(N=8, is_symmetric=False):
    """
    A matrix construction for Neumann BC
    """
    if is_symmetric:
        A = np.zeros((N, N))  # Preallocation of the A matrix

        A[0, 0:2] = [-1, 1]  # First row

        for k in range(1, N-1):
            A[k, (k-1):(k+2)] = [1, -2, 1]  # Intermediate rows

        A[-1, -2:] = [1, -1]  # Last row
        
    else:

        A = np.zeros((N, N))  # Preallocation of the A matrix

        A[0, 0:2] = [-2/3, 2/3]  # First row

        for k in range(1, N-1):
            A[k, (k-1):(k+2)] = [1, -2, 1]  # Intermediate rows

        A[-1, -2:] = [2/3, -2/3]  # Last row

    return A


def get_A_matrix_p(N=8, is_symmetric=False):
    """
    A matrix construction for Periodic BC
    """
    A = np.zeros((N, N))  # Preallocation of the A matrix

    A[0, 0:2] = [-2, 1]  # First row
    A[0, -1] = 1
    for k in range(1, N-1):
        A[k, (k-1):(k+2)] = [1, -2, 1]  # Intermediate rows

    A[-1, -2:] = [1, -2]  # Last row
    A[-1, 0] = 1

    return A


def RMSE(y1, y2, calculate_absolute = True):
    """
    wave_solver

    This function executes the following processes:
        - It computes the root-mean-squarred-error of two vectors.
        - It can also compute the RMSE by taking absolute values of the input vectors.
       

    Input:
        y1 = First vector
        y2 = Second vector
        calculate_absolute = The option that specifies whether the absolute value of the inputs will be taken or not

    Output:
        RMSE = Calculated RMSE error of two vectors

    Usage:
        y1 = np.random.randn(100)
        y2 = np.random.randn(100)
        y_error = RMSE(y1, y2 , calculate_absolute = False)
    """
    if (y1.shape[0] != y2.shape[0]):
        print("Inputs must be in the same size!")
    else:
        if calculate_absolute:
            return np.sqrt(np.sum((np.abs(y1) - np.abs(y2))**2) / (y1.shape[0]))
        else:
            return np.sqrt(np.sum((y1 - y2)**2) / (y1.shape[0]))


if __name__ == "__main__":
    mytests()
