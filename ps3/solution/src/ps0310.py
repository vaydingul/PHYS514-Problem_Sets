import numpy as np
from timeit import timeit
import matplotlib.pyplot as plt
from numpy.core.multiarray import concatenate
from numpy.lib.index_tricks import ix_
import scipy.special as spspec
import scipy.sparse as spspar


def mytests(n_vec=np.arange(1, 50, 4)):
    """
    mytests

    Syntax: mytests()

    This function do the following processes:
      - It computes the eigenvalues and eigenvectors of a predetermined matrix
      - It plots the average elapsed time of Power iteration method and NumPy Built-in method
      - It compares the accuracy of eigenvalues and eigenvectors for iterative and built-in method
    """

    ####### Power Iteration part #################################################

    tolerances = [1e-2] # Tolerance values in which we will iterate
    #tolerances = [1e-2, 1e-4] # To see effect of different tolerances, uncomment it 

    for tolerance in tolerances:
        
        num_iter = 10 # Number of iteration to calculate average
        size_n = n_vec.shape[0] # Size of n_vec
        time_results = np.empty((size_n, 2)) # Timing vector preallocation

        for (idx, n) in enumerate(n_vec):

            A = np.random.randn(n, n)
            # Symmetric matrix construction
            A = A - np.tril(A) + (np.triu(A)).T
            print("Matrix Size [nxn] = ", n, " Tolerance = ", tolerance) # Info about where we are
            # Calculation of eigenvalues and eigenvectors

            # Timing of custom eigenvalue/vector calculator
            time_results[idx, 0] = timeit(lambda: eig_custom(
                A, tolerance=tolerance), number=num_iter)

            # Timing of NumPy built-in calculator
            time_results[idx, 1] = timeit(
                lambda: np.linalg.eig(A), number=num_iter)

        ############# PLOTTING #######################################
        plt.figure()
        plt.plot(n_vec, time_results/num_iter)
        plt.xlabel("Matrix of [n x n]")
        plt.ylabel("Average Elapsed Time [sec]")
        plt.title(
            "Average Elapsed Time for Different Methods under {0} tolerance".format(tolerance))
        plt.legend(["Power Iteration Method", "NumPy Built-in"], loc="best")
        ###############################################################
        
        analysis_n = [100] # Matrix size of which we will compare with the NumPy
        #analysis_n = [16 ,32, 64, 96] To see accuracies of different matrices, uncomment it

        for n in analysis_n:
            A = np.random.randn(n, n)
            A = A - np.tril(A) + (np.triu(A)).T # Symmetric matrix construction

            lambda_vec, eig_matrix = eig_custom(A, tolerance= tolerance) # Calculation of eigenvalues and eigenvectors
            print("Matrix Size [nxn] = ",n," Tolerance = ",tolerance) # Info about where we are

            w, v = np.linalg.eig(A) # NumPy built-in calculation
            ix_sorted = np.argsort(w, axis = 0) # Numpy does not output in a ordered manner, we should sort it by ourselves
            w = w[ix_sorted]
            v = v[:, ix_sorted]

            ####### PLOTTING ############################################
            plt.figure()
            plt.plot(lambda_vec, w)
            plt.xlabel("Eigenvalues calculated by Power Iteration")
            plt.ylabel("Eigenvalues calculated by NumPy")
            plt.title("Eigenvalue Comparison for [{0} x {0}] matrix @ {1} tolerance".format(n, tolerance))

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.set_title("Eigenvector Comparison for [{0} x {0}] matrix @ {1} tolerance".format(n, tolerance))
            ax.set_xlabel("Eigenvectors calculated by Power Iteration")
            ax.set_ylabel("Eigenvectors calculated by NumPy")
            counter = 0
            for k in range(n):
                ax1 = fig.add_subplot(np.ceil(np.sqrt(n)), np.ceil(np.sqrt(n)), counter+1)
                ax1.plot(eig_matrix[:, counter], v[:, counter], "g-o")
                ax1.set_xticks([])
                ax1.set_yticks([])
                counter += 1
            ################################################################
    ########################################################################
    
    ####### Wigner's Semicircle Law #########################################

    partition_size = 100 # Bar count in barplot

    N = [512, 1024, 2048] # Matrix size (N x N)

    for n in N:
        A = np.random.randn(n, n)
        A = A - np.tril(A) + (np.triu(A)).T # Symmetric matrix construction
        w, _ = np.linalg.eig(A) # Eigenvalue calculation

        x = np.linspace(np.min(w), np.max(w), partition_size) # Create linear space for the calculation
        # of Wigner's Semicircle Law

        probs = wigners_prob( x / np.sqrt(n)) # Probability calculation according to Wigner's Semicircle Law

        ################# PLOTTING #######################################
        plt.figure()
        plt.hist(w, bins=partition_size, density= True)
        plt.plot(x[0:-1], probs, "r")
        plt.legend(["Probability Distribution of Eigenvalues", "Wigner's Semicircle Law"])
        plt.xlabel("Eigenvalues")
        plt.ylabel("Probabilities")
        plt.title("Probability Distributions for N = {0}".format(n))
        ########################################################################
    ############################################################################
    
    plt.show()


def eig_largest(A, tolerance=1e-3):
    """
    eig_largest

    Syntax: lambda, v = eig_largest(A, tolerance)

    It computes the largest eigenvalue of a given matrix in absolute value manner using Power Iteration method.

    Inputs:
    A = The matrix whose largest eigenvalue is askes
    tolerance = Iteration termination condition

    Outputs:
    lambda = Largest eigenvalue of A matrix
    v = The corresponding eigenvector of lambda
    """
    n = A.shape[0]  # Size of matrix [n x n]
    v = np.random.randn(n, 1)  # Initial value for eigenvector estimation
    v = v / np.linalg.norm(v)  # Normalization

    criterion = 1  # Initial criterion evaluation to start to loop

    lambda_next = 0  # Initial estimation of lambda

    while criterion > tolerance:

        lambda_prev = lambda_next  # Storing the lambda estimate of previous iteration
        v = A @ v  # Power iteration
        v = v / np.linalg.norm(v)  # Normalization

        # Finding the eigenvalue using Rayleigh's Quotient
        lambda_next = ((A @ v).T @ v) / (v.T @ v)

        # Calculation of termination condition
        criterion = abs(lambda_next - lambda_prev)
    return lambda_next, v


def eig_largest_2(A, tolerance=1e-3):
    """
    eig_largest_2

    Syntax: lambda, v = eig_largest_2(A)

    It computes the largest eigenvalue of a given matrix in absolute value manner using Power Iteration method.

    Inputs:
    A = The matrix whose largest eigenvalue is askes
    tolerance = Iteration termination condition

    Outputs:
    lambda = Largest eigenvalue of A matrix
    v = The corresponding eigenvector of lambda
    """
    n = A.shape[0]  # Size of matrix [n x n]

    v = np.random.randn(n, 1)  # Initial value for eigenvector estimation
    v = v / np.linalg.norm(v)  # Normalization

    norm_A_est = np.sqrt(np.linalg.norm(A, ord=1) *
                         np.linalg.norm(A, ord=np.inf))

    tolerance = tolerance * norm_A_est
    criterion = 1  # Initial criterion evaluation to start to loop

    lambda_est = 0  # Initial estimation of lambda

    Av = A @ v

    while criterion > tolerance:

        v = Av / np.linalg.norm(Av)  # Normalization

        Av = A @ v  # Power iteration

 
        lambda_est = v.T @ Av # Finding the eigenvalue using Rayleigh's Quotient

        criterion = Av - v @ lambda_est
        criterion = np.linalg.norm(criterion)

    return lambda_est, v


def eig_custom(A, tolerance=1e-3):
    """
    eig_custom 


    Syntax: lambda_vec, eig_matrix = eig_custom(A)

    Calculates all the eigenvalues and normalized eigenvectors of the symmetric
    matrix X. Returns the eigenvalues as a column vector with ascending elements (lambda_vec)i and the
    respective normalized eigenvectors as columns of a square matrix (eig_matrix). It uses eigLargest
    and finds the eigenvalues successively using Hotelling's deflation.

    Inputs:
    A = Matrix whose eigenvalues are asked
    tolerance = Iteration termination condition

    Outputs:
    lambda_vec = The sorted (ascending order) vector that stores the eigenvalues of A matrix
    eig_matrix = The matrix whose columns are the corresponding eigenvalues of in the lambda_vec

    """
    N = A.shape[0]  # Size of A [N x N]
    lambda_vec = np.empty((N, 1))
    eig_matrix = np.empty((N, N))

    for n in range(N):

        # It computes the largest eigenvalue of given matrix
        lambda_val, v = eig_largest_2(A, tolerance=tolerance)

        lambda_vec[n] = lambda_val  # Storing the obtained eigenvalue
        eig_matrix[:, n:n+1] = v  # Storing the obtained eigenvector

        A = A - lambda_val * (v @ v.T)  # Apply Hotelling's Deflation

    # Since, the eigenvalues are obtained in absolute value manner.
    # They should be sorted.
    ix_sorted = np.argsort(lambda_vec, axis=0)
    lambda_vec = np.resize(lambda_vec[ix_sorted], (N,))
    # Also, we need to sort the eig_matrix according to the lambda_vec
    eig_matrix = np.resize(eig_matrix[:, ix_sorted], (N, N))
    return lambda_vec, eig_matrix


def wigners_func(x):
    """
    wigners_func 

    Syntax: y = wigners_func(x)

    Long description
        It calculates the probability for a given interval according to the Wigner's Semicircle Law.

    """
    return (1 / 2) * np.sqrt(4 - x ** 2) * x + 2 * np.arcsin(x / 2)


def wigners_prob(x):
    """
    wigners_prob

    Syntax: prob = wigners_prob(x)


    This function calculates the probobility of the existence of an eigenvalue
    for a given interval using Wigner's Semicircle Law.

    Inputs:
    x = Input vector whose probabilities will be determined

    Outputs:
    probs = Output probability vector
    """
    n = x.shape[0]  # Get size of input vector
    prob = np.empty((n-1, ))
    for k in range(n-1):  # Since the probability will be calculated interval-wise,
        # We need to truncate the loop before last element
        x1 = x[k]  # Beginning of interval
        x2 = x[k + 1]  # Enf of interval

        prob[k] = (1/(2 * np.pi))*(wigners_func(x2) - wigners_func(x1))

    return prob


if __name__ == "__main__":
    mytests()
