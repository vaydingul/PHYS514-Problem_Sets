import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from timeit import timeit
import matplotlib.pyplot as plt


def mytests(n_vec=np.arange(1, 50, 4)):
    """
    Syntax: mytests(n_vec)

    This function do the following processes:
      - It calculates the average elapsed time for solution of varying size of matrices with different
        solution methods
      - It plots the results of the average elapsed time
    Inputs:
    n_vec = The dimension vector whose elements are the desired matrix size for elapsed time calculation
    """
    size_n = n_vec.shape[0]  # Size of n_vec
    num_iter = 10  # Number of iteration to calculate timing
    time_vec = np.empty((size_n, 4))  # Timing vector preallocation

    for (idx, n) in enumerate(n_vec):
        print("Matrix Size [n x n] = ", n)

        diagonals = [np.ones(n), np.ones(n-1) * (-0.5), np.ones(n-1) * (-0.5)]
        # Tridiagonal matrix construction
        A_tridiag = sps.diags(diagonals, [0, -1, 1], format="csc").toarray()
        # Sparse matrix construction
        A_sparse = sps.diags(diagonals, [0, -1, 1], format="csc")

        # Timing of Gradient Descent with tridiagonal matrix
        time_vec[idx, 0] = timeit(lambda: lin_solve_gradient_descent_local(A_tridiag,
                                                                           np.random.randn(n, 1)), number=num_iter)
        # Timing of Gradient Descent with sparse matrix
        time_vec[idx, 1] = timeit(lambda: lin_solve_gradient_descent_local(
            A_sparse, np.random.randn(n, 1)), number=num_iter)

        # Timing of Gauss-Seidel with tridiagonal matrix
        time_vec[idx, 2] = timeit(lambda: lin_solve_gauss_seidel(A_tridiag,
                                                                 np.random.randn(n, 1)), number=num_iter)

        # Timing of Gauss-Seidel with sparse matrix
        time_vec[idx, 3] = timeit(lambda: lin_solve_gauss_seidel(
            A_sparse, np.random.randn(n, 1)), number=num_iter)

    ################### PLOTTING ###################################################
    plt.figure()
    plt.plot(n_vec, time_vec/num_iter)
    plt.xlabel("Matrix Size [n x n]")
    plt.ylabel("Elapsed Time [sec]")
    plt.legend(["Gradient Descent Default Matrix", "Gradient Descent Sparse Matrix",
                "Gauss-Seidel Default Matrix", "Gauss-Seidel Sparse Matrix"], loc="best")
    ################################################################################
    plt.show()


def lin_solve_gradient_descent_local(A, b, tolerance=1e-1):
    """
    lin_solve_gradient_descent_local

    Syntax: x = lin_solve_gradient_descent_local(A, b)

    Solves for x satisfying Ax=b for a symmetric
    matrix A using the conjugate gradient method.

    Inputs:
    A = Linear system coefficient matrix
    b = RHS of linear system

    Outputs:
    x = Solution of linear system
    """

    n = A.shape[0]  # Dimension of the A matrix(n x n)
    grad = np.ones((n, 1))  # Gradient initialization
    x = np.zeros((n, 1))  # Initial estimate for x

    while np.linalg.norm(grad) > tolerance:

        grad = A @ x - b  # Gradient calculation
        # Learning rate optimization
        tau = (grad.T @ grad) / (grad.T @ A @ grad)
        x = x - tau * grad  # Jacobi iteration

    return x


def lin_solve_gauss_seidel(A, b, tolerance=1e-1):
    """
    lin_solve_gauss_seidel 
    Syntax: x = lin_solve_gauss_seidel(A, b)

    Solves for x satisfying Ax=b for a symmetric
    matrix A using the Gauss Seidel iterative method.

    Inputs:
    A = Linear system coefficient matrix
    b = RHS of linear system

    Outputs:
    x = Solution of linear system
    """
    n = A.shape[0]  # Dimension of the A matrix(n x n)
    x_prev = np.ones((n, 1))
    x_next = np.zeros((n, 1))  # İnitial estimate

    # In the if statement, that the matrix is either sparse or dense is checked
    if isinstance(A, np.ndarray):
        L_star = np.tril(A)  # Lower triangular part of the A matrix
        U = np.triu(A, k=1)  # Strictly upper triangular part of the A matrix
        while np.linalg.norm(x_next - x_prev) > tolerance:
            x_prev = x_next  # Storing previous estimation for comparison
            # Gauss-Seidel iteration
            x_next = np.linalg.solve(L_star, (b - U @ x_prev))

    elif sps.isspmatrix(A):
        # Lower triangular part of the A matrix
        L_star = sps.tril(A, format="csc")
        # Strictly upper triangular part of the A matrix
        U = sps.triu(A, k=1, format="csc")
        while np.linalg.norm(x_next - x_prev) > tolerance:
            x_prev = x_next  # Storing previous estimation for comparison
            # Gauss-Seidel iteration
            x_next = spsl.spsolve(L_star,  (b - U @ x_prev))

    return x_next


if __name__ == "__main__":
    mytests()
