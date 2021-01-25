function x = linSolveGaussSeidel(A, b)
    %linSolveGaussSeidel 
    %
    % Syntax: x = linSolveGaussSeidel(A, b)
    %
    % Long description
    %   Solves for x satisfying Ax=b for a symmetric
    %   matrix A using the Gauss Seidel iterative method.
    %
    %   Inputs:
    %   A = Linear system coefficient matrix
    %   b = RHS of linear system
    %
    %   Outputs:
    %   x = Solution of linear system

    n = length(A); %dimension of the A matrix

    x_prev = ones(n,1);
    x_next = zeros(n,1); % initial estimate

    L_star = tril(A); % Lower triangular part of the A matrix
    U = triu(A,1); % Strictly upper triangular part of the A matrix

    tolerance = 1e-3; % Criteria for iteration of termination
    

    while( norm(x_next - x_prev) > tolerance )

        x_prev = x_next; % Storing previous estimation for comparison
        x_next = L_star \ (b - U * x_prev); % Gauss-Seidel Iteration
    end
    x = x_next;
    end