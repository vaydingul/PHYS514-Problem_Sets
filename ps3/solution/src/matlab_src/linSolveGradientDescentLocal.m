function x = linSolveGradientDescentLocal(A, b)
    %linSolveGradientDescentLocal
    %
    % Syntax: x = linSolveGradientDescentLocal(A, b)
    %
    % Long description
    %   Solves for x satisfying Ax=b for a symmetric
    %   matrix A using the conjugate gradient method.
    %
    
    n = length(A); % Dimension of the A matrix (n x n)
    grad = ones(n, 1); % Gradient initialization
    x = zeros(n, 1); % Initial estimate
    tolerance = 1e-3; % Condition for iteration termination

    while norm(grad) > tolerance

        grad = A * x - b; % Gradient calculation
        tau = (grad' * grad) / (grad' * A * grad); % Learning rate calculation
        x = x - tau * grad; % Jacobi iteration

    end

end
