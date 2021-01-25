function [lambda, v] = eigLargest(A)
    %eigLargest
    %
    % Syntax: [lambda, v] = eigLargest(A)
    %
    % Long description
    %
    %   It computes the largest eigenvalue of a given matrix in absolute value manner using Power Iteration method.
    %
    %   Inputs:
    %   A = The matrix whose largest eigenvalue is askes
    %
    %   Outputs:
    %   lambda = Largest eigenvalue of A matrix
    %   v = The corresponding eigenvector of lambda

    tolerance = 1e-7; % The tolerance/criteria to terminate the iteration

    v = randn(size(A, 1), 1); % Initial value for eigenvector estimation
    v = v / norm(v); % Normalization

    criterion = 1; % Initial criterion evaluation to start to loop

    lambda_next = 0; % Initial estimation of lambda

    while criterion > tolerance

        lambda_prev = lambda_next; % Storing the lambda estimate of previous iteration
        v = A * v; % Power iteration
        v = v / norm(v); % Normalization
        lambda_next = ((A * v)' * v) / (v' * v); % Finding eigenvalue using Rayleigh's Quotient

        criterion = abs(lambda_next - lambda_prev); % Calculation of termination condition
    end

    lambda = lambda_next;

end
