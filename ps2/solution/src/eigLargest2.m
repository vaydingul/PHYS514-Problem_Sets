function [lambda, v] = eigLargest2(A)
    %eigLargest
    %
    % Syntax: [lambda, v_next] = eigLargest(A)
    %
    % Long description
    %
    %   It computes the largest eigenvalue of a given matrix in absolute value manner using Power Iteration method.
    %   The only difference from the eigLargest() function is that this function uses different termination criteria.
    %
    %   Inputs:
    %   A = The matrix whose largest eigenvalue is askes
    %
    %   Outputs:
    %   lambda = Largest eigenvalue of A matrix
    %   v = The corresponding eigenvector of lambda

    tolerance = 1e-2;

    v_next = randn(size(A, 1), 1); % Initial eigenvector estimation
    v_next = v_next / norm(v_next); % Normalization

    criterion = 1;

    while criterion > tolerance
        v_prev = v_next; % Storing the eigenvector value of previous iteration
        v_next = A * v_prev; % Power iteration
        v_next = v_next / norm(v_next); % Normalization

        criterion = norm(abs(v_next) - abs(v_prev)); % Calculation of terminatio criteria
    end

    lambda = ((A * v_next)' * v_next) / (v_next' * v_next); % Calculation of eigenvalue using Rayleigh's Quotient

    v = v_prev;

end
