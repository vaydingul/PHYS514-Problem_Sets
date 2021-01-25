function [lambda_vec, eig_matrix] = eigCustom(A)
    %eigCustom - Description
    %
    %
    % Syntax: [lambda_vec, eig_matrix] = eigCustom(A)
    %
    % Long description
    %   Calculates all the eigenvalues and normalized eigenvectors of the symmetric
    %   matrix X. Returns the eigenvalues as a column vector with ascending elements (lambda_vec)i and the
    %   respective normalized eigenvectors as columns of a square matrix (eig_matrix). It uses eigLargest
    %   and finds the eigenvalues successively using Hotelling's deflation.
    %
    %   Inputs:
    %   A = Matrix whose eigenvalues are asked
    %
    %   Outputs:
    %   lambda_vec = The sorted (ascending order) vector that stores the eigenvalues of A matrix
    %   eig_matrix = The matrix whose columns are the corresponding eigenvalues of in the lambda_vec

    len = size(A, 1); % size of A (len x len)
    lambda_vec = zeros(len, 1); % Preallocation of lambda_vec
    eig_matrix = zeros(len, len); %Preallocation of eig_matrix

    for n = 1:len

        [lambda, v] = eigLargest(A); % It computes the largest eigenvalue of given matrix

        lambda_vec(n) = lambda; % Store the obtained eigenvalue
        eig_matrix(:, n) = v; % Store the obtained eigenvector

        A = A - lambda * (v * v'); % Apply Hotelling's Deflation

    end

    [lambda_vec, I] = sort(lambda_vec); % Since, the eigenvalues are obtained in absolute value manner.
    % They should be sorted.
    eig_matrix = eig_matrix(:, I); % Also, we need to sort the eig_matrix according to the lambda_vec

end
