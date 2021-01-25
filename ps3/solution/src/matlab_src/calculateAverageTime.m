function average_time = calculateAverageTime(solver, matrix_size, main_diagonal, off_diagonal, num_iter)
    %calculateAverageTime 
    %
    % Syntax: average_time = calculateAverageTime(solver, matrix_size, main_diagonal, off_diagonal, num_iter)
    %
    % Long description
    % This function evaluates the average elapsed time for the solution of linear linear system.
    %
    %   Inputs:
    %   solver = Linear system solution method function handle
    %   matrix_size = Size of the (NxN) matrix
    %   main_diagonal = The value located in main diagonal
    %   off_diagonal = The value located in off diagonal
    %   num_iter = Number of iteration to calculate average time
    %
    %   Outputs:
    %   average_time = Calculated average time

    time_sum = [0, 0]; % cumulative time vector preallocation
    a1 = main_diagonal; % main diagonal element
    a2 = off_diagonal; % off diagonal element
    n = matrix_size; % matrix size (n x n)
    e = ones(n, 1);
    
    A_sparse = spdiags([a2 * e a1 * e a2 * e], -1:1, n, n); % Sparse matrix construction
    A_tridiag = diag(a1 * ones(1, n)) + diag(a2 * ones(1, n - 1), 1) + diag(a2 * ones(1, n - 1), -1); % Tridiagonal matrix construction

    for k = 1:num_iter
        b = randn(n, 1); % Create random b vector at each iteration
        
        tic;
        [~] = solver(A_sparse, b); % Solve Ax=b and store the elapsed time
        time_sparse = toc;

        tic;
        [~] = solver(A_tridiag, b); % Solve Ax=b and store the elapsed time
        time_tridiag = toc;

        time_sum = time_sum + [time_sparse time_tridiag]; % Aggragate results
    end

    average_time = time_sum / num_iter; % Take average
end
