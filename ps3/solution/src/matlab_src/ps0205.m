function ps0205(n_vec)
    %ps0205 
    %
    % Syntax: ps0205(n_vec)
    %
    % Long description
    %   This function do the following processes:
    %       - It calculates the average elapsed time for solution of varying size of matrices with different
    %         solution methods
    %       - It plots the results of the average elapsed time
    %
    %   Inputs:
    %   n_vec = The dimension vector whose elements are the desired matrix size for elapsed time calculation


    num_iter = 100; % Number of iteration to take average
    max_matrix_size = length(n_vec); % Final value in the n_vec

    for n = n_vec
        main_diagonal = 1; % Main diagonal element value of to-be-solved matrix
        off_diagonal = -0.5; % Off diagonal element value of to-be-solved matrix

        time_gradient_descent(n, :) = calculateAverageTime(@linSolveGradientDescentLocal, n, main_diagonal, off_diagonal, num_iter);
        %It solves the system with Gradient Descent method, and outputs the average time elapsed
        time_gauss_seidel(n, :) = calculateAverageTime(@linSolveGaussSeidel, n, main_diagonal, off_diagonal, num_iter);
        %It solves the system with Gauss-Seidel method, and outputs the average time elapsed
            
    end

    %%%%%%%%%%%%%%%%%%%%% PLOTTING SECTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    a = figure;
    hold on;
    plot([1:max_matrix_size], time_gradient_descent(:, 1), "-o");
    plot([1:max_matrix_size], time_gradient_descent(:, 2), "-*");
    plot([1:max_matrix_size], time_gauss_seidel(:, 1), "-^");
    plot([1:max_matrix_size], time_gauss_seidel(:, 2), "-.");
    legend("Gradient Descent - Sparse A Matrix", "Gradient Descent - Default A Matrix", ...
        "Gauss Seidel - Sparse A Matrix", "Gauss Seidel - Default A Matrix")
    xlabel("Size of A Matrix [nxn]");
    ylabel("Elapsed Time [s]");
    title("Time Comparisons");
    hold off;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
