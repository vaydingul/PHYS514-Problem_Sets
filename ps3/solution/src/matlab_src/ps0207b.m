function ps0207b()
    %ps0207b
    %
    % Syntax: ps0207b()
    %
    % Long description
    %   This function do the following processes:
    %       - It computes the eigenvalues and eigenvectors of a predetermined matrix
    %       - It plots the average elapsed time of Power iteration method and MATLAB Built-in method
    %       - It compares the accuracy of eigenvalues and eigenvectors for iterative and built-in method

    n_vec = 1:16; % 1:100; % The dimension vector whose elements are the desired matrix size for elapsed time calculation
    num_iter = 50; % 100;  % Number of iteration to take average

    for n = n_vec
        time_methods_temp = [0, 0]; % Cumulative time array initialization

        for k = 1:num_iter
            A = randn(n);
            A = A - tril(A) + (triu(A))'; % Symmetric matrix construction
            tic;
            [lambda_vec, eig_matrix] = eigCustom(A); % Calculation of eigenvalues and
            % eigenvectors with Power Iteration method
            time_power_iteration = toc;

            tic;
            [V, D] = eig(A); % Calculation of eigenvalues and
            % eigenvectors with MATLAB built-in method
            time_matlab = toc;

            time_methods_temp = time_methods_temp + [time_power_iteration time_matlab]; % Aggragate results
        end

        average_time = time_methods_temp / length(n_vec); % Take average
        time_methods(n, :) = time_methods_temp;
    end

    %%%%%%%%%%%%%%%%%%%%% PLOTTING SECTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%% Time comparison plot
    a = figure;
    hold on;
    plot([n_vec], time_methods(:, 1), "-o");
    plot([n_vec], time_methods(:, 2), "-*");
    legend("Power Iteration Method", "MATLAB")
    xlabel("Size of A Matrix [nxn]");
    ylabel("Average Elapsed Time [s]");
    title("Time Comparisons with Low Accuracy");
    hold off;

    %%%%%%%%%%%%%%%%%% Eigenvalue accuracy plot
    b = figure;
    plot(diag(D), sort(lambda_vec), "-o")
    xlabel("MATLAB Built-in eig() Function Solution")
    ylabel("Power Iteration Solution")
    title("Eigenvalue Comparison")

    %%%%%%%%%%%%%%%% Eigenvector accuracy plot
    d = figure;
    counter = 1;

    for k = 1:sqrt(n)

        for m = 1:sqrt(n)

            subplot(sqrt(n), sqrt(n), counter);
            plot((eig_matrix(:, counter)), (V(:, counter)), "-.");
            counter = counter + 1;

        end

    end

end
