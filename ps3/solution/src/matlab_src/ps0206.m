function ps0206()
    %ps0206 
    %
    % Syntax: ps0206()
    %
    % Long description
    %   This function do the following processes:
    %       - It calculates the average elapsed time for root finding of a given input vector using different
    %         iterative methods.
    %       - It plots the results of the average and total elapsed time for different methods.
    %       - It plots the function evaluations and differences of different iterative methods and MATLAB implementation of 
    %         Lambert W function
    %
    %   Inputs:
    %   n_vec = The dimension vector whose elements are the desired matrix size for elapsed time calculation

    num_iter = 100; % Number of iteration to calculate average of elapsed time
    x = [-1 / exp(1):0.01:10]'; % Input vector
    time_methods = [0 0 0 0]; % Cumulative time vector initilization

    for n = 1:num_iter
        
        tic;
        for k=1:length(x)
        y_newton_elementwise(k) = lambertwCustom(x(k), "n"); % Root finding using Element-by-element Newton-Raphson method
        end
        time_newton_elementwise = toc;

        tic;
        [y_newton] = lambertwCustom(x, "n"); % Root finding using Newton-Raphson method
        time_newton = toc;

        tic;
        [y_halley] = lambertwCustom(x, "h"); % Root finding using Halley's method
        time_halley = toc;

        tic;
        y_matlab = lambertw(x); % Root finding using MATLAB's built-in method
        time_matlab = toc;

        time_methods = time_methods + [time_newton_elementwise time_newton time_halley time_matlab]; % Cumulate results

    end

    average_time = time_methods / num_iter; % Take average

    %%%%%%%%%%%%% Information Section [plots and informations about accuracy and timing] %%%%%%%%%%%%%%%%%


    disp(['Average Timing of Newton-Raphson Method (Element-by-Element): ', num2str(average_time(1))]);
    disp(['Average Timing of Newton-Raphson Method: ', num2str(average_time(2))]);
    disp(['Average Timing of Halleys Method: ', num2str(average_time(3))]);
    disp(['Average Timing of MATLAB Builtin: ', num2str(average_time(4))]);
    disp(['Total Timing of Newton-Raphson Method (Element-by-Element): ', num2str(time_methods(1))]);
    disp(['Total Timing of Newton-Raphson Method: ', num2str(time_methods(2))]);
    disp(['Total Timing of Halleys Method: ', num2str(time_methods(3))]);
    disp(['Total Timing of MATLAB Builtin: ', num2str(time_methods(4))]);
    
    %%%%%%% Undetailed version of Bar plot of average and total elapsed time for different methods
    X = categorical({'Newton-Raphson Method Element-by-Element','Newton-Raphson Method','Halleys Method','MATLAB Builtin'});
    a = figure;
    title(['Time Comparisons over ' num2str(num_iter) ' runs'])
    yyaxis left;
    bar(X, average_time);
    ylabel("Average Elapsed Time [s]");
    yyaxis right;
    bar(X, time_methods, 0.5);
    ylabel("Total Elapsed Time [s]");

    %%%%%%% Detailed version of Bar plot of average and total elapsed time for different methods (omit element-by-element)
    b = figure;
    X = categorical({'Newton-Raphson Method','Halleys Method','MATLAB Builtin'});
    title(['Time Comparisons over ' num2str(num_iter) ' runs (element-by-element is omitted)'])
    yyaxis left;
    bar(X, average_time(2:4));
    ylabel("Average Elapsed Time [s]");
    yyaxis right;
    bar(X, time_methods(2:4), 0.5);
    ylabel("Total Elapsed Time [s]");

    %%%%%% Function evaluation and difference of different methods with repsect to MATLAB Built-in solution
    c = figure;
    subplot(2,1,1);
    hold on;
    plot( x, y_newton, "-o","MarkerIndices", [1:20:length(x)]);
    plot( x, y_halley, "-*","MarkerIndices", [1:20:length(x)]);
    %plot( x, y_matlab, "-^");
    legend("Newton-Raphson Method", "Halley's Method","location","best")
    title("Function Evaluations for Two Methods")
    xlabel("x");
    ylabel("y");
    hold off;
    
    subplot(2,1,2);
    hold on;
    plot( x, abs(y_matlab - y_newton), "-o","MarkerIndices", [1:20:length(x)]);
    plot( x, abs(y_matlab - y_halley), "-*","MarkerIndices", [1:20:length(x)]);
    %plot( x, y_matlab, "-^");
    legend("Difference Between MATLAB and Newton-Raphson Method", "Difference Between MATLAB and Halley's Method","location","best")
    title("Differences Between Two Methods and MATLAB")
    xlabel("x");
    ylabel("y");
    hold off;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
