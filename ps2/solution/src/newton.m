function root = newton(fun, dfun, x_init)
    %newton 
    %
    % Syntax: root = newton(fun, dfun, x_init)
    %
    % Long description
    %   Solves fun(x) = 0 using the Newton-Raphson method. It requires an initial guess
    %   and function handles to fun(x) and its derivative, fun and dfun respectively.
    %
    %   Inputs:
    %   fun = Function whose root is to be determined
    %   dfun = Derivative of fun
    %   x_init = Initial root estimation
    %
    %   Outputs:
    %   root = Estimated root

    x_prev = x_init; % Initial estimation
    x_next = x_prev + 1;
    tolerance = 1e-6; % Condition for iteration termination

    while norm(x_next - x_prev) > tolerance

        x_prev = x_next; % Storing previous estimate for comparison
        x_next = x_prev - fun(x_prev) ./ dfun(x_prev); % Newton-Raphson iteration

    end
    root = x_next;

end
