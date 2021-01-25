function root = halley(fun, dfun, ddfun, x_init)
    %halley - Description
    %
    % Syntax: root = halley(fun, dfun, ddfun, x_init)
    %
    % Long description
    %   Solves fun(x) = 0 using the Halley's method. It requires an initial guess
    %   and function handles to fun(x) and its first and second derivative, fun, dfun and ddfun respectively.
    %
    %   Inputs:
    %   fun = Function whose root is to be determined
    %   dfun = Derivative of fun
    %   ddfun = Derivative of dfun
    %   x_init = Initial root estimation
    %
    %   Outputs:
    %   root = Estimated root

    x_prev = x_init; % Initial estimation allocation
    x_next = x_prev + 1;
    tolerance = 1e-6 % Criteria to terminate iteration

    while norm(x_next - x_prev) > tolerance

        x_prev = x_next; % Storing the previous estimation

        x_next = x_prev - (fun(x_prev) ./ dfun(x_prev)) .* (1 - ((fun(x_prev) .* ddfun(x_prev)) ./ (2 .* dfun(x_prev) .* dfun(x_prev)))).^(-1);
        % Halley's iterative scheme
    end

    root = x_next;

end
