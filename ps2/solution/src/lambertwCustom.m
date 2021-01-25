function y = lambertwCustom(x, method)
    %lambertwCustom - Description
    % Syntax: y = lambertwCustom(x)
    %
    % Long description
    %   It calculates the Lambert W fucntion defined as W(xe^x) = y if
    %   y = xe^x, using the Newton-Raphson method or the Halley's method.
    %
    %   Inputs:
    %   x = The vector of points which the Lambert W function should be evaluated
    %   method = Method selection ==> "h" for Halley's method, "n" for Newton-Raphson Method
    %
    %   Outputs:
    %   y = The vector of evaluated function points

    x_size = length(x); % Size of input vector
    fun_shifted = @(z)(z .* exp(z) - x); % Function whose root is desired
    fun_deriv = @(z)((z + 1) .* exp(z)); % Derivative of fun_shifted
    fun_deriv_deriv = @(z)((z + 2) .* exp(z)); % Derivative fun_deriv

    x_init = ones(x_size, 1); % Initial estimate vector preallocation


    %%%%%%%%%%%%%% Method Logic %%%%%%%%%%%%%%%%%%%%%%%%%%
    if nargin == 1
        y = newton(fun_shifted, fun_deriv, x_init);
    else

        if method == "h"
            y = halley(fun_shifted, fun_deriv, fun_deriv_deriv, x_init);
        else if method == "n"
            y = newton(fun_shifted, fun_deriv, x_init);
        else
            y = newton(fun_shifted, fun_deriv, x_init);
        end

    end

end
