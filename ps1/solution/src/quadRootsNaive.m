function [r1, r2] = quadRootsNaive ( a , b , c )
% QUADROOTSNAIVE Finds the two roots of a qudratic equation ax^2+bx+c=0
% using the conventional discriminant formula .
%
%   Input:
%
%   a, b, c: Coefficients of the quadratic equation in the form of
%               ax^2 + bc + c = 0
%
%   Output:
%   r1, r2: Roots of the quadratic equation

r1 = (-b + sqrt(b^2 - 4 * a *c)) / (2 * a);
r2 = (-b - sqrt(b^2 - 4 * a *c)) / (2 * a);

end