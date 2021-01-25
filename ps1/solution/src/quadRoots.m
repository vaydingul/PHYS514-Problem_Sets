function [r1, r2] = quadRoots ( a , b , c )
% QUADROOTS Finds the two roots of a qudratic equation ax^2+bx+c=0
% using the more correct form of the discriminant formula.
%
%
%   Input:
%
%   a, b, c: Coefficients of the quadratic equation in the form of
%               ax^2 + bc + c = 0
%
%   Output:
%   r1, r2: Roots of the quadratic equation


r1 = (-b - sign(b) * sqrt(b^2 - 4 * a * c)) / (2 * a);
r2 = c / (a * r1);

end