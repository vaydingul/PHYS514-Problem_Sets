function [y] = func0101c(x)
%func0101c - This function calculates the following function:
%   f(x) = cot(x .^ 2) - 1 ./ (x .^ 2
%
% Syntax: y = func0101c(x)
%
% Long description
%
%   This function, additionally, takes care of the precision. Meaning
%   that, if the input number is less than a prespecified value,
%   function changes its calculation method. If it is the case,
%   it uses power expansion to
%   get rid of loss of significance, rather than naive calculation
%
%   Input:
%   x = Input vector
%
%   Output:
%   y = Output vector

%Coefficient list of power series of the function, taken from Wolfram Mathematica
coefficient_list = [0, 0, -1/3];

%Condition for power series expansion rather than naive calculation
power_series_condition = 1e-4;

%Preallocation of output vector with zeros
y = zeros(size(x));

if x > power_series_condition
    
    y = cot(x .^ 2) - 1 ./ (x .^ 2); %Naive calculation
    
else
    
    for n = 1:length(coefficient_list)
        
        y = y + coefficient_list(n) .* x .^(n-1); %Power series expansion
        
    end
    
end

end
