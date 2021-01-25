function [y] = func0101b(x)
%func0101b - This function calculates the following function:
%   f(x) = 1 - ((sqrt(1 + x ^ 2)) / (1 + 0.5 * x ^ 2))
%
% Syntax: y = func0101b(x)
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
coefficient_list = [0, 0, 0, 0, 0.125, 0, -0.125, 0, 0.101563];

%Condition for power series expansion rather than naive calculation
power_series_condition = 1e-2;

%Preallocation of output vector with zeros
y = zeros(size(x));

if x > power_series_condition
    
    y = 1 - ((sqrt(1 + x .^ 2)) ./ (1 + 0.5 * x .^ 2)); %Naive calculation
    
else
    
    for n = 1:length(coefficient_list)
        
        y = y + coefficient_list(n) .* x .^(n-1); %Power series expansion
        
    end
    
end

end
