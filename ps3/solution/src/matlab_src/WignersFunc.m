function y = WignersFunc(x)
%WignersFunc 
%
% Syntax: y = WignersFunc(x)
%
% Long description
%   It calculates the probability for a given interval according to the Wigner's Semicircle Law.

y = (1 / (1))*((1 / 2) .* sqrt(4 - x .^ 2) .* x + 2 * asin(x / 2));
    
end