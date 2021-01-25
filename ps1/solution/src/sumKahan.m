function [ vec_sum ] = sumKahan( vec )
% SUMKAHAN Sums all the elements in vec using Kahan summation to achieve
% roundoff error independent of the length of the vector .
%
%   Input:
%   vec: Vector to be summed up
%
%   Output:
%   vec_sum: Summation of elements of vec

vec_length = length(vec); %Length of vector
compensator = 0; %Initialization of compensator
vec_sum = 0; %%Initialization of summation

for k=1:vec_length
    %Initially, the compensator is zero, however, with the flow of time,
    %the accumulated error coming from low order bits can be stored in
    %compensator.
    
    y = vec(k) - compensator; %Eliminate the effect of error
    
    t = vec_sum + y; %Perform summation
    
    %Below expression is algebraically 0, however, computationally, it
    %stores the loss of significance.
    compensator = (t - vec_sum) - y;
    
    vec_sum = t; %Final summation
    
end

end