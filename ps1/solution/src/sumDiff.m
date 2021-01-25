function diff = sumDiff ( vec )
% SUMDIFF Sums all the elements in a given vector first from beginning to
% the end , and then in the reverse order , and finally takes the absolute
% value of the difference of these sums . This is an estimate of the
% cumulative roundoff error in the sum of the numbers .
%
%   Input:
%   vec: Input vector of which we want to sum elements
%
%   Output:
%   diff: Difference between summation from back and forth
%
%   Note: To use Kahan Summation rather than default MATLAB summation,
%   please uncomment below function handle.
%
%

%sum = @sumKahan; %Fucntion handle to use Kahan Summation, uncomment to use
%rather than default MATLAB summation

vec_length = length(vec); %Length of input vector

sum1 = sum(vec); %Summation of vector elements from beginning to end

vecFlip = vec(vec_length:-1:1); %Flipped form of the input vector

sum2 = sum(vecFlip); %Summation of flipped vector elements from beginning to end

diff = abs(sum1 - sum2); %Absolute value of difference between two summations


end