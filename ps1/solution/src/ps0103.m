function ps0103 ( n_vec )
% PS0103 Main function for solving PHYS414/514 PS01 problem 4.For each
% element n of the integer vector n_vec, it calculates the roundoff error
% in summing n randomly generated numbers uniformly chosen from the
% interval [0 1]. Then these errors are plotted w.r.t.  n. This can be used
% to infer the dependece of the error on n.
%
%   Input:
%   n_vec = Vector of corresponding vector sizes to be summed up
%
%
%

n = length(n_vec); % Length of input vector n_vec

errors = zeros(1,n); % Preallocation of array to store "errors" from each iteration

for nn=n_vec
    
    errors(nn) = sumError(nn); % Call of error function in each iteration
    
end

%%%%%%%%%%%%%%%%%%%%%%% Plotting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(2,1,1)
plot(n_vec, errors);
xlabel("Size of array");
ylabel("Average round-off error")
title("Normal plot")
subplot(2,1,2)
loglog(n_vec, errors);
xlabel("Size of array");
ylabel("Average round-off error")
title("Loglog plot")


end