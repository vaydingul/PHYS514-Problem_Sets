a = 1;
b = 200; %Coefficients of quadratic equation
c = 1.5e-15;

[r1, r2] = quadRootsNaive(a, b, c); %Calculate roots with naive formula
[rr1, rr2] = quadRoots(a, b, c); %Calculate roots with enhanced formula