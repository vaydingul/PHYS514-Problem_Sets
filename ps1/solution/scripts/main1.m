x = logspace(-10, 0, 1000); %Create logarithmically spaced input vector

[y] = func0101b(x); % Calculate function in (1.b)
[yy] = func0101c(x); % Calculate function in (1.b)

%%%%%%%%%%% PLOTTING %%%%%%%%%%%%%%%%%%%%%%%
a = figure;
loglog(x, y);
xlabel("x");
ylabel("y");
legend("f(x)")

b = figure;
loglog(x, yy);
xlabel("x");
ylabel("y");
legend("f(x)")


%saveas(b, "./../report/figures/loglogc.png")
%saveas(a, "./../report/figures/loglogb.png")


