function ps0207c()
    %ps0207c
    %
    % Syntax: ps0207c()
    %
    % Long description
    %   This function do the following processes:
    %       - It computes the eigenvalues and eigenvectors of a predetermined matrix
    %       - 
    %       - I

    N = 2048; % Matrix size (N x N)

    A = randn(N);
    A = A - tril(A) + (triu(A))'; % Symmetric matrix construction

    [~, D] = eig(A); % Calculate eigenvalues
    D = diag(D); 

    x = linspace(min(D), max(D), 1000); % Create a linear space for the calculation of 
    % Wigner's Semicircle Law
    probs = WignersProb(x / sqrt(N));

    %%%%%%%%%%%% PLOTTING SECTION %%%%%%%%%%%%%%%%%%%%%%%%%%%
    c = figure;
    hold on;
    histogram(D, 100, 'Normalization', 'pdf')
    plot(x(1:(end - 1)), probs', "r", "LineWidth", 5)
    legend("Probability Distribution of Eigenvalues", "Wigner's Semicircle Law")
    xlabel("Eigenvalues")
    ylabel("Probabilities")
    title(['Probability Distributions for N=' num2str(N)])
    hold off;
