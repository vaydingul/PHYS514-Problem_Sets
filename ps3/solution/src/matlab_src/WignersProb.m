function prob = WignersProb(x)
    %WignersProb
    %
    % Syntax: prob = WignersProb(x)
    %
    % Long description
    %   This function calculates the probobility of the existence of an eigenvalue
    %   for a given interval using Wigner's Semicircle Law.
    %
    %   Inputs:
    %   x = Input vector whose probabilities will be determined
    %
    %   Outputs:
    %   probs = Output probability vector

    len = length(x); % Get size of input vector

    for k = 1:(len - 1) % Since the probability will be calculated interval-wise,
        % We need to truncate the loop before last element

        x1 = x(k); % Beginning of interval
        x2 = x(k + 1); % End of interval

        prob(k) = WignersFunc(x2) - WignersFunc(x1); % Calculation of probability

    end

end
