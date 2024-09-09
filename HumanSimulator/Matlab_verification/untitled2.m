% Define Pade approximations for time delays
numerator_pade = [-1, 12, -60, 120];
denominator_pade = [1, 12, 60, 120];

% Calculate the numerator and denominator for H_pade_v
num_pade_v = numerator_pade .* tau_v .^ (3:-1:0);
den_pade_v = denominator_pade .* tau_v .^ (3:-1:0);

% Create the transfer function H_pade_v
H_pade_v = tf(num_pade_v, den_pade_v);

% Display the transfer function
disp(H_pade_v);
