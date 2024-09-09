
%% CONSTANTS %%
% Define given constants
K_p = 0.28;
tau_v = 0.34;
omega_nms = 7.87;
zeta_nms = 0.14;
T_l_e = 1.42;
K_f = 0.95;
tau_f = 1.15;
T_l_f = 1.10;
K_v = K_p * T_l_e;
tau_s = 2;
tau_f_star = tau_s - tau_f;
Kn_remnant = 0.03;
Tn_remnant = 0.23;
dt = 0.01;
t_total = 200;

% Define Pade approximations for time delays
numerator_pade = [-1, 12, -60, 120];
denominator_pade = [1, 12, 60, 120];

% H_pade_f = tf(numerator_pade * tau_f_star.^(3:-1:0), denominator_pade * tau_f_star.^(3:-1:0));
% H_pade_v = tf(numerator_pade * tau_v.^(3:-1:0), denominator_pade * tau_v.^(3:-1:0));

% Define Pade approximations for time delays
numerator_pade = [-1, 12, -60, 120];
denominator_pade = [1, 12, 60, 120];

% Calculate the numerator and denominator for H_pade_v
num_pade_v = numerator_pade .* tau_v .^ (3:-1:0);
den_pade_v = denominator_pade .* tau_v .^ (3:-1:0);

% Create the transfer function H_pade_v
H_pade_v = tf(num_pade_v, den_pade_v);

% Define transfer functions
H_remnant = tf(Kn_remnant, [Tn_remnant, 1]);
H_nms = tf(omega_nms^2, [1, 2*zeta_nms*omega_nms, omega_nms^2]);

H_of = tf(K_f, [T_l_f, 1]);
H_ce = tf(5, [1, 0, 0]);
H_vp = tf([K_v, K_p], 1);

% Combine transfer functions for H_comb
H_comb = series(H_vp, H_nms);
H_comb = series(H_comb, H_pade_v);
% H_comb = series(H_comb, H_pade_v);

% Multiply H_comb with H_ce to get overall system
H_total = series(H_comb, H_ce)

% Create the closed-loop system (feedback)
H_closed_loop = feedback(H_total, 1);

% Define input signal (step input)
input_signal = csvread('f_star.csv');

% Define simulation time
t = 0:dt:t_total-dt;


% Simulate the system response
[output_signal, t_out, ~] = lsim(H_closed_loop, input_signal, t);

% Compute the error signal (difference between input and output)
error_signal = input_signal - output_signal;


% python's u_star
u_star = csvread('u_star.csv');

% cut the signals in half
u_star = u_star(5000:end);
output_signal = output_signal(5000:end);

% Plot the results
figure;
subplot(2,1,1);
hold on;
plot(t_out(5000:end), output_signal);
plot(t_out(5000:end), u_star);
hold off;
title('Output Signal');
xlabel('Time (s)');
ylabel('Output');

subplot(2,1,2);
plot(t(5000:end), output_signal-u_star);
title('Difference');
xlabel('Time (s)');
ylabel('Error');
