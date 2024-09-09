%% CONSTANTS %%
K_p = 0.23;
tau_v = 0.33;
omega_nms = 10.40;
zeta_nms = 0.24;
T_l_e = 1.76;
K_f = 1.07;
tau_f = 1.27;
T_l_f = 0.95;
K_v = K_p * T_l_e;
tau_s = 2;
tau_f_star = tau_s - tau_f;
Kn_remnant = 0.03;
Tn_remnant = 0.23;
dt = 0.01;
t_total = 120;
fs=100;

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
input_signal = csvread('f_star_matlab.csv');

% Define simulation time
t = 0:dt:t_total-dt;

% Simulate the system response
[output_signal, t_out, ~] = lsim(H_closed_loop, input_signal, t);

% Compute the error signal (difference between input and output)
error_signal = input_signal - output_signal;

% python's u_star
u_star = csvread('u_star.csv');

%% load van der el's signals
input_signal = csvread('input_signal_ahead.csv');
response_python = csvread('u_star.csv');
output_python = csvread('u_star_matlab.csv');


%% Plot the results
figure;
subplot(2,1,1);
hold on;
plot(t_out, output_signal, 'DisplayName', 'Matlab predicted u star response');
plot(t_out, u_star, 'DisplayName', 'Python predicted u star response');
hold off;
title('Output Signal');
xlabel('Time (s)');
ylabel('Output');
grid on;
legend show;

subplot(2,1,2);
plot(t_out, output_signal - u_star);
title('Difference');
xlabel('Time (s)');
ylabel('Error');
grid on;


%% PERFORM FOURIER ANALYSIS TO CHECK THE RESULTS
N = length(f_star);  % Number of samples in the signal

% Perform FFT
F = fft(u_star);

% Compute the two-sided spectrum
F_mag = abs(F/N);  % Magnitude of FFT
F_mag = F_mag(1:floor(N/2)+1);  % Take the first half of the FFT (positive frequencies)
F_mag(2:end-1) = 2*F_mag(2:end-1);  % Double the magnitude (except for the DC and Nyquist components)

% Generate the frequency axis
frequencies = fs*(0:(N/2))/N;

% Plot the FFT
figure;
loglog(frequencies, F_mag);
title('Magnitude Spectrum of f\_star');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;