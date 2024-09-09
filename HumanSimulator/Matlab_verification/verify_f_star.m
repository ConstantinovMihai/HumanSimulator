%% CONSTANTS %%
% Define given constants
K_e = 0.23;
tau_v = 0.33;
omega_nms = 10.40;
zeta_nms = 0.24;
T_l_e = 1.76;
K_f = 1.07;
tau_f = 1.27;
T_l_f = 0.95;
K_v = K_e * T_l_e;
tau_s = 2;

tau_f_star = tau_s - tau_f;
Kn_remnant = 0.03;
Tn_remnant = 0.23;

dt = 0.01;
t_total = 120;

discard_samples = round(tau_s / dt);
fs = 100;

% Define Pade approximations for time delays
numerator_pade = [-1, 12, -60, 120];
denominator_pade = [1, 12, 60, 120];

% Calculate the numerator and denominator for H_pade_v
num_pade_f = numerator_pade .* tau_f_star .^ (3:-1:0);
den_pade_f = denominator_pade .* tau_f_star .^ (3:-1:0);

% Calculate the numerator and denominator for H_pade_v
num_pade_v = numerator_pade .* tau_v .^ (3:-1:0);
den_pade_v = denominator_pade .* tau_v .^ (3:-1:0);

%% Define transfer functions
H_remnant = tf(Kn_remnant, [Tn_remnant, 1]);
H_nms = tf(omega_nms^2, [1, 2*zeta_nms*omega_nms, omega_nms^2]);

H_of = tf(K_f, [T_l_f, 1]);
H_ce = tf(5, [1, 0, 0]);
H_oe = tf([K_e * T_l_e, K_e], 1);

H_pade_f = tf(num_pade_f, den_pade_f);
H_pade_v = tf(num_pade_v, den_pade_v);

H_f_star = series(H_of, H_pade_f);


[num, den] = tfdata(H_of, 'v');  % 'v' returns the coefficients as vectors

% Normalize by the first coefficient of the denominator
den_normalized = den / den(1);
num_normalized = num / den(1);

% Create the new normalized transfer function
H_f_star_normalized = tf(num_normalized, den_normalized);

% Display the normalized transfer function
disp(H_f_star_normalized)

% Combine transfer functions for H_comb
H_oe_nms = series(H_oe, H_nms);
H_oe_nms_pade_v = series(H_oe_nms, H_pade_v);

% Multiply H_comb with H_ce to get overall system
H_open_loop = series(H_oe_nms_pade_v, H_ce);

% Create the closed-loop system (feedback)
H_closed_loop = feedback(H_open_loop, 1);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('Data/Raw_data/11_PreviewFofuExp_ch6/expdata.mat')
input_signal = ed.PRM.ft(:,1,1);
input_signal = [input_signal(discard_samples+1:end); input_signal(1:discard_samples)];

f_star = csvread('f_star_matlab.csv');

% define simulation time
t = 0:dt:t_total-dt;

% simulate system response
[output_signal, t_out, ~] = lsim(H_f_star, input_signal, t);

figure;
subplot(2,1,1);
hold on;
plot(t_out, output_signal, 'DisplayName', 'Matlab f star');
plot(t, f_star, 'DisplayName', 'Python f star');
hold off;
title('Output Signal');
xlabel('Time (s)');
ylabel('Output');
grid on;
legend show;

subplot(2,1,2);
hold on;
plot(t, output_signal-f_star, 'DisplayName', 'Difference in response');
hold off;
title('Difference');
xlabel('Time (s)');
ylabel('Error');
grid on;
legend show;

%% PERFORM FOURIER ANALYSIS TO CHECK THE RESULTS
f_star = f_star(:);  % Ensure the signal is a column vector

N = length(f_star) / 2;  % Number of samples in the signal

% Perform FFT
F = fft(output_signal);

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


%% let's try again
% Ensure the signal is a column vector
f_star = f_star(:);

% Extract the second half of the signal
second_half_signal = f_star(length(f_star)/2 + 1:end);

% Update the number of samples in the second half
N = length(second_half_signal);

% Perform FFT on the second half of the signal
F = fft(second_half_signal);

% Compute the two-sided spectrum
F_mag = abs(F/N);  % Magnitude of FFT
F_mag = F_mag(1:floor(N/2)+1);  % Take the first half of the FFT (positive frequencies)
F_mag(2:end-1) = 2*F_mag(2:end-1);  % Double the magnitude (except for the DC and Nyquist components)

% Generate the frequency axis
frequencies = fs*(0:(N/2))/N;

% Plot the FFT
figure;
loglog(frequencies, F_mag);
title('Magnitude Spectrum of Second Half of f\_star');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;