
fs = 100; 
f_star = csvread('f_star_matlab.csv')

f_star = f_star(:); 
N = length(f_star); 

% Perform FFT
F = fft(f_star);

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

%% CONSTANTS %%
% Define given constants
K_p = 0.23;
tau_v = 0.33;
omega_nms = 10.40;
zeta_nms = 0.24;
T_l_e = 1.76;
K_f = 1.07;
tau_f = 1.27;
T_l_f = 0.95;

K_v = K_p * T_l_e;

tau_s = 1.5;

tau_f_star = tau_s - tau_f;
Kn_remnant = 0.03;
Tn_remnant = 0.23;

dt = 0.01;
t_total = 120;

discard_samples = round(tau_s / dt);

% Define Pade approximations for time delays
numerator_pade = [-1, 12, -60, 120];
denominator_pade = [1, 12, 60, 120];

% Calculate the numerator and denominator for H_pade_v
num_pade_f = numerator_pade .* tau_f_star .^ (3:-1:0);
den_pade_f = denominator_pade .* tau_f_star .^ (3:-1:0);

% Calculate the numerator and denominator for H_pade_v
num_pade_v = numerator_pade .* tau_v .^ (3:-1:0);
den_pade_v = denominator_pade .* tau_v .^ (3:-1:0);

% Define transfer functions
H_remnant = tf(Kn_remnant, [Tn_remnant, 1]);
H_nms = tf(omega_nms^2, [1, 2*zeta_nms*omega_nms, omega_nms^2]);

H_of = tf(K_f, [T_l_f, 1]);
H_ce = tf(5, [1, 0, 0]);
H_vp = tf([K_v, K_p], 1);

H_pade_f = tf(num_pade_f, den_pade_f);
H_pade_v = tf(num_pade_v, den_pade_v);

H_f_star = series(H_of, H_pade_f);

% Combine transfer functions for H_comb
H_comb = series(H_vp, H_nms);
H_comb = series(H_comb, H_pade_v);

% Multiply H_comb with H_ce to get overall system
H_total = series(H_comb, H_ce);

% Create the closed-loop system (feedback)
H_closed_loop = feedback(H_total, 1);

%% load van der el's signals
idx = 1;

% input_signal = csvread('Data/Clean_CSV_data/van_der_El_CSV_data/PRM/ft.csv');
% input_signal = input_signal(:,idx);
% 
% input_signal = [input_signal(discard_samples+1:end); input_signal(1:discard_samples)];
% 
% real_response = csvread('Data/Clean_CSV_data/van_der_El_CSV_data/PRM/x.csv');
% real_response = real_response(:,idx);
% 
% u_real = csvread('Data/Clean_CSV_data/van_der_El_CSV_data/PRM/u.csv');
% u_real = u_real(:,idx);

load('Data/Raw_data/11_PreviewFofuExp_ch6/expdata.mat')
input_signal = ed.PRM.ft(:,1,1);
input_signal = [input_signal(discard_samples+1:end); input_signal(1:discard_samples)];
real_response=ed.PRM.x(:,1,1);
u_real = ed.PRM.u(:,1,1);
fd_signal = ed.PRM.fd(:,1,1);
u_python = csvread('u_van_der_el.csv');

response_python = csvread('u_star.csv');

%% Simulate the system
% simulate system response
[f_star, ~, ~] = lsim(H_f_star, input_signal, t);

N = length(f_star); 

% Perform FFT
F = fft(f_star);

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