close all
clc 
clear all
%% CONSTANTS %%
% Define given constants
K_e = 1.28;
tau_v = 0.18;
omega_nms = 10.72;
zeta_nms = 0.19;
K_f = 1.01;
tau_f = 0.54;
w_b = 14.09;
K_v = 0;
tau_s = 0.9;
tau_f_star = tau_s - tau_f;

Kn_remnant = 0.03;
Tn_remnant = 1;

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
num_pade_f_star = numerator_pade .* tau_f_star .^ (3:-1:0);
den_pade_f_star = denominator_pade .* tau_f_star .^ (3:-1:0);

% Calculate the numerator and denominator for H_pade_v
num_pade_v = numerator_pade .* tau_v .^ (3:-1:0);
den_pade_v = denominator_pade .* tau_v .^ (3:-1:0);

%% Define transfer functions
H_remnant = tf(Kn_remnant, [Tn_remnant, 1]);
H_nms = tf(omega_nms^2, [1, 2*zeta_nms*omega_nms, omega_nms^2]);

H_of = tf(K_f*w_b, [1, w_b]);
H_ce = tf(1.5, [1, 0]);
H_oe = tf([K_v, K_e], 1);

H_pade_f_star = tf(num_pade_f_star, den_pade_f_star);
H_pade_f = tf(num_pade_f, den_pade_f);
H_pade_v = tf(num_pade_v, den_pade_v);

H_f_star = series(H_of, H_pade_f_star);

% Combine transfer functions for H_comb
H_oe_nms = series(H_oe, H_nms);
H_oe_nms_pade_v = series(H_oe_nms, H_pade_v);

% Multiply H_comb with H_ce to get overall system
H_open_loop = series(H_oe_nms_pade_v, H_ce);

% Create the closed-loop system (feedback)
H_closed_loop = feedback(H_open_loop, 1);

[num, den] = tfdata(H_oe_nms_pade_v, 'v');
H_normalized = tf(num / den(1), den / den(1));
H_normalized
H_pade_v
%% load van der el's signals
run_idx = 1;
index = 1;

file_path_input_signal = sprintf("Data/Clean_CSV_data/David_Li_CSV_data/PreviewDistractionExpData_S%d/PRN/ft.csv", index);
file_path_u_star = sprintf("Data/Clean_CSV_data/David_Li_CSV_data/PreviewDistractionExpData_S%d/PRN/x.csv", index);
file_path_u_real = sprintf("Data/Clean_CSV_data/David_Li_CSV_data/PreviewDistractionExpData_S%d/PRN/u.csv", index);
file_path_fd_signal = sprintf("Data/Clean_CSV_data/David_Li_CSV_data/PreviewDistractionExpData_S%d/PRN/fd.csv", index);


% Read the CSV file
input_signal = csvread(file_path_input_signal);
u_star_real    = csvread(file_path_u_star);
u_real    = csvread(file_path_u_real);
fd_signal = csvread(file_path_fd_signal);

input_signal = input_signal(:,run_idx);
input_signal = [input_signal(discard_samples+1:end); input_signal(1:discard_samples)];
u_star_real = u_star_real(:,run_idx);
u_real = u_real(:,run_idx);
fd_signal = fd_signal(:,run_idx);

%% remnant realisation
% define simulation time
t = 0:dt:t_total-dt;

len = size(u_real, 1);
remnant_realization = randn(len,1);
remnant_filtered = lsim(H_remnant, remnant_realization, t);

%% Simulate the system
[f_star, ~, ~] = lsim(H_f_star, input_signal, t);

[output_signal, t_out, ~] = lsim(H_closed_loop, f_star, t);


%% DO the matrix combination magic
[A1, B1, C1, D1] = ssdata(ss(H_oe_nms_pade_v));
[A2, B2, C2, D2] = ssdata(ss(H_ce));

% Horizontal concatenation for A_comp_int
A_comp_int = [A1, zeros(size(A1, 1), size(A2, 2))];

% Vertical concatenation for A_comp
A_comp = [A_comp_int; [B2 * C1, A2]];

% Vertical concatenation for B_comp
B_comp = [B1; B2 * D1];

% Horizontal and vertical concatenation for C_comp
C_comp = [C1, zeros(size(C1, 1), size(C2, 2))];
C_comp = [C_comp; [D2 * C1, C2]];

% Vertical concatenation for D_comp
D_comp = [D1; D2 * D1];

% Extract the discretized state-space matrices
sys_continuous = ss(A_comp, B_comp, C_comp, D_comp);
sys_discrete = c2d(sys_continuous, dt);

[Ad_comp, Bd_comp, Cd_comp, Dd_comp] = ssdata(sys_discrete);

eigs=eig(Ad_comp) % abs of the max eigenvalue must be lower than 1, otherwise the system is unstable

%% Simulate the system
% Initialize variables
len = length(f_star);            % Length of the input signal f_star
u_sim = zeros(len, 1);           % Pre-allocate u output signal
u_star = zeros(len, 1);          % Pre-allocate u_star output signal
error_signal = zeros(len, 1);    % Pre-allocate error signal
x = zeros(size(Ad_comp, 1), 1);  % Initial state vector, assumed to be zeros
e = zeros(size(Bd_comp, 2), 1);  % Initial error signal, dimension matches Bd_comp

% Main loop
for t = 1:len-1
    % State-space equations
    x = Ad_comp * x + Bd_comp * e;
    y = Cd_comp * x + Dd_comp * e;
    
    % Extract u and u_star from y
    u_sim(t) = y(1);
    u_star(t) = y(2);
    
    % Compute error signal
    e = f_star(t) - (u_star(t) + remnant_filtered(t)*0 + fd_signal(t)); 
    error_signal(t) = e;
end

%% PLOTTING ROUTINES

% for plotting purposes
input_signal = csvread(file_path_input_signal);
input_signal = input_signal(:,run_idx);

t = 0:dt:t_total-dt;
figure;
subplot(3,1,1);
hold on;
plot(t, u_star, 'DisplayName', 'Simulated u star');
plot(t, output_signal, 'DisplayName', 'lsim u star signal')
plot(t, u_star_real, 'DisplayName', 'Real u star')
hold off;
title('Measured Response');
xlabel('Time (s)');
ylabel('Response');
legend show;
grid on;

subplot(3,1,2);
hold on;
plot(t(1:end), u_sim, 'DisplayName', 'Simulated U');
plot(t, u_real, 'DisplayName', "Real U");
title('Simulated Response');
xlabel('Time (s)');
ylabel('Response');
hold off;
legend show;
grid on;

subplot(3,1,3);
margin(H_open_loop);

vaf_real_out=VAF(u_star, output_signal)
vaf_out_input=VAF(output_signal, input_signal)
vaf_u_ureal = VAF(u_real(1000:end), u_sim(1000:end))
disp('Done!');



%% FOURIER ANALYSIS
N = length(u_sim); 

% Perform FFT
F = fft(u_sim);
fs = 100;

% Compute the two-sided spectrum
F_mag = abs(F/N);  % Magnitude of FFT
F_mag = F_mag(1:floor(N/2)+1);  % Take the first half of the FFT (positive frequencies)
F_mag(2:end-1) = 2*F_mag(2:end-1);  % Double the magnitude (except for the DC and Nyquist components)

% Generate the frequency axis
frequencies = fs*(0:(N/2))/N;

% Plot the FFT
figure;
loglog(frequencies, F_mag);
title('Magnitude Spectrum of u (from simulation)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

N = length(u_real); 
% Perform FFT
F_ustar = fft(u_real);
% Compute the two-sided spectrum
F_mag_ustar = abs(F_ustar/N);  % Magnitude of FFT
F_mag_ustar = F_mag_ustar(1:floor(N/2)+1);  % Take the first half of the FFT (positive frequencies)
F_mag_ustar(2:end-1) = 2*F_mag_ustar(2:end-1);  % Double the magnitude (except for the DC and Nyquist components)

% Generate the frequency axis
frequencies_ustar = fs*(0:(N/2))/N;

% Plot the FFT
figure;
loglog(frequencies_ustar, F_mag_ustar);
title('Magnitude Spectrum of u (measured from experiment)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

%% gain margin
% calculate Fourier transforms of signals
fft_ft = fft(input_signal);
fft_e  = fft(error_signal);
fft_u  = fft(u_sim);
fft_x  = fft(u_star);
 
% retrieve timestep and ground frequency from time vector
dt = t(2) - t(1);
w0 = 2*pi/t(end);
 
% calculate how many datapoints we have
N = length(t);
 
% only take the fft points we need 
% (Matlab calculates symmetric fft, so all data is in fft result twice!)
fft_ft = fft_ft(2:N/2+1);
fft_e  = fft_e(2:N/2+1);
fft_u  = fft_u(2:N/2+1);
fft_x  = fft_x(2:N/2+1);
 
% calculate the corresponding frequency vector
w = w0*[1:1:length(fft_ft)].';
 
% calculate periodograms from ffts
Sftft = abs(1/N*conj(fft_ft).*fft_ft);
See   = abs(1/N*conj(fft_e).*fft_e);
Suu   = abs(1/N*conj(fft_u).*fft_u);
Sxx   = abs(1/N*conj(fft_x).*fft_x);
 
% calculate forcing function frequencies
n_ft = find(Sftft > 1);

% calculate pilot response
Hp = fft_x(n_ft)./fft_e(n_ft);

figure()
loglog(frequencies_ustar(n_ft), abs(Hp))
title('Magnitude Spectrum of u');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

%% Generate the Bode plot for eq 4.13-4.14 in van der el's thesis
H_ox = H_oe * H_nms * H_pade_v;
H_ot = H_of * H_oe * H_pade_f * H_nms * H_pade_v;
H_ol_t = H_ot * H_ce / (1 + (H_ox-H_ot)*H_ce);

figure();
[mag, phase, wout] = bode(H_ol_t); 
mag = squeeze(mag); 
semilogx(wout, mag, 'b-');
hold on;

% Convert frequencies to rad/s for overlaying points on Bode plot
frequencies_rad_s = frequencies_ustar * 2 * pi;

semilogx(frequencies_rad_s(n_ft), 20*log10(abs(Hp)), 'ro');

title('Crossover Frequency');
xlabel('Frequency (rad/s)');
ylabel('Magnitude (dB)');
grid on;
hold off;


%% UTILITY FUNCTIONS
function vaf_value = VAF(y, y_hat)
    % Compute the variance of the difference
    y = y(:);
    y_hat = y_hat(:);
    var_y_diff = var(y - y_hat);

    % Compute the variance of the original signal
    var_y = var(y);

    % Compute the VAF
    vaf_value = 1 - var_y_diff / var_y;
end