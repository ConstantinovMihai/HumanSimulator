close all
clc 
clear all

load('Data/Raw_data/expdata_SI_fofuexp.mat')

%% CONSTANTS %%
exp_idx = 8; % this correspond to PRM
person_idx = 9;
run_idx = 5;

% Define given constants
K_e = ed.parest.Ke(person_idx,exp_idx)
tau_v = ed.parest.tv(person_idx,exp_idx)
omega_nms = ed.parest.wnm(person_idx,exp_idx)
zeta_nms = ed.parest.dnm(person_idx,exp_idx)
K_f = ed.parest.Kf(person_idx,exp_idx)
tau_f = ed.parest.tpf(person_idx, exp_idx)
T_l_f = ed.parest.Tlf(person_idx,exp_idx)
K_v = 0

% for comparison, the values used to verify david li
% K_e = 1.28;
% tau_v = 0.18;
% omega_nms = 10.72;
% zeta_nms = 0.19;
% T_l_e = 0;
% K_f = 1.01;
% tau_f = 0.54;
% w_b = 14.09;
% K_v = 0;

tau_s = 1.5;
tau_f_star = tau_s - tau_f
Kn_remnant = 0;
Tn_remnant = 1;
dt = 0.01;
t_total = 120;

discard_samples = round(tau_s / dt)

% Define Pade approximations for time delays
numerator_pade = [-1, 12, -60, 120];
denominator_pade = [1, 12, 60, 120];

% Calculate the numerator and denominator for H_pade_v
num_pade_f = numerator_pade .* tau_f .^ (3:-1:0);
den_pade_f = denominator_pade .* tau_f .^ (3:-1:0);

% Calculate the numerator and denominator for H_pade_v
num_pade_f_star = numerator_pade .* tau_f_star .^ (3:-1:0);
den_pade_f_star = denominator_pade .* tau_f_star .^ (3:-1:0);

% Calculate the numerator and denominator for H_pade_v
num_pade_v = numerator_pade .* tau_v .^ (3:-1:0);
den_pade_v = denominator_pade .* tau_v .^ (3:-1:0);

%% Define transfer functions
H_remnant = tf(Kn_remnant, [Tn_remnant, 1]);
H_nms = tf(omega_nms^2, [1, 2*zeta_nms*omega_nms, omega_nms^2]);

H_of = tf(K_f, [T_l_f, 1]);
H_ce = tf(1.5, [1, 0]);
H_oe = tf([K_v, K_e], 1);

H_pade_f_star = tf(num_pade_f_star, den_pade_f_star);
H_pade_v = tf(num_pade_v, den_pade_v);
H_pade_f = tf(num_pade_f, den_pade_f);

H_f_star = series(H_of, H_pade_f_star);

% Combine transfer functions for H_comb
H_oe_nms = series(H_oe, H_nms);
H_oe_nms_pade_v = series(H_oe_nms, H_pade_v);

% Multiply H_comb with H_ce to get overall system
H_open_loop = series(H_oe_nms_pade_v, H_ce);

% Create the closed-loop system (feedback)
H_closed_loop = feedback(H_open_loop, 1);
% 
% [num, den] = tfdata(H_f_star, 'v');
% H_normalized = tf(num / den(1), den / den(1))
% H_pade_f_star
% H_of

%% load signals
input_signal = ed.PRM.ft(:,run_idx,person_idx);
input_signal = [input_signal(discard_samples+1:end); input_signal(1:discard_samples)];
real_response=ed.PRM.x(:,run_idx,person_idx);
u_real = ed.PRM.u(:,run_idx,person_idx);
fd_signal = ed.PRM.fd(:,run_idx,person_idx);

%% remnant realisation
% define simulation time
t = 0:dt:t_total-dt;

len = size(u_real, 1);
remnant_realization = randn(len,1)/sqrt(dt);
remant_filtered = lsim(H_remnant, remnant_realization, t);

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
u_sim = zeros(len, 1);               % Pre-allocate u output signal
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
    u_star(t) = y(2) + fd_signal(t);
    
    % Compute error signal
    e = f_star(t) - (u_star(t) + remant_filtered(t) ); 
    error_signal(t) = e;
end

%% check using the lsim method
t = 0:dt:t_total-dt;
[f_star, ~, ~] = lsim(H_f_star, input_signal, t);
f_star_python = csvread('f_star_sasha.csv');

[u_star_lsim, t_out, ~] = lsim(H_closed_loop, f_star_python, t);

du_star_lsim_dt = diff(u_star_lsim)/dt;
% u_sim = diff(du_star_lsim_dt)/dt / 5;
% for plotting purposes

%% PLOTTING ROUTINES
input_signal = ed.PRM.ft(:,run_idx,person_idx);

t = 0:dt:t_total-dt;
figure;
subplot(3,1,1);
hold on;
plot(t, u_star, 'DisplayName', 'Simulated (mine) u star');
plot(t, input_signal, 'DisplayName', 'Input signal')
plot(t, real_response, 'DisplayName', 'Sasha u star')
hold off;
title('U star (x) signal. Sasha vs mine');
xlabel('Time (s)');
ylabel('Response');
legend show;
grid on;

subplot(3,1,2);
hold on;
plot(t, u_sim, 'DisplayName', 'Simulated (mine) U');
plot(t, u_real, 'DisplayName', "Sasha U");
title('U signal. Sasha vs mine');
xlabel('Time (s)');
ylabel('Response');
hold off;
legend show;
grid on;

u_sim_python = csvread('u_sasha.csv');
input_signal_python = csvread('input_signal_sasha.csv');
f_star_python = csvread('f_star_sasha.csv');
u_star_python = csvread('u_star_sasha.csv');

subplot(3,1,3);
hold on;
plot(t, u_star, 'DisplayName', 'Simulated (mine) U');
% plot(t, u_real, 'DisplayName', "Sasha U");
plot(t, u_star_python, 'DisplayName', "Python (mine) U");
title('U signal. Sasha vs mine');
xlabel('Time (s)');
ylabel('Response');
hold off;
legend show;
grid on;

diff_signals = norm(f_star - f_star_python)

%% VAF calculations
vaf_real_out=VAF(real_response, output_signal)
vaf_out_input=VAF(output_signal, input_signal)
vaf_u_ureal = VAF(u_real(1000:end), u_sim(1000:end))
disp('Done!');

%% gain margin
% calculate Fourier transforms of signals
fft_ft = fft(input_signal);
fft_fd = fft(fd_signal);
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
fft_fd = fft_fd(2:N/2+1);
fft_e  = fft_e(2:N/2+1);
fft_u  = fft_u(2:N/2+1);
fft_x  = fft_x(2:N/2+1);
 
% calculate the corresponding frequency vector
w = w0*[1:1:length(fft_ft)].';
 
% calculate periodograms from ffts
Sftft = abs(1/N*conj(fft_ft).*fft_ft);
Sfdfd = abs(1/N*conj(fft_fd).*fft_fd);
See   = abs(1/N*conj(fft_e).*fft_e);
Suu   = abs(1/N*conj(fft_u).*fft_u);
Sxx   = abs(1/N*conj(fft_x).*fft_x);
 
% calculate forcing function frequencies
n_ft = find(Sftft > 1);
n_fd = find(Sfdfd > 1e-6);

% calculate pilot response
Hp = fft_x(n_ft)./fft_e(n_ft);

figure()
loglog(w(n_ft), abs(Hp))
title('Magnitude Spectrum of u');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

%% FOURIER ANALYSIS
N = length(u_sim); 

% Perform FFT
F_usim = fft(u_sim);
fs = 100;

% Compute the two-sided spectrum
F_mag_usim = abs(F_usim/N);  % Magnitude of FFT
F_mag_usim = F_mag_usim(1:floor(N/2)+1);  % Take the first half of the FFT (positive frequencies)
F_mag_usim(2:end-1) = 2*F_mag_usim(2:end-1);  % Double the magnitude (except for the DC and Nyquist components)

% Generate the frequency axis
frequencies = fs*(0:(N/2))/N;

% Plot the FFT
figure;
loglog(frequencies, F_mag_usim);
hold on
loglog(frequencies(n_ft+1), F_mag_usim(n_ft+1), 'or');
loglog(frequencies(n_fd+1), F_mag_usim(n_fd+1), 'og');
title('Magnitude Spectrum of u (from simulation)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

N = length(u_real); 
% Perform FFT
F_ureal = fft(u_real);

% Compute the two-sided spectrum
F_mag_ureal = abs(F_ureal/N);  % Magnitude of FFT
F_mag_ureal = F_mag_ureal(1:floor(N/2)+1);  % Take the first half of the FFT (positive frequencies)
F_mag_ureal(2:end-1) = 2*F_mag_ureal(2:end-1);  % Double the magnitude (except for the DC and Nyquist components)

% Generate the frequency axis
frequencies_ustar = fs*(0:(N/2))/N;

% Plot the FFT
figure;
loglog(frequencies_ustar, F_mag_ureal);
hold on
loglog(frequencies(n_ft+1), F_mag_ureal(n_ft+1), 'or');
loglog(frequencies(n_fd+1), F_mag_ureal(n_fd+1), 'og');
title('Magnitude Spectrum of u (measured from experiment)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

%% Pn calculations 
% calculate fft of signal               
V = fft(u_sim);
V = V(2:N/2+1,:);

% calculate variance at target and disturbance primes
var_v_fd = sum((abs(V(n_fd))/(N/2)).^2)/2;
absasdas = V(n_fd)

var_v_ft = sum((abs(V(n_ft))/(N/2)).^2)/2;

dw = 2 * pi / (N * dt);
% calculate total variance
var_v = sum(dw.*(V(1:end-1).*conj(V(1:end-1)))/N*dt)/pi;

% calculate remaining variance
var_v_n = var_v - var_v_fd - var_v_ft;

% calculate noise power metric
Pn = var_v_n/var_v

%% Generate the Bode plot for eq 4.13-4.14 in van der el's thesis
H_ox = H_oe * H_nms * H_pade_v;
H_ot = H_of * H_oe / H_pade_f * H_nms * H_pade_v;

H_ol_t = H_ot * H_ce / (1 + (H_ox-H_ot)*H_ce);

figure();
bode(H_ox)

figure();
bode(H_ot)

figure();
bode(H_ol_t)
title('H ol t')

figure();
[mag, phase, wout] = bode(H_ol_t); 
mag = squeeze(mag); 
semilogx(wout, 20*log10(mag), 'b-');
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