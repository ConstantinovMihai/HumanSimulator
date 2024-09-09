close all
clc 
clear all
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

%% Define transfer functions
H_remnant = tf(Kn_remnant, [Tn_remnant, 1]);
H_nms = tf(omega_nms^2, [1, 2*zeta_nms*omega_nms, omega_nms^2]);

H_of = tf(K_f, [T_l_f, 1]);
H_ce = tf(5, [1, 0, 0]);
H_oe = tf([K_e * T_l_e, K_e], 1);

H_pade_f = tf(num_pade_f, den_pade_f);
H_pade_v = tf(num_pade_v, den_pade_v);

H_f_star = series(H_of, H_pade_f);

% Combine transfer functions for H_comb
H_oe_nms = series(H_oe, H_nms);
H_oe_nms_pade_v = series(H_oe_nms, H_pade_v);

% Multiply H_comb with H_ce to get overall system
H_open_loop = series(H_oe_nms_pade_v, H_ce);

% Create the closed-loop system (feedback)
H_closed_loop = feedback(H_open_loop, 1);

%% load van der el's signals
load('Data/Raw_data/11_PreviewFofuExp_ch6/expdata.mat')
input_signal = ed.PRM.ft(:,1,1);
input_signal = [input_signal(discard_samples+1:end); input_signal(1:discard_samples)];
real_response=ed.PRM.x(:,1,1);
fd_signal = ed.PRM.fd(:,1,1);
u_real = ed.PRM.u(:,1,1);

u_python = csvread('u_van_der_el.csv');

response_python = csvread('u_star.csv');

%% remnant realisation
% define simulation time
t = 0:dt:t_total-dt;

len = size(u_real, 1);
remnant_realization = randn(len,1);
remnant_filtered = lsim(H_remnant, remnant_realization, t);

%% Simulate the system
[f_star, ~, ~] = lsim(H_f_star, input_signal, t);
[u_star_lsim, t_out, ~] = lsim(H_closed_loop, f_star, t);

du_star_lsim_dt = diff(u_star_lsim)/dt;
u_lsim = diff(du_star_lsim_dt) / dt / 5;

input_signal = ed.PRM.ft(:,1,1);

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
u = zeros(len, 1);               % Pre-allocate u output signal
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
    u(t) = y(1);
    u_star(t) = y(2);
    
    % Compute error signal
    e = f_star(t) - u_star(t); 
    error_signal(t) = e;
end

% Final output signals
u_cascaded = u;
u_star_cascaded = u_star;

%% plot the signals
t = 0:dt:t_total-dt;
figure;
subplot(2,1,1);
hold on;
plot(t, u_star_cascaded, 'DisplayName', 'Simulated u star');
plot(t, u_star_lsim, 'DisplayName', 'Lsim u star')
hold off;
title('Measured Response');
xlabel('Time (s)');
ylabel('Response');
legend show;
grid on;

% Simulated response
subplot(2,1,2);
hold on;
plot(t, u_cascaded, 'DisplayName', 'Cascaded u');
%% WHERE iS THIS 5 COMING FROM??
plot(t(1:end-2), u_lsim, 'DisplayName', 'Lsim u') 
title('Simulated Response');
xlabel('Time (s)');
ylabel('Response');
hold off;
legend show;
grid on;