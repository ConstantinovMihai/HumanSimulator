v_python = csvread('v.csv');;
input_signal_test_python = csvread('input_signal_test.csv');

figure
hold on;
plot(v_python)
plot(u_sim)
hold off;
legend show;
grid on;

    
V = fft(v_python);
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

%% FFT plots
% Compute Frequency Axis
Fs = 1 / dt;                % Sampling frequency
f = (1:N/2) * (Fs / N);     % Frequency axis

% Plot the Frequency Components
figure;
loglog(f, abs(V*2/N));
title('Frequency Components of V');
xlabel('Frequency (Hz)');
ylabel('Magnitude |V(f)|');
grid on;