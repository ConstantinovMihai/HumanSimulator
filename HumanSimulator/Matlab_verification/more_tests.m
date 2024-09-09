% Parameters for the Signal
Fs = 1000;              % Sampling frequency (Hz)
T = 1/Fs;               % Sampling period (seconds)
L = 1000;               % Length of signal (number of samples)
t = (0:L-1)*T;          % Time vector (seconds)

% Define the Frequencies of the Two Sine Waves
f1 = 50;                % Frequency of the first sine wave (Hz)
f2 = 120;               % Frequency of the second sine wave (Hz)

% Create the Signal Composed of Two Sine Waves
signal = 0.7 * sin(2 * pi * f1 * t) + sin(2 * pi * f2 * t);

% Add Some Noise to the Signal
signal = signal + 2 * randn(size(t));

% Compute the FFT of the Signal
Y = fft(signal);

% Compute the Two-Sided Spectrum P2. Then Compute the Single-Sided Spectrum P1
P2 = abs(Y / L);           % Two-sided spectrum
P1 = P2(1:L/2+1);          % Single-sided spectrum
P1(2:end-1) = 2*P1(2:end-1); % Double the non-DC components

% Define the Frequency Axis for the Plot
f = Fs * (0:(L/2)) / L;

% Plot the Single-Sided Amplitude Spectrum
figure;
plot(f, P1);
title('Single-Sided Amplitude Spectrum of Signal');
xlabel('Frequency (Hz)');
ylabel('|P1(f)|');