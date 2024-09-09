import numpy as np
import matplotlib.pyplot as plt
from systems_functions import *
from constants import *

ct = SimulationConstants()
f_star = np.loadtxt("f_star.csv", delimiter=",")
f_star = remove_transient_behaviour(f_star, ct)

processed_signal = np.loadtxt("processed_signal.csv", delimiter=",")

# Example lengths (replace with your actual lengths)
len_processed_signal = len(processed_signal)
len_f_star = len(f_star)
discard_samples = ct.discard_samples

# Calculate lengths
zeros_before_length = len_processed_signal - len_f_star - discard_samples
zeros_after_length = discard_samples

# Create arrays with appropriate shapes
zeros_before = np.zeros((zeros_before_length,))  # Shape (zeros_before_length,)
zeros_after = np.zeros((zeros_after_length,))    # Shape (zeros_after_length,)

# Concatenate arrays
concatenated_array = np.concatenate((zeros_before, f_star.flatten(), zeros_after))


processed_signal = np.loadtxt("processed_signal.csv", delimiter=",")

#plt.plot(np.concatenate((np.zeros((len(processed_signal) - len(f_star) - ct.discard_samples, 1)), f_star, np.zeros((ct.discard_samples, 1)))), label="f_star")
plt.plot(concatenated_array, label="f_star")
plt.plot(processed_signal, label="processed signal")
plt.grid()
plt.legend()
plt.show()

print(f"Rms difference {np.sqrt(np.mean((concatenated_array - processed_signal)**2))}")
