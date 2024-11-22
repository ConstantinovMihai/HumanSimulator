import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

input_signal = np.loadtxt(r'HumanSimulator/Simulated_signals/input_signal.csv', delimiter=',')
error_signal = np.loadtxt(r'HumanSimulator/Simulated_signals/error_signal.csv', delimiter=',')
f_star = np.loadtxt(r'HumanSimulator/Simulated_signals/f_star.csv', delimiter=',')
distraction_times = np.loadtxt(r'HumanSimulator/Simulated_signals/distraction_times.csv', delimiter=',').astype(int)
tc = np.loadtxt(r'HumanSimulator/Simulated_signals/tc.csv', delimiter=',')
u_star = np.loadtxt(r'HumanSimulator/Simulated_signals/u_star.csv', delimiter=',')

plt.plot(input_signal, "r", label="original signal")
for distraction_time in distraction_times:
    plt.axvline(distraction_time, color='k', linestyle='--', alpha=0.35)
    

for i in range(0, len(distraction_times)-1, 2):
    plt.axvspan(distraction_times[i], distraction_times[i+1], alpha=0.3, color='lightblue')
    

if len(distraction_times) % 2 == 1:
    plt.axvspan(distraction_times[i-1], len(f_star), alpha=0.3, color='lightblue')

plt.plot(u_star, label="u_star")

# create nice legend
blue_patch = mpatches.Patch(color='lightblue', alpha=0.3, label='distraction interval')
red_line = plt.Line2D([0], [0], color='red', label='original signal')
u_star_line = plt.Line2D([0], [0], color='blue', label='u_star')
plt.legend(handles=[blue_patch, red_line, u_star_line])

plt.xlabel("Time step")
plt.ylabel("Output signal")
plt.title("Response of the closed-loop system")
plt.grid()
plt.show()