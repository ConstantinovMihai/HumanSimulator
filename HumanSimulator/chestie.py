import numpy as np

len_signal = 100
nb_distractions = 4
distraction_times = np.sort(np.random.randint(0, len_signal, nb_distractions))

tc = np.zeros(len_signal)

# place the distractions in the signal (1 for distraction, 0 for no distraction)
# from the even index to the odd signal
for i in range(0, nb_distractions, 2):
    print(f"i {i}")
    tc[distraction_times[i]:distraction_times[i+1]] = 1

print(f"distraction times {distraction_times}")
print(tc)