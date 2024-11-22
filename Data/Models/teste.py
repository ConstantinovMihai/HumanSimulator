import re
import numpy as np
import matplotlib.pyplot as plt

fdist = np.loadtxt('/home/mihai/Thesis/Data/Clean_CSV_data/updated_data/PRDPE/fdist.csv', delimiter=',')
mdist = np.loadtxt('/home/mihai/Thesis/Data/Clean_CSV_data/updated_data/PRDPE/mdist.csv', delimiter=',')

plt.plot(fdist[:,19], label='fdist')
plt.plot(mdist[:,19], label='mdist')
plt.legend()
plt.grid()
plt.show()