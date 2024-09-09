"""
Simulate the signal presented to the ideal human operator model
"""

import systems_functions as sf
import numpy as np
import matplotlib.pyplot as plt
import constants
from input_signal import plot_time_frequency_signal, get_fft_values


def simulate_f_star(input_signal, ct):
    """
    Simulate the signal presented to the ideal human operator model.

    Parameters:
    input_signal (array_like): Input signal.
    sim_consts (constants.SimulationConstants): Object containing simulation constants.

    Returns:
    f_star (array_like): Processed signal.
    """
    # move the signal to the left (to take into account the look ahead)
    # deal with the edge case tau_s = 0 
    if ct.tau_s > 0:
        input_signal_ahead = np.roll(input_signal, -ct.discard_samples)
    else:
        input_signal_ahead = input_signal

    #np.savetxt("input_signal_ahead.csv", input_signal_ahead, delimiter=",")
    
    # add the pade delay
    H = sf.multiply_transfer_functions(ct.H_of, ct.H_pade_f)

    # Convert transfer function to state-space
    A, B, C, D = sf.convert_tf_ss_ccf(H)

    # Discretize state-space system
    Ad, Bd, Cd, Dd = sf.cont2discrete_zoh(ct.dt, A, B, C, D)

    # Simulate the discrete system
    f_star, _ = sf.simulate_discrete_system(Ad, Bd, Cd, Dd, input_signal_ahead)
 
    return f_star.flatten().reshape(-1, 1)


def test_f_star_signal(ct, input_signal, f_star):
    """
    Test the f_star signal.
    """
   
    # calculate the FFT of the signals
    frequency_values, fft_values = get_fft_values(input_signal, ct.fs, len(input_signal))

    # plot the time and frequency domain of the signal
    plot_time_frequency_signal(np.arange(0, len(input_signal)*ct.dt, ct.dt), input_signal, frequency_values, fft_values, name="input signal")

    print(f"frequency with most energy in input signal {frequency_values[np.argmax(fft_values)]} Hz")

    # get rid of the transient response
    # f_star = sf.remove_transient_behaviour(f_star, ct)
    frequency_values, fft_values = get_fft_values(f_star, ct.fs, len(f_star))

    # plot the time and frequency domain of the signal
    plot_time_frequency_signal(np.arange(0, len(f_star) * ct.dt, ct.dt), f_star, frequency_values, fft_values, name="processed signal")
    
    print(f"frequency with most energy in f_star signal {frequency_values[np.argmax(fft_values)]} Hz")
    print(f"first few frequencies values {frequency_values[:10]}")

    # plot the response
    plt.plot(input_signal, "r", label="original signal")
    plt.plot(f_star, label="processed signal")
    plt.legend()
    plt.grid()
    plt.xlabel('Time step')
    plt.ylabel('Output signal')
    plt.title('Response of the discrete-time system')
    plt.show()


if __name__ == "__main__":
    # get the simulation constants
    ct = constants.SimulationConstantsVanderEl()
    
    # # import the input signal from input_signal.csv
    input_signal = np.loadtxt("input_signal.csv", delimiter=",").reshape(-1, 1)
    #input_frequencies = np.loadtxt("frequencies.csv", delimiter=",").astype(int)
    
    # input_signals = sf.read_undistracted_data(f'Data/Clean_CSV_data/van_der_El_CSV_data/PRM/ft.csv')
    # input_signal = input_signals[0]
    # np.savetxt("input_signal.csv", input_signal, delimiter=",") 
    input_signal = sf.read_undistracted_data(f'Data/Clean_CSV_data/van_der_El_CSV_data/PRM/ft.csv')[0]
    
    # simulate the signal presented to the ideal human operator model
    f_star = simulate_f_star(input_signal, ct)

    #np.savetxt("f_star_matlab.csv", f_star, delimiter=",")
    test_f_star_signal(ct, input_signal, f_star)
