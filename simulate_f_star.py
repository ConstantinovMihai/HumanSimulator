"""
Simulate the signal presented to the ideal human operator model
"""

import systems_functions as sf
import numpy as np
import matplotlib.pyplot as plt
import constants
from input_signal import plot_time_frequency_signal, get_fft_values
from simulate_closed_loop import open_loop_identification


def simulate_f_star(input_signal, H, sim_consts):
    """
    Simulate the signal presented to the ideal human operator model.

    Parameters:
    input_signal (array_like): Input signal.
    H (dict): Transfer function parameters {'num': numerator coefficients, 'den': denominator coefficients, 'd0': direct feedthrough term}.
    sim_consts (constants.SimulationConstants): Object containing simulation constants.

    Returns:
    f_star (array_like): Processed signal.
    """
    # move the signal to the left (to take into account the look ahead)
    # deal with the edge case tau_s = 0 
    if sim_consts.tau_s == 0:
        input_signal_ahead = input_signal
    else:
        input_signal_ahead = np.roll(input_signal, -sim_consts.discard_samples)

    # Convert transfer function to state-space
    A, B, C, D = sf.convert_tf_ss_ccf(H)

    # Discretize state-space system
    Ad, Bd, Cd, Dd = sf.cont2discrete_zoh(sim_consts.dt, A, B, C, D)
     
    # Simulate the discrete system
    f_star = sf.simulate_discrete_system(Ad, Bd, Cd, Dd, input_signal_ahead)

    return f_star.flatten().reshape(-1, 1)


def generate_f_star(input_signal: np.ndarray, ct: constants.SimulationConstants):
    """
    Generate the f_star signal using the input signal and the transfer function H_of.

    Parameters:
    ct (constants.SimulationConstants): Object containing simulation constants.
    input_signal (np.ndarray): Input signal.

    Returns:
    f_star (np.ndarray): Processed signal.
    """
    
    # add the pade delay
    H_of = sf.multiply_transfer_functions(ct.H_of, ct.H_pade_f)

    # simulate the signal presented to the ideal human operator model
    f_star = simulate_f_star(input_signal, H_of, ct)

    return f_star


def test_f_star_signal(ct, input_signal, f_star):
    """
    Test the f_star signal.
    """
    # add the pade delay
    H_of = sf.multiply_transfer_functions(ct.H_of, ct.H_pade_f)

    # calculate the FFT of the signals
    frequency_values, fft_values = get_fft_values(input_signal, ct.fs, len(input_signal))

    # plot the time and frequency domain of the signal
    plot_time_frequency_signal(np.arange(0, len(input_signal)*ct.dt, ct.dt), input_signal, frequency_values, fft_values, name="input signal")

    print(f"frequency with most energy in input signal {frequency_values[np.argmax(fft_values)]} Hz")

    # get rid of the transient response
    f_star = sf.remove_transient_behaviour(f_star, ct)
    frequency_values, fft_values = get_fft_values(f_star, ct.fs, len(f_star))

    # plot the time and frequency domain of the signal
    plot_time_frequency_signal(np.arange(0, len(f_star) * ct.dt, ct.dt), f_star, frequency_values, fft_values, name="processed signal")
    
    print(f"frequency with most energy in f_star signal {frequency_values[np.argmax(fft_values)]} Hz")

    # plot the response
    plt.plot(input_signal, "r", label="original signal")
    plt.plot(np.concatenate((np.zeros((len(input_signal) - len(f_star), 1)), f_star)), label="processed signal")
    plt.legend()
    plt.grid()
    plt.xlabel('Time step')
    plt.ylabel('Output signal')
    plt.title('Response of the discrete-time system')
    plt.show()

    # run the open loop identification
    # read from frequencies.csv
    input_frequencies = np.loadtxt("frequencies.csv", delimiter=",").astype(int)

    # intput_frequencies = np.astype(input_frequencies, int)
    open_loop_identification(f_star, input_signal[len(input_signal)//2:], input_frequencies, H_of, ct)


if __name__ == "__main__":
    # get the simulation constants
    ct = constants.SimulationConstants()
    
    # import the input signal from input_signal.csv
    input_signal = np.loadtxt("input_signal.csv", delimiter=",").reshape(-1, 1)

    # simulate the signal presented to the ideal human operator model
    f_star = generate_f_star(input_signal, ct)
    np.savetxt("f_star.csv", f_star, delimiter=",")

    test_f_star_signal(ct, input_signal, f_star)

   