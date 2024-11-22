""" Take f_star signal and simulate the closed-loop system. 
""" 

import systems_functions as sf
import numpy as np
import matplotlib.pyplot as plt
import constants
from input_signal import plot_time_frequency_signal, get_fft_values
import bode
import simulate_f_star as sfs

# for now, set seed
np.random.seed(20054232)

def generate_state_space_remnant(ct: constants.SimulationConstants, show_bode=False):
    """
    Generates a state-space representation of the remnant.
    Returns:
        tuple: A tuple containing four elements:
            - A (ndarray): State transition matrix.
            - B (ndarray): Input matrix.
    """

    # define the filter parameters
    H = ct.H_remnant
    
    # convert tf to ss
    A, B, C, D = sf.convert_tf_ss_ccf(H)

    frequencies, magnitude, phase = bode.generate_bode_plot(A, B, C, D)

    if show_bode:
        bode.plot_bode_plot(frequencies, magnitude, phase, show=True, title="Remnant Bode plot")

    return A, B, C, D


def generate_remnant_realisation(nb_samples : int, ct: constants.SimulationConstants) -> float:
    """Generates a remnant realisation.
        The remnant is a white noise filtered through a first-order low-pass filter with a cut-off frequency of 0.1 Hz
    Returns:
        float: Remnant realisation.
    """
    
    # generate white noise of length nb_samples
    white_noise = np.random.normal(0, 1, nb_samples) / np.sqrt(ct.dt) #TODO: generate this properly

    # filter the white noise
    A, B, C, D = generate_state_space_remnant(ct)
    
    Ad, Bd, Cd, Dd = sf.cont2discrete_zoh(ct.dt, A, B, C, D)
    
    # initialize state vector
    x = np.zeros((Ad.shape[0], 1))
    
    # initialize output signal
    y = np.zeros((Cd.shape[0], nb_samples))

    # simulate system
    for t in range(nb_samples):
        # update state vector
        x = Ad @ x + Bd * white_noise[t]
        # compute output
        y[:, t] = Cd @ x + Dd * white_noise[t]

    return y.flatten().reshape(-1, 1), white_noise


# in the for loop there might be changes in those matrices, 
# so it has to run an extra function that calls simulate_closed_loop_dynamics when the transfer functions are not modified, 
# and then update the transfer function, 
# use the state of the system as initial conditions for the new simulation 
# and run the simulation with the new values
def simulate_closed_loop_dynamics(f_star, H1, H2, remnant, fd_signal, ct: constants.SimulationConstants, print_matrices_dim=False, x0=None):
    """Simulates the closed-loop dynamics of a system with two transfer functions H1 and H2.

    Args:
        f_star (array-like): Desired output signal over time.
        H1 (tf): Transfer function of the first system component.
        H2 (tf): Transfer function of the second system component.
        remnant (np.array) : Remnant realisation.
        x0 (np.array) : Initial state vector (optional)

    Returns:
        tuple: A tuple containing two elements:
            - u (ndarray): The control input signal over time.
            - u_star (ndarray): The output signal of the second system component over time.
            - error_signal (ndarray): The error signal over time.
            - x (ndarray): The last state vector of the system.
    """
    # first, check that both function have more poles than zeros

    # convert tf to ss
    A1, B1, C1, D1 = sf.convert_tf_ss_ccf(H1)
    A2, B2, C2, D2 = sf.convert_tf_ss_ccf(H2)

    # print the size of the matrices
    if print_matrices_dim:
        print(f"A1: {A1.shape}, B1: {B1.shape}, C1: {C1.shape}, D1: {D1.shape}")
        print(f"A2: {A2.shape}, B2: {B2.shape}, C2: {C2.shape}, D2: {D2.shape}")

    A_comp_int = np.hstack((A1, np.zeros((A1.shape[0], A2.shape[1]))))
    A_comp = np.vstack((A_comp_int, np.hstack((B2 @ C1, A2))))

    B_comp = np.vstack((B1, B2 @ D1))

    C_comp = np.hstack((C1, np.zeros((C1.shape[0], C2.shape[1]))))
    C_comp = np.vstack((C_comp, np.hstack((D2 @ C1, C2))))
                    
    D_comp = np.vstack((D1, D2 @ D1))

    # discretize
    Ad_comp, Bd_comp, Cd_comp, Dd_comp = sf.cont2discrete_zoh(ct.dt, A_comp, B_comp, C_comp, D_comp)

    # initialize state vector
    if x0 is None:
        x = np.zeros((Ad_comp.shape[0], 1))
    else:
        x = x0

    # initialize output signals
    u = np.zeros((len(f_star), 1))
    u_star = np.zeros((len(f_star), 1))

    # initialize error signal
    e = np.array([[0]]).reshape(-1, 1)
    error_signal = np.zeros((len(f_star), 1))

    assert x.shape[0] == Ad_comp.shape[0] and x.shape[1] == 1, "x has the wrong shape"

    for t in range(len(f_star)-1):
    
        x = Ad_comp @ x + Bd_comp @ e
        y = Cd_comp @ x + Dd_comp @ e

        u[t] = y[0]
        u_star[t] = y[1] + fd_signal[t]
        
        # compute error signal
        e = (f_star[t] - (u_star[t] + remnant[t])).reshape(-1, 1) # NOTE: the 0 is the disturbance signal (not implemented yet) 
        error_signal[t] = e 

    
    return u, u_star, error_signal, x


def open_loop_identification(u, e, input_frequencies_index, H, ct: constants.SimulationConstants, show_plot=True):    

    frequency_values, fft_values_u = get_fft_values(u, ct.fs, len(u))
    frequency_values, fft_values_e = get_fft_values(e, ct.fs, len(e))
    
    # Index fft_values_u with this mask
    U_w = fft_values_u[input_frequencies_index]
    E_w = fft_values_e[input_frequencies_index]

    # compute Yp(w) = U(w) / E(w)
    Yp_w = U_w / E_w

    # plot the Yp obtained at the given frequencies (of the multi-sine signal)
    # on top of the true transfer function H's bode plot
    A, B, C, D = sf.convert_tf_ss_ccf(H)
    frequencies, magnitude, phase = bode.generate_bode_plot(A, B, C, D)

    if show_plot:
        plt.loglog(frequency_values, np.abs(fft_values_e), label="fft")
        plt.loglog(frequency_values[input_frequencies_index], np.abs(E_w), marker='o', color='red', label="E(w)")
        plt.grid()
        plt.legend()
        plt.show()

        plt.loglog(frequency_values, np.abs(fft_values_u), label="fft")
        plt.loglog(frequency_values[input_frequencies_index], np.abs(U_w), marker='o', color='red', label="U(w)")
        plt.grid()
        plt.legend()
        plt.show()

    # plot the bode plot of the system
    plt.figure()
    plt.loglog(frequencies, magnitude, label="True transfer function")  # Bode magnitude plot with lines
    plt.loglog(2*np.pi*frequency_values[input_frequencies_index], np.abs(Yp_w), 'o', color='red', label="Yp(w)", zorder=5) # Points without lines
    plt.title("True transfer function H - Magnitude")
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Magnitude [dB]')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.show()


    plt.figure()
    plt.semilogx(frequencies, phase, label="True transfer function")  # Bode magnitude plot with lines
    plt.semilogx(2*np.pi*frequency_values[input_frequencies_index], np.rad2deg(np.unwrap(np.angle(Yp_w))), 'o', color='red', label="Yp(w)", zorder=5) # Points without lines
    plt.title("True transfer function H - Phase")
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Phase [deg]')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.show()


def plot_time_response(signal_1, signal_2, label_1="F_star signal", label_2="u_star signal"):
    """Plots the time response of the closed-loop system.
    Args:
        u (ndarray): Control input signal over time.
        u_star (ndarray): Output signal of the second system component over time.
        f_star (ndarray): Desired output signal over time.
    """

    plt.plot(signal_1, label=label_1)
    plt.plot(signal_2, label=label_2)
    plt.grid()
    plt.legend()
    plt.xlabel("Time step")
    plt.ylabel("Output signal")
    plt.title("Response of the closed-loop system")
    plt.show()


def read_undistracted_data(data_signal_string):
    data = np.loadtxt(data_signal_string, delimiter=',').transpose()
    return data

def run_test( plot_bode = False, plot_response = False, inject_remnant=True, test_remnant=False, check_open_loop=False):
    # get the simulation constants
    ct = constants.SimulationConstantsVanderEl()

    # f_star = np.loadtxt("f_star.csv", delimiter=",")
    input_signals = read_undistracted_data(f'Data/Clean_CSV_data/van_der_El_CSV_data/PRM/ft.csv')
    fd_signals = read_undistracted_data(f'Data/Clean_CSV_data/van_der_El_CSV_data/PRM/fd.csv')

    for input in input_signals:
        input_signal = input
        fd_signal = fd_signals[0]*0
        break

    f_star = sfs.simulate_f_star(input_signal, ct)
    input_frequencies = sf.extract_forcing_frequencies(input_signal)

    remnant_realisation, white_noise = generate_remnant_realisation(len(f_star), ct)
    remnant_realisation *= 1 if inject_remnant else 0

    # prepare FFT calculation

    u, u_star, error_signal, x = simulate_closed_loop_dynamics(f_star, ct.H_comb, ct.H_ce, remnant_realisation, fd_signal, ct)
    np.savetxt("u_star.csv", u_star, delimiter=",")
    
    np.savetxt(r"HumanSimulator/Simulated_signals/focused_u.csv", u, delimiter=",")
    np.savetxt(r"HumanSimulator/Simulated_signals/focused_x.csv", u_star, delimiter=",")
    np.savetxt(r"HumanSimulator/Simulated_signals/focused_error_signal.csv", error_signal, delimiter=",")
    np.savetxt(r"HumanSimulator/Simulated_signals/focused_remnant_signal.csv", remnant_realisation, delimiter=",")
    np.savetxt(r"HumanSimulator/Simulated_signals/focused_input_signal.csv", input_signal, delimiter=",")
    # np.savetxt(r"HumanSimulator/Simulated_signals/focused_tc.csv", tc, delimiter=",")
    # np.savetxt(r"HumanSimulator/Simulated_signals/focused_distraction_times.csv", distraction_times, delimiter=",")
    np.savetxt(r"HumanSimulator/Simulated_signals/focused_f_star.csv", f_star, delimiter=",")

    if plot_response:
        #remnant_realisation = sf.remove_transient_behaviour(remnant_realisation, ct)
        N = len(remnant_realisation)
        frequency_values, fft_values = get_fft_values(remnant_realisation, ct.fs, N)
        plot_time_frequency_signal(np.arange(0, N*ct.dt, ct.dt), remnant_realisation, frequency_values, fft_values, name="Remnant realisation")
        
        plot_time_response(f_star, u_star, label_1="F_star signal", label_2="u_star signal")

    if test_remnant:
        open_loop_identification(remnant_realisation, white_noise, input_frequencies, ct.H_remnant, ct, show_plot=True)

    if check_open_loop:
        open_loop_identification(u_star, u, input_frequencies, ct.H_ce, ct, show_plot=True)
        open_loop_identification(u, error_signal, input_frequencies, ct.H_comb, ct, show_plot=True)

    frequency_values, fft_values = get_fft_values(u_star, ct.fs, len(u_star))
    # plot_time_frequency_signal(np.arange(0, N*ct.dt, ct.dt), u_star, frequency_values, fft_values, name="U star")

    if plot_bode:
        A1, B1, C1, D1 = sf.convert_tf_ss_ccf(ct.H_comb)
        A2, B2, C2, D2 = sf.convert_tf_ss_ccf(ct.H_ce)
        frequencies, magnitude, phase = bode.generate_bode_plot(A2, B2, C2, D2)
        bode.plot_bode_plot(frequencies, magnitude, phase)
        frequencies, magnitude, phase = bode.generate_bode_plot(A1, B1, C1, D1)
    
        bode.plot_bode_plot(frequencies, magnitude, phase)
    

if __name__ == "__main__":
    run_test(plot_response=True, inject_remnant=False, test_remnant=False, plot_bode=False, check_open_loop=True)
    print("Done!")