import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import bode
import constants
from simulate_closed_loop import generate_remnant_realisation, get_fft_values, simulate_closed_loop_dynamics, plot_time_frequency_signal, open_loop_identification
import systems_functions as sf

def load_constants():
    return constants.SimulationConstants()

def load_f_star(filename="f_star.csv"):
    return np.loadtxt(filename, delimiter=",")

def prepare_fft(signal, ct):
    N = len(signal)
    return get_fft_values(signal, ct.fs, N)

def define_transfer_functions(ct):
    H = {"num": np.array([ct.K_f]), "den": np.array([ct.T_l_f, 1]), "d0": np.array([0])}
    H_ce = {"num": np.array([ct.H_ce_K]), "den": np.array([ct.T_L_e, 1]), "d0": np.array([0])}
    return H, H_ce

def plot_signals(time_steps, u_star, remnant_realisation, frequency_values, fft_values, plot_bode, plot_response, H, H_ce, f_star, ct : constants.SimulationConstants):
    plot_time_frequency_signal(time_steps, u_star, frequency_values, fft_values, name="U star")

    if plot_bode:
        plot_bode_plot(H, H_ce)
    
    if plot_response:
        plot_response_plot(time_steps, remnant_realisation, f_star, u_star, ct)

def plot_bode_plot(H, H_ce):
    A1, B1, C1, D1 = sf.convert_tf_ss_ccf(H)
    A2, B2, C2, D2 = sf.convert_tf_ss_ccf(H_ce)
    
    for A, B, C, D in [(A2, B2, C2, D2), (A1, B1, C1, D1)]:
        frequencies, magnitude, phase = bode(A, B, C, D)
        bode_plot(frequencies, magnitude, phase)

def test_remnant(ct: constants.SimulationConstants):

    H = {"num": np.array([ct.Kn_remnant]), "den": np.array([ct.Tn_remnant, 1]), "d0": np.array([0])} 
    A, B, C, D = sf.convert_tf_ss_ccf(H)
    # select 10 values from frequency_values.csv, from about the middle, equidistant
    input_frequencies = np.loadtxt("frequency_values.csv", delimiter=",")

    input_frequencies = input_frequencies[::50]
    y, white_noise = generate_remnant_realisation(1000, ct)

    # plot the frequency spectrum of y and white_noise
    N = len(y)
    frequency_values, fft_values_y = get_fft_values(y, ct.fs, N)
    frequency_values, fft_values_white_noise = get_fft_values(white_noise, ct.fs, N)

    # plot the time and frequency domain of the signal
    plot_time_frequency_signal(np.arange(0, N*ct.dt, ct.dt), y, frequency_values, fft_values_y, name="Remnant realisation")
    plot_time_frequency_signal(np.arange(0, N*ct.dt, ct.dt), white_noise, frequency_values, fft_values_white_noise, name="White noise")


    open_loop_identification(y, white_noise, input_frequencies, H, ct)


def bode_plot(frequencies, magnitude, phase):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.semilogx(frequencies, magnitude)
    plt.title('Bode plot')
    plt.ylabel('Magnitude (dB)')
    plt.subplot(2, 1, 2)
    plt.semilogx(frequencies, phase)
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Phase (degrees)')
    plt.grid()
    plt.show()


def plot_response_plot(time_steps, remnant_realisation, f_star, u_star, ct : constants.SimulationConstants):
    frequency_values, fft_values = get_fft_values(remnant_realisation, ct.fs, time_steps)
    plot_time_frequency_signal(time_steps, remnant_realisation, frequency_values, fft_values, name="Remnant realisation")
    plt.plot(f_star, label="f_star")
    plt.plot(u_star, label="u_star")
    plt.grid()
    plt.legend()
    plt.xlabel("Time step")
    plt.ylabel("Output signal")
    plt.title("Response of the closed-loop system")
    plt.show()



def run_test(plot_bode = False, plot_response = False, no_remnant=True):
    ct = load_constants()
    f_star = load_f_star()
    
    remnant_realisation, _ = generate_remnant_realisation(len(f_star), ct)
    
    remnant_realisation *= 0 if no_remnant else 1

    frequency_values, fft_values = prepare_fft(remnant_realisation, ct)
    
    H, H_ce = define_transfer_functions(ct)
    
    u, u_star, _, _ = simulate_closed_loop_dynamics(f_star, H, H_ce, remnant_realisation, ct)
    
    time_steps = np.arange(0, len(u_star) * ct.dt, ct.dt)

    # remove transient behaviour
    u_star = sf.remove_transient_behaviour(u_star, ct)
    f_star = sf.remove_transient_behaviour(f_star, ct)
    remnant_realisation = sf.remove_transient_behaviour(remnant_realisation, ct)
    time_steps = sf.remove_transient_behaviour(time_steps, ct)

    frequency_values, fft_values = prepare_fft(u_star, ct)

    plot_signals(time_steps, u_star, remnant_realisation, frequency_values, fft_values, plot_bode, plot_response, H, H_ce, f_star, ct)

if __name__ == "__main__":
    run_test(no_remnant=False)