"""
This module is responsible for generating the input signal for the human simulator, using multisines signals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import copy
import constants

np.random.seed(20051310)

def generate_multi_sign(T, fs, n, A, nfwm, phi, discard_samples=0):
    """
    Generate a multisine signal.
    :param T: float, duration of the signal in seconds
    :param fs: float, sampling frequency in Hz
    :param n: int, number of multisines
    :param A: list, amplitude of the multisines
    :param nfwm: list, nf*wm of the multisines
    :param phi: list, phase of the multisines
    :param discard_samples: int, number of samples to discard due to look ahead time
    :return: numpy array, multisine signal
    """

    N = int(T * fs)
    t = np.arange(0, T, 1/fs)
    # input signal
    u = np.zeros(N)

    for i in range(n):
        u += A[i] * np.sin(nfwm[i] * t + phi[i])

    return t, u


def zero_pad_signal(u, N):
    """
    Zero pad the input signal.
    :param u: numpy array, input signal
    :param N: int, desired number of samples
    :return: numpy array, zero padded signal
    """
    u = np.concatenate((u, np.zeros(N - len(u))))
    return u


def get_fft_values(y_values : np.array, fs: float, N : int):
    """ Get the FFT values of a signal.

    Args:
        y_values (numpy array): input signal
        fs (float): sampling frequency in Hz (1/T)
        N (int): number of samples
    Returns:
        frequency_values (numpy array): frequency values
        fft_values (numpy array): FFT values
    """
    # copy y_values and flatten it
    y_values = copy.deepcopy(y_values).flatten()
    frequency_values = np.arange(0.0, fs/2, fs/N)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * fft_values_[0:N//2]

    return frequency_values, fft_values


def plot_time_frequency_signal(t, u, frequency_values, fft_values, name=None, show=True):
    """
    Plot the time and frequency domain of the signal.
    :param t: numpy array, time values
    :param u: numpy array, input signal
    :param frequency_values: numpy array, frequency values
    :param fft_values: numpy array, FFT values
    """
    # Plot the input signal
    plt.subplot(2, 1, 1)
    plt.plot(t, u)
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Input signal")
    plt.title(f"Time domain of the {name} signal")

    # Plot the frequency domain of the signal in log-log scale
    plt.subplot(2, 1, 2)
    plt.plot(frequency_values * 2 * np.pi, np.abs(fft_values), 'o', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Angular Frequency [rad/s]')
    plt.grid()
    plt.ylabel('Amplitude')
    plt.title(f"Frequency domain of the signal {name} (log-log scale)")
    
    if show: 
        plt.show()


def compute_crest_factor(u):
    """
    Compute the crest factor of the input signal.
    :param u: numpy array, input signal
    :return: float, crest factor
    """
    return np.max(np.abs(u)) / np.sqrt(np.mean(u ** 2))


def generate_multi_sign_from_df(ct, df_name='plaetinck_params.csv'):
    """
    Determine the parameters of a multisine signal.
    :param df: pandas DataFrame, data frame containing the parameters of the multisine signal
    :return: tuple, parameters of the multisine signal
    """
    # Extract parameters from the DataFrame
    df = pd.read_csv(df_name)
    df.columns = df.columns.str.strip()

    n = len(df)
    A = df['Af'].values
    nf = df['nf'].values
    nfwm = nf * 2*np.pi / ct.t_measurement
    phi = df['phi_f'].values

    # Generate the input signal
    t, u = generate_multi_sign(ct.t_total, ct.fs, n, A, nfwm, phi, discard_samples=ct.discard_samples)

    return u

if __name__ == "__main__":
    ct = constants.SimulationConstants()
    # Read data from CSV file
    df = pd.read_csv('plaetinck_params.csv')

    # Strip any leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()

    # Extract parameters from the DataFrame
    n = len(df)
    N = int(ct.t_total * ct.fs)  # number of samples

    # Amplitude (Column3), Frequency (Column4), Phase (Column5)
    A = df['Af'].values
    nf = df['nf'].values
    nfwm = nf * 2*np.pi / ct.t_measurement
    phi = df['phi_f'].values
    
    # save the input frequencies nf/ct.t_total in a file
    np.savetxt("frequencies.csv", nf, delimiter=",")

    # Generate the input signal
    t, u = generate_multi_sign(ct.t_total, ct.fs, n, A, nfwm, phi, discard_samples=ct.discard_samples)

    # Get FFT values of the input signal
    frequency_values, fft_values = get_fft_values(u, ct.fs, N)
    np.savetxt("input_signal.csv", u, delimiter=",")
    
    print(f"Crest factor: {compute_crest_factor(u)}")

    plot_time_frequency_signal(t, u, frequency_values, fft_values, name="Example forcing function")
