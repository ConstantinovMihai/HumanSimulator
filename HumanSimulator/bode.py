""" 
Bode plots routines for systems with state-space representation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, bode, ss2tf
import systems_functions as sf

def generate_bode_plot(A, B, C, D):
    """
    Generate Bode plot of a continuous-time system with state-space representation (A, B, C, D).

    Parameters:
    A (array_like): State transition matrix.
    B (array_like): Input matrix.
    C (array_like): Output matrix.
    D (array_like): Feedthrough matrix.
    """
    # Create a TransferFunction object
    numerator, denominator = ss2tf(A, B, C, D)
    system = TransferFunction(numerator, denominator)

    # Compute Bode plot
    frequencies, magnitude, phase = bode(system)

    return frequencies, 10**(magnitude/20), phase


def plot_bode_plot(frequencies, magnitude, phase, show=True, title=None):
    """
    Plots the Bode plot for a given set of frequencies, magnitude, and phase data.

    Args:
        frequencies (array-like): The frequencies at which the magnitude and phase are evaluated.
        magnitude (array-like): The magnitude values (in dB) corresponding to the frequencies.
        phase (array-like): The phase values (in degrees) corresponding to the frequencies.
        show (bool, optional): If True, displays the plot immediately. Defaults to True.
        title (str, optional): The title of the plot. If None, defaults to 'Bode Plot'. Defaults to None.
    """
    # Plot magnitude
    plt.figure()
    plt.loglog(frequencies, magnitude)  # Bode magnitude plot
    plt.title(title + ' - Magnitude' if title else 'Bode Plot - Magnitude')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Magnitude [dB]')
    plt.grid(which='both', axis='both')

    # Plot phase
    plt.figure()
    plt.semilogx(frequencies, phase)  # Bode phase plot
    plt.title(title + ' - Phase' if title else 'Bode Plot - Phase')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Phase [degrees]')
    plt.grid(which='both', axis='both')

    if show:
        plt.show()

if __name__ == "__main__":
    # plot a bode plot with very high frequencies to check the phase behaviour
    # choose a system with phase going under 180 degrees (add a delay for example)

    omega_n = 1.0  # natural frequency
    zeta = 0.1    # damping ratio

    # State-space representation with differentiator
    A = np.array([[0, 1, 0], 
                    [0, 0, 1], 
                    [0, -omega_n**2, -2*zeta*omega_n]])
    B = np.array([[0], [0], [1]])
    C = np.array([[1, 0, 0]])
    D = np.array([[0]])

    H = {"num": np.array([1.5]), "den": np.array([1, 0]), "d0": np.array([0])}
    A, B, C, D = sf.convert_tf_ss_ccf(H)

    frequencies, magnitude, phase = generate_bode_plot(A, B, C, D)
    plot_bode_plot(frequencies, magnitude, phase, title="Simple system")
    plt.show()