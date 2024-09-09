"""
This module contains functions for converting transfer functions to state-space models and simulating discrete-time systems.
"""

import numpy as np
import scipy.linalg as linalg
from typing import  Tuple
import copy
import sympy as sp
from typing import Tuple
import time
from scipy.fftpack import fft

def convert_tf_ss_ccf_sympy(H: dict):
    """Converts a transfer function to a state-space model using the controllable canonical form.

    Parameters:
    H (dict): A dictionary where 'num' contains the coefficients of the numerator,
              'den' contains the coefficients of the denominator, and 'd0' contains the direct feedthrough term.

    Returns:
    tuple: A tuple containing the state-space representation matrices (A, B, C, D) in controllable canonical form.
    """
    # Ensure transfer function is proper
    assert len(H['den']) > len(H['num']), "The transfer function must have more poles than zeros"

    # Highest degree of polynomial is the first entry in the denominator and numerator 
    num = copy.deepcopy(H['num'])
    den = copy.deepcopy(H['den'])

    # Order of the system is determined by the length of the denominator
    order = len(den) - 1

    # Pad the numerator with zeros if it has fewer coefficients than the denominator
    if len(num) < order:
        num = [0] * (order - len(num)) + num

    # Normalize coefficients so that the leading coefficient of the denominator is 1
    num = [coeff / den[0] for coeff in num[::-1]]
    den = [coeff / den[0] for coeff in den[::-1]]
    
    # A matrix
    A = sp.zeros(order, order)
    A[:-1, 1:] = sp.eye(order - 1)

    A[-1, :] = -sp.Matrix(den[:-1]).transpose()
    
    # B matrix
    B = sp.zeros(order, 1)
    B[-1, 0] = 1
    
    # C matrix
    C = sp.Matrix(num).transpose()
    
    # D matrix
    D = sp.Matrix([H['d0']])
    
    return A, B, C, D


def convert_tf_ss_ccf(H: dict):
    """Converts a transfer function to a state-space model using the controllable canonical form.

    Parameters:
    H (dict): A dictionary where 'num' contains the coefficients of the numerator,
              'den' contains the coefficients of the denominator, and 'd0' contains the direct feedthrough term.

    Returns:
    tuple: A tuple containing the state-space representation matrices (A, B, C, D) in controllable canonical form.
    """
    # IMPORTANT: IT IS ASSUMED THAT
    # THE TRANSFER FUNCTION IS PROPER AND
    # assert len(H['den']) > len(H['num']), "The transfer function must have more poles than zeros"
    

    # highest degree of polynomial is the first entry in 
    # the denominator and numerator 
    num = copy.deepcopy(H['num'])
    den = copy.deepcopy(H['den'])

    # # bring the transfer functions in cannonical controllable form, i.e: 
    # # H(s) = (b0 + b1s + b2s^2 + ... + bn-1*s^(n-1) ) / (a0 + a1s + a2s^2 + ... + ans^n) + d
    # # where n is the order of the system
    # # The order of the system is determined by the length of the denominator
    order = len(den) - 1
  
    # Pad the numerator with zeros if it has fewer coefficients than the denominator
    # in order to bring it to the general form
    if len(num) < order:
        num = np.append(np.zeros(order - len(num)), num)

    # Normalize coefficients so that the leading coefficient of the denominator is 1
    num = (num / den[0])[::-1]
    den = (den / den[0])[::-1]
    
    # A matrix
    A = np.zeros((order, order))
    A[:-1, 1:] = np.eye(order - 1)
    A[-1, :] = -den[:-1]
    
    # B matrix
    B = np.zeros((order, 1))
    B[-1, 0] = 1
    
    # C matrix
    C = num.reshape(1, -1)
    
    # D matrix
    D = copy.deepcopy(H['d0']).reshape(-1, 1)
    
    return A, B, C, D




def remove_transient_behaviour(signal, ct):
    """
    Remove the transient behaviour and discarded samples from a signal.

    This function cuts the signal to remove the transient behaviour and the discarded samples 
    due to the look-ahead time specified in the `ct` object. It returns the signal starting 
    from the middle minus the discarded samples length to the end minus the discarded samples length.

    Parameters:
    signal (np.array): The input signal array to process.
    ct: An object containing parameters, specifically `discard_samples` which defines the number 
        of samples to discard from the end.

    Returns:
    np.array: The processed signal with transient behaviour removed.
    """
    return signal[len(signal)//2-ct.discard_samples:-ct.discard_samples]


def remove_transient_behaviour_array(signal_array, ct):
    """
    Apply transient behaviour removal to each signal in an array of signals.

    This function processes each signal in the provided array, cutting it to remove the transient 
    behaviour and discarded samples based on the parameters in the `ct` object. It utilizes 
    `remove_transient_behaviour_aux` to process each individual signal.

    Parameters:
    signal_array (list of np.array): A list of input signal arrays to process.
    ct: An object containing parameters, specifically `discard_samples` which defines the number 
        of samples to discard from the end.

    Returns:
    list of np.array: A list of processed signals with transient behaviour removed.
    """
 
    return [remove_transient_behaviour(arr, ct) for arr in signal_array]


def read_undistracted_data(data_signal_string):
    data = np.loadtxt(data_signal_string, delimiter=',').transpose()
    return data


def print_transfer_function(H : dict, name = "Transfer function"):
    """ Print the transfer function in a human-readable format.

    Parameters:
    H (dict): A dictionary where 'num' contains the coefficients of the numerator,
              'den' contains the coefficients of the denominator, and 'd0' contains the direct feedthrough term.
    """

    num = copy.deepcopy(H['num'])
    den = copy.deepcopy(H['den'])

    num = [coeff / den[0] for coeff in num]
    den = [coeff / den[0] for coeff in den]
    
    print(f"{name}: {num} / {den}")


def print_state_space(A, B, C, D, only_shape=False):
    """ Print the state-space representation in a human-readable format.

    Parameters:
    A (array_like): State transition matrix.
    B (array_like): Input matrix.
    C (array_like): Output matrix.
    D (array_like): Feedthrough matrix.
    """
    if not only_shape:
        print("State-space representation:")
        print(f"A = {A}")
        print(f"B = {B}")
        print(f"C = {C}")
        print(f"D = {D}")

    print("Matrix shapes:")
    print(f"Shape of A: {A.shape}")
    print(f"Shape of B: {B.shape}")
    print(f"Shape of C: {C.shape}")
    print(f"Shape of D: {D.shape}")


def simulate_discrete_system(A, B, C, D, u, x0=None):
    """
    Simulate a discrete-time system with state-space representation (A, B, C, D)
    and an input signal u.

    Parameters:
    A (array_like): State transition matrix.
    B (array_like): Input matrix.
    C (array_like): Output matrix.
    D (array_like): Feedthrough matrix.
    u (array_like): Input signal.
    x0 (array_like, optional): Initial state vector of shape (n_states, 1). Default is None (initialized with zeros).

    Returns:
    y (array_like): Output signal.
    x (array like): Last state vector.
    """
    num_samples = len(u)
    num_states = A.shape[0]
    num_outputs = C.shape[0]

    # Initialize state vector
    if x0 is None:
        x = np.zeros((num_states, 1))
    else:
        x = x0

    # Initialize output signal
    y = np.zeros((num_outputs, num_samples))

    for t in range(num_samples):  
        # Update state vector
        x = A@x + B*u[t]
       
        y[:, t] = C@x + D*u[t]
      
    return y, x

def cont2discrete_zoh_sympy(
    dt: sp.Symbol,
    A: sp.Matrix,
    B: sp.Matrix,
    C: sp.Matrix,
    D: sp.Matrix,
) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix]:
    """
    Discretize continuous-time system using zero-order hold.

    Args:
        dt: time step of the discrete-time system (symbolic)
        A: continuous-time state transition matrix of shape (n, n)
        B: continuous-time input matrix of shape (n, m)
        C: continuous-time output matrix of shape (p, n)
        D: continuous-time feed-through matrix of shape (p, m)

    Returns:
        Ad: discrete-time state transition matrix of shape (n, n)
        Bd: discrete-time input matrix of shape (n, m)
        Cd: discrete-time output matrix of shape (p, n)
        Dd: discrete-time feed-through matrix of shape (p, m)
    """
    # Build an exponential matrix
    em_upper = sp.Matrix.hstack(A, B)

    # Stack zeros under the A and B matrices
    em_lower = sp.Matrix.hstack(sp.zeros(B.shape[1], A.shape[0]), 
                                sp.zeros(B.shape[1], B.shape[1]))


    em = sp.Matrix.vstack(em_upper, em_lower)
    ms = sp.simplify(sp.exp(dt * em))
    
    # Extract discrete-time matrices
    Ad = ms[:A.shape[0], :A.shape[1]]
    Bd = ms[:A.shape[0], A.shape[1]:]
    Cd = C
    Dd = D

    return Ad, Bd, Cd, Dd

def cont2discrete_zoh(
    dt: np.array,
    A: np.array,
    B: np.array,
    C: np.array,
    D: np.array,
) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Discretize continuous-time system using zero-order hold.
    Please refer to the Scipy documentation of the `cont2discrete` function for some inspiration:
        https://github.com/scipy/scipy/blob/v1.9.3/scipy/signal/_lti_conversion.py#L335-L532

    Args:
        dt: time step of the discrete-time system
        A: continuous-time state transition matrix of shape (4, 4)
        B: continuous-time input matrix of shape (4, 2)
        C: continuous-time output matrix of shape (2, 4)
        D: continuous-time feed-through matrix of shape (2, 2)

    Returns:
        Ad: discrete-time state transition matrix of shape (4, 4)
        Bd: discrete-time input matrix of shape (4, 2)
        Cd: discrete-time output matrix of shape (2, 4)
        Dd: discrete-time feed-through matrix of shape (2, 2)

    """
    # Build an exponential matrix
    em_upper = np.hstack((A, B))

    # Need to stack zeros under the a and b matrices
    em_lower = np.hstack((np.zeros((B.shape[1], A.shape[0])),
                            np.zeros((B.shape[1], B.shape[1]))))

    em = np.vstack((em_upper, em_lower))
    # print(f"{em.shape=}")
    ms = linalg.expm(dt * em)

    # Dispose of the lower rows
    ms = ms[:A.shape[0], :]

    Ad = ms[:, 0:A.shape[1]]
    Bd = ms[:, A.shape[1]:]

    Cd = C
    Dd = D

    return Ad, Bd, Cd, Dd


def multiply_transfer_functions(H1, H2):
    """Multiplies two transfer functions.

    Parameters:
    H1 (dict): A dictionary where 'num' contains the coefficients of the numerator,
               'den' contains the coefficients of the denominator, and 'd0' contains the direct feedthrough term.
    H2 (dict): A dictionary where 'num' contains the coefficients of the numerator,
                'den' contains the coefficients of the denominator, and 'd0' contains the direct feedthrough term.

    Returns:
    dict: A dictionary containing the coefficients of the resulting transfer function.
    """

    num1 = H1['num']
    den1 = H1['den']
    num2 = H2['num']
    den2 = H2['den']

    num = np.polymul(num1, num2)
    den = np.polymul(den1, den2)

    normalization_factor = den[0]
    num = num / normalization_factor
    den = den / normalization_factor


    return {'num': num, 'den': den, 'd0': np.array([0])}

def multiply_transfer_functions_sympy(H1, H2):
    """Multiplies two transfer functions.

    Parameters:
    H1 (dict): A dictionary where 'num' contains the coefficients of the numerator,
               'den' contains the coefficients of the denominator, and 'd0' contains the direct feedthrough term.
    H2 (dict): A dictionary where 'num' contains the coefficients of the numerator,
                'den' contains the coefficients of the denominator, and 'd0' contains the direct feedthrough term.

    Returns:
    dict: A dictionary containing the coefficients of the resulting transfer function.
    """
    num1 = sp.Array(H1['num'])
    den1 = sp.Array(H1['den'])
    num2 = sp.Array(H2['num'])
    den2 = sp.Array(H2['den'])

    num_poly1 = sp.Poly(num1, sp.symbols('s'))
    den_poly1 = sp.Poly(den1, sp.symbols('s'))
    num_poly2 = sp.Poly(num2, sp.symbols('s'))
    den_poly2 = sp.Poly(den2, sp.symbols('s'))

    num_poly_result = num_poly1 * num_poly2
    den_poly_result = den_poly1 * den_poly2

    num_result = num_poly_result.all_coeffs()
    den_result = den_poly_result.all_coeffs()

    return {'num': sp.Array(num_result), 'den': sp.Array(den_result), 'd0': sp.Array([0])}


def add_transfer_functions(H1, H2):
    """Adds two transfer functions.

    Parameters:
    H1 (dict): A dictionary where 'num' contains the coefficients of the numerator,
               'den' contains the coefficients of the denominator, and 'd0' contains the direct feedthrough term.
    H2 (dict): A dictionary where 'num' contains the coefficients of the numerator,
                'den' contains the coefficients of the denominator, and 'd0' contains the direct feedthrough term.

    Returns:
    dict: A dictionary containing the coefficients of the resulting transfer function.
    """

    num1 = H1['num']
    den1 = H1['den']
    num2 = H2['num']
    den2 = H2['den']

    num = np.polynomial.polynomial.polyadd(num1, num2)
    den = np.polynomial.polynomial.polyadd(den1, den2)

    return {'num': num, 'den': den, 'd0': np.array([0])}

def pad_signal_with_zeros(signal, length, discarded_samples=0):
    """
    Pad the signal with zeros to the desired length.
    :param signal: np.array, signal to pad
    :param length: int, desired length of the signal
    :return: np.array, padded signal
    """
    return np.concatenate((np.zeros((length - len(signal)-discarded_samples, 1)), signal, np.zeros((discarded_samples, 1))))


def extract_forcing_frequencies(input_signal):
    fft_ft = fft(input_signal)
    N = len(input_signal)

    # Only take the FFT points we need
    fft_ft = fft_ft[1:N//2+1]

    # Calculate periodograms from FFTs
    Sftft = np.abs(1/N * np.conj(fft_ft) * fft_ft)

    # Calculate forcing function frequencies
    n_ft = np.where(Sftft > 1)[0] + 1 # +1 to account for the zero frequency

    return n_ft

def test_extract_frequencies():
    input_signal = read_undistracted_data(f'Data/Clean_CSV_data/van_der_El_CSV_data/PRM/ft.csv')[1]
    n_ft = extract_forcing_frequencies(input_signal)
    print(f"n_ft: {n_ft}")


if __name__ == "__main__":
    # Example usage
    H1 = {
        "num": np.array([1, 1]),  # s + 2
        #"den": np.array([1, 3, 2]),  # s^2 + 3s + 2
        "den": np.array([1, 0, 1]),  # s^2 + 1
        "d0": np.array([0])
    }

    H2 = {
        "num": np.array([1, 12, 60, 120]),  # 2s + 3
        # "den": np.array([1, 1]),  # s + 1
        "den": np.array([2, 7]),  # 2s+7
        "d0": np.array([0])
    }

    test_extract_frequencies()

    H_product = multiply_transfer_functions(H1, H2)
    print("Product Transfer Function:", H_product)