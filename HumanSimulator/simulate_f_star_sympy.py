"""
Simulate the signal presented to the ideal human operator model
"""

import systems_functions as sf
import numpy as np
import matplotlib.pyplot as plt
import constants
from simpy_tests import SimulationConstants
import sympy as sp
import time


def simulate_f_star_numeric(A, B, C, D, input_signal: np.ndarray, ct, x0=None):
    """Convert the symbols to numeric values and simulate the system.

    This function takes symbolic state-space matrices (A, B, C, D) and an initial state vector x0,
    converts them to their numeric equivalents based on the constants provided in the 'ct' class,
    and then simulates the discrete system response to the given input signal.

    Args:
        input_signal (np.ndarray): The input signal to the system as a NumPy array.
        A (sympy.Matrix): The symbolic state matrix A.
        B (sympy.Matrix): The symbolic input matrix B.
        C (sympy.Matrix): The symbolic output matrix C.
        D (sympy.Matrix): The symbolic feedthrough matrix D.
        x0 (np.ndarray): The initial state vector of the system as a NumPy array.
        ct (SimulationConstants): An instance of the SimulationConstants class containing 
                                  the numeric values of the symbolic variables.

    Returns:
        f_star (np.ndarray): The simulated output of the system as a column vector.
        x_f (np.ndarray): The final state vector of the system after simulation.
    """
    [A_num, B_num, C_num, D_num] = ct.evaluate_matrices([A, B, C, D])
    
    Ad, Bd, Cd, Dd = sf.cont2discrete_zoh(ct.dt, A_num, B_num, C_num, D_num)
    # fill in the numerical values in the matrices

    f_star, x_f = sf.simulate_discrete_system(Ad, Bd, Cd, Dd, input_signal, x0=x0)
 
    return f_star.flatten().reshape(-1, 1), x_f


def simulate_f_star_numeric_alt(A, B, C, D, input_signal: np.ndarray, ct, x0=None):
    """Convert the symbols to numeric values and simulate the system.

    This function takes symbolic state-space matrices (A, B, C, D) and an initial state vector x0,
    converts them to their numeric equivalents based on the constants provided in the 'ct' class,
    and then simulates the discrete system response to the given input signal.

    Args:
        input_signal (np.ndarray): The input signal to the system as a NumPy array.
        A (sympy.Matrix): The symbolic state matrix A.
        B (sympy.Matrix): The symbolic input matrix B.
        C (sympy.Matrix): The symbolic output matrix C.
        D (sympy.Matrix): The symbolic feedthrough matrix D.
        x0 (np.ndarray): The initial state vector of the system as a NumPy array.
        ct (SimulationConstants): An instance of the SimulationConstants class containing 
                                  the numeric values of the symbolic variables.

    Returns:
        f_star (np.ndarray): The simulated output of the system as a column vector.
        x_f (np.ndarray): The final state vector of the system after simulation.
    """
 
    [Ad, Bd, Cd, Dd] = ct.evaluate_matrices([A, B, C, D])
    # fill in the numerical values in the matrices

    f_star, x_f = sf.simulate_discrete_system(Ad, Bd, Cd, Dd, input_signal, x0=x0)
 
    return f_star.flatten().reshape(-1, 1), x_f



def simulate_f_star(input_signal: np.ndarray, ct: SimulationConstants, x0=None):
    """
    Generate the f_star signal using the input signal and the transfer function H_of.

    Parameters:
    ct (constants.SimulationConstants): Object containing simulation constants.
    input_signal (np.ndarray): Input signal.

    Returns:
    f_star (np.ndarray): Processed signal.
    """

    if ct.tau_s > 0:
        input_signal_ahead = np.roll(input_signal, -ct.discard_samples)
    else:
        input_signal_ahead = input_signal
    

    A, B, C, D = sf.convert_tf_ss_ccf_sympy(ct.H_of_del)
   
    # now plan the scheduling of the transfer function
    f_star, x_f = simulate_f_star_numeric(A, B, C, D, input_signal_ahead, ct, x0=x0)

    return f_star, x_f


def process_transition_period(input_signal, timestamp, ct):
    """
    Process the transition period of the input signal.

    Parameters:
    input_signal (array_like): Input signal.
    timestamp (int): Time at which the transition occurs.
    ct (constants.SimulationConstants): Object containing simulation constants.

    Returns:
    f_star (array_like): Processed signal.
    """
    # Split the input signal into two parts
    input_signal_before = input_signal[:timestamp]
    input_signal_after = input_signal[timestamp:]

    # Simulate the f_star signal for the first part of the input signal
    f_star_before, _ = simulate_f_star(input_signal_before, ct)

    # Simulate the f_star signal for the second part of the input signal
    f_star_after, _ = simulate_f_star(input_signal_after, ct)

    # Combine the two parts of the f_star signal
    f_star = np.concatenate((f_star_before, f_star_after), axis=0)

    return f_star


def schedule_distractions(distraction_times, ct, input_signal):
    x_f = None
    f_star_list = []  # Initialize an empty list to store f_star outputs

    if ct.tau_s > 0:
        input_signal_ahead = np.roll(input_signal, -ct.discard_samples)
    else:
        input_signal_ahead = input_signal


    A, B, C, D = sf.convert_tf_ss_ccf_sympy(ct.H_of_del)
    
    # Break the input signal into chunks based on the distraction times
    input_chunks = np.split(input_signal_ahead, distraction_times)
    
    new_gain = {0: 1, 1: 0.1}
    start = time.time()
    curr_time = start
    transition_time = 0
    constant_time = 0 

    for idx, input_signal_chunk in enumerate(input_chunks):
        print(f"Processing chunk {idx + 1}/{len(input_chunks)}. Took {np.round(time.time() - curr_time, 2)} seconds")
        curr_time = time.time()
        # simulate the transition period
        input_signal_chunk_transition = input_signal_chunk[:200]
        input_signal_chunk_rest = input_signal_chunk[200:]
        # f_star_transition, x_f = simulate_f_star_numeric(A, B, C, D, input_signal_chunk_transition, ct, x0=x_f)

        transition_time = time.time()
        f_star_transition, x_f = simulate_transition_part(A, B, C, D, input_signal_chunk_transition, idx % 2, ct, x_f)
        print(f"Transition time: {np.round(time.time() - transition_time, 2)} seconds")

        f_star_list.append(f_star_transition)  # Append each f_star to the list

        # when driver is distracted, their gains go to zero
        # ct.K_f_val = new_gain[idx % 2]
        print(f"K_f_val: {ct.K_f_val}")
        # ct.initialize_transfer_functions()
        
        constant_time = time.time()
        f_star, x_f = simulate_f_star_numeric(A, B, C, D, input_signal_chunk_rest, ct, x0=x_f)
        print(f"Constant time: {np.round(time.time() - constant_time, 2)} seconds")
        # f_star, x_f = simulate_f_star(input_signal_chunk, ct, x0=x_f)
        f_star_list.append(f_star)  # Append each f_star to the list

    # Concatenate all f_star outputs into a single array
    f_star_combined = np.concatenate(f_star_list, axis=0)

    return f_star_combined


def simulate_transition_part(A,B,C,D, input_signal, state_transition, ct, x_f):
    # use simulate_f_star but each time step is a different transfer function
    # and each input signal is only one time step
    f_star_out = None
    gain_dict = {0: [0.1, 1], 1 : [1, 0.1]}
    P1, P2 = gain_dict[state_transition]
    # when going from distraction to non distraction P2=0.1, P1=1
    # when going from non distraction to distraction P2=1, P1=0.1
    P_schedule = schedule_transition_period(P1, P2, len(input_signal))
    
    print("let's see")
    # Ad, Bd, Cd, Dd = sf.cont2discrete_zoh_sympy(ct.dt, A, B, C, D)
    #print(Ad, Bd, Cd, Dd)
    for t, K_f_val_idx in enumerate(P_schedule):
        ct.K_f_val = K_f_val_idx
        ct.initialize_transfer_functions()

        f_star, x_f = simulate_f_star_numeric(A, B, C, D, input_signal[t], ct, x0=x_f)

        # append the f_star to the output signal
        if t == 0:
            f_star_out = f_star
        else:
            f_star_out = np.concatenate((f_star_out, f_star))
    
    return f_star_out, x_f


def schedule_transition_period(P1=1, P2=0.1, length=200):
    t = np.linspace(0, 2, min(length, 200))  # Time points from 0 to 10 
   
    M = 1.0   # Time of maximum rate of change
    G = 9.0   # Maximum rate of change

    P_sch = sigmoid_schedule(t, P1, P2, M, G)

    return P_sch


def sigmoid_schedule(t, P1, P2, M, G):
    return P1 + (P2 - P1) / (1 + np.exp(-G * (t - M)))


def test_time_varying_symbols():
    # get the simulation constants
    ct = SimulationConstants()
    ct.initialize_transfer_functions()
    assert ct.discard_samples > 0
    # import the input signal from input_signal.csv
    input_signal = np.loadtxt("input_signal.csv", delimiter=",").reshape(-1, 1)
    # add a distraction at half time

    N = len(input_signal)
    print(np.sort(np.random.randint(0, N, 3)))
    distraction_times = np.sort(np.random.randint(0, N, 3))

    print(np.sort(np.random.randint(0, N, 3)))

    # place three random distraction in the signal
    # distraction_times = np.random.randint(0, N, 3)

    f_star = schedule_distractions(distraction_times, ct, input_signal)

    # plot the response
    plt.plot(input_signal, "r", label="original signal")
    plt.plot(f_star, label="processed signal")

    for distraction_time in distraction_times:
        plt.axvline(distraction_time, color='k', linestyle='--', alpha=0.5, label="Distraction time")

    plt.legend()
    plt.grid()
    plt.xlabel('Time step')
    plt.ylabel('Output signal')
    plt.title('Response of the discrete-time system')
    plt.show()


def test_time_constant_symbols():
    # get the simulation constants
    ct = SimulationConstants()
    ct.initialize_transfer_functions()
    assert ct.discard_samples > 0
    # import the input signal from input_signal.csv
    input_signal = np.loadtxt("input_signal.csv", delimiter=",").reshape(-1, 1)

    # simulate the signal presented to the ideal human operator model
    f_star, _ = simulate_f_star(input_signal, ct)
    f_star = sf.remove_transient_behaviour(f_star, ct)

    # plot the response
    plt.plot(input_signal, "r", label="original signal")
   # Calculate lengths
    zeros_before_length = len(input_signal) - len(f_star) - ct.discard_samples
    zeros_after_length = ct.discard_samples

    # Create arrays with appropriate shapes
    zeros_before = np.zeros((zeros_before_length,))  # Shape (zeros_before_length,)
    zeros_after = np.zeros((zeros_after_length,))    # Shape (zeros_after_length,)

    # Concatenate arrays
    processed_signal = np.concatenate((zeros_before, f_star.flatten(), zeros_after))
    plt.plot(processed_signal, label="processed signal")
    plt.legend()
    plt.grid()
    plt.xlabel('Time step')
    plt.ylabel('Output signal')
    plt.title('Response of the discrete-time system')
    plt.show()

    np.savetxt("processed_signal.csv", processed_signal, delimiter=",")


if __name__ == "__main__":
    #test_time_constant_symbols()
    test_time_varying_symbols()




