""" Take f_star signal and simulate the closed-loop system. 
""" 

import systems_functions as sf
import numpy as np
import matplotlib.pyplot as plt
import simpy_tests
from input_signal import get_fft_values, plot_time_frequency_signal
import bode
import sympy as sp
from simulate_f_star_sympy import schedule_transition_period
import matplotlib.patches as mpatches

# for now, set seed
np.random.seed(20054232)

def generate_remnant_realisation(nb_samples, ct: simpy_tests.SimulationConstants):
    """
    Generates a state-space representation of the remnant.
    Returns:
        tuple: A tuple containing four elements:
            - A (ndarray): State transition matrix.
            - B (ndarray): Input matrix.
    """

    # convert tf to ss
    A, B, C, D = sf.convert_tf_ss_ccf_sympy(ct.H_remnant)
    remnant_realisation, white_noise, _ = generate_remnant_realisation_num(A, B, C, D, nb_samples, ct)

    return remnant_realisation, white_noise


def generate_remnant_realisation_num(A,B,C,D, nb_samples: int, ct: simpy_tests.SimulationConstants, x0 = None) -> tuple:
    """Generates a remnant realisation.
    The remnant is a white noise filtered through a first-order low-pass filter with a cut-off frequency of 0.1 Hz.

    Args:
        nb_samples (int): Number of samples to generate.
        ct (SimulationConstants): Instance of SimulationConstants containing symbolic parameters.

    Returns:
        tuple: A tuple containing:
            - 2D array (numpy array) of the generated output signal y.
            - 1D array (numpy array) of the generated white noise.
            - float of the last state vector of the system.
    """
    # discretize
    [A_num, B_num, C_num, D_num] = ct.evaluate_matrices([A, B, C, D])
    Ad, Bd, Cd, Dd = sf.cont2discrete_zoh(ct.dt, A_num, B_num, C_num, D_num)
   
    # generate white noise of length nb_samples
    white_noise = np.random.normal(0, 1, nb_samples) / np.sqrt(ct.dt) #TODO: generate this properly

    if x0 is None:
        # initialize state vector
        x = np.zeros((Ad.shape[0], 1))
    else:
        x = x0

    # initialize output signal
    y = np.zeros((Cd.shape[0], nb_samples))

    # simulate system
    for t in range(nb_samples):
        # update state vector
        x = Ad @ x + Bd * white_noise[t]
        # compute output
        y[:, t] = Cd @ x + Dd * white_noise[t]
    
    return y.flatten().reshape(-1, 1), white_noise, x


def generate_combined_system(H1, H2, ct: simpy_tests.SimulationConstants, print_matrices_dim=False):
    
    # convert tf to ss
    A1, B1, C1, D1 = sf.convert_tf_ss_ccf_sympy(H1)
    A2, B2, C2, D2 = sf.convert_tf_ss_ccf_sympy(H2)
    
    # print the size of the matrices
    if print_matrices_dim:
        print(f"A1: {A1.shape}, B1: {B1.shape}, C1: {C1.shape}, D1: {D1.shape}")
        print(f"A2: {A2.shape}, B2: {B2.shape}, C2: {C2.shape}, D2: {D2.shape}")

    A_comp_int = sp.Matrix.hstack(A1, sp.zeros(A1.shape[0], A2.shape[1]))
    A_comp = sp.Matrix.vstack(A_comp_int, sp.Matrix.hstack(B2 * C1, A2))

    B_comp = sp.Matrix.vstack(B1, B2 * D1)

    C_comp = sp.Matrix.hstack(C1, sp.zeros(C1.shape[0], C2.shape[1]))
    C_comp = sp.Matrix.vstack(C_comp, sp.Matrix.hstack(D2 * C1, C2))
                    
    D_comp = sp.Matrix.vstack(D1, D2 * D1)

    return A_comp, B_comp, C_comp, D_comp


def simulate_closed_loop_dynamics_num(f_star, A, B, C, D, remnant,  ct: simpy_tests.SimulationConstants,  x0=None, e0=None):
    # discretize
    
    [A_num, B_num, C_num, D_num] = ct.evaluate_matrices([A, B, C, D])
    Ad_comp, Bd_comp, Cd_comp, Dd_comp = sf.cont2discrete_zoh(ct.dt, A_num, B_num, C_num, D_num)

    # initialize state vector
    if x0 is None:
        x = np.zeros((Ad_comp.shape[0], 1))
    else:
        x = x0

    # initialize output signals
    u = np.zeros((len(f_star), 1))
    u_star = np.zeros((len(f_star), 1))

    # initialize error signal
    if e0 is None:
        e = np.array([[f_star[0]]]).reshape(-1, 1)
    else:
        e = e0
    
    error_signal = np.zeros((len(f_star), 1))

    assert x.shape[0] == Ad_comp.shape[0] and x.shape[1] == 1, "x has the wrong shape"
    assert len(f_star) == len(remnant), "f_star and remnant have different lengths"

    for t in range(len(f_star)):
        
        x = Ad_comp @ x + Bd_comp @ e
        y = Cd_comp @ x + Dd_comp @ e

        u[t] = y[0]
        u_star[t] = y[1]
        
        # compute error signal
        e = (f_star[t] - (u_star[t] + remnant[t] + 0)).reshape(-1, 1) # NOTE: the 0 is the disturbance signal (not implemented yet) 
        error_signal[t] = e 

   
    return u, u_star, error_signal, x, e


def simulate_transition_part_closed_loop(f_star, A,B,C,D, remnant, state_transition, ct, x_f):
    gain_dict = {0: [0.1, 1], 1 : [1, 0.1]}
    P1, P2 = gain_dict[state_transition]
    # when going from distraction to non distraction P2=0.1, P1=1
    # when going from non distraction to distraction P2=1, P1=0.1
    P_schedule = schedule_transition_period(P1, P2, len(f_star))
    
    # initialize state vector
    if x_f is None:
        x = np.zeros((A.shape[0], 1))
    else:
        x = x_f

    u_out = None
    u_star_out = None
    error_signal_out = None

    e = None

    for t, K_p_val_idx in enumerate(P_schedule):
        ct.K_p_val = K_p_val_idx
        ct.initialize_transfer_functions()
        #print(f"{f_star.shape=}; {remnant.shape=}; {t=}")
        u, u_star, error_signal, x, e = simulate_closed_loop_dynamics_num(np.array(f_star[t]).reshape(-1, 1), A,B,C,D, np.array(remnant[t]).reshape(-1, 1), ct, x, e)
        if t== 0:
            u_out = u
            u_star_out = u_star
            error_signal_out = error_signal
        else:
            u_out = np.concatenate((u_out, u))
            u_star_out = np.concatenate((u_star_out, u_star))

            error_signal_out = np.concatenate((error_signal_out, error_signal))
    if t == 100:
        assert np.sum(u_star_out) != 0, f"u_star_out is zero {np.sum(u_star_out)}"    

    return u_out, u_star_out, error_signal_out, x


def simulate_time_varying_closed_loop(distraction_times, ct, f_star, remnant_signal, transition_time=200):

    A_comp, B_comp, C_comp, D_comp = generate_combined_system(ct.H_comb, ct.H_ce, ct)
    # u, u_star, error_signal, x, _ = simulate_closed_loop_dynamics_num(f_star, A_comp, B_comp, C_comp, D_comp, remnant_signal, ct, x0)
    x_f = None
    u_list = []
    u_star_list = []
    error_signal_list = []
    
    f_star_chunks = np.split(f_star, distraction_times)
    remnant_chunks = np.split(remnant_signal, distraction_times)

    for idx, chunk in enumerate(f_star_chunks):
        print(f"Chunk {idx} of {len(f_star_chunks)}")
        input_signal_chunk_transition = chunk[:transition_time]
        input_signal_chunk_rest = chunk[transition_time:]

        remnant_chunk_transition = remnant_chunks[idx][:transition_time]
        remnant_chunk_rest = remnant_chunks[idx][transition_time:]

        u, u_star, error_signal, x_f = simulate_transition_part_closed_loop(input_signal_chunk_transition, A_comp, B_comp, C_comp, D_comp, remnant_chunk_transition, idx % 2, ct, x_f)
        
        u_list.append(u)
        u_star_list.append(u_star)
        error_signal_list.append(error_signal)

        u, u_star, error_signal, x_f, _ = simulate_closed_loop_dynamics_num(input_signal_chunk_rest, A_comp, B_comp, C_comp, D_comp, remnant_chunk_rest, ct, x_f, e0=None)
        
        u_list.append(u)
        u_star_list.append(u_star)
        error_signal_list.append(error_signal)

    return np.concatenate(u_list), np.concatenate(u_star_list), np.concatenate(error_signal_list), x_f
    

def test_distractions():
    ct = simpy_tests.SimulationConstants()
    ct.initialize_transfer_functions()

    assert ct.discard_samples > 0

    # import the input signal from input_signal.csv
    f_star = np.loadtxt("f_star.csv", delimiter=",").reshape(-1, 1)
    # add a distraction at half time

    N = len(f_star)
    print(np.sort(np.random.randint(0, N, 3)))
    distraction_times = np.sort(np.random.randint(0, N, 3))

    # u, u_star, error_signal, x = simulate_distractions()

        # plot the response
    plt.plot(f_star, "r", label="original signal")

    for distraction_time in distraction_times:
        plt.axvline(distraction_time, color='k', linestyle='--', alpha=0.5, label="Distraction time")

    plt.legend()
    plt.grid()
    plt.xlabel('Time step')
    plt.ylabel('Output signal')
    plt.title('Response of the discrete-time system')
    plt.show()


def simulate_transition_part_remnant(A,B,C,D, input_signal, state_transition, ct, x_f):
    # use simulate_f_star but each time step is a different transfer function
    # and each input signal is only one time step
    remnant_realisation_out = None
    white_noise_out = None
    gain_dict = {0: [0.105, 0.205], 1 : [0.205, 0.105]}
    P1, P2 = gain_dict[state_transition]
    # when going from distraction to non distraction P2=0.1, P1=1
    # when going from non distraction to distraction P2=1, P1=0.1
    P_schedule = schedule_transition_period(P1, P2)
    
    print("let's see")
    # Ad, Bd, Cd, Dd = sf.cont2discrete_zoh_sympy(ct.dt, A, B, C, D)
    #print(Ad, Bd, Cd, Dd)
    for t, Kn_remnant_val_idx in enumerate(P_schedule):
        ct.Kn_remnant_val = Kn_remnant_val_idx
        ct.initialize_transfer_functions()

        remnant_realisation, white_noise, x_f = generate_remnant_realisation_num(A, B, C, D, nb_samples=1, ct=ct, x0=x_f)

        # append the f_star to the output signal
        if t == 0:
            remnant_realisation_out = remnant_realisation
            white_noise_out = white_noise
        else:
            remnant_realisation_out = np.concatenate((remnant_realisation_out, remnant_realisation))
            white_noise_out = np.concatenate((white_noise_out, white_noise))
    
    return remnant_realisation_out, white_noise_out, x_f


def time_varying_remnant(distraction_times, ct, input_signal):
    A, B, C, D = sf.convert_tf_ss_ccf_sympy(ct.H_remnant)
    x_f = None
    remnant_realisation_list = []
    white_noise_list = []

    input_chunks = np.split(input_signal, distraction_times)

    for idx, chunk in enumerate(input_chunks):
        input_signal_chunk_transition = chunk[:200]
        input_signal_chunk_rest = chunk[200:]

        remnant_realisation, white_noise, x_f = simulate_transition_part_remnant(A, B, C, D, len(input_signal_chunk_transition), idx % 2, ct, x_f)
        
        remnant_realisation_list.append(remnant_realisation)
        white_noise_list.append(white_noise)

        remnant_realisation, white_noise, x_f = generate_remnant_realisation_num(A, B, C, D, len(input_signal_chunk_rest), ct, x_f)

        remnant_realisation_list.append(remnant_realisation)
        white_noise_list.append(white_noise)

    return np.concatenate(remnant_realisation_list, axis=0), np.concatenate(white_noise_list, axis=0)


def generate_tc_from_distraction_times(distraction_times, len_signal):
    tc = np.zeros(len_signal)
    for i in range(0, len(distraction_times), 2):
        tc[distraction_times[i]:distraction_times[i+1]] = 1

    return tc

if __name__ == "__main__":
    # run_test(plot_response=True, inject_remnant=True, test_remnant=True, plot_bode=False)
    ct = simpy_tests.SimulationConstants()
    ct.initialize_transfer_functions()

    f_star = np.loadtxt("f_star.csv", delimiter=",")
    distraction_times = np.sort(np.random.randint(0, len(f_star), 2))
    # print(distraction_times)
    # distraction_times = np.array([1000, len(f_star)-1001])
    # print(f"Distraction times: {distraction_times}")
    # distraction_times = np.array([0, len(f_star)])
    # print(f"Distraction times: {distraction_times}")
    tc = generate_tc_from_distraction_times(distraction_times, len(f_star))

    # remnant_signal, _ = time_varying_remnant(distraction_times, ct, f_star)

    remnant_signal, _ = generate_remnant_realisation(len(f_star), ct)

    assert len(remnant_signal) == len(f_star), f"Remnant signal and f_star have different lengths {len(remnant_signal)} != {len(f_star)}"

    print(f"Simulating the closed loop system!")
    u, u_star, error_signal, x = simulate_time_varying_closed_loop(distraction_times, ct, f_star, remnant_signal)
    input_signal = np.loadtxt("input_signal.csv", delimiter=",").reshape(-1, 1)
    
    print(f"{len(u)=}")

    # # save the output signals
    # np.savetxt(r"HumanSimulator/Simulated_signals/cd_u.csv", u, delimiter=",")
    # np.savetxt(r"HumanSimulator/Simulated_signals/cd_x.csv", u_star, delimiter=",")
    # np.savetxt(r"HumanSimulator/Simulated_signals/cd_error_signal.csv", error_signal, delimiter=",")
    # np.savetxt(r"HumanSimulator/Simulated_signals/cd_remnant_signal.csv", remnant_signal, delimiter=",")
    # np.savetxt(r"HumanSimulator/Simulated_signals/cd_input_signal.csv", input_signal, delimiter=",")
    # np.savetxt(r"HumanSimulator/Simulated_signals/cd_tc.csv", tc, delimiter=",")
    # np.savetxt(r"HumanSimulator/Simulated_signals/cd_distraction_times.csv", distraction_times, delimiter=",")
    # np.savetxt(r"HumanSimulator/Simulated_signals/cd_f_star.csv", f_star, delimiter=",")
    
    plt.plot(input_signal, "r", label=r"Input signal ($f_t$)")
    for distraction_time in distraction_times:
        plt.axvline(distraction_time, color='k', linestyle='--', alpha=0.35)
        

    for i in range(0, len(distraction_times)-1, 2):
        plt.axvspan(distraction_times[i], distraction_times[i+1], alpha=0.3, color='lightblue')
        

    if len(distraction_times) % 2 == 1:
        plt.axvspan(distraction_times[i-1], len(f_star), alpha=0.3, color='lightblue')
    
    plt.plot(u_star, label="u_star")

    # create nice legend
    blue_patch = mpatches.Patch(color='lightblue', alpha=0.3, label='Distraction interval')
    red_line = plt.Line2D([0], [0], color='red', label=r"Input signal $f_t$")
    u_star_line = plt.Line2D([0], [0], color='blue', label=r'Control output signal $x$')
    plt.legend(handles=[blue_patch, red_line, u_star_line], loc='best')

    plt.xlabel("Time step")
    plt.ylabel("Signal amplitude")
    plt.title("Simulated response of a distracted human operator")
    plt.grid()
    plt.show()

    print("Done!")