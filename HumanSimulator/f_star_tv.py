"""
Simulate the signal presented to the ideal human operator model
And test various time varying implementation scenarios
"""
import systems_functions as sf
import numpy as np
import matplotlib.pyplot as plt
import constants
from sigmoid_schedule import sigmoid_schedule


def simulate_f_star(input_signal, sim_consts, x0=None):
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
    A, B, C, D = sf.convert_tf_ss_ccf(sim_consts.H_of_del)

    # Discretize state-space system
    Ad, Bd, Cd, Dd = sf.cont2discrete_zoh(sim_consts.dt, A, B, C, D)
     
    # Simulate the discrete system
    f_star, x_f = sf.simulate_discrete_system(Ad, Bd, Cd, Dd, input_signal_ahead, x0)

    return f_star.flatten().reshape(-1, 1), x_f


def generate_non_stationary_f_star(input_signal : np.array, ct : constants.SimulationConstantsTimeVarying):
    """ 
    Generate time varying f_star signal using the input signal and the transfer function H_of.

    Parameters:
    ct (constants.SimulationConstants): Object containing simulation constants.
    input_signal (np.ndarray): Input signal.

    Returns:
    f_star (np.ndarray): Processed signal.
    """
    
    # add the pade delay
    H_of = sf.multiply_transfer_functions(ct.H_of, ct.H_pade_f)

    input_signal_1 = input_signal[:int(len(input_signal)/2)]
    input_signal_2 = input_signal[int(len(input_signal)/2):]

    # simulate the signal presented to the ideal human operator model
    f_star_1, x_f = simulate_f_star(input_signal_1, H_of, ct)

    ct.update_transfer_function('H_of', new_num=np.array([25]), new_den=np.array([1, 25]))
    H_of = sf.multiply_transfer_functions(ct.H_of, ct.H_pade_f)

    f_star_2, _ = simulate_f_star(input_signal_2, H_of, ct, x0=x_f)

    print(f"{f_star_1.shape}; {f_star_2.shape}")

    # return the appended signal
    return np.concatenate((f_star_1, f_star_2))


def simulate_transition_part(input_signal, H_schedule, ct):
    # use simulate_f_star but each time step is a different transfer function
    # and each input signal is only one time step
    f_star_out = None
    x_f = None
    for t, H in enumerate(H_schedule):
        
        f_star, x_f = simulate_f_star(np.array([input_signal[t]]), ct, x0=x_f)

        # append the f_star to the output signal
        if t == 0:
            f_star_out = f_star
        else:
            f_star_out = np.concatenate((f_star_out, f_star))
    
    return f_star_out



# now try to create a schedule of a transfer function
# and simulate the signal presented to the ideal human operator model
# in the time varying context
def test_f_star_signal(input_signal, change_schedule, ct):
    # assume that there is a time when the transfer function changes
    # the transition goes according to a sigmoid schedule
    
    # first step, break the input signal into parts according to the change schedule
    
    # step 1.5: simulate the transition part of the signal

    # second step, simulate each part of the signal using the corresponding transfer function

    # third step, append the signals together
    pass


def test_time_varying_signal():
    # time varying constants
    ct = constants.SimulationConstants()
    ct_tv = constants.SimulationConstantsTimeVarying()

    # import the input signal from input_signal.csv
    input_signal = np.loadtxt("input_signal.csv", delimiter=",").reshape(-1, 1)

    f_star = generate_non_stationary_f_star(input_signal, ct_tv)
    
    print(len(f_star))

    # plot the response
    plt.plot(input_signal, "r", label="original signal")
    plt.plot(np.concatenate((np.zeros((len(input_signal) - len(f_star), 1)), f_star)), label="processed signal")
    plt.legend()
    plt.grid()
    
    # plot a red dotted line to show the change in the signal
    plt.axvline(len(input_signal)//2, c='red', ls='dotted')

    plt.xlabel('Time step')
    plt.ylabel('Output signal')
    plt.title('Response of the discrete-time system')
    plt.show()


if __name__ == "__main__":    
    ct = constants.SimulationConstants()
    ct_tv = constants.SimulationConstantsTimeVarying()
    input_signal = np.loadtxt("input_signal.csv", delimiter=",").reshape(-1, 1)
    
    t = np.linspace(0, 10, 1000)  # Time points from 0 to 10
    P1 = 10  # Initial parameter value
    P2 = 15  # Final parameter value
    M = 5.0   # Time of maximum rate of change
    G = 1.0   # Maximum rate of change
    P_sch = sigmoid_schedule(t, P1, P2, M, G)
    
    H_schedule = []
    for P in P_sch:
        H_schedule.append({'num': np.array([ct_tv.K_f * P]), 'den': np.array([1, P]), "d0" : np.array([0])})

    f_star_trans = simulate_transition_part(input_signal[:len(t)], H_schedule, ct_tv)

    test_time_varying_signal()