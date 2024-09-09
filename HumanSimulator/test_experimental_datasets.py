import systems_functions as sf
import constants
from input_signal import *
import simulate_closed_loop as scl
from simulate_f_star import simulate_f_star
from input_signal import plot_time_frequency_signal, get_fft_values
from scipy.fftpack import fft as sp_fft
from scipy.optimize import fmin, bisect
import time

def read_undistracted_data(data_signal_string):
    data = np.loadtxt(data_signal_string, delimiter=',').transpose()
    return data


def generate_simulation_signals(ct, input_signal, fd_signal):
    """
    Generate the simulation signals for the closed-loop system.
    :param ct: SimulationConstants, simulation constants
    :param df_name: str, name of the DataFrame
    :return: tuple, simulation signals
    """

    # import the input signal from input_signal.csv
    f_star = simulate_f_star(input_signal, ct)
    remnant_realisation, _ = scl.generate_remnant_realisation(len(f_star), ct)

    u, u_star, error_signal, _ = scl.simulate_closed_loop_dynamics(f_star, ct.H_comb, ct.H_ce, remnant_realisation, fd_signal=fd_signal, ct=ct)
   
    # u, u_star, error_signal, f_star, remnant_realisation, input_signal = sf.remove_transient_behaviour_array(
    #     [u, u_star, error_signal, f_star, remnant_realisation, input_signal], ct)

    return u, u_star, error_signal, f_star, remnant_realisation, input_signal


def VAF(y, y_hat, show_plot=False):
    """
    Compute the Variation Accounted For (VAF) between two signals.
    
    :param y: np.array, first signal
    :param y_hat: np.array, second signals
    :return: float, VAF
    """

    # assert y.shape == y_hat.shape (
    # f"The signals must have the same form: shape {y.shape} != {y_hat.shape}")

    var_y_diff = np.var(y.reshape(-1,1) - y_hat.reshape(-1,1))
    var_y = np.var(y)

    if show_plot:
        plt.plot(y, label="y")
        plt.plot(y_hat, label="y_hat")
        plt.grid()
        plt.legend()
        plt.show()
   
    return 1 - var_y_diff / var_y


def vaf_david_li(generate_plots=False):
    # get the simulation constants
    ct = constants.SimulationConstants()
    variances_accounted_for = []
    for i in range(1, 4):
        input_signals = read_undistracted_data(f'Data/Clean_CSV_data/David_Li_CSV_data/PreviewDistractionExpData_S{i}/PRN/ft.csv')
        real_responses = read_undistracted_data(f'Data/Clean_CSV_data/David_Li_CSV_data/PreviewDistractionExpData_S{i}/PRN/u.csv')
        fd_signals = read_undistracted_data(f'Data/Clean_CSV_data/David_Li_CSV_data/PreviewDistractionExpData_S{i}/PRN/fd.csv')

        for idx, input_signal in enumerate(input_signals):
            # input_signal  = sf.remove_transient_behaviour(input_signal, ct)
            # real_response = sf.remove_transient_behaviour(real_responses[idx], ct)
            # fd_signal = sf.remove_transient_behaviour(fd_signals[idx], ct)

            real_response = real_responses[idx]
            fd_signal = fd_signals[idx]
         
            input_frequencies = sf.extract_forcing_frequencies(input_signal)
            
            u, u_star, error_signal, f_star, remnant_realisation, input_signal = generate_simulation_signals(ct, input_signal, fd_signal)


            if generate_plots:
                plt.plot(input_signal, label="input_signal")
                plt.plot(f_star, label="f_star")
                plt.xlabel("Time step")
                plt.ylabel("Response")
                plt.title("Input signal vs f_star")
                plt.grid()
                plt.legend()
                plt.savefig('input_vs_f_star.png')
                plt.show()

                # Plot real_response and u
                plt.plot(real_response, label="Experimental (real) HO control signal")
                plt.plot(u, label="Simulated HO control signal")
                plt.xlabel("Time step")
                plt.ylabel("Response")
                plt.title("real_response vs simulated response. HO control signal")
                plt.grid()
                plt.legend()
                plt.savefig('real_vs_u.png')
                plt.show()

                print("now")

            variances_accounted_for.append(VAF(real_response, u, show_plot=generate_plots))
            print(f"VAF: {np.round(variances_accounted_for[-1], 3)}")

    print(f"Real response vs simulated response in the PRN case: \n Average VAF: {np.round(np.mean(variances_accounted_for), 3)} +- {np.round(np.std(variances_accounted_for), 3)}")


def vaf_van_der_el(generate_plots=False):
    # get the simulation constants
    ct = constants.SimulationConstantsVanderEl()
    variances_accounted_for = []

    input_signals = read_undistracted_data(f'Data/Clean_CSV_data/van_der_El_CSV_data/PRM/ft.csv')
    real_responses = read_undistracted_data(f'Data/Clean_CSV_data/van_der_El_CSV_data/PRM/u.csv')
    fd_signals = read_undistracted_data(f'Data/Clean_CSV_data/van_der_El_CSV_data/PRM/fd.csv')

    for idx, input_signal in enumerate(input_signals):
        if idx % 5 == 0 and idx != 0:
            print(f"Intermediary VAF: {np.round(np.mean(variances_accounted_for[-5:]), 3)}") 
            print("====================================")
        
        fd_signal = fd_signals[idx]

        # input_signal  = sf.remove_transient_behaviour(input_signal, ct)
        #real_response = sf.remove_transient_behaviour(real_responses[idx], ct)
        real_response = real_responses[idx]

        u, u_star, error_signal, f_star, remnant_realisation, input_signal = generate_simulation_signals(ct, input_signal, fd_signal)
        input_frequencies = sf.extract_forcing_frequencies(input_signal)
        # np.savetxt('u_van_der_el.csv', u, delimiter=',')
        #scl.open_loop_identification(f_star, input_signal, input_frequencies, ct.H_of, ct, show_plot=False)

        if generate_plots:
            plt.plot(input_signal, label="input_signal")
            plt.plot(real_response, label="real_response")
            plt.xlabel("Time step")
            plt.ylabel("Response")
            plt.title("Input signal vs real_response")
            plt.grid()
            plt.legend()
            plt.show()
            plt.savefig('input_vs_f_star.png')
            plt.close()

            # Plot real_response and u
            plt.plot(real_response, label="Experimental (real) HO control signal")
            plt.plot(u, label="Simulated HO control signal")
            plt.xlabel("Time step")
            plt.ylabel("Response")
            plt.title("real_response vs simulated response. HO control signal")
            plt.grid()
            plt.legend()
            # plt.savefig('real_vs_u.png')
            # plt.close()
            plt.show()

            plt.plot(real_response, label="U_star HO control signal")
            plt.plot(input_signal, label="input_signal")
            plt.xlabel("Time step")
            plt.ylabel("Response")
            plt.title("U_star vs input signal")
            plt.grid()
            plt.legend()
            plt.show()

        variances_accounted_for.append(VAF(real_response, u, show_plot=False))
        print(f"VAF: {np.round(variances_accounted_for[-1], 3)}")

    print(f"Real response vs simulated response in the PRN case: \n Average VAF: {np.round(np.mean(variances_accounted_for), 3)} +- {np.round(np.std(variances_accounted_for), 3)}")


def read_sasha_parameters(ct : constants.SimulationConstants, folder=r'/home/mihai/Thesis/Data/Clean_CSV_data/Sasha_SI_fofuexp/parest/', exp_idx=7, person_idx=0):
    ct.tau_f = np.loadtxt(folder + 'tpf.csv', delimiter=',')[person_idx, exp_idx]
    ct.T_l_f = np.loadtxt(folder + 'Tlf.csv', delimiter=',')[person_idx, exp_idx]
    ct.K_p = np.loadtxt(folder + 'Ke.csv', delimiter=',')[person_idx, exp_idx]
    ct.K_f = np.loadtxt(folder + 'Kf.csv', delimiter=',')[person_idx, exp_idx]
    ct.omega_nms =  np.loadtxt(folder + 'wnm.csv', delimiter=',')[person_idx, exp_idx]
    ct.zeta_nms = np.loadtxt(folder + 'dnm.csv', delimiter=',')[person_idx, exp_idx]
    ct.tau_v = np.loadtxt(folder + 'tv.csv', delimiter=',')[person_idx, exp_idx]

    ct.update_transfer_functions()




def compute_noise_power_metric(v, input_signal, fd_signal, dt):
    """
    Compute the noise power metric Pn for a given signal.
    Parameters:
    v : numpy array - The input signal.
    nd : numpy array -Indices for disturbance frequencies.
    nt : numpy array - Indices for target frequencies.
    dt : float - The time step or sampling interval.
    Returns: float - The noise power metric Pn.
    """
    N = len(v) # Number of points in the signal
   
    frequency_values, fft_values = get_fft_values(v, 1/dt, N)
    
    # remove the DC component
    frequency_values = frequency_values[1:]
    V = fft_values[1:]

    # Compute FFT of the signals
    fft_ft = sp_fft(input_signal)
    fft_fd = sp_fft(fd_signal)

    # Select the relevant portion of the FFT results
    fft_ft = fft_ft[1:N // 2 + 1]
    fft_fd = fft_fd[1:N // 2 + 1]

    Sftft = np.abs((1 / N) * np.conj(fft_ft) * fft_ft)
    Sfdfd = np.abs((1 / N) * np.conj(fft_fd) * fft_fd)

    # Find indices where conditions are met
    n_ft = np.where(Sftft > 1)[0]
    n_fd = np.where(Sfdfd > 1e-6)[0]

    # Calculate variance at target and disturbance primes
    var_v_fd = np.sum((np.abs(V[n_fd]) / (N / 2)) ** 2) / 2
    var_v_ft = np.sum((np.abs(V[n_ft]) / (N / 2)) ** 2) / 2

    # Calculate frequency weighting
    dw = 2 * np.pi / (N * dt)

    # Calculate total variance
    var_v = np.sum(dw * np.abs(V[:-1] * np.conj(V[:-1])) / N * dt) / np.pi

    var_v_n = var_v - var_v_fd - var_v_ft
    Pn = var_v_n / var_v
   
    return Pn


def tune_gain_parameter(ct : constants.SimulationConstants, exp_idx=7, idx=0, person_idx=0, folder=r'Data/Clean_CSV_data/Sasha_SI_fofuexp/PRM/', maxiter=15):
    # iterate over the gain parameter
    
    # 1. simulate multiple realisations of the noise
    # 2. compute the multiple 'u' realisations
    # 3. compute the average P_u (see Dr. Pool's code)
    # 4. Check if P_u computed this way is close to the P_u desired
    # 5. Return the gain parameter that gives the closest P_u_avg to the desired P_u 
    

    input_signals = read_undistracted_data(folder + r'ft.csv')
    real_responses = read_undistracted_data(folder + r'u.csv')
    fd_signals = read_undistracted_data(folder + r'fd.csv')
    
    # for starters, tune only the first person's performance
    
    input_signal = input_signals[idx]
    real_response = real_responses[person_idx]
    fd_signal = fd_signals[person_idx]

    def compute_noise_power_metric_aux(Kn, ct, input_signal, fd_signal, dt, Pn_real, n_realizations=10):
        pn_list = []
        ct.Kn_remnant = Kn
        ct.update_transfer_functions()

        for _ in range(int(n_realizations)):
            
            u, _, _, _, _, input_signal = generate_simulation_signals(ct, input_signal, fd_signal)
            # compute the noise power metric
            Pn = compute_noise_power_metric(u, input_signal, fd_signal, dt)
            pn_list.append(Pn)
        
        print(f"Pn: {np.average(pn_list)}")

        return Pn_real - np.average(pn_list)

    # compute the real pn
    Pn_real = compute_noise_power_metric(real_response, input_signal, fd_signal, ct.dt)

    
    Kn_opt = fmin(compute_noise_power_metric_aux, 0.1, args=(ct, input_signal, fd_signal,  ct.dt, Pn_real), maxiter=maxiter)
    np.savetxt('Kn_opt.csv', Kn_opt, delimiter=',')

    return Kn_opt


def tune_sasha_optimal_kn(folder=r'Data/Clean_CSV_data/Sasha_SI_fofuexp/PRM/'):
    input_signals = read_undistracted_data(folder + r'ft.csv')
    real_responses = read_undistracted_data(folder + r'u.csv')
    fd_signals = read_undistracted_data(folder + r'fd.csv')
    
    ct = constants.SimulationConstants()
    read_sasha_parameters(ct)

    person_idx = 0
    Kn_opt_list = []   

    for idx, input_signal in enumerate(input_signals):
        print(f"{idx=}")
        Kn = tune_gain_parameter(ct, idx=idx, person_idx=person_idx)
        Kn_opt_list.append(Kn)
        
        if idx % 5 == 0 and idx != 0:
            print(f"====================================\n Person number: {person_idx+1}")
            person_idx += 1
            read_sasha_parameters(ct, person_idx=person_idx)

    return Kn_opt_list


def verify_sasha(generate_plots=False, folder=r'Data/Clean_CSV_data/Sasha_SI_fofuexp/PRM/'):

    variances_accounted_for = []
    input_signals = read_undistracted_data(folder + r'ft.csv')
    real_responses = read_undistracted_data(folder + r'u.csv')
    fd_signals = read_undistracted_data(folder + r'fd.csv')
    

    ct = constants.SimulationConstants()
    read_sasha_parameters(ct)

    person_idx = 0

    for idx, input_signal in enumerate(input_signals):
        
        if idx % 5 == 0 and idx != 0:
            print(f"====================================\n Person number: {person_idx+1}")
            person_idx += 1
            read_sasha_parameters(ct, person_idx=person_idx)

        real_response = real_responses[idx]
        fd_signal = fd_signals[idx]

        Kn = tune_gain_parameter(ct, idx=idx, person_idx=person_idx)
        ct.Kn_remnant = Kn
        ct.update_transfer_functions()
        u, u_star, error_signal, f_star, remnant_realisation, input_signal = generate_simulation_signals(ct, input_signal, fd_signal)


        np.savetxt('u_sasha.csv', u, delimiter=',')
        np.savetxt('f_star_sasha.csv', f_star, delimiter=',')
        np.savetxt('input_signal_sasha.csv', input_signal, delimiter=',')
        np.savetxt('u_star_sasha.csv', u_star, delimiter=',')

        if generate_plots:
            plt.plot(input_signal, label="input_signal")
            plt.plot(f_star, label="f_star")
            plt.xlabel("Time step")
            plt.ylabel("Response")
            plt.title("Input signal vs f_star")
            plt.grid()
            plt.legend()
            plt.savefig('input_vs_f_star.png')
            plt.show()

            # Plot real_response and u
            plt.plot(real_response, label="Experimental (real) HO control signal")
            plt.plot(u, label="Simulated HO control signal")
            plt.xlabel("Time step")
            plt.ylabel("Response")
            plt.title("real_response vs simulated response. HO control signal")
            plt.grid()
            plt.legend()
            plt.savefig('real_vs_u.png')
            plt.show()


        variances_accounted_for.append(VAF(real_response[1000:], u[1000:], show_plot=generate_plots))
        print(f"VAF: {np.round(variances_accounted_for[-1], 3)}")

    print(f"Real response vs simulated response in the PRN case: \n Average VAF: {np.round(np.mean(variances_accounted_for), 3)} +- {np.round(np.std(variances_accounted_for), 3)}")


if __name__ == "__main__":
    
    tune_gain_parameters = True
    run_david = False
    run_van_der_el = False
    run_sasha = False

    if tune_gain_parameters:
        print("Tuning the gain parameter")
        ct = constants.SimulationConstants()
        tune_gain_parameter(ct)

    if run_david:
        print("Running David Li's dataset simulation")
        vaf_david_li(generate_plots=False)

    if run_van_der_el:
        print("Running Van der El's dataset simulation")
        vaf_van_der_el(generate_plots=False)

    if run_sasha:
        print("Running Sasha's dataset simulation")
        verify_sasha(generate_plots=False)

 