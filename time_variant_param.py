import systems_functions as sf
import constants
from input_signal import *
import simulate_closed_loop as scl
from simulate_f_star import generate_f_star


def generate_simulation_signals(ct, df_name='plaetinck_params.csv'):
    """
    Generate the simulation signals for the closed-loop system.
    :param ct: SimulationConstants, simulation constants
    :param df_name: str, name of the DataFrame
    :return: tuple, simulation signals
    """

    # import the input signal from input_signal.csv
    input_signal = generate_multi_sign_from_df(ct, df_name=df_name).reshape(-1, 1)
    f_star = generate_f_star(input_signal, ct)
    remnant_realisation, _ = scl.generate_remnant_realisation(len(f_star), ct)

    u, u_star, error_signal = scl.simulate_closed_loop_dynamics(f_star, ct.H_comb, ct.H_ce, remnant_realisation, ct)

    u, u_star, error_signal, f_star, remnant_realisation = sf.remove_transient_behaviour_array(
        [u, u_star, error_signal, f_star, remnant_realisation], ct)

    return u, u_star, error_signal, f_star, remnant_realisation


if __name__ == "__main__":
    # get the simulation constants
    ct = constants.SimulationConstants()
    df_name = 'plaetinck_params.csv'

    # import the input signal from input_signal.csv
    u, u_star, error_signal, f_star, remnant_realisation = generate_simulation_signals(ct, df_name)
    scl.plot_time_response(f_star, u_star, "f_star", "u_star")  