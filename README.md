# Human Simulator

This project contains a Python implementation of Van der El's cybernetic model. For detailed information on the model, please refer to [this article](https://www.sciencedirect.com/science/article/pii/S2405896316320602). The simulation generates time traces of control signals based on:

1. The parameters of the multisine signals, stored in a DataFrame.
2. The simulation parameters, stored in a dataclass (e.g., `constants.py`).

## Main Files

1. **bode.py**: Contains routines for systems with state-space representation.
2. **constants.py**: Defines the constants needed for the simulation within a dataclass.
3. **input_signal.py**: Generates the multisine signal based on parameters in a DataFrame and includes testing routines for Fourier analysis.
4. **simulate_closed_loop.py**: Implements and performs open-loop identification on the closed-loop part of the model.
5. **simulate_f_star.py**: Implements and performs tests on the open-loop part of the model.
6. **systems_functions.py**: Contains system theory functions implemented using bare numpy.
7. **time_variant.py**: The main file to run the simulation, combining both open and closed-loop parts to produce the desired signals.
8. **unit_test.py**: Contains unit tests for the simulation.
9. **verification_closed_loop.py**: Contains tests for the closed-loop part of the model.
