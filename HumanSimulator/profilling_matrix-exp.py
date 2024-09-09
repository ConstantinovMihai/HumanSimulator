import sympy as sp
import cProfile
import pstats
import scipy.linalg
import time
import numpy as np
from simpy_tests import *
import scipy

def compute_matrix_exponential(aux=False):
    # Define the symbolic variable
    w_b = sp.symbols('w_b')

    # Define the matrix em
    em = sp.Matrix([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [-960.0, -240.0 - 960.0, -24.0 - 240.0, -1.0 - 24.0, 1],
        [0, 0, 0, 0, 0]
    ])

    if aux:
        em = sp.Matrix([
            [0, 1, 0, 0,],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-960.0*w_b, -240.0*w_b - 960.0, -24.0*w_b - 240.0, -1.0*w_b - 24.0, 1],
            [0, 0, 0, 0]
        ])

    # Define dt
    dt = 0.01

    # Compute dt * em
    dt_em = dt * em
    print(f"dt*em = {dt_em}")

    # Compute the matrix exponential
    ms = sp.exp(dt_em)
    
    # Simplify the result
    ms_simplified = sp.simplify(ms)

    # Print the matrix exponential
    print("Matrix exponential:")
    print(ms_simplified)


def compute_matrix_exponential_numeric():
    # Define the symbolic variable
    w_b = sp.symbols('w_b')

    # Define the matrix em
    em = sp.Matrix([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [-960.0*w_b, -240.0*w_b - 960.0, -24.0*w_b - 240.0, -1.0*w_b - 24.0, 1],
        [0, 0, 0, 0, 0]
    ])

    # Define dt
    dt = 0.01

    # Compute dt * em
    dt_em = dt * em
    print(f"dt*em = {dt_em}")

    # Convert the symbolic matrix to a numerical matrix with a specific value for w_b
    w_b_value = 1  # You can set this to any value you need
    dt_em_num = dt_em.evalf(subs={w_b: w_b_value})
    dt_em_num_np = np.array(dt_em_num).astype(np.float64)

    # Compute the matrix exponential numerically
    start_time = time.time()
    ms_num = scipy.linalg.expm(dt_em_num_np)
    end_time = time.time()

    print("Matrix exponential computed numerically:")
    print(ms_num)
    print(f"Time taken for numerical computation: {end_time - start_time} seconds")


def run_and_profile_discretization(A, B, C, D, dt, nb_runs=10000, use_scipy=True):
    """
    Run and profile the discretization of the given matrices.

    Parameters:
    A, B, C, D (array-like): State space representation matrices.
    dt (float): Sampling time.
    nb_runs (int): Number of times to run the discretization for profiling.
    use_scipy (bool): If True, use scipy for discretization. Otherwise, use custom discretization.
    """
    def run_scipy_discretization(A, B, C, D, dt, nb_runs):
        for _ in range(nb_runs):
            (Ad, Bd, Cd, Dd, dt_new) = scipy.signal.cont2discrete((A, B, C, D), dt)
        return Ad, Bd, Cd, Dd

    def run_own_discretization(A, B, C, D, dt, nb_runs):
        for _ in range(nb_runs):
            Ad, Bd, Cd, Dd = sf.cont2discrete_zoh(dt, A, B, C, D)
        return Ad, Bd, Cd, Dd

    # Select the function to use for discretization
    func = run_scipy_discretization if use_scipy else run_own_discretization

    # Profile the selected function
    pr = cProfile.Profile()
    pr.enable()
    result = func(A, B, C, D, dt, nb_runs)
    pr.disable()

    # Print profiling results
    ps = pstats.Stats(pr).sort_stats(pstats.SortKey.TIME)
    ps.print_stats(100)  # Print top 100 time-consuming calls

    return result

if __name__ == "__main__":
    # Profile the function
    pr = cProfile.Profile()
    ct = SimulationConstants()
    ct.initialize_transfer_functions()

    A, B, C, D = sf.convert_tf_ss_ccf_sympy(ct.H_of_del)
    [A_num, B_num, C_num, D_num] = ct.evaluate_matrices([A, B, C, D])

    print(f"condition number A {np.linalg.cond(A_num)}")
    print(f"condition number B {np.linalg.cond(B_num)}")
    print(f"condition number C {np.linalg.cond(C_num)}")
    print(f"condition number D {np.linalg.cond(D_num)}")

    print(f"H {ct.H_of_del}")
    print(ct.evaluate_transfer_function(ct.H_of_del))
    
    print("Scipy implementation for discretization:")
    result_scipy = run_and_profile_discretization(A_num, B_num, C_num, D_num, 0.01, nb_runs=100000, use_scipy=True)

    print("Own implementation for discretization:")
    result_own = run_and_profile_discretization(A_num, B_num, C_num, D_num, 0.01, nb_runs=100000, use_scipy=False)

    # numeric = False
    # pr.enable() 
    # compare_discretization_functions(A_num, B_num, C_num, D_num, dt=0.01)
    # # if numeric:
    # #     compute_matrix_exponential_numeric()
    # # else:
    # #     compute_matrix_exponential()
    # pr.disable()

    # # Print profiling results
    # ps = pstats.Stats(pr).sort_stats(pstats.SortKey.TIME)
    # ps.print_stats(100)  # Print top 10 time-consuming calls

