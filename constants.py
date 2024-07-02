from dataclasses import dataclass
import numpy as np
import systems_functions as sf  

@dataclass(frozen=True)
class SimulationConstants:
    # CE model parameters
    H_ce_K: float = 5  
    T_L_e: float = 0.3  # seconds (s)
    w_b : float = 13  # rad/s

    # Human operator model parameters
    omega_nms: float = 10.5 # rad/s
    zeta_nms: float = 0.2 
    K_f: float = 1
    K_p : float = 1
    K_v : float = 0
    T_l_f: float = 1  # seconds (s)
    tau_f: float = 0.5 #0.5 # seconds (s)  
    tau_v: float = 0 #0.4 # seconds (s) 
    tau_s : float = 0.5 # seconds (s)
    tau_f_star = tau_s + tau_f # aparent time delay seconds (s)

    # Simulation parameters
    dt : float = 0.01  # seconds (s)
    t_total: float = 200  # seconds (s)
    t_measurement: float = 100 # seconds (s)
    settling_time : float = 2 # seconds (s)
    discard_samples = int(tau_s / dt) # number of samples to discard due to the look ahead time
    
    # Remnant model parameters
    Kn_remnant : float = 0.205
    Tn_remnant : float = 0.2

    fs : float = 1/dt # Hz

    # Define the Pade delay transfer function components
    numerator_pade = np.array([-1, 12, -60, 120])  # Coefficients of the numerator
    denominator_pade = np.array([1, 12, 60, 120])  # Coefficients of the denominator

    # Define the Pade delay transfer function as a dictionary
    H_pade_f = {
        "num": numerator_pade * tau_f_star ** np.arange(3, -1, -1),
        "den": denominator_pade * tau_f_star ** np.arange(3, -1, -1),
        "d0": np.array([0])
    }

    H_pade_v = {
        "num": numerator_pade * tau_v ** np.arange(3, -1, -1),
        "den": denominator_pade * tau_v ** np.arange(3, -1, -1),
        "d0": np.array([0])
    }

    # remnant filter 
    H_remnant = {"num": np.array([Kn_remnant]), "den": np.array([Tn_remnant, 1]), "d0": np.array([0])} 

    # neuromuscular dynamics
    H_nms = {"num": np.array([omega_nms**2]), "den": np.array([1, 2*zeta_nms*omega_nms, omega_nms**2]), "d0": np.array([0])}

    # Define the transfer function
    H_of = {"num": np.array([K_f*w_b]), "den": np.array([1, w_b]), "d0" : np.array([0])}
    H_ce = {"num": np.array([1.5]), "den": np.array([1, 0]), "d0": np.array([0])}

    H_vp = {"num": np.array([K_v, K_p]), "den": np.array([1]), "d0": np.array([0])}
    
    # obtained the combined transfer function by adding the delays
    H_comb = sf.multiply_transfer_functions(H_vp, H_nms)
    H_comb = sf.multiply_transfer_functions(H_comb, H_pade_v)
        