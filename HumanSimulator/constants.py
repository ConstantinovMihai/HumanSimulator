from dataclasses import dataclass, field
import numpy as np
import systems_functions as sf  
import sympy as sp


# @dataclass(frozen=True)
class SimulationConstants:

    # # System parameters
    tau_f: float = 0.54 # [s]  
    w_b : float =  14.09  # [rad/s]
    K_p : float = 1.28 # [-]
    K_f: float = 1.01 # [-]
    omega_nms: float = 10.72 # [rad/s]
    zeta_nms: float = 0.19 # [-]
    tau_v: float =  0.18 # [s]
    T_l_f : float = 1/w_b # [s]
    tau_s = 0.9 # [s]
    K_v = 0
    # tau_s = 2 # [s]
    # tau_f_star = tau_s - tau_f # apparent time delay seconds (s)
    
    # Remnant model parameters
    Kn_remnant : float = 0
    Tn_remnant : float = 1

    # Simulation parameters
    dt : float = 0.01  # seconds (s)
    t_total: float = 200  # seconds (s)
    t_measurement: float = 100 # seconds (s)
    settling_time : float = 0.9 # seconds (s)
    
    fs : float = 1/dt # Hz

    def __init__(self):
        self.update_transfer_functions()

    def update_transfer_functions(self):
        self.discard_samples = int(self.tau_s / self.dt) # number of samples to discard due to the look ahead time
        self.tau_f_star = self.tau_s - self.tau_f # apparent time delay seconds (s)
        # Define the Pade delay transfer function components
        self.numerator_pade = np.array([-1, 12, -60, 120])  # Coefficients of the numerator
        self.denominator_pade = np.array([1, 12, 60, 120])  # Coefficients of the denominator

        # Define the Pade delay transfer function as a dictionary
        self.tau_f_vector = np.array([self.tau_f_star**3, self.tau_f_star**2, self.tau_f_star, 1])
        self.tau_v_vector = np.array([self.tau_v**3, self.tau_v**2, self.tau_v, 1])
        
        self.H_pade_f = {
            "num": self.numerator_pade * self.tau_f_vector,
            "den": self.denominator_pade * self.tau_f_vector,
            "d0": np.array([0])
        }

        self.H_pade_v = {
            "num": self.numerator_pade * self.tau_v_vector,
            "den": self.denominator_pade * self.tau_v_vector,
            "d0": np.array([0])
        }

        # remnant filter 
        self.H_remnant = {"num": np.array([self.Kn_remnant]), "den": np.array([self.Tn_remnant, 1]), "d0": np.array([0])} 

        # neuromuscular dynamics
        self.H_nms = {"num": np.array([self.omega_nms**2]), "den": np.array([1, 2*self.zeta_nms*self.omega_nms, self.omega_nms**2]), "d0": np.array([0])}

        # Define the transfer function
        self.H_of = {"num": np.array([self.K_f]), "den": np.array([self.T_l_f, 1]), "d0" : np.array([0])}
        self.H_ce  = {"num": np.array([1.5]), "den": np.array([1, 0]), "d0": np.array([0])}
        
        self.H_vp = {"num": np.array([self.K_v, self.K_p]), "den": np.array([1]), "d0": np.array([0])}
        
        # obtained the combined transfer function by adding the delays
        self.H_comb = sf.multiply_transfer_functions(self.H_vp, self.H_nms)
        self.H_comb = sf.multiply_transfer_functions(self.H_comb, self.H_pade_v)


@dataclass(frozen=True)
class SimulationConstantsVanderEl(): 
    # System parameters
    K_p : float = 0.23 # [-]
    tau_v: float =  0.33 # [s]
    omega_nms: float = 10.40 # [rad/s]
    zeta_nms: float = 0.24 # [-]
    T_l_e = 1.76 # [s]
    K_f: float = 1.07 # [-]
    tau_f: float = 1.27 # [s]  
    T_l_f : float = 0.95 # [s] 
    K_v = K_p * T_l_e   
    tau_s = 2 # [s]
    tau_f_star = tau_s - tau_f # apparent time delay seconds (s)
    
    # Remnant model parameters
    Kn_remnant : float = 0.03
    Tn_remnant : float = 0.23

    # Simulation parameters
    dt : float = 0.01  # seconds (s)
    t_total: float = 200  # seconds (s)
    t_measurement: float = 100 # seconds (s)
    settling_time : float = 2 # seconds (s)
    discard_samples = int(tau_s / dt) # number of samples to discard due to the look ahead time
    
    fs : float = 1/dt # Hz

    # Define the Pade delay transfer function components
    numerator_pade = np.array([-1, 12, -60, 120])  # Coefficients of the numerator
    denominator_pade = np.array([1, 12, 60, 120])  # Coefficients of the denominator

    # Define the Pade delay transfer function as a dictionary
    tau_f_vector = np.array([tau_f_star**3, tau_f_star**2, tau_f_star, 1])
    tau_v_vector = np.array([tau_v**3, tau_v**2, tau_v, 1])

    H_pade_f = {
        "num": numerator_pade * tau_f_vector,
        "den": denominator_pade * tau_f_vector,
        "d0": np.array([0])
    }

    H_pade_v = {
        "num": numerator_pade * tau_v_vector,
        "den": denominator_pade * tau_v_vector,
        "d0": np.array([0])
    }

    # remnant filter 
    H_remnant = {"num": np.array([Kn_remnant]), "den": np.array([Tn_remnant, 1]), "d0": np.array([0])} 

    # neuromuscular dynamics
    H_nms = {"num": np.array([omega_nms**2]), "den": np.array([1, 2*zeta_nms*omega_nms, omega_nms**2]), "d0": np.array([0])}

    # Define the transfer function
    H_of = {"num": np.array([K_f]), "den": np.array([T_l_f, 1]), "d0" : np.array([0])}

    H_ce = {"num": np.array([5]), "den": np.array([1, 0, 0]), "d0": np.array([0])}
    #H_ce  = {"num": np.array([1.5]), "den": np.array([1, 0]), "d0": np.array([0])}
    
    H_vp = {"num": np.array([K_v, K_p]), "den": np.array([1]), "d0": np.array([0])}
    
    # obtained the combined transfer function by adding the delays
    H_comb = sf.multiply_transfer_functions(H_vp, H_nms)
    H_comb = sf.multiply_transfer_functions(H_comb, H_pade_v)

