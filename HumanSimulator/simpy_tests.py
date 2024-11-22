import sympy as sp
from typing import Dict
import systems_functions as sf
import numpy as np
from dataclasses import dataclass, field

@dataclass()
class SimulationConstants:
    # Define symbolic variables
    H_ce_K, T_L_e, w_b, omega_nms, zeta_nms, K_f, K_p, K_v, T_l_f, tau_f, tau_v, Kn_remnant, Tn_remnant = sp.symbols(
        'H_ce_K T_L_e w_b omega_nms zeta_nms K_f K_p K_v T_l_f tau_f tau_v Kn_remnant Tn_remnant')
    
    # Assign actual numerical values to the symbolic variables
    H_ce_K_val = 5  
    T_L_e_val = 0.3  # seconds (s)
    w_b_val = 13  # rad/s
    omega_nms_val = 10.5  # rad/s
    zeta_nms_val = 0.2 
    K_f_val = 1
    K_p_val = 1
    K_v_val = 0
    T_l_f_val = 1  # seconds (s)
    tau_f_val = 0.5  # seconds (s)  
    tau_v_val = 0  # seconds (s) 
    tau_s = 1  # seconds (s)
    dt = 0.01  # seconds (s)
    t_total = 240  # seconds (s)
    t_measurement = 120  # seconds (s)
    settling_time = 2  # seconds (s)
    Kn_remnant_val = 0.205
    Tn_remnant_val = 0.2
    discard_samples = int(tau_s / dt)
    tau_f_star_val = tau_s - tau_f_val  # apparent time delay seconds (s)
    fs = 1 / dt  # Hz
    H_pade_f = None
    H_pade_v = None
    H_of_del = None
    

    @classmethod
    def define_pade_delay(cls):
        # Define the Pade delay transfer function components
        numerator_pade = sp.Array([-1, 12, -60, 120])  # Coefficients of the numerator
        denominator_pade = sp.Array([1, 12, 60, 120])  # Coefficients of the denominator

        # Define the Pade delay transfer function as a dictionary
        H_pade_f = {
            "num": sp.Array([numerator_pade[i] * cls.tau_f_star_val**(len(numerator_pade) - 1 - i) for i in range(len(numerator_pade))]),
            "den": sp.Array([denominator_pade[i] * cls.tau_f_star_val**(len(denominator_pade) - 1 - i) for i in range(len(denominator_pade))]),
            "d0": sp.Array([0])
        }

        H_pade_v = {
            "num": sp.Array([numerator_pade[i] * cls.tau_v_val**(len(numerator_pade) - 1 - i) for i in range(len(numerator_pade))]),
            "den": sp.Array([denominator_pade[i] * cls.tau_v_val**(len(denominator_pade) - 1 - i) for i in range(len(denominator_pade))]),
            "d0": sp.Array([0])
        }

        return H_pade_f, H_pade_v

    @classmethod
    def initialize_transfer_functions(cls):
        cls.H_pade_f, cls.H_pade_v = cls.define_pade_delay()
        
        # Define other transfer functions
        cls.H_remnant = {"num": sp.Array([cls.Kn_remnant]), "den": sp.Array([cls.Tn_remnant, 1]), "d0": sp.Array([0])} 

        cls.H_nms = {"num": sp.Array([cls.omega_nms_val**2]), "den": sp.Array([1, 2*cls.zeta_nms_val*cls.omega_nms_val, cls.omega_nms_val**2]), "d0": sp.Array([0])}

        cls.H_of = {"num": sp.Array([cls.K_f*cls.w_b]), "den": sp.Array([1, cls.w_b]), "d0": sp.Array([0])}
        cls.H_ce = {"num": sp.Array([1.5]), "den": sp.Array([1, 0]), "d0": sp.Array([0])}

        cls.H_vp = {"num": sp.Array([cls.K_v, cls.K_p]), "den": sp.Array([1]), "d0": sp.Array([0])}

        # Combined transfer function by adding the delays
        cls.H_comb = sf.multiply_transfer_functions_sympy(cls.H_vp, cls.H_nms)
        cls.H_comb = sf.multiply_transfer_functions_sympy(cls.H_comb, cls.H_pade_v)
        cls.H_of_del = sf.multiply_transfer_functions_sympy(cls.H_of, cls.H_pade_f)

    def evaluate_transfer_function(self, H): 
        # Create a dictionary mapping symbolic variables to their values
        substitutions = {
            self.H_ce_K: self.H_ce_K_val,
            self.T_L_e: self.T_L_e_val,
            self.w_b: self.w_b_val,
            self.omega_nms: self.omega_nms_val,
            self.zeta_nms: self.zeta_nms_val,
            self.K_f: self.K_f_val,
            self.K_p: self.K_p_val,
            self.K_v: self.K_v_val,
            self.T_l_f: self.T_l_f_val,
            self.tau_f: self.tau_f_val,
            self.tau_v: self.tau_v_val,
            self.Kn_remnant: self.Kn_remnant_val,
            self.Tn_remnant: self.Tn_remnant_val,
        }
      
        # Evaluate num, den, and d0 arrays
        num_eval = [sp.Array([term.subs(substitutions).evalf() for term in H["num"]])]
        den_eval = [sp.Array([term.subs(substitutions).evalf() for term in H["den"]])]
        #d0_eval = [sp.Array([term.subs(substitutions).evalf() for term in H["d0"]])]
        
        return {"num": num_eval, "den": den_eval, "d0": sp.Matrix([0])}

    # Method to evaluate matrices
    def evaluate_matrices(self, matrices):
        # Create a dictionary mapping symbolic variables to their values
        substitutions = {
            self.H_ce_K: self.H_ce_K_val,
            self.T_L_e: self.T_L_e_val,
            self.w_b: self.w_b_val,
            self.omega_nms: self.omega_nms_val,
            self.zeta_nms: self.zeta_nms_val,
            self.K_f: self.K_f_val,
            self.K_p: self.K_p_val,
            self.K_v: self.K_v_val,
            self.T_l_f: self.T_l_f_val,
            self.tau_f: self.tau_f_val,
            self.tau_v: self.tau_v_val,
            self.Kn_remnant: self.Kn_remnant_val,
            self.Tn_remnant: self.Tn_remnant_val,
        }

          # Evaluate each matrix and convert to numeric if necessary
        evaluated_matrices = []
        for matrix in matrices:
            
            evaluated_matrix = matrix.subs(substitutions).evalf()
            if isinstance(evaluated_matrix, sp.Matrix):

                evaluated_matrix = np.array(evaluated_matrix.tolist(), dtype=np.float64)
            evaluated_matrices.append(evaluated_matrix)
        
        return evaluated_matrices

if __name__ == "__main__":
    # Initialize the transfer functions
    ct = SimulationConstants()
    ct.initialize_transfer_functions()

    # Define the matrices
    Ad = sp.Matrix([[1.0 * sp.exp(-0.01 * SimulationConstants.w_b)]])
    Bd = sp.Matrix([[1.0 / SimulationConstants.w_b - 1.0 * sp.exp(-0.01 * SimulationConstants.w_b) / SimulationConstants.w_b]])
    Cd = sp.Matrix([[SimulationConstants.K_f * SimulationConstants.w_b]])
    Dd = sp.Matrix([[0]])

    # Store the matrices in a list
    matrices = [Ad, Bd, Cd, Dd]

    # Evaluate the matrices numerically using the method in the class
    evaluated_matrices = ct.evaluate_matrices(matrices)

    # Print the results
    for i, matrix in enumerate(evaluated_matrices):
        print(f"Matrix {i+1}:")
        sp.pprint(matrix)
        print()
