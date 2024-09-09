import numpy as np
import matplotlib.pyplot as plt

def sigmoid_schedule(t, P1, P2, M, G):
    """
    Calculates the time-varying parameter P(t) using a sigmoid scheduling function.

    Args:
        t (float or np.ndarray): Time or array of times at which to evaluate the function.
        P1 (float): Initial parameter value.
        P2 (float): Final parameter value.
        M (float): Time of maximum rate of change.
        G (float): Maximum rate of change.

    Returns:
        np.ndarray: Parameter values corresponding to each time point in t.
    """
    return P1 + (P2 - P1) / (1 + np.exp(-G * (t - M)))


if __name__ == "__main__":
    # Example usage
    t = np.linspace(0, 10, 100)  # Time points from 0 to 10
    P1 = 0.5  # Initial parameter value
    P2 = 1.5  # Final parameter value
    M = 5.0   # Time of maximum rate of change
    G = 1.0   # Maximum rate of change

    # Calculate time-varying parameter P(t)
    P_t = sigmoid_schedule(t, P1, P2, M, G)

    plt.plot(t, P_t)
    plt.grid()
    plt.show()

    # Print or use P_t as needed
    print("Parameter values at different times:")
    print(P_t)
