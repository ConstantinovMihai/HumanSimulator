import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Function to plot an ellipse based on covariance matrix
def plot_cov_ellipse(cov, pos, nstd=1, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the covariance matrix `cov`
    at position `pos`.
    """
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Get the angle for the ellipse
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    
    # Width and height of the ellipse
    width, height = 2 * nstd * np.sqrt(eigvals)
    
    # Draw the ellipse
    ellipse = Ellipse(xy=pos, width=width, height=height, angle=angle, **kwargs)
    if ax is None:
        ax = plt.gca()
    ax.add_patch(ellipse)

# Parameters for the bivariate normal distribution
mean = np.array([0, 0])
covariance = np.array([[3, 1], [1, 2]])

# Create the plot
fig, ax = plt.subplots()

# Plot the 95% confidence ellipse (safe zone limit)
plot_cov_ellipse(covariance, mean, nstd=2.4477, ax=ax, 
                 edgecolor='blue', linestyle='-', linewidth=1.5, alpha=0.7, 
                 label="Safe zone limit")

# Plot the 68% confidence ellipse (restart limit)
plot_cov_ellipse(covariance, mean, nstd=2, ax=ax, 
                 edgecolor='lightcoral', linestyle='--', linewidth=1.5, alpha=0.7, 
                 label="Restart limit")

# Formatting the plot
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Adding legend
ax.legend()

# Show the plot
plt.grid(True, alpha=0.3)
plt.title('Example of confidence ellipses')
plt.show()
