import numpy as np
import matplotlib.pyplot as plt
from tsai.all import *
import sklearn.preprocessing as pp
import scipy as sp
import scipy.spatial.distance as distance
from sklearn.preprocessing import StandardScaler
from utils import *
from scipy.stats import norm, multivariate_normal, chi2


class MultivariateNormalDistribution:
    def __init__(self, mean, covariance):
        self.mean = mean
        self.cov = covariance
        self.inv_cov = sp.linalg.inv(covariance)
    

def fit_gaussian(data):
    """
    Fit a multivariate Gaussian to the data.
    
    Parameters:
    data (array-like): 2D array of shape (n_samples, n_features).
    
    Returns:
    tuple: Mean vector and covariance matrix of the Gaussian.
    """
    mean = np.mean(data, axis=0)
    cov_val = np.cov(data, rowvar=False)
    return mean, cov_val


def plot_gaussian_1d(data, mean, std_dev, color='red', title='1D Gaussian Fit'):
    """
    Plot a 1D Gaussian distribution on top of the data histogram.
    
    Parameters:
    data (array-like): The original data used to fit the Gaussian.
    mean (float): Mean of the Gaussian.
    std_dev (float): Standard deviation of the Gaussian.
    color (str): Color for the Gaussian curve.
    """
    # Create a range of x values
    x = np.linspace(min(data), max(data), 1000)
    
    # Get the corresponding y values for the Gaussian PDF
    gaussian_pdf = norm.pdf(x, mean, std_dev)

    # Plot the data histogram
    plt.hist(data, bins=30, density=True, alpha=0.6, color='blue')
    
    # Plot the Gaussian curve
    plt.plot(x, gaussian_pdf, color=color, lw=2)
    plt.title(title)
    plt.xlabel('Data')
    plt.ylabel('Probability Density')
    plt.show()


def compute_mahalanobis_z_squared(sample, mean, cov):
    """
    Computes the Mahalanobis distance (equivalent to multivariate z score)
    for a given sample with respect to a multivariate normal distribution.

    Parameters:
    - sample: The data point (1D array) for which to compute the Mahalanobis distance.
    - mean: The mean of the multivariate normal distribution (1D array).
    - cov: The covariance matrix of the multivariate normal distribution (2D array).

    Returns:
    - Mahalanobis distance (score).
    """
    # Create the multivariate normal distribution object
    mvn = multivariate_normal(mean=mean, cov=cov)

    # Compute the Mahalanobis distance
    mahalanobis_distance = mvn.mahalanobis(sample)

    # Return the squared Mahalanobis distance (equivalent to z^2)
    return mahalanobis_distance


def process_data_array(signal, runs_per_person=4):
    # List for storing grouped arrays (4 per group) for error_signal
    processed_signal_list = []

    # Loop through error_signal in steps of 4
    for i in range(0, signal.shape[1], runs_per_person):
        # Take 4 columns at a time and append them as a list to error_not_distracted_list

        signal_group = np.array([signal[:, j] for j in range(i, min(i + runs_per_person, signal.shape[1]))])
        processed_signal_list.append(signal_group)

    return np.array(processed_signal_list)


def read_process_data(error_signal_path=r'/home/mihai/Thesis/Data/Clean_CSV_data/updated_data/PRN/e.csv', 
                        u_signal_path=r'/home/mihai/Thesis/Data/Clean_CSV_data/updated_data/PRN/u.csv', 
                    x_signal_path=r'/home/mihai/Thesis/Data/Clean_CSV_data/updated_data/PRN/x.csv',
                tc_path=r'/home/mihai/Thesis/Data/Clean_CSV_data/updated_data/PRN/mdist.csv'):
    # Load data
    error_signal = np.loadtxt(error_signal_path, delimiter=',')
    u_signal = np.loadtxt(u_signal_path, delimiter=',')
    x_signal = np.loadtxt(x_signal_path, delimiter=',')

    tc = np.loadtxt(tc_path, delimiter=',')
    
    # Process data
    processed_error_signal = process_data_array(error_signal)
    processed_u_signal = process_data_array(u_signal)
    processsed_x_signal = process_data_array(x_signal)
    processed_tc = process_data_array(tc)

    return processed_error_signal, processed_u_signal, processsed_x_signal, processed_tc


def compute_priors(data_signal, write_to_file=False):
    # first, go through each person and compute the priors
    
    prior_pdf_list = []
    
    for person_idx in range(len(data_signal)):
        # combine the entire list in the priors
        data_prior = np.concatenate(data_signal[person_idx])
        mean_prior, cov_prior = fit_gaussian(data_prior)
        prior_pdf_list.append(MultivariateNormalDistribution(mean_prior, cov_prior))

    if write_to_file:
        for idx, prior_pdf in enumerate(prior_pdf_list):
            np.savetxt(f'prior_pdf_{idx}.csv', prior_pdf)

    return prior_pdf_list


def run_detector(data_signal, prior_pdf_list, runs_per_person=4, write_to_file=False):
    distances_from_prior_per_run = []


    for person_idx in range(data_signal.shape[0]):
        print(f"Processing person {person_idx + 1}...")
        
        mean_prior = prior_pdf_list[person_idx].mean
        inv_cov_prior = prior_pdf_list[person_idx].inv_cov
        

        # Iterate through each run for the current person
        for run_idx in range(runs_per_person):
            data_run = data_signal[person_idx][run_idx]
            
            # Compute Mahalanobis distances for each sample in the run
            distances_from_prior = np.array([distance.mahalanobis(sample, mean_prior, inv_cov_prior) for sample in data_run])
            np.savetxt(f'distances_{person_idx}_{run_idx}.csv', distances_from_prior, delimiter=',')
            # Append to the list of distances
            distances_from_prior_per_run.append(distances_from_prior)

    distances_from_prior_per_run = np.vstack(distances_from_prior_per_run)

    if write_to_file:
        np.savetxt('MD_distances_per_run.csv', distances_from_prior_per_run, delimiter=',')

    return distances_from_prior_per_run

def compute_changepoints(tc_signal):
    cps_array = []
    for col_idx in range(tc_signal.shape[1]):
        cps =  np.where(np.diff(tc_signal[:,col_idx], prepend=np.nan) == 1)[0]
        cps_array.append(cps)

    return cps_array


def compute_changepoints_list(tc_signal_all_runs):
    cps_list = []
    for tc_signal in tc_signal_all_runs:
        cps_list.append(compute_changepoints(tc_signal))

    return cps_list


def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return np.loadtxt(file_path, delimiter=',')


def process_prior_data(error_file, u_file, x_file):
    """
    Load and process prior data (error and u signals).
    """
    error_prior = load_data(error_file)
    u_prior = load_data(u_file)
    x_prior = load_data(x_file)

    processed_error_prior = process_data_array(error_prior)
    processed_u_prior = process_data_array(u_prior)
    processed_x_prior = process_data_array(x_prior)

    data_signal_prior = np.stack((processed_error_prior, processed_u_prior, processed_x_prior), axis=-1)
    data_signal_prior = normalize_data_signal(data_signal_prior)

    return data_signal_prior


def process_test_data(error_file, u_file, x_file, tc_file):
    """
    Load and process test data (error, u signals and changepoints).
    """
    error_signal_test = load_data(error_file)
    u_signal_test = load_data(u_file)
    x_signal_test = load_data(x_file)

    tc_test = load_data(tc_file)

    cps_test = compute_changepoints(tc_test)
    
    processed_error_test = process_data_array(error_signal_test)
    processed_u_test = process_data_array(u_signal_test)
    processed_x_test = process_data_array(x_signal_test)

    processed_tc_test = process_data_array(tc_test)

    data_signal_test = np.stack((processed_error_test, processed_u_test, processed_x_test), axis=-1)
    data_signal_test = normalize_data_signal(data_signal_test)

    return data_signal_test, processed_tc_test, cps_test


def run_analysis(prior_error_file, prior_u_file, prior_x_file, test_error_file, test_u_file, test_x_file, test_tc_file):
    """
    Run the entire analysis process:
    - Load and process prior data
    - Compute priors
    - Load and process test data
    - Run the detector
    """
    # Process prior data and compute priors
    data_signal_prior = process_prior_data(prior_error_file, prior_u_file, prior_x_file)
    pdf_list = compute_priors(data_signal_prior)
    

    # Process test data and run detector
    data_signal_test, tc_test, cps_test = process_test_data(test_error_file, test_u_file, test_x_file, test_tc_file)

    run_detector(data_signal_test, pdf_list, write_to_file=True)


def generate_data_signal_with_derivatives(error_file, u_file, x_file):
    error_signal = load_data(error_file)
    u_signal = load_data(u_file)
    x_signal = load_data(x_file)

    processed_error = process_data_array(error_signal)
    processed_u = process_data_array(u_signal)
    processed_x = process_data_array(x_signal)

    error_prior_derivative = np.diff(processed_error, axis=2)
    u_prior_derivative = np.diff(processed_u, axis=2)
    x_prior_derivative = np.diff(processed_x, axis=2)

    velocity_trajectory = np.sqrt(error_prior_derivative**2 + u_prior_derivative**2 + x_prior_derivative**2)

    data_signal_prior = np.stack((processed_error[:,:,1:], processed_u[:,:,1:], processed_x[:,:,1:],
                                  velocity_trajectory), axis=-1)

    data_signal_prior = normalize_data_signal(data_signal_prior)

    return data_signal_prior
    

def compute_priors_with_derivatives(error_file, u_file, x_file):
    """
    Load and process prior data (error and u signals).
    """
   
    data_signal_prior = generate_data_signal_with_derivatives(error_file, u_file, x_file)

    pdf_priors = compute_priors(data_signal_prior, write_to_file=False)

    return pdf_priors


def run_detector_with_derivatives(error_file, u_file, x_file, pdf_priors):

    data_signal_test = generate_data_signal_with_derivatives(error_file, u_file, x_file)
    data_signal_test = normalize_data_signal(data_signal_test)
    distances = run_detector(data_signal_test, pdf_priors, runs_per_person=4, write_to_file=False)

    return distances


if __name__ == '__main__':
    # File paths for prior and test data
    prior_error_file = r'/home/mihai/Thesis/Data/Clean_CSV_data/updated_data/PRN/e.csv'
    prior_u_file = r'/home/mihai/Thesis/Data/Clean_CSV_data/updated_data/PRN/u.csv'
    prior_x_file = r'/home/mihai/Thesis/Data/Clean_CSV_data/updated_data/PRN/x.csv'

    test_error_file = r'/home/mihai/Thesis/Data/Clean_CSV_data/updated_data/PRDPE/e.csv'
    test_u_file = r'/home/mihai/Thesis/Data/Clean_CSV_data/updated_data/PRDPE/u.csv'
    test_x_file = r'/home/mihai/Thesis/Data/Clean_CSV_data/updated_data/PRDPE/x.csv'
    test_tc_file = r'/home/mihai/Thesis/Data/Clean_CSV_data/updated_data/PRDPE/mdist.csv'

    prior_without_derivatives = True
    prior_with_derivatives = False

    if prior_without_derivatives:
        # Process test data and run detector
        data_signal_test, processed_tc_test, cps_test = process_test_data(test_error_file, test_u_file, test_x_file, test_tc_file)
        # find the total number of changepoints
        print(f"Total number of changepoints: {sum([len(cps) for cps in cps_test])}")

        # Run the analysis
        run_analysis(prior_error_file, prior_u_file, prior_x_file, test_error_file, test_u_file, test_x_file, test_tc_file)
    
    if prior_with_derivatives:
        pdf_priors = compute_priors_with_derivatives(prior_error_file, prior_u_file, prior_x_file) 
        md_distances = run_detector_with_derivatives(test_error_file, test_u_file, test_x_file, pdf_priors)
        np.savetxt('md_distances_with_derivatives.csv', md_distances, delimiter=',')

    print("Done!")
