from scipy import stats
from bocd import read_data
import numpy as np
import matplotlib.pyplot as plt

def normality_tests(data):
    """Return normality tests for data.
    """
    # Shapiro-Wilk test.
    shapiro_stat, shapiro_p = stats.shapiro(data)
    
    # D'Agostino's K-squared test.
    dagostino_stat, dagostino_p = stats.normaltest(data)
    
    ks = stats.kstest(data,
             stats.norm.cdf)
    print(f"ks: {ks}")
    # Anderson-Darling test.
    anderson_stat = stats.anderson(data, dist='norm')
    
    return shapiro_stat, shapiro_p, dagostino_stat, dagostino_p, anderson_stat


def plot_data(T, data, cps):
    fig, axes = plt.subplots(2, 1, figsize=(20,10))

    ax1, ax2 = axes

    ax1.scatter(range(0, T), data)
    ax1.plot(range(0, T), data)
    ax1.set_xlim([0, T])
    ax1.margins(0)
    
   
    for cp in cps:
        ax1.axvline(cp, c='red', ls='dotted')
        ax2.axvline(cp, c='red', ls='dotted')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data, csp, tc = read_data('CSV_data/PreviewDistractionExpData_S1/PSN/ft.csv')

    # decimate data
    # data = data[::10]
    # tc   = tc[::10]

    # get the data where tc=1
    data_distracted = data[tc==1]

    # get the data where tc=0
    data_not_distracted = data[tc==0]

    # Calculate mean and standard deviation for both datasets
    mean_not_distracted = np.mean(data_not_distracted)
    std_not_distracted = np.std(data_not_distracted)
    mean_distracted = np.mean(data_distracted)
    std_distracted = np.std(data_distracted)


    T = len(data)
    print(T)
    x_axis = np.linspace(np.min(data), np.max(data), 100)

    plt.subplot(2, 1, 1)
    plt.hist(data_not_distracted, bins=100, density=True, alpha=0.5, color='b')
    plt.plot(x_axis, stats.norm.pdf(x_axis, mean_not_distracted, std_not_distracted), color='r')
    plt.title('Not Distracted')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.subplot(2, 1, 2)
    plt.hist(data_distracted, bins=100, density=True, alpha=0.5, color='g')
    plt.plot(x_axis, stats.norm.pdf(x_axis, mean_distracted, std_distracted), color='r')
    plt.title('Distracted')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

    shapiro_stat, shapiro_p, dagostino_stat, dagostino_p, anderson_stat = normality_tests(data_not_distracted)
    print(f"Shapiro-Wilk test: {shapiro_stat}, p-value: {shapiro_p}")
    print(f"D'Agostino's K-squared test: {dagostino_stat}, p-value: {dagostino_p}")
    print(f"Anderson-Darling test: {anderson_stat}")