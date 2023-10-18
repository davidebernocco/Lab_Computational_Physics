"""
Now I have to do everything from the beginning again

@author: david
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from numba import jit, njit, int32, float64
import math


#-- ES 1 --
#---------- Random numbers with non-uniform distribution: Inverse Trasformation Method (ITM)

num_rand = 10**6
data = np.random.rand(num_rand)
a=3
exp_data = -(1/a)*np.log(data)

IQR = np.percentile(data, 75) - np.percentile(data, 25)
bins = int((max(data) - min(data)) / (2 * IQR * len(data)**(-1/3)))

hist, bins = np.histogram(exp_data, bins, density=False)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_widths = np.diff(bins)
density = hist / sum(hist)

plt.bar(bins[:-1], density, width=bin_widths, alpha=0.5, color='b', label='$p(x) = 3e^{-3x}$')
plt.xlabel('Counts')
plt.ylabel('Probability Density')
plt.title('Normalized Histogram - exponentially distributed variable')
plt.legend()
plt.grid()
plt.show()

log_counts = np.log(hist / sum(hist))

plt.bar(bins[:-1], log_counts, width=bin_widths, alpha=0.5, color='b')
plt.xlabel('Counts')
plt.ylabel('log(Probability Density)')
plt.title('Log Normalized histogram')
plt.grid()
plt.show()

plt.bar(bins[:-1], np.log(hist), width=bin_widths, alpha=0.5, color='b')
plt.xlabel('Counts')
plt.ylabel('log(Probability Density)')
plt.title('Log histogram')
plt.grid()
plt.show()
"""
def line(x, m, q):
    return m*x + q

params, covariance = curve_fit(line, bin_centers, hist)

plt.hist(data, bins=round(math.sqrt(num_rand)), color='blue', alpha=0.7, density=True, label='Histogram')
plt.plot(bin_centers, line(bin_centers, *params), 'r', label='Linear Fit')

plt.xlabel('Values')
plt.ylabel('Frequency / Probability Density')
plt.title('Histogram with Linear Fit')
plt.legend()
plt.grid(True)
plt.show()
"""

