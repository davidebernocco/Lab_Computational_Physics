"""
Now I have to do everything from the beginning again

@author: david
"""

#Figures now render in the Plots pane by default.
#To make them also appear inline in the Console,
# uncheck "Mute Inline Plotting" under the Plots pane options menu.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

data = np.random.randn(1000)
hist, bins = np.histogram(data, bins=20, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2)) / (std_dev * np.sqrt(2 * np.pi))

params, covariance = curve_fit(gaussian, bin_centers, hist)

plt.hist(data, bins=20, color='blue', alpha=0.7, density=True, label='Histogram')
plt.plot(bin_centers, gaussian(bin_centers, *params), 'r', label='Gaussian Fit')

plt.xlabel('Values')
plt.ylabel('Frequency / Probability Density')
plt.title('Histogram with Gaussian Fit')
plt.legend()
plt.grid(True)
