"""
Now I have to do everything from the beginning again

@author: david
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from numba import jit, njit, int32, float64
import math
import time


#-- ES 1 --
#---------- Random numbers with non-uniform distribution: Inverse Trasformation Method (ITM)
"""
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


def line(x, m, q):
    return m*x + q

params, covariance = curve_fit(line, bin_centers[:80], log_counts[:80])

plt.bar(bins[:-1], log_counts, width=bin_widths, alpha=0.5, color='b')
plt.plot(bin_centers[:80], line(bin_centers[:80], *params), 'r', label='Linear Fit')
plt.xlabel('Counts')
plt.ylabel('log(Probability Density)')
plt.title('Histogram with Linear Fit')
plt.legend()
plt.grid(True)
plt.show()
"""

#-- ES 2 --
#---------- Random numbers with non-uniform distribution: comparison between different algorithms

#2.1) with ITM
"""
num_rand = 10**6

start_time1 = time.time()
data = np.random.rand(num_rand)
exp_data = np.sin(math.pi*(2*data - 1))

IQR = np.percentile(data, 75) - np.percentile(data, 25)
bins = int((max(data) - min(data)) / (2 * IQR * len(data)**(-1/3)))

hist, bins = np.histogram(exp_data, bins, density=False)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_widths = np.diff(bins)
density = hist / sum(hist)

plt.bar(bins[:-1], density, width=bin_widths, alpha=0.5, color='b', label=r'$ p(x) = \frac{1}{\pi} \frac{1}{\sqrt{1 - x^{2}}} $')
plt.xlabel('Counts')
plt.ylabel('Probability Density')
plt.title('Normalized Histogram - U-shaped distributed variable (ITM)')
plt.legend()
plt.grid()
plt.show()

end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print(f"Elapsed time1: {elapsed_time1:.4f} seconds")


#2.2) with an algorithm that uses only elementary operations
start_time2 = time.time()
varU = np.random.rand(num_rand)
varV = np.random.rand(num_rand)

@jit
def var_x(U,V):
    x = []
    for i in range(num_rand):
        if U[i]**2 + V[i]**2 <= 1 :
            x.append((U[i]**2 - V[i]**2)/(U[i]**2 + V[i]**2))
    return x

IQR2 = np.percentile(var_x(varU, varV), 75) - np.percentile(var_x(varU, varV), 25)
bins2 = int((max(var_x(varU, varV)) - min(var_x(varU, varV))) / (2 * IQR2 * len(var_x(varU, varV))**(-1/3)))

hist2, bins2 = np.histogram(var_x(varU, varV), bins, density=False)
bin_centers2 = (bins2[:-1] + bins2[1:]) / 2
bin_widths2 = np.diff(bins2)
density2 = hist2 / sum(hist2)

plt.bar(bins2[:-1], density2, width=bin_widths2, alpha=0.5, color='b', label=r'$ p(x) = \frac{1}{\pi} \frac{1}{\sqrt{1 - x^{2}}} $')
plt.xlabel('Counts')
plt.ylabel('Probability Density')
plt.title('Normalized Histogram - U-shaped distributed variable')
plt.legend()
plt.grid()
plt.show()

end_time2 = time.time()
elapsed_time2 = end_time2 - start_time2
print(f"Elapsed time2: {elapsed_time2:.4f} seconds")
"""

#-- ES 3 --
#---------- Random numbers with gaussian distribution: the box Muller algorithm

num_rand = 10**6

start_time3 = time.time()
X = np.random.rand(num_rand)
Y = np.random.rand(num_rand)
x  = np.sqrt(-2*np.log(X))*np.cos(2*math.pi*Y)
y = np.sqrt(-2*np.log(X))*np.sin(2*math.pi*Y)

IQR = np.percentile(x, 75) - np.percentile(x, 25)
bins = int((max(x) - min(x)) / (2 * IQR * len(x)**(-1/3)))

hist, bins = np.histogram(x, bins, density=False)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_widths = np.diff(bins)
density = hist / sum(hist)

plt.bar(bins[:-1], density, width=bin_widths, alpha=0.5, color='b', label=r'$ p(x) = \frac{1}{\sqrt{2\pi \sigma}} e^{-\frac{(x-\mu)^{2}}{2\sigma^{2}}} $')
plt.xlabel('Counts x')
plt.ylabel('Probability Density')
plt.title('Normalized Histogram - gaussian distributed variable (box Muller)')
plt.legend()
plt.grid()
plt.show()

IQR2 = np.percentile(y, 75) - np.percentile(y, 25)
bins2 = int((max(y) - min(y)) / (2 * IQR2 * len(y)**(-1/3)))

hist2, bins2 = np.histogram(y, bins2, density=False)
bin_centers2 = (bins2[:-1] + bins2[1:]) / 2
bin_widths2 = np.diff(bins2)
density2 = hist2 / sum(hist2)

plt.bar(bins2[:-1], density2, width=bin_widths2, alpha=0.5, color='b', label=r'$ p(y) = \frac{1}{\sqrt{2\pi \sigma}} e^{-\frac{(y-\mu)^{2}}{2\sigma^{2}}} $')
plt.xlabel('Counts y')
plt.ylabel('Probability Density')
plt.title('Normalized Histogram - gaussian distributed variable (box Muller)')
plt.legend()
plt.grid()
plt.show()

end_time3 = time.time()
elapsed_time3 = end_time3 - start_time3
print(f"Elapsed time3: {elapsed_time3:.4f} seconds")




