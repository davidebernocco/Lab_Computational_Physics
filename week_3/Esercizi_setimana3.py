"""
Now I have to do everything from the beginning again

@author: david
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math
import time
import random
from Funz3 import boxmuller, R, var_x, decay



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

pack = var_x(varU, varV, num_rand)

IQR2 = np.percentile(pack, 75) - np.percentile(pack, 25)
bins2 = int((max(pack) - min(pack)) / (2 * IQR2 * len(pack)**(-1/3)))

hist2, bins2 = np.histogram(pack, bins, density=False)
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

# OSS: with the algoritms 3.1 and 3.2 showed in the Appendix_3 file,  we end up each time with two statistically independent and normal-distributed variables.
# If the goal is producing a single set of random variables as usual, this procedure is evedently inefficient!

# => Optimized algorithm: every two calls uses one of the two random numbers already generated in the previous call
"""
num_rand = 10**6
start_time5 = time.time()

gen_lst = boxmuller(num_rand)

IQR5 = np.percentile(gen_lst, 75) - np.percentile(gen_lst, 25)
bins5 = int((max(gen_lst) - min(gen_lst)) / (2 * IQR5 * len(gen_lst)**(-1/3)))

hist5, bins5 = np.histogram(gen_lst, bins5, density=False)
bin_centers5 = (bins5[:-1] + bins5[1:]) / 2
bin_widths5 = np.diff(bins5)
density5 = hist5 / sum(hist5)

plt.bar(bins5[:-1], density5, width=bin_widths5, alpha=0.5, color='b', label=r'$ p(x) = \frac{1}{\sqrt{2\pi \sigma}} e^{-\frac{(x-\mu)^{2}}{2\sigma^{2}}} $')
plt.xlabel('Counts')
plt.ylabel('Probability Density')
plt.title('Normalized Histogram - gaussian distributed variable (box Muller mod.)')
plt.legend()
plt.grid()
plt.show()
print(np.mean(gen_lst), np.std(gen_lst))

end_time5 = time.time()
elapsed_time5 = end_time5 - start_time5
print(f"Elapsed time5: {elapsed_time5:.5f} seconds")
"""


#-- ES 3 --
#---------- Simulation of radioactive decay

Lambda = 0.3
Nstart = 10

numero = decay(Nstart, Lambda)[0]
t = decay(Nstart, Lambda)[1]

print(numero, len(numero), t, len(t))

"""
plt.scatter(t, numero, label=f'N(0) = {Nstart} and $\lambda$ = {Lambda}', color='blue', marker='o')
plt.xlabel('Time t')
plt.ylabel('N(t)')
plt.title('Simulation of radioactive decay')
plt.legend()
plt.show()                
"""








