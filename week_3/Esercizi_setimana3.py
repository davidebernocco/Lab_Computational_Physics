"""
Plots and other numerical estimations (3rd week)

@author: david
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math
import time
from Funz3 import boxmuller, var_x, decay, line, multiple_plot



# -----------------------------------------------------------------------------
# RANDOM NUMBERS WITH NON-UNIFORM DISTRIBUTION: INVERSE TRANSFORMATION METHOD
# -----------------------------------------------------------------------------


# --------------------------
# Generates a sequence of point that follow the exponential distribution with ITM
# (See slides for more formal details).
# The fundamental quantities to plot an histogram are built 

num_rand = 10**6
data = np.random.rand(num_rand)
a=3
exp_data = -(1/a)*np.log(data)

IQR = np.percentile(data, 75) - np.percentile(data, 25)
nbins = int((max(data) - min(data)) / (2 * IQR * len(data)**(-1/3)))

hist, bins = np.histogram(exp_data, nbins, density=False)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_widths = np.diff(bins)
density = hist / (num_rand * bin_widths[0])





# Plots the normalized (to 1) histogram
plt.bar(bins[:-1], density, width=bin_widths, alpha=0.5, color='b', label='$p(x) = 3e^{-3x}$')
plt.xlabel('Counts')
plt.ylabel('Probability Density')
#plt.title('Normalized Histogram - exponentially distributed variable')
plt.legend()
plt.grid()
plt.show()





# Plots an histogram taking the logarithm of columns.
# A small quantity of 1**(-10) is added not to have issues associated to log(empty bins)
log_counts = np.log((hist + 1**(-10)) / (num_rand * bin_widths[0]))
plt.bar(bins[:-1], log_counts, width=bin_widths, alpha=0.5, color='b')
plt.xlabel('Counts')
plt.ylabel('ln(Probability Density)')
#plt.title('Log Normalized histogram')
plt.grid()
plt.show()





# Take the log histogram and perform a linear interpolation.

params, covariance = curve_fit(line, bin_centers[:80], log_counts[:80])

plt.bar(bins[:-1], log_counts, width=bin_widths, alpha=0.5, color='b')
plt.plot(bin_centers[:80], line(bin_centers[:80], *params), 'r', label='Linear Fit')
plt.xlabel('Counts')
plt.ylabel('ln(Probability Density)')
#plt.title('Histogram with Linear Fit')
plt.legend()
plt.grid(True)
plt.show()





# -----------------------------------------------------------------------------
# RANDOM NUMBERS WITH NON-UNIFORM DISTRIBUTION: COMPARISON BETWEEN DIFFERENT ALGORITHMS
# -----------------------------------------------------------------------------

# The considered distribution is p(x) = 1\pi *  1\sqrt(1 - x^{2})   (U-shaped)


# -------------------------
# 2.1) A sequence of points distributed according to p(x) is generated with ITM.
# The normalized histogram is built

num_rand = 10**6

start_time1 = time.time()
data1 = np.random.rand(num_rand)
sin_data = np.sin(math.pi*(2*data1 - 1))

IQR1 = np.percentile(sin_data, 75) - np.percentile(sin_data, 25)
nbins1 = int((max(sin_data) - min(sin_data)) / (2 * IQR1 * len(sin_data)**(-1/3)))

hist1, bins1 = np.histogram(sin_data, nbins1, density=False)
bin_centers1 = (bins1[:-1] + bins1[1:]) / 2
bin_widths1 = np.diff(bins1)
density1 = hist1 / (num_rand * bin_widths1[0])

plt.bar(bins1[:-1], density1, width=bin_widths1, fill=False, edgecolor='blue', linewidth=1, label='with ITM')
plt.xlabel('Counts')
plt.ylabel('Probability density1')
#plt.title('Normalized Histogram - U-shaped distributed variable (ITM)')
plt.legend()
plt.grid()
#plt.show()

end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print(f"Elapsed time1: {elapsed_time1:.4f} seconds")





# -------------------------
# 2.2) A sequence of points distributed according to p(x) is generated  with 
# an algorithm that uses only elementary operations.
# The normalized histogram is built and compared to the previous one.

start_time2 = time.time()
varU = np.random.rand(num_rand)
varV = np.random.rand(num_rand)

pack = var_x(varU, varV, num_rand)

IQR2 = np.percentile(pack, 75) - np.percentile(pack, 25)
nbins2 = int((max(pack) - min(pack)) / (2 * IQR2 * len(pack)**(-1/3)))

hist2, bins2 = np.histogram(pack, nbins2, density=False)
bin_centers2 = (bins2[:-1] + bins2[1:]) / 2
bin_widths2 = np.diff(bins2)
density2 = hist2 / (len(pack) * bin_widths2[0])

plt.bar(bins2[:-1], density2, width=bin_widths2, fill=False, edgecolor='red', linewidth=1, label='with elementary operations')
plt.xlabel('Counts')
plt.ylabel('Probability density1')
#plt.title('Normalized Histogram - U-shaped distributed variable')
plt.legend()
plt.grid()
plt.show()

end_time2 = time.time()
elapsed_time2 = end_time2 - start_time2
print(f"Elapsed time2: {elapsed_time2:.4f} seconds")





# -----------------------------------------------------------------------------
# RANDOM NUMBERS WITH NON-UNIFORM DISTRIBUTION: BOX-MULLER ALGORITHM
# -----------------------------------------------------------------------------

# OSS: with the algoritms 3.1 and 3.2 showed in the Appendix_3 file, we end up
# each time with two statistically independent and normal-distributed variables.
# If the goal is producing a single set of random variables as usual, 
# this procedure is evedently inefficient!
# => Optimized algorithm: every two calls uses one of the two random numbers 
#    already generated in the previous call

# Here the algorithm is used to generate a sample that follows specifically
# the gaussian distribution.

num_rand = 10**6
start_time5 = time.time()

gen_lst = boxmuller(num_rand)

IQR5 = np.percentile(gen_lst, 75) - np.percentile(gen_lst, 25)
bins5 = int((max(gen_lst) - min(gen_lst)) / (2 * IQR5 * len(gen_lst)**(-1/3)))

hist5, bins5 = np.histogram(gen_lst, bins5, density=False)
bin_centers5 = (bins5[:-1] + bins5[1:]) / 2
bin_widths5 = np.diff(bins5)
density5 = hist5 / (num_rand * bin_widths5[0])

plt.bar(bins5[:-1], density5, width=bin_widths5, alpha=0.5, color='b', label=r'$ p(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^{2}}{2\sigma^{2}}} $')
plt.xlabel('Counts')
plt.ylabel('Probability Density')
#plt.title('Normalized Histogram - gaussian distributed variable (box Muller mod.)')
plt.legend()
plt.grid()
plt.show()
print(np.mean(gen_lst), np.std(gen_lst))

end_time5 = time.time()
elapsed_time5 = end_time5 - start_time5
print(f"Elapsed time5: {elapsed_time5:.5f} seconds")





# -----------------------------------------------------------------------------
# SIMULATION ON RADIOACTIVE DECAY
# -----------------------------------------------------------------------------


# -------------------------
# 3.1) The code implements an easy dynamic typical of radioactive decay.
# Once fixed lambda, a run is launched: taking the log of the counts, a linear
# fit is performed in order to check whether the slope coincides with lambda.

Lambda = 0.1
Nstart = 10**3

result = decay(Nstart, Lambda)
numero = result[0]
t = result[1]

plt.scatter(t, numero, label=f'N(0) = {Nstart},  $\lambda$ = {Lambda}', color='blue', marker='o')
plt.xlabel('Time t')
plt.ylabel('N(t)')
#plt.title('Simulation of radioactive decay')
plt.legend()
plt.grid(True)
plt.show()                


log_num = np.log(numero)
params, covariance = curve_fit(line, t, log_num)

plt.scatter(t, log_num, label=f'N(0) = {Nstart},  $\lambda$ = {Lambda}', color='blue', marker='o')
plt.plot(t, line(t, *params), 'r', label='Linear Fit')
plt.xlabel('Time t')
plt.ylabel('ln( N(t) )')
#plt.title('Simulation of radioactive decay')
plt.legend()
plt.grid(True)
plt.show()





# -------------------------
# 3.2) Two separate series of simulation are implemented: in the first the 
# initial concentration N(0) is varied keeping lambda costant. In the second, 
# it was done the opposite.

multiple_plot([10, 100, 1000, 10000, 100000], [0.1, 0.1, 0.1, 0.1, 0.1] )

multiple_plot([1000 for i in range(9)], [0.1*i for i in range(1,10)])









