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
from numba import jit, njit, int32, float64
import math

# COULD BE INTERESTING AND USEFUL FOR THE FUTURE LEARNING HOW TO HANDLE DICTIONARIES

"""
#-- ES 1 --
#---------- Linear congruent method and periodicity

@jit(int32[:](int32, int32, int32, int32, int32))
def lin_cong_met(x0, a, c, M, n):
    gen_lst = [x0]
    x = x0
    for i in range(n):
       x = (a*x + c)%M
       gen_lst.append(x)
    return np.asarray(gen_lst, int32)
#If I didn't use numba, it would take >10 times more to evaluate this function!! (proved)


@njit
def lin_cong_met_period(x0, a, c, M, n):
    gen_lst = [x0]
    x = x0
    for i in range(n):
       x = (a*x + c)%M
       if x != x0:
           gen_lst.append(x)
       else:
           return np.asarray(gen_lst, dtype=np.int32), np.int32(len(gen_lst))
    return np.asarray(gen_lst, dtype=np.int32), np.int32(len(gen_lst))
#It seems synthax has to be slightly changed if I want to output an array and an integer
"""


#-- ES 2 --
#---------- Intrinsic generators: uniformity and correlation (qualitative test)
"""
num_rand = 1000
data = np.random.rand(num_rand)

hist, bins = np.histogram(data, bins = round(math.sqrt(num_rand)), density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

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


even_entries = data[::2]
odd_entries = data[1::2]

plt.scatter(even_entries, odd_entries, color='blue', marker='o', label='Data from Python PRNG')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Correlation test - Pairs of consecutive numbers')
plt.legend()
plt.grid(True)
plt.show()
"""

#-- ES 3 --
#---------- Intrinsic generators: uniformity and correlation (quantitative test)

@njit
def momentum_oreder_k(lst, k):
    result = 0
    for x in lst:
        result += x ** k
    Delta = abs(result - (1/(1+k)) )
    return np.float64(result/len(lst)), np.float64(Delta)










