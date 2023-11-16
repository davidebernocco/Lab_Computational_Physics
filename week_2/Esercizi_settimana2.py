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
from numba import jit, njit, int32
import math
from Funz2 import lin_cong_met_period, line,  momentum_order_k, BruteForce

# COULD BE INTERESTING AND USEFUL FOR THE FUTURE LEARNING HOW TO HANDLE DICTIONARIES

"""
#-- ES 1 --
#---------- Linear congruent method and periodicity

x0 = 3
a = 4
c = 1
M = 9
num = 100

lst_LCM = lin_cong_met_period(x0, a, c, M, num)
print('\n---------- Exercise 1, 2nd week:-----------')
print('\nThe generated sequence with the L.C.M. is:', lst_LCM[0].tolist())
print('It has a finite period of: ', lst_LCM[1])



#-- ES 2 --
#---------- Intrinsic generators: uniformity and correlation (qualitative test)

# OSS: If we are interested in reproducibility, it could be useful to fix a SEED in the prng!!!

num_rand = 1000
data = np.random.rand(num_rand)

# Here the Freedman Diaconis method is used to estimate a proper number of bins in the histogram, provided the data
IQR = np.percentile(data, 75) - np.percentile(data, 25) 
N_bins = int((max(data) - min(data)) / (2 * IQR * len(data)**(-1/3)))

#The histogram is created
hist, bins = np.histogram(data, N_bins, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

#Fitting the histogram with the build in function that uses the "Least Squares Method"
params, covariance = curve_fit(line, bin_centers, hist)

#Plotting the histogram and the interpolated line
plt.hist(data, bins=round(math.sqrt(num_rand)), color='blue', alpha=0.7, density=True, label='Histogram')
plt.plot(bin_centers, line(bin_centers, *params), 'r', label='Linear Fit')
plt.xlabel('Values')
plt.ylabel('Frequency / Probability Density')
plt.title('Uniformity test - Histogram with Linear Fit')
plt.legend()
plt.grid(True)
plt.show()

print('\n---------- Exercise 2, 2nd week:-----------')
print('\nQualitative test for the Uniformity of the distribution (from intrinsic python generator).')
print('See the histogram generated above: "Uniformity test - Histogram with Linear Fit". ')
print('The slope of the line interpoling the histogram is: ', params[0], '±', math.sqrt(covariance[0][0]) )


#The generated numbers are paired conscutively forming a set of 2D cartesian coordinates
even_entries = data[::2]
odd_entries = data[1::2]

#Plotting the points
plt.scatter(even_entries, odd_entries, color='blue', marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Correlation test - Pairs of consecutive numbers')
plt.grid(True)
plt.show()

print('\nQualitative test for the Correlation of the distribution (from intrinsic python generator).')
print('See the scatter plot generated above: "Correlation test - Pairs of consecutive numbers" ')
print('If no ordered patterns appear in this plot, then the Intrinsic Pseudorandom numbers generator can be considered reliable.')
print('The same trick could be performed pairing not consecutive numbers to check possible correlations on larger scales.')

"""

#-- ES 3 --
#---------- Intrinsic generators: uniformity and correlation (quantitative test)


#3.1.A) Brute force method: evaluation of k-momentum using several sequences of increasing length

num_N = [20*i for i in range(1,5000)]
menu_k1 = BruteForce(num_N, 1)
menu_k3 = BruteForce(num_N, 3)
menu_k7 = BruteForce(num_N, 7)


plt.scatter(np.log(num_N), np.log(menu_k1), color='orange', marker='+', label=r'$\Delta_{N}(k=1)$')
plt.scatter(np.log(num_N), np.log(menu_k3), color='olive', marker='o', label=r'$\Delta_{N}(k=3)$')
plt.scatter(np.log(num_N), np.log(menu_k7), color='blue', marker='*', label=r'$\Delta_{N}(k=7)$')
plt.xlabel('log(N)')
plt.ylabel('log(Delta)')
plt.title('Uniformity test - Brute force method')
plt.legend()
plt.grid(True)
plt.show()


#3.1.B) Partial sums method: evaluation of k-moment using a single sequence
"""
Data = np.random.rand(num_N[-1])
population = [i for i in range(1,len(Data)+1)] 

@njit
def PartialSums(lst, k):
    dinner = []
    claw = 0
    for i in range(len(lst)):
        claw = claw + lst[i]**k
        dinner.append(abs(claw/(i+1) - (1/(1+k)) ))
    return dinner
 
plt.scatter(np.log(population), np.log(PartialSums(Data, 1)), color='magenta', marker='+', label=r'$\Delta_{N}(k=1)$')
plt.scatter(np.log(population), np.log(PartialSums(Data, 3)), color='green', marker='o', label=r'$\Delta_{N}(k=3)$')
plt.scatter(np.log(population), np.log(PartialSums(Data, 7)), color='cyan', marker='*', label=r'$\Delta_{N}(k=7)$')
plt.xlabel('log(N)')
plt.ylabel('log(Delta)')
plt.title('Uniformity test - Partial sums')
plt.legend()
plt.grid(True)
plt.show()


#3.2 Correlation test using directly the Partial sums method

@njit
def Correlation_PS(lst, k):
    population = [i for i in range(1+k,len(Data)+1)] 
    dinner = []
    claw = 0
    for i in range(len(lst)-k):
        claw = claw + lst[i]*lst[i+k]
        dinner.append(abs(claw/(i+1) - 1/4 ))
    return population, dinner, claw/(i+1)

plt.scatter(np.log(Correlation_PS(Data, 1)[0]), np.log(Correlation_PS(Data, 1)[1]), color='maroon', marker='+', label=r'$\Delta_{N}(k=1)$')
plt.scatter(np.log(Correlation_PS(Data, 3)[0]), np.log(Correlation_PS(Data, 3)[1]), color='lime', marker='o', label=r'$\Delta_{N}(k=3)$')
plt.scatter(np.log(Correlation_PS(Data, 7)[0]), np.log(Correlation_PS(Data, 7)[1]), color='navy', marker='*', label=r'$\Delta_{N}(k=7)$')
plt.xlabel('log(N)')
plt.ylabel('log(Delta)')
plt.title('Correlation test - Partial sums')
plt.legend()
plt.grid(True)
plt.show()
"""

