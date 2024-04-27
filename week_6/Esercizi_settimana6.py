"""
Plots and other numerical estimations (6th week)

@author: david
"""

from Funz6 import int_GaussLeg, integranda, int_trap, int_Simpson, clt_distr, unif_distr, exp_distr, clt_lorentz
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy



# -----------------------------------------------------------------------------
# GAUSS-LEGENDRE QUADRATURE: NUMERICAL INTEGRATION
# -----------------------------------------------------------------------------


# 1.1) A first simple example of numerical integration with GL method.
# it provides the actual error

xi = 0
xf = 1
poly_deg = 2

Bach = int_GaussLeg(integranda, xi, xf, [poly_deg], False, 0)

print(f'Actual error between the known analythic results of the integral and the numerical integration with Gauss Legendre at order {poly_deg}:', abs(Bach[0] - (math.e - 1)))





# 1.2) "Trapezoidal", "Simpson" and "Gauss - Legendre quadrature" comparison.

n_list = np.asarray( [2*k for k in range(1, 32)], dtype=np.int32 )
n_list2 = np.asarray( [k for k in range(1, 65)], dtype=np.int32 )


Rossini = int_trap(integranda, 0, 1, n_list, True, math.e -1)
Doninzetti = int_Simpson(integranda, 0, 1, n_list, True,   math.e -1)
Bellini = int_GaussLeg(integranda, 0, 1, n_list2, True, math.e -1)

plt.scatter(np.log(n_list), np.log(Rossini[1]), label='Trapezoidal', marker='s', s=50)
plt.scatter(np.log(n_list), np.log(Doninzetti[1]), label='Simpson', marker='o', s=50)
plt.scatter(np.log(n_list2), np.log(Bellini[1]), label='Gauss-Leg.', marker='^', s=50)
plt.xlabel('log(n)', fontsize=12)
plt.ylabel(r'$ \log(\Delta_{n}) $', fontsize=12)
plt.legend(loc='lower left')
plt.grid(True)
plt.show()





# -----------------------------------------------------------------------------
# RANDOM NUMBERS WITH GAUSSIAN DISTRIBUTION: THE CENTRAL LIMIT THEOREM
# -----------------------------------------------------------------------------


# 2.1), 2.2), 2.3)  Verifies, as predicted by the Central Limit Theorem (CLT),
# that taking a large number of averages from uniform distributions reproduces
# a gaussian distribution

Num = 500
num = 10000

Beethoven = clt_distr( unif_distr, -1, 1, Num, num)

print('\n', 'Numerical average of the averages: ', Beethoven[2])
print('\n', 'Expected mu value: ', 0)
print('\n', f'Numerical stdev of the averages for N = {Num}: ', Beethoven[3])
print('\n', f'Expected sigma value for N = {Num}: ', (1 / math.sqrt(3))/math.sqrt(Num))
print('\n', ' < Z_N ^ 4 > = ', Beethoven[4], ' 3 * < Z_N ^ 2 > ^ 2 = ', 3 * Beethoven[5]) 


IQR = np.percentile(Beethoven[0], 75) - np.percentile(Beethoven[0], 25)
nbins = int((max(Beethoven[0]) - min(Beethoven[0])) / (2 * IQR * len(Beethoven[0])**(-1/3)))

hist, bins = np.histogram(Beethoven[0], nbins, density=False)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_widths = np.diff(bins)
density = hist / (num * bin_widths[0])

plt.bar(bins[:-1], density, width=bin_widths, alpha=0.5, color='b', label=r'$PDF^{num}$')

plt.xlabel(r'$\bar{X}$', fontsize=12)
plt.ylabel('Probability density', fontsize=12)
plt.grid(True)

x = np.linspace(min(Beethoven[0]), max(Beethoven[0]), 1000)
y = norm.pdf(x, 0, (1 / math.sqrt(3))/math.sqrt(Num))
plt.plot(x, y, label=r'$PDF^{theo}$', color='black')
plt.legend()

plt.show()





# 2.4)  The same as before, but taking the averages from exponential distributions

Chopin = clt_distr( exp_distr, 0, 1, Num, num)

print('\n', 'Numerical average of the averages: ', Chopin[2])
print('\n', 'Expected mu value: ', 1 )
print('\n', f'Numerical stdev of the averages for N = {Num}: ', Chopin[3])
print('\n', f'Expected sigma value for N = {Num}: ', 1 / math.sqrt(Num))
print('\n', ' < Z_N ^ 4 > = ', Chopin[4], ' 3 * < Z_N ^ 2 > ^ 2 = ', 3 * Chopin[5]) 


IQRe = np.percentile(Chopin[0], 75) - np.percentile(Chopin[0], 25)
nbinse = int((max(Chopin[0]) - min(Chopin[0])) / (2 * IQRe * len(Chopin[0])**(-1/3)))

histe, binse = np.histogram(Chopin[0], nbinse, density=False)
bin_centerse = (binse[:-1] + binse[1:]) / 2
bin_widthse = np.diff(binse)
densitye = histe / (num * bin_widthse[0])

plt.bar(binse[:-1], densitye, width=bin_widthse, alpha=0.5, color='b', label=r'$PDF^{num}$')

plt.xlabel(r'$\bar{X}$', fontsize=12)
plt.ylabel('Probability density', fontsize=12)
plt.grid(True)

x = np.linspace(min(Chopin[0]), max(Chopin[0]), 1000)
y = norm.pdf(x, 1, 1 / math.sqrt(Num))
plt.plot(x, y, label=r'$PDF^{theo}$', color='black')
plt.legend()

plt.show()





# 2.5) Now from a series of Cauchy-Lorentz distributions
# Here the CLT in his traditional formulation does not hold, since we cannot  
# define a proper mean (1-momentum) for the distributions we start from.

Debussy = clt_lorentz( -1, 1, Num, num)

"""
# More generally, if X1, X2,.., Xn are independent and Cauchy distributed with 
# location parameters x01, x02,..., x0n and scales gamma1, gamma2,..., gamman,
# and a1, a2,..., an are real numbers, then Sum_i(ai * Xi) is Cauchy distributed
# with location Sum_i(ai * x0i) and scale Sum_i(|ai| * gammai)
"""

print('\n', 'Median as numerical estimationfor the central value x0: ', Debussy[2])
print('\n', 'Expected value for x0: ', 0 )
print('\n', f'Half IQR as numerical estimator for the scale parameter gamma for N = {Num}: ', Debussy[3])
print('\n', 'Expected gamma: ', 1)


IQRd = np.percentile(Debussy[0], 75) - np.percentile(Debussy[0], 25)
nbinsd = int((max(Debussy[0]) - min(Debussy[0])) / (2 * IQRd * len(Debussy[0])**(-1/3)))

histd, binsd = np.histogram(Debussy[0], nbinsd, density=False)
bin_centersd = (binsd[:-1] + binsd[1:]) / 2
bin_widthsd = np.diff(binsd)
densityd = histd / (num * bin_widthsd[0])

plt.bar(binsd[:-1], densityd, width=bin_widthsd, alpha=0.5, color='b', label=r'$PDF^{num}$')

plt.xlabel(r'$\bar{X}$', fontsize=12)
plt.ylabel('Probability density', fontsize=12)
plt.grid(True)

x = np.linspace(-40, 40, 10000)
y = cauchy.pdf(x, loc = Debussy[2], scale = Debussy[3])
plt.plot(x, y, label=r'$PDF^{theo}$', color='black')
y2 = norm.pdf(x, 0, 1.3)
plt.plot(x, y2, label='Gaussian', color='red', linestyle='dashed')
plt.xlim(-20, 20)
plt.legend()

plt.show()





    