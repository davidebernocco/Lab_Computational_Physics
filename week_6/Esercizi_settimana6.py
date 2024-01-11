"""
Now I have to do everything from the beginning again

@author: david
"""

from Funz6 import int_GaussLeg, integranda, int_trap, int_Simpson, clt_distr, unif_distr, exp_distr, clt_lorentz
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy

#-- ES 1 --
#---------- 1D integration: "Gauss - Legendre quadrature"


# ---- 1.1) A first simple example
"""
xi = 0
xf = 1
poly_deg = 2

Bach = int_GaussLeg(integranda, xi, xf, poly_deg, False, 0)

print(f'Actual error between the known analythic results of the integral and the numerical integration with Gauss Legendre at order {poly_deg}:', abs(Bach[0] - (math.e - 1)))
"""


# ---- 1.2) "Trapezoidal", "Simpson" and "Gauss - Legendre quadrature" comparison
"""
n_list = np.asarray( [2 ** k for k in range(1, 11)], dtype=np.int32 )


Rossini = int_trap(integranda, 0, 1, n_list, True, math.e -1)
Doninzetti = int_Simpson(integranda, 0, 1, n_list, True,   math.e -1)
Bellini = int_GaussLeg(integranda, 0, 1, n_list, True, math.e -1)

plt.scatter(np.log(n_list), np.log(Rossini[1]), label='Trapezoidal method', marker='s', s=50)
plt.scatter(np.log(n_list), np.log(Doninzetti[1]), label='Simpson method', marker='o', s=50)
plt.scatter(np.log(n_list), np.log(Bellini[1]), label='Gauss Legendre quadrature', marker='^', s=50)
plt.xlabel('log(n)', fontsize=12)
plt.ylabel(r'$ \log(\Delta_{n}) $', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
"""




#-- ES 2 --
#---------- Random numbers with gaussian distribution: the CENTRAL LIMIT THEOREM


# ---- 2.1), 2.2), 2.3)  Central Limit Theorem from UNIFORM DISTRIBUTION


Num = 500
num = 1000

"""
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

plt.xlabel('x', fontsize=12)
plt.ylabel('Probability density', fontsize=12)
plt.grid(True)


x = np.linspace(min(Beethoven[0]), max(Beethoven[0]), 1000)
y = norm.pdf(x, Beethoven[2], Beethoven[3])
plt.plot(x, y, label=r'$Gaussian$', color='black')
plt.legend()

plt.show()




# ---- 2.4)  Central Limit Theorem from EXPONENTIAL DISTRIBUTION

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

plt.xlabel('x', fontsize=12)
plt.ylabel('Probability density', fontsize=12)
plt.grid(True)


x = np.linspace(min(Chopin[0]), max(Chopin[0]), 1000)
y = norm.pdf(x, Chopin[2], Chopin[3])
plt.plot(x, y, label=r'$Gaussian$', color='black')
plt.legend()

plt.show()

"""

# ---- 2.5)  Central Limit Theorem from LORENTZ DISTRIBUTION

Debussy = clt_lorentz( -1, 1, Num, num)

# More generally, if X1, X2,.., Xn are independent and Cauchy distributed with 
# location parameters x01, x02,..., x0n and scales gamma1, gamma2,..., gamman,
# and a1, a2,..., an are real numbers, then Sum_i(ai * Xi) is Cauchy distributed
# with location Sum_i(ai * x0i) and scale Sum_i(|ai| * gammai)

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

plt.xlabel('x', fontsize=12)
plt.ylabel('Probability density', fontsize=12)
plt.grid(True)


x = np.linspace(-40, 40, 10000)
y = cauchy.pdf(x, loc = Debussy[2], scale = Debussy[3])
plt.plot(x, y, label='Cauchy Lorentz', color='black')
plt.xlim(-30, 30)
plt.legend()

plt.show()


"""
FOR THE SAVED IMAGE, IT IS THE DISTRIBUTION OF THE VARIABLE "SAMPLE MEAN",
WITH X0 = 0.013 AND GAMMA = 0.978
"""












    