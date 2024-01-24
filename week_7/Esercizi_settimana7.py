"""
Now I have to do everything from the beginning again

@author: david
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import time
from Funz7 import Metropolis, gauss_func, plot_histo, acc_ratio, n_dep
from Funz7 import equil_time, accuracy, dir_sampl_ground_state
from Funz7 import Metro_sampl_ground_state, corr, boxmuller
from Funz7 import Metropolis_Boltzmann, Metro_sampl_Boltzmann
from Funz7 import Metropolis_Boltzmann_N, Metro_sampl_Boltzmann_N



#-- ES 1 --
#---------- Random numbers with gaussian distribution: the METROPOLIS algorithm


# ---- 1.1) Which n gives a satisfactory accordance with expected result?
"""
sigma = 1
n_step = np.asarray([10**2, 10**3, 10**4, 10**5])

            
start_time1 = time.time()

parrot = plot_histo(n_step, Metropolis, sigma, 5*sigma)

end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print(f"Elapsed time1: {elapsed_time1:.4f} seconds")
"""



# ---- 1.2) Dependence of acceptance ratio on delta, keeping n fixed
"""
sigma = 1

delta_arr = np.asarray([0.5*sigma + i for i in range(10)])
delta_over_sigma = delta_arr/sigma
acceptance = acc_ratio(0, 10000, 1, delta_arr)

ursula = np.linspace(min(delta_over_sigma), max(delta_over_sigma), 10)
plt.plot(ursula, [1/2 for _ in range(10)], label='Ideal range limit', color='red')
plt.plot(ursula,  [1/3 for _ in range(10)], color='red')
plt.scatter(delta_over_sigma, acceptance, label='Data', marker='o', s=50)
plt.xlabel(r'$ \delta / \sigma $', fontsize=12)
plt.ylabel(' Acceptance ratio ', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()
"""


#  ---- 1.3) |Numerical variance - exact variance| vs n
"""
num_arr = np.asarray([100*i for i in range(1, 301)])
num_dependence = n_dep(0, num_arr, 1, 5)

jasmine = np.linspace(min(num_arr), max(num_arr), 10)
plt.plot(jasmine, [0.05 for _ in range(10)], label='Equilibration limit', color='red', linewidth=2)
plt.scatter(num_arr, num_dependence, marker='o', s=50)
plt.xlabel('n', fontsize=15)
plt.ylabel(r'$ | \sigma_{num}^2 - \sigma_{exp}^2 | $', fontsize=15)
plt.grid(True)
plt.show()
"""



# ---- 1.4) Equilibration time for fixed delta
"""
num_arr = np.asarray([100*i for i in range(1, 301)])
equilibration = equil_time(0, num_arr, 1, 5, 1000)
print("Simulation completed with average n =",  equilibration)
"""





#-- ES 2 --
#---------- Sampling physical quantities: direct sampling and METROPOLIS SAMPLING


# ---- 2.1), 2.2) Direct sampling
"""
lista_n = np.asarray([2 ** i for i in range(7, 20)])
acc = accuracy(1, lista_n, dir_sampl_ground_state)

plt.scatter(np.log(lista_n), np.log(acc[0]), label=r'$ \Delta_n( \langle \sigma^2 \rangle) $', marker='s', s=50)
plt.scatter(np.log(lista_n), np.log(acc[1]), label=r'$ \Delta_n( \langle E_{Pot} \rangle) $', marker='o', s=50)
plt.scatter(np.log(lista_n), np.log(acc[2]), label=r'$ \Delta_n( \langle E_{Kin} \rangle) $', marker='^', s=50)
plt.scatter(np.log(lista_n), np.log(acc[3]), label=r'$ \Delta_n( \langle E_{Tot} \rangle) $', marker='v', s=50)
plt.xlabel('log(n)', fontsize=12)
plt.ylabel(r'$ \log(\Delta_{n}) $', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
"""


# ---- 2.3) Metropolis sampling
"""
lista_n = np.asarray([2 ** i for i in range(7, 20)])
acc_Metro = accuracy(1, lista_n, Metro_sampl_ground_state)

plt.scatter(np.log(lista_n), np.log(acc_Metro[0]), label=r'$ \Delta_n( \langle \sigma^2 \rangle) $', marker='s', s=50)
plt.scatter(np.log(lista_n), np.log(acc_Metro[1]), label=r'$ \Delta_n( \langle E_{Pot} \rangle) $', marker='o', s=50)
plt.scatter(np.log(lista_n), np.log(acc_Metro[2]), label=r'$ \Delta_n( \langle E_{Kin} \rangle) $', marker='^', s=50)
plt.scatter(np.log(lista_n), np.log(acc_Metro[3]), label=r'$ \Delta_n( \langle E_{Tot} \rangle) $', marker='v', s=50)
plt.xlabel('log(n)', fontsize=12)
plt.ylabel(r'$ \log(\Delta_{n}) $', fontsize=12)
plt.legend(loc='lower left')
plt.grid(True)
plt.show()
"""



#-- ES 3 --
#---------- CORRELATIONS 


# ---- 3.1) Correlations for n fixed, varying delta
"""
delta_arr = [0.5, 1, 5, 10, 15] #0.5,1,5,8,10

for i in range(len(delta_arr)):
    dracula = Metropolis(0, delta_arr[i], 10000, 1)[0]
    jekyll = corr(10000, 50, dracula)
    
    plt.scatter(np.arange(50), jekyll, label=f'C(j) for delta = {delta_arr[i]} ', marker='o', s=50)

plt.xlabel('j', fontsize=12)
plt.ylabel(r'$ C(j) $', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
"""


# ---- 3.2) Correlations comparison: Box-Muller vs Metropolis

house = boxmuller(10000)
garden = Metropolis(0, 5, 10000, 1)[0]

castle = corr(10000, 50, house)
park = corr(10000, 50, garden)

plt.scatter(np.arange(50), castle, label='C(j) BoxMuller', marker='s', s=50, color='m')
plt.scatter(np.arange(50), park, label='C(j) Metropolis delta/sigma = 5', marker='o', s=50, color='green')
plt.xlabel('j', fontsize=12)
plt.ylabel(r'$ C(j) $', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()





#-- ES 4 --
#---------- Verification of the BOLTZMAN DISTRIBUTION


# ---- 4.1), 4.3) SINGLE CLASSICAL PARTICLE 1D IN THERMAL EQUILIBRIUM

"""
npoints = 100000
MB_distr1D = Metropolis_Boltzmann(0, 2, npoints, 1, 1, 1)



IQR = np.percentile(MB_distr1D[0], 75) - np.percentile(MB_distr1D[0], 25)
nbins = int((max(MB_distr1D[0]) - min(MB_distr1D[0])) / (2 * IQR * len(MB_distr1D[0])**(-1/3)))

hist, bins = np.histogram(MB_distr1D[0], nbins, density=False)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_widths = np.diff(bins)
density = hist / (npoints * bin_widths[0])

plt.bar(bins[:-1], density, width=bin_widths, alpha=0.5, color='b', label=r'$ f(v)^{num} $')
plt.xlabel(r'$v$', fontsize=12)
plt.ylabel('Probability density', fontsize=12)
plt.grid(True)
plt.legend()
plt.show() 



IQRE = np.percentile(MB_distr1D[1], 75) - np.percentile(MB_distr1D[1], 25)
nbinsE = int((max(MB_distr1D[1]) - min(MB_distr1D[1])) / (2 * IQRE * len(MB_distr1D[1])**(-1/3)))

histE, binsE = np.histogram(MB_distr1D[1], nbinsE, density=False)
bin_centersE = (binsE[:-1] + binsE[1:]) / 2
bin_widthsE = np.diff(binsE)
densityE = histE / (npoints * bin_widthsE[0])

plt.bar(binsE[:-1], densityE, width=bin_widthsE, alpha=0.5, color='b', label=r'$ f(E)^{num} $')
plt.xlabel(r'$E$', fontsize=12)
plt.ylabel('Probability density', fontsize=12)
plt.grid(True)
plt.legend()
plt.show() 
"""


# Mean vel tends to zero beacause of symmetry (no preferential direction +/- x)
# Mean energy changes in time until thermalization is reached
"""
cri_cri = Metro_sampl_Boltzmann(0, 2, npoints, 1, 1, 1)
print( "Mean velocity= ", cri_cri[0]) 
print( "Mean Energy", cri_cri[1] / 2)
print( "Expected values: 0, (kb * T) / (2 * m)")
"""

# ---- 4.2) Check P(E) follows the expected behavoir

"""
plt.scatter(np.log(bin_centersE), np.log(densityE), color='g', label=r'$ ln( f(E)^{num}) $',  marker='o', s=50)
plt.plot(np.log(bin_centersE), np.log(list(map(lambda x: (1/np.sqrt(math.pi*x))*math.e**(-x), bin_centersE))), color='red', label=r'$ ln( f(E)^{theo}) $' ) 
plt.xlabel(r'$ln(E)$', fontsize=12)
plt.ylabel('ln( Probability density) ', fontsize=12)
plt.grid(True)
plt.legend()
plt.show() 
"""



# ---- 4.5) IDEAL 1D CLASSICAL GAS OF N PARTICLES IN THERMAL EQUILIBRIUM


n_step = 10000
N_part = 1000
"""
miao = Metro_sampl_Boltzmann_N(2, 13, n_step, 1, 75, 1, N_part)
print( "Mean particle velocity = ", miao[0])
print( "Mean particle Energy", miao[1] / 2)
print( "Expected values: 0, (kb * T) / (2 * m)")
print( "\n", "The acceptance ratio depends on T!!")
"""


# ---- 4.6) P(<E>/N) (Should asymptotically tends to a gaussian centered in the <E> over all microstates)

"""
bau = Metropolis_Boltzmann_N(10, 13, n_step, 1, 10, 1, N_part)

IQRE = np.percentile(bau[1]/N_part, 75) - np.percentile(bau[1]/N_part, 25)
nbinsE = int((max(bau[1]/N_part) - min(bau[1]/N_part)) / (2 * IQRE * len(bau[1]/N_part)**(-1/3)))

histE, binsE = np.histogram(bau[1]/N_part, nbinsE, density=False)
bin_centersE = (binsE[:-1] + binsE[1:]) / 2
bin_widthsE = np.diff(binsE)
densityE = histE / (n_step * bin_widthsE[0])

plt.bar(binsE[:-1], densityE, width=bin_widthsE, alpha=0.5, color='b', label=r'$ f(\epsilon)^{num} $')
plt.xlabel(r'$ \epsilon = \langle E \rangle / N$', fontsize=12)
plt.ylabel('Probability density', fontsize=12)
plt.grid(True)
plt.legend()
plt.show() 
"""


# 4.7) ---- Heat capacity (should be constant!)

"""
mano1 = Metropolis_Boltzmann_N(10, 13, n_step, 1, 10, 1, N_part)
mano2 = Metropolis_Boltzmann_N(10, 13, n_step, 1, 20, 1, N_part)
mano3 = Metropolis_Boltzmann_N(10, 13, n_step, 1, 30, 1, N_part)
mano9 = Metropolis_Boltzmann_N(10, 13, n_step, 1, 90, 1, N_part)
mano10 = Metropolis_Boltzmann_N(10, 13, n_step, 1, 100, 1, N_part)
mano11 = Metropolis_Boltzmann_N(10, 13, n_step, 1, 110, 1, N_part)
print((np.mean(mano2[1]) - np.mean(mano1[1])) / 10)
print((np.mean(mano3[1]) - np.mean(mano2[1])) / 10)
print((np.mean(mano10[1]) - np.mean(mano9[1])) / 10)
print((np.mean(mano11[1]) - np.mean(mano10[1])) / 10)
"""

# 4.8) Mean square energy fluctuations

"""
braccio1 = Metropolis_Boltzmann_N(10, 13, n_step, 1, 10, 1, N_part)
braccio2 = Metropolis_Boltzmann_N(10, 13, n_step, 1, 100, 1, N_part)

print("Mean square fluctuation for T = 10K: ", np.var(braccio1[1]/N_part))
print("Mean square fluctuation for T = 100K: ", np.var(braccio2[1]/N_part))
# To be compared with the two corresp. histos at 4.6) !!
"""



