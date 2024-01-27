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
"""
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
"""





#-- ES 4 --
#---------- Verification of the BOLTZMAN DISTRIBUTION


# ---- 4.1), 4.3) SINGLE CLASSICAL PARTICLE 1D IN THERMAL EQUILIBRIUM

"""
npoints = 100000

MB_distr1D = Metropolis_Boltzmann(0, 10, npoints, 1, 1, 1)



IQR = np.percentile(MB_distr1D[0], 75) - np.percentile(MB_distr1D[0], 25)
nbins = int((max(MB_distr1D[0]) - min(MB_distr1D[0])) / (2 * IQR * len(MB_distr1D[0])**(-1/3)))

hist, bins = np.histogram(MB_distr1D[0], nbins, density=False)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_widths = np.diff(bins)
density = hist / (npoints * bin_widths[0])

fig_v, ax_v = plt.subplots(figsize=(6.2, 4.5))
ax_v.bar(bins[:-1], density, width=bin_widths, alpha=0.5, color='b')
ax_v.set_xlabel(r'$v_x$', fontsize=15)
ax_v.set_ylabel(r'$ f(v_x)_{num} $', fontsize=15)
ax_v.grid(True)
plt.show() 



IQRE = np.percentile(MB_distr1D[1], 75) - np.percentile(MB_distr1D[1], 25)
nbinsE = int((max(MB_distr1D[1]) - min(MB_distr1D[1])) / (2 * IQRE * len(MB_distr1D[1])**(-1/3)))

histE, binsE = np.histogram(MB_distr1D[1], nbinsE, density=False)
bin_centersE = (binsE[:-1] + binsE[1:]) / 2
bin_widthsE = np.diff(binsE)
densityE = histE / (npoints * bin_widthsE[0])

fig_E, ax_E = plt.subplots(figsize=(6.2, 4.5))
ax_E.bar(binsE[:-1], densityE, width=bin_widthsE, alpha=0.5, color='b')
ax_E.set_xlabel(r'$E$', fontsize=15)
ax_E.set_ylabel(r'$ P(E)_{num} $', fontsize=15)
ax_E.grid(True)
plt.show() 



cri_cri = Metro_sampl_Boltzmann(0, 10, npoints, 1, 1, 1, int(npoints/1000))
print( "Mean velocity= ", cri_cri[0], "+/-", cri_cri[2]) 
print( "Mean Energy", cri_cri[1] / 2, "+/-", cri_cri[3]/2)
print( "Expected values: 0, (kb * T) / 2 ")
"""


# ---- 4.2) Check P(E) follows the expected behavoir
"""
fig_lnE, ax_lnE = plt.subplots(figsize=(6.2, 4.5))
ax_lnE.scatter(np.log(bin_centersE), np.log(densityE), color='g', label=r'$ ln( P(E)_{num}) $',  marker='o', s=50)
ax_lnE.plot(np.log(bin_centersE), np.log(list(map(lambda x: (1/np.sqrt(math.pi*x))*math.e**(-x), bin_centersE))), color='red', label=r'$ ln( P(E)_{theo}) $' ) 
ax_lnE.set_xlabel(r'$ln(E)$', fontsize=12)
ax_lnE.set_ylabel(r'$ ln( P(E)) $', fontsize=12)
ax_lnE.grid(True)
ax_lnE.legend()
plt.show() 
"""



# ---- 4.5) IDEAL 1D CLASSICAL GAS OF N PARTICLES IN THERMAL EQUILIBRIUM

"""
n_step = 100000
N_part = 200

# Mean vel tends to zero beacause of symmetry (no preferential direction +/- x)
# Mean energy changes in time due to typical canonical E fluctuation. However stay always around <E>
miao = Metro_sampl_Boltzmann_N(10, 30, n_step, 1, 100, 1, N_part, int((N_part * n_step)/20000))
print( "Mean particle velocity = ", miao[0], "+/-", miao[2])
print( "Mean particle Energy", miao[1]/2, "+/-", miao[3]/2)
print( "Expected values: 0, (kb * T) / 2")
print( "\n", "The acceptance ratio depends on T and delta!!")
"""


# ---- 4.6) P(<E>/N) (Should asymptotically tends to a gaussian centered in the <E> over all microstates)
"""
n_step = 100000
N_part = 1000

bau = Metropolis_Boltzmann_N(10, 30, n_step, 1, 100, 1, N_part)
#print(bau[2])


IQRE = np.percentile(bau[1]/N_part, 75) - np.percentile(bau[1]/N_part, 25)
nbinsE = int((max(bau[1]/N_part) - min(bau[1]/N_part)) / (2 * IQRE * len(bau[1]/N_part)**(-1/3)))

histE, binsE = np.histogram(bau[1]/N_part, nbinsE, density=False)
bin_centersE = (binsE[:-1] + binsE[1:]) / 2
bin_widthsE = np.diff(binsE)
densityE = histE / (n_step * N_part * bin_widthsE[0])

fig_EN, ax_EN =  plt.subplots(figsize=(6.2, 4.5))
ax_EN.bar(binsE[:-1], densityE, width=bin_widthsE, alpha=0.5, color='b', label=r'$ P(\epsilon): N=1000, T=100 K $')
ax_EN.set_xlabel(r'$ \epsilon = \langle E_i \rangle / N$', fontsize=15)
ax_EN.set_ylabel('Probability density', fontsize=15)
ax_EN.grid(True)
ax_EN.legend()
plt.show() 
"""


# 4.7) ---- Heat capacity (should be constant!)
"""
n_step = 100000
N_part = 200

mano1 = Metropolis_Boltzmann_N(10, 10, n_step, 1, 10, 1, N_part)
mano2 = Metropolis_Boltzmann_N(10, 13, n_step, 1, 20, 1, N_part)
mano3 = Metropolis_Boltzmann_N(10, 17, n_step, 1, 30, 1, N_part)
mano9 = Metropolis_Boltzmann_N(10, 28, n_step, 1, 90, 1, N_part)
mano10 = Metropolis_Boltzmann_N(10, 30, n_step, 1, 100, 1, N_part)
mano11 = Metropolis_Boltzmann_N(10, 33, n_step, 1, 110, 1, N_part)
print((np.mean(mano2[1]) - np.mean(mano1[1])) / 10)
print((np.mean(mano3[1]) - np.mean(mano2[1])) / 10)
print((np.mean(mano10[1]) - np.mean(mano9[1])) / 10)
print((np.mean(mano11[1]) - np.mean(mano10[1])) / 10)
"""

# 4.8) Mean square energy fluctuations

"""
lista_N = np.asarray([2**i for i in range(6,12)])
fluct = np.zeros(len(lista_N), dtype = np.float32)
for i in range(len(lista_N)):
    peppe = np.std(Metropolis_Boltzmann_N(10, 30, n_step, 1, 100, 1, lista_N[i])[1].flatten())
    fluct[i] = peppe/((100/2) * lista_N[i])    


fig_fl, ax_fl =  plt.subplots(figsize=(6.2, 4.5))
ax_fl.scatter(np.log(lista_N), np.log(fluct), marker='*', s=50)
ax_fl.set_xlabel(r'$ ln(N) $', fontsize=15)
ax_fl.set_ylabel(r'$ ln(\langle \Delta E^2 \rangle / \langle E \rangle) $', fontsize=15)
ax_fl.grid(True)
plt.show() 
"""


