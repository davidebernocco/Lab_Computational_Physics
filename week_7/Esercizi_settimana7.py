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
acceptance = acc_ratio(0, 5000, 1, delta_arr)

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


plt.scatter(num_arr, num_dependence, marker='o', s=50)
plt.xlabel('n', fontsize=12)
plt.ylabel(r'$ | \sigma_{num}^2 - \sigma_{exp}^2 | $', fontsize=12)
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
lista_n = np.asarray([2 ** i for i in range(7, 18)])
acc = accuracy(1, lista_n, dir_sampl_ground_state)

plt.scatter(np.log(lista_n), np.log(acc[0]), label='Var. accuracy', marker='s', s=50)
plt.scatter(np.log(lista_n), np.log(acc[1]), label='Pot en. accuracy', marker='o', s=50)
plt.scatter(np.log(lista_n), np.log(acc[2]), label='Kin en. accuracy', marker='^', s=50)
plt.scatter(np.log(lista_n), np.log(acc[3]), label='Tot en. accuracy', marker='v', s=50)
plt.xlabel('log(n)', fontsize=12)
plt.ylabel(r'$ \log(\Delta_{n}) $', fontsize=12)
plt.legend(loc='lower left')
plt.grid(True)
plt.show()



# ---- 2.3) Metropolis sampling
   
#lista_n = np.asarray([2 ** i for i in range(7, 18)])
acc_Metro = accuracy(1, lista_n, Metro_sampl_ground_state)

plt.scatter(np.log(lista_n), np.log(acc_Metro[0]), label='Var. accuracy', marker='s', s=50)
plt.scatter(np.log(lista_n), np.log(acc_Metro[1]), label='Pot en. accuracy', marker='o', s=50)
plt.scatter(np.log(lista_n), np.log(acc_Metro[2]), label='Kin en. accuracy', marker='^', s=50)
plt.scatter(np.log(lista_n), np.log(acc_Metro[3]), label='Tot en. accuracy', marker='v', s=50)
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
    
    plt.scatter(np.arange(1,51), jekyll, label=f'C(j) for delta = {delta_arr[i]} ', marker='o', s=50)

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

plt.scatter(np.arange(1,51), castle, label='C(j) BoxMuller', marker='o', s=50)
plt.scatter(np.arange(1,51), park, label='C(j) Metropolis delta/sigma = 5', marker='s', s=50)
plt.xlabel('j', fontsize=12)
plt.ylabel(r'$ C(j) $', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
"""





