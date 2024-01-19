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
from Funz7 import equil_time



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







