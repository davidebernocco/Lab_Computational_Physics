"""
Now I have to do everything from the beginning again

@author: david
"""

import math 
from numba import jit, njit
import numpy as np
from Funz5 import int_trap, int_Simpson
import matplotlib.pyplot as plt


#-- ES 1 --
#---------- Equispaced points: comparison between "trapezoidal" and "Simpson" method

exact = math.e - 1

n_intervals = np.asarray([2**j for j in range(1, 11)], dtype = np.int32)
"""
trapezoidal = int_trap(n_intervals, exact)

plt.scatter(np.log(n_intervals), np.log(trapezoidal[1]), label='Trapezoidal method')
plt.xlabel('log(n)')
plt.ylabel(r'$ \log(\Delta_{n}) $', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
"""

Simpson = int_Simpson(n_intervals, exact)

plt.scatter(np.log(n_intervals), np.log(Simpson[1]), label='Simpson method')
plt.xlabel('log(n)')
plt.ylabel(r'$ \log(\Delta_{n}) $', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


