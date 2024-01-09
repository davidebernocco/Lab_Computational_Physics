"""
Now I have to do everything from the beginning again

@author: david
"""

from Funz6 import int_GaussLeg, integranda, int_trap, int_Simpson
import math
import numpy as np
import matplotlib.pyplot as plt


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






    