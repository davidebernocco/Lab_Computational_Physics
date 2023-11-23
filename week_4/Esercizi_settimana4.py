"""
Now I have to do everything from the beginning again

@author: david
"""

from Funz4 import  RW1D_average, iter_plot, line, Accuracy, graphNwalk_N,  graphMsdN
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math
from numba import jit
#-- ES 1 --
#---------- 1D Random Walks (RW)

# 1.1) Properties
"""
ocean = RW1D_average(100, 64, 0, 0.5)

iter_plot(ocean, 0, 64, 100, 0.5, 'Istantaneous position $x_i$', False)

iter_plot(ocean, 1, 64, 100, 0.5, 'Istantaneous squared position $x_i ^2$', False)

iter_plot(ocean, 0, 64, 100, 0.5, 'Istantaneous position $x_i$', True)

iter_plot(ocean, 1, 64, 100, 0.5, 'Istantaneous squared position $x_i ^2$', True)

print( ocean[2][-1])
print( ocean[3][-1])


t = np.array([i for i in range(1,65)])



paramsx, covariancex = curve_fit(line, t, ocean[2])

plt.scatter(t, ocean[2], label='Data', color='black')
plt.plot(t, line(t, *paramsx), color='red', label='Linear Fit')
plt.xlabel('i')
plt.ylabel(r'$\langle x_i \rangle^{num} $', fontsize=12)
plt.legend()
plt.grid(True)
plt.yticks(np.arange(min(ocean[2])-0.5, max(ocean[2])+0.5, 0.25))
plt.show()


paramsx2, covariancex2 = curve_fit(line, t, ocean[3])

plt.scatter(t, ocean[3], label='Data', color='black')
plt.plot(t, line(t, *paramsx2), color='red', label='Linear Fit')
plt.xlabel('i')
plt.ylabel(r'$\langle x_{i}^2 \rangle^{num}$', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


paramsDx, covarianceDx = curve_fit(line, t, ocean[4])

plt.scatter(t, ocean[4], label='Data', color='black')
plt.plot(t, line(t, *paramsDx), color='red', label='Linear Fit')
plt.xlabel('i')
plt.ylabel(r'$\langle (\Delta x_{i})^2 \rangle^{num}$', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
"""


# ---- 1.3) and 1.4) Accuracy of the mean square distance
"""
minimum_Nwalk = Accuracy(100000, 0.05, 0, 64, 10, 10, 0.5)
   
plot_varyingN = graphNwalk_N()
"""



# ---- 1.5) Dependence of the Mean square distance on N
"""
plot_MSDvsN = graphMsdN()
"""

