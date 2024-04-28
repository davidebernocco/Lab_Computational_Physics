"""
Plots and other numerical estimations (4th week)

@author: david
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from Funz4 import RW1D_average, iter_plot, line, Accuracy, graphNwalk_N 
from Funz4 import graphMsdN, Histo_gauss, RW1D_average_random_l



# -----------------------------------------------------------------------------
# 1D RANDOM WALK
# -----------------------------------------------------------------------------


# --------------------------
# 1.1) For a set of RWs of a given number it plots all the instantaneous position
# x(t) and mean square distance dx^2(t) alongside the expected behaviours of
# their averaged values (averages meant instant by instant on all the walkers).
# Furthermore, a linear fit is applied on both the numerical averages <x(t)>
# and <dx^2(t)>.

ocean = RW1D_average(100, 64, 0, 0.5)

iter_plot(ocean, 0, 64, 100, 0.5, 'Istantaneous position $x_i$', False)

iter_plot(ocean, 1, 64, 100, 0.5, 'Istantaneous square position $x_i ^2$', False)

iter_plot(ocean, 0, 64, 100, 0.5, 'Istantaneous position $x_i$', True)

iter_plot(ocean, 1, 64, 100, 0.5, 'Istantaneous square position $x_i ^2$', True)

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





# --------------------------
#  1.3), 1.4) Estimates the minimum number of walkers needed, on average, to
# give an accuracy less or equal to 5%.
# In addition it is investigated a possible dependence of accuracy on N_steps.

minimum_Nwalk = Accuracy(100000, 0.05, 0, 64, 10, 10, 0.5)
   
plot_varyingN = graphNwalk_N()





# --------------------------
# 1.5) It plots the dependence of the Mean square distance on N_steps

plot_MSDvsN = graphMsdN()





# --------------------------
# 1.6) It verifies in a qualitative way that for sufficiently large N_steps
# the distribution of final positions P_N(x) can be approximated with a gaussian

plot_Histo = Histo_gauss()





# --------------------------
# 1.7) Provides key quantities to study the behaviour of RWs with steps of
# different length (taken from a certain distribution, no more 1 constant!)

random_l = RW1D_average_random_l(10000, 64, 0, 0.5)
#From this tuple we can repeat all the procedures done from 1.1 to 1.7. Just play!


