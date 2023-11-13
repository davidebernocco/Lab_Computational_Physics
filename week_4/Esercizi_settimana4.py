"""
Now I have to do everything from the beginning again

@author: david
"""
import random
import numpy as np
from numba import njit, jit, int32, float64
import matplotlib.pyplot as plt

#-- ES 1 --
#---------- 1D Random Walks (RW)

# 1.1) Properties

Prob_l = 0.5 # clearly:  Prob_r = 1 - Prob_l

@njit
def RW_1D(N, x0, Pl):
    xi = x0
    position = [x0]
    square_pos = [x0**2]
    steps = [0]
    for i in range(N):
        l = np.random.rand()
        if l <= Pl:
            xi -= 1
        else:
            xi += 1
        position.append(xi)
        square_pos.append(xi**2)
        steps.append(i+1)
    return position, square_pos, steps

sea = RW_1D(64, 0, 0.5)


plt.plot(sea[2], sea[0], label=r'$P_{left}$ = 0.5 , N = 64')
plt.xlabel('Iteration steps i')
plt.ylabel('Istantaneous position $x_i$')
plt.title('1D Random Walk')
plt.legend()
plt.show()    

plt.plot(sea[2], sea[1], label=r'$P_{left}$ = 0.5 , N = 64')
plt.xlabel('Iteration steps i')
plt.ylabel('Istantaneous squared position $x_i ^2$')
plt.title('1D Random Walk')
plt.legend()
plt.show()    
    