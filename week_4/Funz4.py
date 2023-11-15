"""
Library of self-made functions needed for the codes implemented for the exercises of the 3rd week

@author: david
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt



@njit
def RW_1D(N, x0, Pl):
    xi = x0
    position = [x0] 
    square_pos = [x0**2]
    for i in range(N):
        l = np.random.rand()
        if l <= Pl:
            xi -= 1
        else:
            xi += 1
        position.append(xi)
        square_pos.append(xi**2)
    return np.asarray(position, dtype=np.int32), np.asarray(square_pos, dtype=np.int32)



@njit
def RW1D_average(N_w, N, x0, Pl):
    
    position = np.full((N_w, N + 1), x0, dtype=np.int32)
    square_pos = np.full((N_w, N + 1), x0**2, dtype=np.int32)
    cumul_x = np.zeros(N, dtype=np.float32)
    cumul_x2 = np.zeros(N, dtype=np.float32)
    P_N = np.zeros(2*N +1, dtype=np.int32)

    for j in range(N_w):
        xi = x0
        l = np.random.uniform(0, 1, N)
        for i in range(N):
            if l[i] <= Pl:
                xi -= 1
            else:
                xi += 1
            position[j, i + 1] = xi
            square_pos[j, i + 1] = xi**2
            cumul_x[i] += xi
            cumul_x2[i] += xi**2
        P_N[N + xi] += 1
    average_x = cumul_x / N_w
    average_x2 = cumul_x2 / N_w

    return position, square_pos, average_x, average_x2, average_x2 - average_x**2, P_N



def iter_plot(vect, N, N_w, Pl, string):
    
    t = [i for i in range(N+1)]
    
    for i in range(N_w):
        
        plt.plot(t, vect[i])
    plt.xlabel('Iteration steps i')
    plt.ylabel(string)
    plt.title(fr'1D Random Walks $P_{{\mathrm{{left}}}} = {Pl}$, $N = {N}$')
    plt.show()    
    
    return









