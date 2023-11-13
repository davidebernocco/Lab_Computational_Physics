"""
Library of self-made functions needed for the codes implemented for the exercises of the 3rd week

@author: david
"""

import random
import numpy as np
from numba import njit, jit, int32, float64


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
    return position, square_pos

def RW1D_average(N_w, N, x0, Pl):
    for j in range(N_w):
        
            