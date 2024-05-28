"""
Library of self-made functions needed for the final project exercises

@author: david
"""

import numpy as np
from numba import njit


@njit
def tent_map(x, r):
    
    if x < 1/2:
        x = r*x
        
    else:
        x = r - r*x
    
    return x



@njit
def iteration_tent(r, n0, n):
    trajectory =  np.zeros((n0 + n), dtype = np.float32)
    x0 = np.random.rand()
    trajectory[0] = x0
    x = x0
    
    for i in range(1, n0):
        x = tent_map(x, r)
        trajectory[i] = x
        
    for i in range(n):
        x = tent_map(x, r)
        trajectory[i + n0] = x 
        
    return trajectory




@njit
def bifurcation(r, n0, n):
    accum = np.zeros((len(r), n), dtype = np.float32)
    
    
    for i in range(len(r)):
        
        x = np.float32(0.5)
        
        for j in range(1, n0):
            x = np.float32(tent_map(x, r[i]))

        for k in range(n):
            x = np.float32(tent_map(x, r[i]))
            accum[i][k] = x
            
    return accum







