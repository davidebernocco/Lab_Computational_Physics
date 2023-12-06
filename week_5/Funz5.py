"""
Library of self-made functions needed for the codes implemented for the exercises of the 5th week

@author: david
"""

from numba import jit, njit
import numpy as np
import math 



@njit
def int_trap(num, I):
    
    results = np.zeros(len(num), dtype = np.float32)
    Delta = np.zeros(len(num), dtype = np.float32)
    
    for j in range(len(num)):
        xi = np.linspace(0, 1, num[j] + 1)
        h = xi[1] - xi[0]
        somma = 0
        
        for i in range(len(xi) - 1):
            somma += (math.e ** xi[i]) * h
        
        results[j] = somma
        Delta[j] = abs(somma - I)
    
    return results, Delta



@njit
def int_Simpson(num, I): # N.B. Unlike the "trpazoidal method" here the number of sub intervals must be even (2^n is ok)!
    
    results = np.zeros(len(num), dtype = np.float32)
    Delta = np.zeros(len(num), dtype = np.float32)
    
    for j in range(len(num)):
        xi = np.linspace(0, 1, num[j] + 1)
        h = xi[1] - xi[0]
        somma = 0
        
        for i in range(1, num[j]/2 + 1):
            somma += math.e ** xi[2*i - 2] + 4*math.e ** xi[2*i - 1] + math.e ** xi[2*i]
        
        results[j] = (h/3) * somma
        Delta[j] = abs((h/3) * somma - I)
    
    return results, Delta




