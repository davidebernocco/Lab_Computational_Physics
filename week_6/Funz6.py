"""
Library of self-made functions needed for the codes implemented for the exercises of the 5th week

@author: david
"""

import numpy as np
#from numba import njit
import math




def integranda(x):
    
    return math.e ** x




def int_GaussLeg(func, a, b, num, condition, I):
    
    results = np.zeros(len(num), dtype = np.float64)
    Delta = np.zeros(len(num), dtype = np.float64)
    
    for j in range(len(num)):
        
        nodes, weights = np.polynomial.legendre.leggauss(num[j])
        nodes = 0.5 * (b - a) * nodes + 0.5 * (b + a)
        weights = 0.5 * (b - a) * weights
        
        somma = 0
        
        for i in range(num[j]):
            somma += func(nodes[i]) * weights[i]
            
        results[j] = somma
        
        if condition: 
            Delta[j] = abs(somma - I)
    
    return results, Delta





def int_trap(func, a, b, num, condition, I):
    
    results = np.zeros(len(num), dtype = np.float32)
    Delta = np.zeros(len(num), dtype = np.float32)
    
    for j in range(len(num)):
        xi = np.linspace(a, b, num[j] + 1)
        h = xi[1] - xi[0]
        somma = ( func(xi[0]) + func(xi[-1]) ) * ( 1 / 2 )
        
        for i in range(1, len(xi) - 1):
            somma += func(xi[i])
        
        results[j] = h * somma
        
        if condition:
            Delta[j] = abs(h * somma - I)
       
    return results, Delta





def int_Simpson(func, a, b, num, condition, I): # N.B. Unlike the "trpazoidal method" here the number of sub intervals must be even (2^n is ok)!
    
    results = np.zeros(len(num), dtype = np.float32)
    Delta = np.zeros(len(num), dtype = np.float32)
    
    for j in range(len(num)):
        xi = np.linspace(a, b, num[j] + 1)
        h = xi[1] - xi[0]
        somma = 0
        
        for i in range(1, int(num[j]/2) + 1):
            somma += func(xi[2*i - 2]) + 4 * func(xi[2*i - 1]) + func(xi[2*i])
        
        results[j] = (h/3) * somma
        
        if condition:    
            Delta[j] = abs((h/3) * somma - I)
    
    return results, Delta