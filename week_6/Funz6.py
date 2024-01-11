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





def unif_distr(a, b, N):
    
    return np.random.uniform(a, b, N)




def exp_distr(a, b, N):
    
    x = np.random.uniform(a, b, N)
    
    return -np.log(x)





def clt_distr(func, a, b, N, n_rep):
    
    aver = np.zeros(n_rep, dtype = np.float32)
    stdev =  np.zeros(n_rep, dtype = np.float32)
    z_N = np.zeros(n_rep, dtype = np.float32)
    
    for i in range(n_rep):
        np.random.seed(i)
        r = func(a, b, N)
        
        aver[i] += np.mean(r)
        stdev[i] += np.std(r)
        
    aver_aver = np.mean(aver)
    stdev_aver_aver = np.std(aver)
    
    z_N = (aver - aver_aver) / stdev_aver_aver
    
    return aver, stdev, aver_aver, stdev_aver_aver, np.mean(z_N ** 4), np.mean(z_N ** 2) ** 2



# CLT does not hold for Lorentz distribution! We use median end IQR to estimate x0 and gamma
def clt_lorentz( a, b, N, n_rep): 
    
    aver = np.zeros(n_rep, dtype = np.float32)
    half_IQR =  np.zeros(n_rep, dtype = np.float32)
    
    for i in range(n_rep):
        x = np.random.uniform(a, b, N)
        r = np.tan( math.pi * (x - 1/2) )
        
        aver[i] = np.mean(r)
        half_IQR[i] =  0.5 * np.percentile(r, 75) - np.percentile(r, 25)
        
    medianaTot = np.median(aver)
    half_IQRTot =  0.5 * (np.percentile(aver, 75) - np.percentile(aver, 25))
    
    
    return aver, half_IQR, medianaTot, half_IQRTot







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




