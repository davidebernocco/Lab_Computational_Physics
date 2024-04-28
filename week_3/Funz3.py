
"""
Library of self-made functions needed for the 3rd week exercises

@author: david
"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, int32, float64



# -----------------------------------------------------------------------------
# RANDOM NUMBERS WITH NON-UNIFORM DISTRIBUTION: INVERSE TRANSFORMATION METHOD
# -----------------------------------------------------------------------------


# Linear function used in fit
@jit
def line(x, m, q):
    return m*x + q





# -----------------------------------------------------------------------------
# RANDOM NUMBERS WITH NON-UNIFORM DISTRIBUTION: COMPARISON BETWEEN DIFFERENT ALGORITHMS
# -----------------------------------------------------------------------------


# Function defined through only elementary operations, used to generate points
# according to the U-shaped distribution
@jit
def var_x(U,V,n):
    
    x = []
    
    for i in range(n):
        if U[i]**2 + V[i]**2 <= 1 :
            x.append((U[i]**2 - V[i]**2)/(U[i]**2 + V[i]**2))
            
    return np.asarray(x, dtype=np.float64)





# -----------------------------------------------------------------------------
# RANDOM NUMBERS WITH NON-UNIFORM DISTRIBUTION: BOX-MULLER ALGORITHM
# -----------------------------------------------------------------------------


# Box-Muller algorithm that generates a single gaussian distribution.
@jit(float64[:](int32))
def boxmuller(fagioli):
    
    sacchetto = []
    
    for i in range(fagioli):
        gaus_stored = False
        g = 0.0
        
        if gaus_stored:
            rnd = g
            gaus_stored = False
        else:
            while True:
                x = random.uniform(-1,1) #Alternatively: x = 2.0 * random.random() - 1.0
                y = random.uniform(-1,1) #Alternatively: y = 2.0 * random.random() - 1.0
                r2 = x**2 + y**2
                if r2 > 0.0 and r2 < 1.0:
                    break
            r2 = math.sqrt(-2.0 * math.log(r2) / r2)
            rnd = x * r2
            g = y * r2
            gaus_stored = True
            
        sacchetto.append(rnd)   
        
    return np.asarray(sacchetto, float64)





# -----------------------------------------------------------------------------
# SIMULATION ON RADIOACTIVE DECAY
# -----------------------------------------------------------------------------


# Given an initial number of particles in the sample, at each iteretion 
# (corresponding to an actual time interval) one particle can decay if a
# randomly (uniformly) generated number is lower then lambda (lambda in [0,1])
@jit
def decay(Ni, l):
    
    can = [Ni]
    Time = [0]
    t=0
    Nleft = Ni
    
    while Nleft > 0:
        chickpeas = 0
        for i in range(Nleft):
            r = random.random()
            if r <= l:
                chickpeas += 1
        Nleft -= chickpeas
        t += 1
        can.append(Nleft)
        Time.append(t)
        
    return np.asarray(can[:-1], int32), np.asarray(Time[:-1], int32)





# Just allows to plot different dataset with different labels
def multiple_plot(lst, l):
    
    for j in range(len(lst)):
        pino = decay(lst[j], l[j])
        pigna = pino[0]
        ago = pino[1]
        plt.scatter(ago, np.log(pigna), label=f'N(0) = {lst[j]},  $\lambda$ = {round(l[j],1)}', marker='o')
        plt.xlabel('Time t')
        plt.ylabel('ln( N(t) )')
        plt.legend(fontsize='7')
        plt.grid(True)
        del pino
        
    plt.show()
    
    return

