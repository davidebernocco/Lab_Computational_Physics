"""
Library of self-made functions needed the 10th week exercises

@author: david
"""

import numpy as np
import random
from numba import njit
import math



# -----------------------------------------------------------------------------
# LATTICE GAS MODEL
# -----------------------------------------------------------------------------




# (randomly) initializes particles positions on square lattice
def random_gas_lattice(Lo, Lv, Np):
    lattice = np.zeros((Lo, Lv), dtype=int)
    lattice_dictionary = {}
    
    for k in range(1, Np + 1):
        i = random.randint(0, Lo - 1)
        j = random.randint(0, Lv - 1)
        while lattice[i, j] != 0:
            i = random.randint(0, Lo - 1)
            j = random.randint(0, Lv - 1)
        lattice[i, j] = k
        lattice_dictionary[k] = (i, j)
        
    return lattice, lattice_dictionary





# Implement one step of the lattice gas dynamics for all the particles
# and output the new configuration alongside the <dR^2(t)>.
# Dictionary used to store labelled particle positions on lattice.
def trial_move(lattice, dictionary, Np, delta_R, particles, directions):
    Lo, Lv = lattice.shape
    
    for k in range(Np):
        p = random.choice(particles)
        trial = random.choice(directions)
        i, j = dictionary[p]
        
        if trial == 1:
            new_i = (i + 1) % Lo
            if lattice[new_i, j] == 0:
                lattice[i, j] = 0
                lattice[new_i, j] = p
                dictionary[p] = (new_i, j)
                delta_R[k][1] += 1
        elif trial == 2:
            new_j = (j + 1) % Lv
            if lattice[i, new_j] == 0:
                lattice[i, j] = 0
                lattice[i, new_j] = p
                dictionary[p] = (i, new_j)
                delta_R[k][0] += 1
        elif trial == 3:
            new_i = (i - 1) % Lo
            if lattice[new_i, j] == 0:
                lattice[i, j] = 0
                lattice[new_i, j] = p
                dictionary[p] = (new_i, j)
                delta_R[k][1] -= 1
        elif trial == 4:
            new_j = (j - 1) % Lv
            if lattice[i, new_j] == 0:
                lattice[i, j] = 0
                lattice[i, new_j] = p
                dictionary[p] = (i, new_j)
                delta_R[k][0] -= 1

    return lattice, dictionary, delta_R





# Perform an equilibration sequence before data are effectively collected.
# It looks at absolute difference between block-averaged means: until it is
# less than the treshold 10^(-3) it keeps discarding points.          
def equil_sequence(Np, block_size, latt, latt_dict, dR, particles, directions):
    dm = 1
    m0 = 0
    ns = 0
    
    while dm > 10**(-3):
        d_acc = 0
        
        for k in range(block_size):
            latt, latt_dict, dR = trial_move(latt, latt_dict, Np, dR, particles, directions)
            dR2 = dR**2
            dr2 = np.sum(np.mean(dR2, axis=0) - np.mean(dR, axis=0)**2)
            d_acc += dr2 / (4*(k+1+(ns*block_size)))
            
        m1 = d_acc / block_size
        dm = abs(m1 - m0)
        m0 = m1
        ns += 1
        
    dR = np.zeros((Np, 2))
    
    return dR, latt, latt_dict
         
  
   
  

# Main part of the code: after the equilibration sequence has been removed,
# <dR^2(t)> and D(t) are collected for a total of Nmc steps .         
def MC_iteration(Lo, Lv, Np, Nmc, equilibration, block_size):
    dR = np.zeros((Np, 2))
    DR2_aver = np.zeros(Nmc)
    D = np.zeros(Nmc)
    DT = np.zeros(Nmc)
    
    latt, latt_dict = random_gas_lattice(Lo, Lv, Np)
    particles = list(range(1, Np + 1))
    directions = np.arange(1, 5)
    
    if equilibration:
        dR, latt, latt_dict = equil_sequence(Np, block_size, latt, latt_dict, dR, particles, directions)      
    
    for i in range(Nmc):
        latt, latt_dict, dR = trial_move(latt, latt_dict, Np, dR, particles, directions)
        dR2 = dR**2
        DR2_aver[i] = np.sum(np.mean(dR2, axis=0) - np.mean(dR, axis=0)**2)
        D[i] = DR2_aver[i] / ((i+1))
        DT[i] = np.mean(D[:i+1])
    
    return DR2_aver, D/4 , DT/4        





# Perform mean and stdv of a given array through block-averages
@njit
def block_average(lst, s):
    
    aver =  np.zeros(s, dtype = np.float32)
    block_size = int(len(lst) / s)

    for k in range(s):
        aver[k] = np.mean(lst[(k * block_size):((k + 1) * block_size)])
        
    Sigma_s = np.std(aver)
    Aver_tot = np.mean(aver)
        
    return Aver_tot, Sigma_s / math.sqrt(s)





# It simply returns the sub-mean and sub-stdv for each block
def block(lst, s):

    sigma = np.zeros(s, dtype = np.float32)
    aver =  np.zeros(s, dtype = np.float32)
    block_size = int(len(lst) / s)

    for k in range(s):
        sigma[k] = np.std(lst[(k * block_size):((k + 1) * block_size)])
        aver[k] = np.mean(lst[(k * block_size):((k + 1) * block_size)])

    return aver, sigma





# Estimates the autodiffusion coefficient D as an average over the values 
# obtained through block-averages from single runs
def aver_DT(L, Np, Nmc, s, Naver):
    D_aver = 0
    sD_aver = 0
    
    for i in range(Naver):
        Dt = MC_iteration(L, L, Np, Nmc, True, 10**3)[1]
        D, sD = block_average(Dt, s)
        D_aver += D
        sD_aver += sD**2
        
    return D_aver / Naver, math.sqrt(sD_aver)
    




# Returns fluctuations (sigma_D) for single runs with increasing size L (rho cost)
def sD_N(L, r, Nmc, s):
    sD_arr = np.zeros(len(L))
    
    for i in range(len(L)):
        Np = int(r * L[i]**2)
        Dt = MC_iteration(L[i], L[i], Np, int(Nmc[i]), True, 10**3)[1]
        D, sD = block_average(Dt, int(s[i]))
        sD_arr[i] = sD
        
    return sD_arr





# Returns D averaged on multiple runs for different rho (L cost)
def D_vs_rho(L, r, Nmc, s, Naver):
    D_arr = np.zeros(len(r))
    sD_arr = np.zeros(len(r))
    
    for i in range(len(r)):
        Np = int(r[i] * L**2)
        D_arr[i], sD_arr[i] = aver_DT(L, Np, Nmc, s, Naver)
        
    return D_arr, sD_arr




# Linear function used for fit
def line(x, a, b):
    return a + b*x





# -----------------------------------------------------------------------------
# SIMULATED ANNEALING
# -----------------------------------------------------------------------------




# Function to be minimized (this one has multiple local minima)
def function(x):
    return (x + 0.2) * x + np.cos(14.5 * x - 0.3)





# Main code: starting from x0, with a Metropolis-like algorithm new points are
# proposed and tested using Boltzmann function with a certain T. Every time
# f(x_new) < f(x_local_minimum) x_new is updated as local minimum: the procedure
# goes on for N steps, at the end of which T is decreased and the whole algorithm
# is iterated until a certain T_threshold is reached. The last saved value of 
# x is considered the best estimation of function minimum.
# (In generating x_new an extra factor sqrt(T) has been added to the usual 
# random uniform pick in order to make the code more efficient)
def simulated_annealing(N, x0, T0, Tfact):
    x = x0
    fx = function(x0)
    T = T0
    fx_min = fx
    temp = []
    minimum = []
    func_min = []
    
    while(T > 10**(-5)):
        
        for _ in range(N):
            x_new = x + np.sqrt(T)*np.random.uniform(-0.5, 0.5)
            fx_new = function(x_new)
            boltzmann = np.exp(- (fx_new - fx) / T)
            a = np.random.rand()
            
            if boltzmann > a:
                x = x_new
                fx = fx_new
            
            if fx < fx_min:
                x_min = x
                fx_min = fx
                temp.append(T)
                minimum.append(x_min)
                func_min.append(fx_min)
                
        T *= Tfact
        
    temp = np.asarray(temp, dtype=np.float32)
    minimum = np.asarray(minimum, dtype=np.float32)
    func_min = np.asarray(func_min, dtype=np.float32)
   
    return temp, minimum, func_min

