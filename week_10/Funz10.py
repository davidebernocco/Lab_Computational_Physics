"""
Library of self-made functions needed for the codes implemented for the exercises of the 10th week

@author: david
"""

import numpy as np
import random
from numba import njit
import math



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
        


def trial_move(lattice, dictionary, Np, delta_R):
    Lo, Lv = lattice.shape
    directions = np.arange(1, 5)
    particles = [i for i in range(1, Np +1)]
    
    for k in range(Np):
        p = random.sample(particles, 1)[0]
        trial = random.choice(directions)
        i,j = dictionary[p]
        
        if trial == 1 and lattice[(i+1)%Lo, j] == 0:
            lattice[i, j] = 0
            lattice[(i+1)%Lo, j] = p
            dictionary[p] = ((i+1)%Lo, j)
            delta_R[k][1] += 1
        elif trial == 2 and lattice[i,(j+1)%Lv] == 0:
            lattice[i, j] = 0
            lattice[i,(j+1)%Lv] = p
            dictionary[p] = (i, (j+1)%Lv)
            delta_R[k][0] += 1
        elif trial == 3 and lattice[(i-1)%Lo, j] == 0:
            lattice[i, j] = 0
            lattice[(i-1)%Lo, j] = p
            dictionary[p] = ((i-1)%Lo, j)
            delta_R[k][1] -= 1
        elif trial == 4 and lattice[i,(j-1)%Lv]  == 0:
            lattice[i, j] = 0
            lattice[i,(j-1)%Lv] = p
            dictionary[p] = (i, (j-1)%Lv)
            delta_R[k][0] -= 1
        else:
            delta_R[k][0] += 0
            delta_R[k][1] += 0

    return lattice, dictionary, delta_R
            
            
         
            
def MC_iteration(Lo, Lv, Np, Nmc, equilibration, block_size):
    dR = np.zeros((Np, 2))
    DR2_aver = np.zeros(Nmc)
    D = np.zeros(Nmc)
    
    latt, latt_dict = random_gas_lattice(Lo, Lv, Np)
    
    if equilibration:
        dm = 1
        m0 = 0
        ns = 0
        while dm >= 0.005:
            d_acc = 0
            for k in range(block_size):
                latt, latt_dict, dR = trial_move(latt, latt_dict, Np, dR)
                dR2 = dR**2
                dr2 = np.sum(np.mean(dR2, axis=0) - np.mean(dR, axis=0)**2)
                d_acc += dr2 / (4*(k+1+(ns*block_size)))
            m1 = d_acc / block_size
            dm = m1 - m0
            m0 = m1
            ns += 1
        #print(ns)
        dR = np.zeros((Np, 2))
    
    for i in range(Nmc):
        latt, latt_dict, dR = trial_move(latt, latt_dict, Np, dR)
        dR2 = dR**2
        DR2_aver[i] = np.sum(np.mean(dR2, axis=0) - np.mean(dR, axis=0)**2)
        D[i] = DR2_aver[i] / ((i+1))
    
    return DR2_aver, D/4           



@njit
def block_average(lst, s):
    
    sigma = np.zeros(s, dtype = np.float32)
    aver =  np.zeros(s, dtype = np.float32)
    block_size = int(len(lst) / s)

    for k in range(s):
        sigma[k] = np.std(lst[(k * block_size):((k + 1) * block_size)])
        aver[k] = np.mean(lst[(k * block_size):((k + 1) * block_size)])
        
    Sigma_s = np.std(sigma)
    Aver_tot = np.mean(aver)
        
    return Aver_tot, Sigma_s / math.sqrt(s)




def aver_DT(L, Np, Nmc, s, Naver):
    D_aver = 0
    sD_aver = 0
    
    for i in range(Naver):
        Dt = MC_iteration(L, L, Np, Nmc, True, 10**3)[1]
        D, sD = block_average(Dt, s)
        D_aver += D
        sD_aver += sD**2
        
    return D_aver / Naver, math.sqrt(sD_aver)
    



def sD_N(L, r, Nmc, s):
    sD_arr = np.zeros(len(L))
    
    for i in range(len(L)):
        Np = int(r * L[i]**2)
        Dt = MC_iteration(L[i], L[i], Np, Nmc, True, 10**3)[1]
        D, sD = block_average(Dt, s)
        sD_arr[i] = sD
        
    return sD_arr




# -----------------------------------------------------------------------------
# SIMULATED ANNEALING
# -----------------------------------------------------------------------------


def function(x):
    return (x + 0.2) * x + np.cos(14.5 * x - 0.3)



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
            
            # Possible scaling factor within x_new
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



