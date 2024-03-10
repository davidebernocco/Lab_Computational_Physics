"""
Library of self-made functions needed for the codes implemented for the exercises of the 9th week

@author: david 
"""

import numpy as np
from PIL import Image
from ipywidgets import interact
from numba import njit
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

def random_spin_lattice(N, M):
    return np.random.choice([-1,1], size=(N,M))


@njit
def initial_energy(s):
    N, M = s.shape
    total = 0
    for i in range(N):
        for j in range(M):
            total += -s[i, j] * (s[(i-1)%N, j] + s[i, (j+1)%M])   
    return total
            


@njit
def Ising_conditions(s, beta):
    N, M = s.shape
    i = np.random.randint(0, N)
    j =  np.random.randint(0, M)
    
    NNsum = s[(i+1)%N, j] + s[i,(j+1)%M] + s[(i-1)%N, j] + s[i,(j-1)%M]
    
    dE = 2 * s[i, j] * NNsum
    
    if dE <= 0:
        s[i, j] *= -1
        deltaE = dE
        dM = 2 * s[i, j]
    elif np.exp(-dE * beta) > np.random.rand():
        s[i, j] *= -1
        deltaE = dE
        dM = 2 * s[i, j]
    
    return s, deltaE, dM




def accumulation(No, Nv, beta, eqSteps, mcSteps):
    L = No * Nv
    config = random_spin_lattice(No, Nv)
    
    for i in range(eqSteps):
        config = Ising_conditions(config, beta)[0]
    
    Ener = initial_energy(config)
    Mag = np.sum(config)
    E, E2 =  np.zeros(mcSteps), np.zeros(mcSteps)
    M, M2 =  np.zeros(mcSteps), np.zeros(mcSteps)
    E[0], E2[0], M[0], M2[0] = Ener, Ener * Ener, Mag, Mag * Mag
      
    for i in range(mcSteps):
        config, dE, dM = Ising_conditions(config, beta)
        Ener += dE
        Mag += dM
        
        E[i] = Ener
        E2[i] = Ener * Ener
        M[i] = Mag
        M2[i] = Mag * Mag
        
    return M / L, M2 / L, E / L, E2 / L
        



results = accumulation(10, 10, 1.5, 3500, 10000)
Nsteps_lst = np.arange(10000)
fig_m, ax_m = plt.subplots(figsize=(6.2, 4.5))
ax_m.plot(Nsteps_lst, results[0], marker='o' )
ax_m.set_xlabel(r'$ MCsteps $', fontsize=15)
ax_m.set_ylabel(r'$ M / N $', fontsize=15)
ax_m.grid(True)
plt.show()






def averaged_quantities(No, Nv, beta, eqSteps, mcSteps):
    
    val = accumulation(No, Nv, beta, eqSteps, mcSteps)
    fatt_aver = 1 / (mcSteps)
    e_aver = np.sum(val[2]) * fatt_aver
    e2_aver = np.sum(val[3]) * fatt_aver
    m_aver = np.sum(val[0]) * fatt_aver
    mABS_aver = np.sum(abs(val[0])) * fatt_aver
    m2_aver = np.sum(val[1]) * fatt_aver
    C = (e2_aver - e_aver * e_aver) * beta ** 2
    X = (m2_aver - m_aver * m_aver) * beta 

    return e_aver, mABS_aver, C, X







"""
@njit
def block_average(lst, s):
    
    aver = np.zeros(s, dtype = np.float32)
    block_size = int(len(lst) / s)

    for k in range(s):
        aver[k] = np.std(lst[(k * block_size):((k + 1) * block_size)])
        
    Sigma_s = np.std(aver)
        
    return Sigma_s / math.sqrt(s)







# ------------------------- ESEMPIO DA INTERNET -------------------------------
def random_spin_lattice(N, M):
    return np.random.choice([-1,1], size=(N,M))

def display_spin_lattice(lattice):
    return Image.fromarray(np.uint8((lattice + 1) * 0.5 * 255))


def Ising_update(lattice, n, m, beta):
    total = 0
    N, M = lattice.shape
    for i in range(n-1, n+2):
        for j in range(m-1, m+2):
             if i == n and j == m:
                 continue
             total += lattice[i % N, j % M]
             
    dE = 2 * lattice[n, m] * total
    
    if dE <= 0:
        lattice[n, m] *= -1
    elif np.exp(-dE * beta) > np.random.rand():
        lattice[n, m] *= -1
    


def Ising_step(lattice, beta):
    N, M = lattice.shape
    for n_offset in range(2):
        for m_offset in range(2):
            for n in range(n_offset, N, 2):
                for m in range(m_offset, M, 2):
                    Ising_update(lattice, n, m, beta)
    return lattice


def display_sequence(images):
    def _show(frame=(0, len(images) - 1)):
        return display_spin_lattice(images[frame])
    return interact(_show)

# -----------------------------------------------------------------------------





@njit
def Metropolis_Boltzmann( v0, dvmax, n, kb, T, m):
    
    acc = 0
    velocity = np.zeros(n, dtype = np.float32)
    energy = np.zeros(n, dtype = np.float32)
    velocity[0] = v0
    energy[0] = (m / 2) * v0 ** 2
    
    E_t = (m / 2) * v0 ** 2
    v_t = v0
    
    for i in range(1, n):
        v_star = np.random.uniform(v_t - dvmax, v_t + dvmax)
        E_star = (m / 2) * v_star ** 2
        
        esp1v = ( -m * v_star ** 2 / ( 2 * kb * T) )  
        esp2v = ( -m * v_t ** 2 / ( 2 * kb * T) )    
        alphav = math.e ** (esp1v - esp2v)           
        
        if alphav >= np.random.rand() :
            v_t = v_star
            acc += 1
        
        esp1E = ( - E_star / ( kb * T) )  
        esp2E = ( - E_t / ( kb * T) )    
        alphaE = math.e ** (esp1E - esp2E) 
               
        
        if alphaE >= np.random.rand() :
            E_t = E_star
            
        velocity[i] = v_t
        energy[i] = E_t
            
    return velocity, energy, acc/n

"""







