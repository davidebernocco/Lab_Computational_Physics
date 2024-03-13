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
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

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
    E, E2 =  np.zeros(int(mcSteps/(No*Nv))+1), np.zeros(int(mcSteps/(No*Nv))+1)
    M, M2 =  np.zeros(int(mcSteps/(No*Nv))+1), np.zeros(int(mcSteps/(No*Nv))+1)
    E[0], E2[0], M[0], M2[0] = Ener, Ener * Ener, Mag, Mag * Mag
    
    j= 1
    
    for i in range(1, mcSteps + 1):
        config, dE, dM = Ising_conditions(config, beta)
        Ener += dE
        Mag += dM
        
        if i%L == 0:
            E[j] = Ener
            E2[j] = Ener * Ener
            M[j] = Mag
            M2[j] = Mag * Mag
            j += 1
        
    return M / L, M2 / L, E / L, E2 / L
        



"""
# ----------------------------------------------------------------------------
# CONFIGURATION SEQUENCE ANIMATION
# ----------------------------------------------------------------------------

def display_spin_lattice(lattice):
    return Image.fromarray(np.uint8((lattice + 1) * 0.5 * 255))


def animation_Ising(No, Nv, beta, mcSteps, plot_name):
    L = No * Nv
    config = random_spin_lattice(No, Nv)
    fig = plt.figure()
    
    spin_color = mcolors.ListedColormap(['blue', 'red'])
    im = plt.imshow(display_spin_lattice(config), cmap=spin_color)

    plt.title('2D Ising model (no H)')
    plt.axis('off')
    
    metadata = dict(title='Movie', artist='codinglikened')
    writer = PillowWriter(fps=10, metadata=metadata)
    
    with writer.saving(fig, plot_name, 100):
        for i in range(1, mcSteps + 1):
            config, _, _ = Ising_conditions(config, beta)
            
            if i % L == 0:
                im.set_data(display_spin_lattice(config))
                writer.grab_frame()
                
    return


animation_Ising(30, 30, 0.5, 200000, 'Ising_2D.gif')
plt.close('all')

# -----------------------------------------------------------------------------
"""

"""
Beta = 1.5
neqs = 75000
nmcs = 100000
n1 = n2 = 23


results = accumulation(n1, n2, Beta, neqs, nmcs)
Nsteps_lst = np.arange(int(nmcs/(n1*n2))+1)
fig_m, ax_m = plt.subplots(figsize=(6.2, 4.5))
ax_m.plot(Nsteps_lst, results[0], marker='o' )
ax_m.set_xlabel(r'$ MCsteps/100 $', fontsize=15)
ax_m.set_ylabel(r'$ M / N $', fontsize=15)
ax_m.grid(True)
plt.show()






def averaged_quantities(No, Nv, beta, eqSteps, mcSteps):
    
    val = accumulation(No, Nv, beta, eqSteps, mcSteps)
    fatt_aver = 1 / (len(val[0]))
    e_aver = np.sum(val[2]) * fatt_aver
    e2_aver = np.sum(val[3]) * fatt_aver
    m_aver = np.sum(val[0]) * fatt_aver
    mABS_aver = np.sum(abs(val[0])) * fatt_aver
    m2_aver = np.sum(val[1]) * fatt_aver
    C = (e2_aver - (No*Nv) * e_aver * e_aver) * beta ** 2 
    X = (m2_aver - (No*Nv) * mABS_aver * mABS_aver) * beta

    return e_aver, mABS_aver, C, X



def T_variation(No, Nv, T_m, T_M, d_T):
    arrT = np.arange(T_m, T_M, d_T)
    e, m = np.zeros(len(arrT)),  np.zeros(len(arrT))
    c, x = np.zeros(len(arrT)), np.zeros(len(arrT))
    for i in range(len(arrT)):
        e[i], m[i], c[i], x[i] = averaged_quantities(No, Nv, 1/arrT[i], 10**6, 10**6)
        
    return e, m, c, x


resultsT = T_variation(40, 40, 1, 4, 0.25)
Nsteps_lst =  np.arange(1, 4, 0.25)

fig_mT, ax_mT = plt.subplots(figsize=(6.2, 4.5))
ax_mT.plot(Nsteps_lst, resultsT[1], marker='o' )
ax_mT.set_xlabel(r'$ T [K] $', fontsize=15)
ax_mT.set_ylabel(r'$ \langle |M| \rangle / N $', fontsize=15)
ax_mT.grid(True)
plt.show()

fig_eT, ax_eT = plt.subplots(figsize=(6.2, 4.5))
ax_eT.plot(Nsteps_lst, resultsT[0], marker='o' )
ax_eT.set_xlabel(r'$ T [K] $', fontsize=15)
ax_eT.set_ylabel(r'$ \langle E \rangle / N $', fontsize=15)
ax_eT.grid(True)
plt.show()

fig_cT, ax_cT = plt.subplots(figsize=(6.2, 4.5))
ax_cT.plot(Nsteps_lst, resultsT[2], marker='o' )
ax_cT.set_xlabel(r'$ T [K] $', fontsize=15)
ax_cT.set_ylabel(r'$ c_V / N $', fontsize=15)
ax_cT.grid(True)
plt.show()

fig_xT, ax_xT = plt.subplots(figsize=(6.2, 4.5))
ax_xT.plot(Nsteps_lst, resultsT[3], marker='o' )
ax_xT.set_xlabel(r'$ T [K] $', fontsize=15)
ax_xT.set_ylabel(r'$ \chi / N $', fontsize=15)
ax_xT.grid(True)
plt.show()
"""





"""
@njit
def block_average(lst, s):
    
    aver = np.zeros(s, dtype = np.float32)
    block_size = int(len(lst) / s)

    for k in range(s):
        aver[k] = np.std(lst[(k * block_size):((k + 1) * block_size)])
        
    Sigma_s = np.std(aver)
        
    return Sigma_s / math.sqrt(s)

"""












