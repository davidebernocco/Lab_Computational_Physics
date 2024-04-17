"""

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from Funz10 import MC_iteration, block_average, aver_DT, sD_N
from Funz10 import function, simulated_annealing


# -----------------------------------------------------------------------------
# LATTICE GAS MODEL
# -----------------------------------------------------------------------------


# --------------------------
L = 20
MCsteps = 10**5
Nsteps_lst =  np.arange(1, MCsteps + 1, 1)

the = MC_iteration(L, L, 80, MCsteps, False, 0)

fig_dR2, ax_dR2 = plt.subplots(figsize=(6.2, 4.5))
ax_dR2.scatter(Nsteps_lst, the[0], marker='o', s=50)
ax_dR2.set_xlabel(r'$ t [a.u.] $', fontsize=15)
ax_dR2.set_ylabel(r'$ \langle \Delta R^2(t) \rangle $', fontsize=15)
ax_dR2.grid(True)
plt.show()

fig_D, ax_D = plt.subplots(figsize=(6.2, 4.5))
ax_D.scatter(Nsteps_lst, the[1], marker='o', s=50)
ax_D.set_xlabel(r'$t [a.u.] $', fontsize=15)
ax_D.set_ylabel(r'$ D(t) $', fontsize=15)
ax_D.grid(True)
plt.show()





# ------------------------
# With multiple values of density rho

L = 100
MCsteps = 2500
Nsteps_lst =  np.arange(1, MCsteps + 1, 1)

fig_dR2, ax_dR2 = plt.subplots(figsize=(6.2, 4.5))
fig_D, ax_D = plt.subplots(figsize=(6.2, 4.5))

for i in [0.03, 0.1, 0.2, 0.3, 0.5, 0.7]:
    the = MC_iteration(L, L, int(i*L**2), MCsteps, True, 10**3)
    
    ax_dR2.scatter(Nsteps_lst, the[0], marker='o', s=50, label=r'$ \rho = {} $'.format(i))
    ax_D.scatter(Nsteps_lst, the[1], marker='o', s=50)
    
ax_dR2.set_xlabel(r'$ t [a.u.] $', fontsize=15)
ax_dR2.set_ylabel(r'$ \langle \Delta R^2(t) \rangle $', fontsize=15)
ax_dR2.grid(True)
ax_dR2.legend()
plt.show()  

ax_D.set_xlabel(r'$t [a.u.] $', fontsize=15)
ax_D.set_ylabel(r'$ D(t) $', fontsize=15)
ax_D.grid(True)
plt.show()
    
    


# ----------------
the = MC_iteration(30, 30, 27, 225, True, 10**3)
D_aver, sD_aver = block_average(the[1], 10)
Nsteps_lst =  np.arange(1, 225 + 1, 1)

fig_D, ax_D = plt.subplots(figsize=(6.2, 4.5))
ax_D.scatter(Nsteps_lst, the[1], marker='o', s=50)
ax_D.plot([1,225], [D_aver, D_aver], label=r'$ \langle D(t) \rangle_T $')
ax_D.set_xlabel(r'$ t [a.u.] $', fontsize=15)
ax_D.set_ylabel(r'$ D(t) $', fontsize=15)
ax_D.grid(True)
plt.show()


the = MC_iteration(100, 100, 300, 2500, True, 10**3)
D_aver, sD_aver = block_average(the[1], 10)
Nsteps_lst =  np.arange(1, 2500 + 1, 1)

fig_D, ax_D = plt.subplots(figsize=(6.2, 4.5))
ax_D.scatter(Nsteps_lst, the[1], marker='o', s=50)
ax_D.plot([1,2500], [D_aver, D_aver], label=r'$ \langle D(t) \rangle_T $')
ax_D.set_xlabel(r'$ t [a.u.] $', fontsize=15)
ax_D.set_ylabel(r'$ D(t) $', fontsize=15)
ax_D.grid(True)
plt.show()





# ------------- Instantaneous fluctuations of D(t)
# Estimate instantaneous sigma trhough block with fixed size s, making it slide
# with time

def instant_fluct(d_t, block_size):
    s = len(d_t) - block_size
    fluct = np.zeros(s)
    
    for k in range(s):
        fluct[k] = np.std(d_t[k : (s+k)])
    
    lst_N = np.arange(block_size, len(d_t))
    
    return lst_N, fluct
    

data = MC_iteration(20, 20, 80, 10**5, False, 0)
drugo = instant_fluct(data[1], 100)
    
fig_fluct, ax_fluct = plt.subplots(figsize=(6.2, 4.5))
ax_fluct.scatter(drugo[0], drugo[1], marker='o', s=50, label='Block fluctuations')
ax_fluct.set_xlabel(r'$ t [a.u.] $', fontsize=15)
ax_fluct.set_ylabel(r'$ \Delta(t) $', fontsize=15)
ax_fluct.grid(True)
ax_fluct.legend()
plt.show()   




# --------------------- Alternative approach on D(t) fluctuations

def block(lst, s):

    sigma = np.zeros(s, dtype = np.float32)
    aver =  np.zeros(s, dtype = np.float32)
    block_size = int(len(lst) / s)

    for k in range(s):
        sigma[k] = np.std(lst[(k * block_size):((k + 1) * block_size)])
        aver[k] = np.mean(lst[(k * block_size):((k + 1) * block_size)])

    return aver, sigma
    

drugo = block(the[1], 100)
au = np.arange(1,101)

fig_fluct, ax_fluct = plt.subplots(figsize=(6.2, 4.5))
ax_fluct.scatter(au, drugo[0], marker='o', s=50, label='Block averages')
ax_fluct.set_xlabel(r'$ s [a.u.] $', fontsize=15)
ax_fluct.set_ylabel(r'$ \langle D(t) \rangle_s $', fontsize=15)
ax_fluct.grid(True)
ax_fluct.legend()
plt.show()

fig_fluct, ax_fluct = plt.subplots(figsize=(6.2, 4.5))
ax_fluct.scatter(au, drugo[1], marker='o', s=50, label='Block stdv')
ax_fluct.set_xlabel(r'$ s [a.u.] $', fontsize=15)
ax_fluct.set_ylabel(r'$ \sigma_s $', fontsize=15)
ax_fluct.grid(True)
ax_fluct.legend()
plt.show()








# -------------- D average estimation
# Remember that probably it is better to stop the simulation at a point in which
# <dR2(t)> = (L/2)^2 and so Nmc ~ (L/2)^2. Otherwise PBC effects afflict measure
# => I insert an equilibration sequence code indide the algorithm!


pilato = aver_DT(20, 80, 150, 15, 100)





# ------------- Fluctuations of D = <D(t)> at rho = cost as L is increased
lst_L = np.arange(20, 110, 20)

barabba = sD_N(lst_L, 0.03, 150, 15)

fig_fluct, ax_fluct = plt.subplots(figsize=(6.2, 4.5))
ax_fluct.scatter(lst_L, barabba, marker='o', s=50, label='Block fluctuations')
ax_fluct.set_xlabel(r'$ L $', fontsize=15)
ax_fluct.set_ylabel(r'$ \sigma_D $', fontsize=15)
ax_fluct.grid(True)
ax_fluct.legend()
plt.show()




# ------------- How D changes with rho at fixed L














# -----------------------------------------------------------------------------
# SIMULATED ANNEALING
# -----------------------------------------------------------------------------
                 
data_ann = simulated_annealing(10**4, 1,10,0.9)
x_list = np.arange(-1.5, 1.5, 10**3)

fig_min, ax_min = plt.subplots(figsize=(6.2, 4.5))
ax_min.scatter(data_ann[1], data_ann[2], marker='o', s=50, label='Local minima')
ax_min.plot(x_list, function(x_list), label='Analytic function')
ax_min.set_xlabel(r'$ x $', fontsize=15)
ax_min.set_ylabel(r'$ f(x) $', fontsize=15)
ax_min.grid(True)
ax_min.legend()
plt.show()









