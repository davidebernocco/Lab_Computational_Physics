"""

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from scipy.optimize import curve_fit
from Funz10 import MC_iteration, block_average, aver_DT, sD_N, block, D_vs_rho, line
from Funz10 import function, simulated_annealing


# -----------------------------------------------------------------------------
# LATTICE GAS MODEL
# -----------------------------------------------------------------------------



# --------------------------
# Plot <dR2(t)> and D(t) for a fixed L and rho

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

fig_DT, ax_DT = plt.subplots(figsize=(6.2, 4.5))
ax_DT.scatter(Nsteps_lst, the[2], marker='o', s=50)
ax_DT.set_xlabel(r'$ T [a.u.] $', fontsize=15)
ax_DT.set_ylabel(r'$ \langle D \rangle_T $', fontsize=15)
ax_DT.grid(True)
plt.show()





# -----------------------
# Plot D(t) for a couple of values of L
# => A piece of code has been put indide the algorithm to remove equilibration!
# (Valid from now on)

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





# ------------------------
# Plot <dR2(t)> and D(t) with multiple values of density rho

L = 100
MCsteps = 2500
Nsteps_lst =  np.arange(1, MCsteps + 1, 1)

fig_dR2, ax_dR2 = plt.subplots(figsize=(6.2, 4.5))
fig_D, ax_D = plt.subplots(figsize=(6.2, 4.5))

for i in [0.03, 0.1, 0.2, 0.3, 0.5, 0.7]:
    the = MC_iteration(L, L, int(i*L**2), MCsteps, True, 10**3)
    
    ax_dR2.scatter(Nsteps_lst, the[0], marker='o', s=50, label=r'$ \rho = {} $'.format(i))
    ax_D.scatter(Nsteps_lst, the[1], marker='o', s=50, label=r'$ \rho = {} $'.format(i))
    
ax_dR2.set_xlabel(r'$ t [a.u.] $', fontsize=15)
ax_dR2.set_ylabel(r'$ \langle \Delta R^2(t) \rangle $', fontsize=15)
ax_dR2.grid(True)
ax_dR2.legend()
plt.show()  

ax_D.set_xlabel(r'$t [a.u.] $', fontsize=15)
ax_D.set_ylabel(r'$ D(t) $', fontsize=15)
ax_D.grid(True)
plt.show()
    
    



# --------------------- 
# D(t) fluctuations
# Evaluating the std over contiguous blocks of fixed length

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





# --------------------- 
# D average estimation
# Remember that probably it is better to stop the simulation at a point in which
# <dR2(t)> = (L/2)^2 and so Nmc ~ (L/2)^2. Otherwise PBC effects afflict measure

pilato = aver_DT(20, 80, 150, 15, 100)





# --------------------
#  Fluctuations of D = <D(t)> at rho = cost as L is increased

lst_L = np.arange(20, 110, 5)

barabba = sD_N(lst_L, 0.2, (lst_L/2)**2, lst_L/10)

log_N = np.log((lst_L**2)*0.03)
log_fluct = np.log(barabba)

parDN, covDN = curve_fit(line, log_N, log_fluct)
a_DN, b_DN = parDN

fig_fluct, ax_fluct = plt.subplots(figsize=(6.2, 4.5))
ax_fluct.scatter(log_N, log_fluct, marker='o', s=50, label='Block fluctuations')
ax_fluct.plot(log_N, line(log_N, a_DN, b_DN),  label='Fit curve', color='red')
ax_fluct.set_xlabel(r'$log( N_p) $', fontsize=15)
ax_fluct.set_ylabel(r'$ log(\sigma_D) $', fontsize=15)
ax_fluct.grid(True)
ax_fluct.legend()
plt.show()





# --------------------
#  How D changes with rho at fixed L

rho_lst = np.asarray([0.05*i for i in range(1,20)])

caifa =  D_vs_rho(50, rho_lst, 2500, 25, 30)

parD, covD = curve_fit(line, rho_lst, caifa[0], sigma=caifa[1])
a_D, b_D = parD

fig_Drho, ax_Drho = plt.subplots(figsize=(6.2, 4.5))
ax_Drho.scatter(rho_lst, caifa[0], marker='o', s=50, label='Numerical data')
ax_Drho.errorbar(rho_lst, caifa[0], yerr=caifa[1], fmt='.', capsize=5, color='black')
ax_Drho.plot(rho_lst, line(rho_lst, a_D, b_D),  label='Fit curve', color='limegreen')
ax_Drho.set_xlabel(r'$ \rho $', fontsize=15)
ax_Drho.set_ylabel(r'$ D $', fontsize=15)
ax_Drho.grid(True)
ax_Drho.legend()
plt.show()









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









