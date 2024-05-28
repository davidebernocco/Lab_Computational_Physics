"""
Plots and other numerical estimations (final project)

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

from FuncF import iteration_tent, bifurcation



        

Neq = 10
Niter = 50
traiett = iteration_tent(0.5, Neq, Niter)
Ntot = np.arange(Neq + Niter)

fig_traj, ax_traj = plt.subplots(figsize=(6.2, 4.5))
ax_traj.scatter(Ntot, traiett, marker='o', s=50)
ax_traj.set_xlabel(r'$ i $', fontsize=15)
ax_traj.set_ylabel(r'$ x_i $', fontsize=15)
ax_traj.grid(True)
plt.show()








Neq = 200
Niter = 50
r_arr = np.arange(0.2, 2.005, 0.005, dtype = np.float32)
bif_data = bifurcation(r_arr,Neq, Niter)


r_repeated = np.repeat(r_arr, Niter)
bif_data_flat = bif_data.flatten()

fig_bif, ax_bif = plt.subplots(figsize=(6.2, 4.5))
ax_bif.scatter(r_repeated, bif_data_flat, marker='o', s=1)
ax_bif.set_xlabel(r'$ r $', fontsize=15)
ax_bif.set_ylabel(r'$ x_i $', fontsize=15)
ax_bif.grid(True)
plt.show()

















