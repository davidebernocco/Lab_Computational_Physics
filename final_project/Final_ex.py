"""
Plots and other numerical estimations (final project)

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

from FuncF import iteration_tent, bifurcation, bifurcation_image, bifurcation_diagram




# -----------------------------------------------------------------------------
# STUDY OF THE TENT MAP
# -----------------------------------------------------------------------------
    
    
# ------------
# Visualizing transition sequence for tent map
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




# --------------
# Raw bifurcation diagram: all the points with same intensity (tent map)
Neq = 200
Niter = 50
r_arr = np.arange(0.2, 2.005, 0.005, dtype = np.float32)
bif_data = bifurcation(r_arr,Neq, Niter)


r_repeated = np.repeat(r_arr, Niter)
bif_data_flat = bif_data.flatten()

fig_bif, ax_bif = plt.subplots(figsize=(6.2, 4.5))
ax_bif.scatter(r_repeated, bif_data_flat)
ax_bif.set_xlabel(r'$ r $', fontsize=15)
ax_bif.set_ylabel(r'$ x_i $', fontsize=15)
ax_bif.grid(True)
plt.show()




# --------------
# Bifurcation diagram with scatter plot (tent map)
# Colored map

Neq = 1000
Niter = 1000
r_arr = np.linspace(0.2, 2.0, 1000, dtype = np.float32)
x_arr = np.arange(0, 1000, 1, dtype = np.int32)
bif_data = bifurcation_image(r_arr, Neq, Niter)

X, Y = np.meshgrid(r_arr, x_arr/1000)

fig_bif, ax_bif = plt.subplots(figsize=(6.2, 4.5))
ax_bif.scatter(X, Y, c=bif_data, cmap='Blues', s=0.1, alpha=0.8)
ax_bif.set_xlabel(r'$ r $', fontsize=15)
ax_bif.set_ylabel(r'$ x_i $', fontsize=15)
ax_bif.grid(True)
plt.show()




# --------------
# Bifurcation diagram with pixels (tent map)
# Colored map

bifurcation_image = bifurcation_diagram()

plt.figure(figsize=(10, 10))
plt.imshow(bifurcation_image, extent=[0.2, 2.0, 0.0, 1.0], aspect='auto', cmap='Blues', vmin=0, vmax=255, origin='lower')
plt.xlabel(r'$ r $', fontsize=15)
plt.ylabel(r'$ x_i $', fontsize=15)
plt.show()













