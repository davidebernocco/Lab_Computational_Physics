"""
Plots and other numerical estimations (final project)

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

from FunzF import tent_map, sine_map, logistic_map
from FuncF import iteration_tent, iteration_sine, iteration_logistic
from FuncF import bifurcation_tent, bifurcation_sine, bifurcation_logistic
from FuncF import bifurcation_image, bifurcation_diagram



# -----------------------------------------------------------------------------
# STUDY OF THE TENT MAP
# -----------------------------------------------------------------------------
    
    
# ------------
# Visualizing transition sequence (tent map)
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
Neq = 1000
Niter = 10000
r_arr = np.linspace(0.2, 2.0, 1000, dtype = np.float32)
bif_data = bifurcation_tent(r_arr,Neq, Niter)


r_repeated = np.repeat(r_arr, Niter)
bif_data_flat = bif_data.flatten()

fig_bif, ax_bif = plt.subplots(figsize=(6.2, 4.5))
ax_bif.scatter(r_repeated, bif_data_flat, s=0.1)
ax_bif.set_xlabel(r'$ r $', fontsize=15)
ax_bif.set_ylabel(r'$ x_i $', fontsize=15)
ax_bif.grid(True)
plt.show()




# --------------
# Bifurcation diagram with scatter plot (tent map)
# Colored map

Neq = 1000
Niter = 10000
x_arr = np.arange(0, 1000, 1, dtype = np.int32)
bif_data = bifurcation_image(r_arr, Neq, Niter, tent_map)

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

bifurcation_image = bifurcation_diagram(r_arr, Neq, Niter, tent_map)

plt.figure(figsize=(10, 10))
plt.imshow(bifurcation_image, extent=[0.2, 2.0, 0.0, 1.0], aspect='auto', cmap='Blues', vmin=0, vmax=255, origin='lower')
plt.xlabel(r'$ r $', fontsize=15)
plt.ylabel(r'$ x_i $', fontsize=15)
plt.show()







# -----------------------------------------------------------------------------
# STUDY OF THE SINE MAP
# -----------------------------------------------------------------------------
    
    
# ------------
# Visualizing transition sequence (sine map)
Neq = 10
Niter = 50
traiett = iteration_sine(0.5, Neq, Niter)
Ntot = np.arange(Neq + Niter)

fig_traj, ax_traj = plt.subplots(figsize=(6.2, 4.5))
ax_traj.scatter(Ntot, traiett, marker='o', s=50)
ax_traj.set_xlabel(r'$ i $', fontsize=15)
ax_traj.set_ylabel(r'$ x_i $', fontsize=15)
ax_traj.grid(True)
plt.show()




# --------------
# Raw bifurcation diagram: all the points with same intensity (sine map)
Neq = 1000
Niter = 10000
r_arr = np.linspace(0.2, 1.0, 2000, dtype = np.float32)
bif_data = bifurcation_sine(r_arr,Neq, Niter)


r_repeated = np.repeat(r_arr, Niter)
bif_data_flat = bif_data.flatten()

fig_bif, ax_bif = plt.subplots(figsize=(6.2, 4.5))
ax_bif.scatter(r_repeated, bif_data_flat, s=0.1)
ax_bif.set_xlabel(r'$ r $', fontsize=15)
ax_bif.set_ylabel(r'$ x_i $', fontsize=15)
ax_bif.grid(True)
plt.show()




# --------------
# Bifurcation diagram with scatter plot (sine map)
# Colored map

Neq = 1000
Niter = 10000
x_arr = np.arange(0, 1000, 1, dtype = np.int32)
bif_data = bifurcation_image(r_arr, Neq, Niter, sine_map)

X, Y = np.meshgrid(r_arr, x_arr/1000)

fig_bif, ax_bif = plt.subplots(figsize=(6.2, 4.5))
ax_bif.scatter(X, Y, c=bif_data, cmap='Blues', s=0.1, alpha=0.8)
ax_bif.set_xlabel(r'$ r $', fontsize=15)
ax_bif.set_ylabel(r'$ x_i $', fontsize=15)
ax_bif.grid(True)
plt.show()




# --------------
# Bifurcation diagram with pixels (sine map)
# Colored map

bifurcation_image = bifurcation_diagram(r_arr, Neq, Niter, sine_map)

plt.figure(figsize=(10, 10))
plt.imshow(bifurcation_image, extent=[0.2, 1.0, 0.0, 1.0], aspect='auto', cmap='Blues', vmin=0, vmax=255, origin='lower')
plt.xlabel(r'$ r $', fontsize=15)
plt.ylabel(r'$ x_i $', fontsize=15)
plt.show()










# -----------------------------------------------------------------------------
# STUDY OF THE LOGISTIC MAP
# -----------------------------------------------------------------------------
    
    
# ------------
# Visualizing transition sequence (logistic map)
Neq = 10
Niter = 50
traiett = iteration_logistic(0.5, Neq, Niter)
Ntot = np.arange(Neq + Niter)

fig_traj, ax_traj = plt.subplots(figsize=(6.2, 4.5))
ax_traj.scatter(Ntot, traiett, marker='o', s=50)
ax_traj.set_xlabel(r'$ i $', fontsize=15)
ax_traj.set_ylabel(r'$ x_i $', fontsize=15)
ax_traj.grid(True)
plt.show()




# --------------
# Raw bifurcation diagram: all the points with same intensity (logistic map)
Neq = 1000
Niter = 10000
r_arr = np.linspace(0.2, 4.0, 4000, dtype = np.float32)
bif_data = bifurcation_logistic(r_arr,Neq, Niter)


r_repeated = np.repeat(r_arr, Niter)
bif_data_flat = bif_data.flatten()

fig_bif, ax_bif = plt.subplots(figsize=(6.2, 4.5))
ax_bif.scatter(r_repeated, bif_data_flat, s=0.1)
ax_bif.set_xlabel(r'$ r $', fontsize=15)
ax_bif.set_ylabel(r'$ x_i $', fontsize=15)
ax_bif.grid(True)
plt.show()




# --------------
# Bifurcation diagram with scatter plot (logistic map)
# Colored map

Neq = 1000
Niter = 1000
x_arr = np.arange(0, 1000, 1, dtype = np.int32)
bif_data = bifurcation_image(r_arr, Neq, Niter, logistic_map)

X, Y = np.meshgrid(r_arr, x_arr/1000)

fig_bif, ax_bif = plt.subplots(figsize=(6.2, 4.5))
ax_bif.scatter(X, Y, c=bif_data, cmap='Blues', s=0.1, alpha=0.8)
ax_bif.set_xlabel(r'$ r $', fontsize=15)
ax_bif.set_ylabel(r'$ x_i $', fontsize=15)
ax_bif.grid(True)
plt.show()




# --------------
# Bifurcation diagram with pixels (logistic map)
# Colored map

bifurcation_image = bifurcation_diagram(r_arr, Neq, Niter, logistic_map)

plt.figure(figsize=(10, 10))
plt.imshow(bifurcation_image, extent=[0.2, 2.0, 0.0, 1.0], aspect='auto', cmap='Blues', vmin=0, vmax=255, origin='lower')
plt.xlabel(r'$ r $', fontsize=15)
plt.ylabel(r'$ x_i $', fontsize=15)
plt.show()













