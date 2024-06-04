"""
Plots and other numerical estimations (final project)

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

from FunzF import tent_map, sine_map, logistic_map, non_predicibility_vs_chaos
from FuncF import iteration_tent, iteration_sine, iteration_logistic
from FuncF import bifurcation_tent, bifurcation_sine, bifurcation_logistic
from FuncF import bifurcation_image, bifurcation_diagram
from FuncF import lyapunov_sine, lyapunov_logistic, entropy, beta_function
from FuncF import iteration_Henon, Lyapunov_spectrum_2D


# -----------------------------------------------------------------------------
# STUDY OF THE TENT MAP
# -----------------------------------------------------------------------------
    
    
# ------------
# Visualizing transition sequence (tent map)
# Try 4 values of r: 0.3, 1.07, 1.3, 1.9
# For 1<r<sqrt(2) there are 8 separate sets of x that correspond to the Julia set
# for 1<r<2 there are 2 unstable fixed points: x=0 and x=r/(r+1)
# for r=2 non-periodic dynamics happens only for x0 irrational
# for r=3 Certain points leave the interval [0,1] after a number of iterations. 
#         The points that never leave I form a so-called non-wandering set (or better Julia set).
#         This non-wandering set turns out to be a Cantor set

Neq = 10
Niter = 50
traiett1 = iteration_tent(0.37, 0.3, Neq, Niter)
traiett2 = iteration_tent(0.370001, 0.3, Neq, Niter)
Ntot = np.arange(Neq + Niter)

fig_traj, ax_traj = plt.subplots(figsize=(6.2, 4.5))
ax_traj.plot(Ntot, traiett1, color='b', label='x0 = 0.37')
ax_traj.plot(Ntot, traiett2, color='g', label='x0 = 0.370001')
ax_traj.set_xlabel(r'$ i $', fontsize=20)
ax_traj.set_ylabel(r'$ x_i $', fontsize=20)
legend = ax_traj.legend()
legend.set_title('Tent map (r = 0.3)', prop={'size': 12, 'weight': 'bold'})
ax_traj.grid(True)
plt.show()




# --------------
# Raw bifurcation diagram: all the points with same intensity (tent map)
# Zoom in the region 1<r<sqrt(2) to see the Julia set
Neq = 1000
Niter = 10000
r_arr = np.linspace(0.2, 2.0, 1000, dtype = np.float32)
bif_data = bifurcation_tent(np.float32(0.5), r_arr,Neq, Niter)


r_repeated = np.repeat(r_arr, Niter)
bif_data_flat = bif_data.flatten()

fig_bif, ax_bif = plt.subplots(figsize=(6.2, 4.5))
ax_bif.scatter(r_repeated, bif_data_flat, s=0.1)
ax_bif.set_xlabel(r'$ \mu $', fontsize=15)
ax_bif.set_ylabel(r'$ x_i $', fontsize=15)
ax_bif.grid(True)
plt.show()




# --------------
# Bifurcation diagram with scatter plot (tent map)
# Colored map

Neq = 1000
Niter = 10000
x_arr = np.arange(0, 1000, 1, dtype = np.int32)
bif_data = bifurcation_image(np.float32(np.sqrt(2)/5), r_arr, Neq, Niter, tent_map)

X, Y = np.meshgrid(r_arr, x_arr/1000)

fig_bif, ax_bif = plt.subplots(figsize=(6.2, 4.5))
ax_bif.scatter(X, Y, c=bif_data, cmap='Blues', s=0.1, alpha=0.8)
ax_bif.set_xlabel(r'$ \mu $', fontsize=15)
ax_bif.set_ylabel(r'$ x_i $', fontsize=15)
ax_bif.grid(True)
plt.show()




# --------------
# Bifurcation diagram with pixels (tent map)
# Colored map

bifurcation_image = bifurcation_diagram(np.float32(np.sqrt(2)/5), r_arr, Neq, Niter, tent_map)

plt.figure(figsize=(10, 10))
plt.imshow(bifurcation_image, extent=[0.2, 2.0, 0.0, 1.0], aspect='auto', cmap='Blues', vmin=0, vmax=255, origin='lower')
plt.xlabel(r'$ \mu $', fontsize=15)
plt.ylabel(r'$ x_i $', fontsize=15)
plt.show()







# -----------------------------------------------------------------------------
# STUDY OF THE SINE MAP
# -----------------------------------------------------------------------------
    
    
# ------------
# Visualizing transition sequence (sine map)
# Try 4 values of r: 1, 3.1, 3.48, 3.995
Neq = 10
Niter = 50
traiett1 = iteration_sine(0.37, 3.1, Neq, Niter)
traiett2 = iteration_sine(0.370001, 3.1, Neq, Niter)
Ntot = np.arange(Neq + Niter)

fig_traj, ax_traj = plt.subplots(figsize=(6.2, 4.5))
ax_traj.plot(Ntot, traiett1, color='b', label='x0 = 0.37')
ax_traj.plot(Ntot, traiett2, color='g', label='x0 = 0.370001')
ax_traj.set_xlabel(r'$ i $', fontsize=20)
ax_traj.set_ylabel(r'$ x_i $', fontsize=20)
legend = ax_traj.legend()
legend.set_title('Sine map (r = 3.1)', prop={'size': 12, 'weight': 'bold'})
ax_traj.grid(True)
plt.show()




# --------------
# Raw bifurcation diagram: all the points with same intensity (sine map)
Neq = 1000
Niter = 10000
r_arr = np.linspace(0.2, 1.0, 500, dtype = np.float32)
bif_data = bifurcation_sine(np.float32(0.5), r_arr,Neq, Niter)


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
bif_data = bifurcation_image(np.float32(np.sqrt(2)/5), r_arr, Neq, Niter, sine_map)

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

bifurcation_image = bifurcation_diagram(np.float32(np.sqrt(2)/5), r_arr, Neq, Niter, sine_map)

plt.figure(figsize=(10, 10))
plt.imshow(bifurcation_image, extent=[0.2, 1.0, 0.0, 1.0], aspect='auto', cmap='Blues', vmin=0, vmax=255, origin='lower')
plt.xlabel(r'$ r $', fontsize=15)
plt.ylabel(r'$ x_i $', fontsize=15)
plt.show()




# --------------
# Lyapunov exponent as function of r (sine map)
Neq = 1000
Niter = 1000
r_arr = np.linspace(0.2, 1.0, 500, dtype = np.float32)
l_data = lyapunov_sine(0.25, r_arr,Neq, Niter)

fig_lyap, ax_lyap = plt.subplots(figsize=(6.2, 4.5))
ax_lyap.scatter(r_arr, l_data, s=0.2)
ax_lyap.set_xlabel(r'$ r $', fontsize=15)
ax_lyap.set_ylabel(r'$ \lambda $', fontsize=15)
ax_lyap.grid(True)
plt.show()






# -----------------------------------------------------------------------------
# STUDY OF THE LOGISTIC MAP
# -----------------------------------------------------------------------------
    
    
# ------------
# Visualizing transition sequence (logistic map)
# Try 4 values of r: 1, 3.1, 3.48, 3.995
# For 0<=r<=1 one fixed point at x=0
# For 1<r<=3 one fixed point at x = (r-1)/r
# For 3<r<3.56995 the attractor is made of a discrete number of points (periodical doubling)
# For r=4 from almost all initial conditions the iterate sequence is chaotic. 
# Nevertheless, there exist an infinite number of initial conditions that lead to cycles
# Number 0f cycles of minimal lenght k:  (need to solve f(f(f......)) = x )
    #k=1 -> n=2 (1:x=0, 2:x=3/4)
    #k=2 -> n=1 (oscillates between x=(5-sqrt(5))/8 and x=(5+sqrt(5))/8 )
    #k=3 -> n=2 (1: oscillates between x=1/2*(1 + np.cos(np.pi/9)) , x~0.116978 , x~0.413176 )
    #           (2: oscillates between x~0.950484 , x~0.188255 , x~0.61126 )
    #etc...
Neq = 10
Niter = 50
traiett1 = iteration_logistic(0.37, 3.995, Neq, Niter)
traiett2 = iteration_logistic(0.370001, 3.995, Neq, Niter)
Ntot = np.arange(Neq + Niter)

fig_traj, ax_traj = plt.subplots(figsize=(6.2, 4.5))
ax_traj.plot(Ntot, traiett1, color='b', label='x0 = 0.37')
ax_traj.plot(Ntot, traiett2, color='g', label='x0 = 0.370001')
ax_traj.set_xlabel(r'$ i $', fontsize=20)
ax_traj.set_ylabel(r'$ x_i $', fontsize=20)
legend = ax_traj.legend()
legend.set_title('Logistic map (r = 3.995)', prop={'size': 12, 'weight': 'bold'})
ax_traj.grid(True)
plt.show()





# ---------------
# Chaos and randomness

arr_chaos, arr_random = non_predicibility_vs_chaos(3.995, 3000)

fig_cr = plt.figure(figsize=(6.2, 4.5))
ax_cr = fig_cr.add_subplot(111, projection='3d')
ax_cr.scatter(arr_chaos[0], arr_chaos[1], arr_chaos[2], marker='*', s=50, color='r', label='Chaotic trajectory')
ax_cr.scatter(arr_random[0], arr_random[1], arr_random[2], marker='o', s=2, color='b', label='Random trajectory')
ax_cr.set_xlabel(r'$ x_i $', fontsize=15)
ax_cr.set_ylabel(r'$ x_{i+1} $', fontsize=15)
ax_cr.set_zlabel(r'$ x_{i+2} $', fontsize=15)
ax_cr.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
ax_cr.grid(True)
plt.show()







# --------------
# Raw bifurcation diagram: all the points with same intensity (logistic map)
# Self-similarities:
# Zooming around r=3.8494344 we should see a period-doubling approach to chaos 3,6,12 ..

Neq = 1000
Niter = 10000
r_arr = np.linspace(0.2, 4.0, 2000, dtype = np.float32)
bif_data = bifurcation_logistic(np.float32(0.5), r_arr,Neq, Niter)


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
bif_data = bifurcation_image(np.float32(np.sqrt(2)/5), r_arr, Neq, Niter, logistic_map)

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

bifurcation_image = bifurcation_diagram(np.float32(np.sqrt(2)/5), r_arr, Neq, Niter, logistic_map)

plt.figure(figsize=(10, 10))
plt.imshow(bifurcation_image, extent=[0.2, 2.0, 0.0, 1.0], aspect='auto', cmap='Blues', vmin=0, vmax=255, origin='lower')
plt.xlabel(r'$ r $', fontsize=15)
plt.ylabel(r'$ x_i $', fontsize=15)
plt.show()



# --------------
# Lyapunov exponent as function of r (logistic map)
Neq = 1000
Niter = 1000
r_arr = np.linspace(0.2, 4.0, 2000, dtype = np.float32)
l_data = lyapunov_logistic(0.25, r_arr,Neq, Niter)

fig_lyap, ax_lyap = plt.subplots(figsize=(6.2, 4.5))
ax_lyap.scatter(r_arr, l_data, s=0.2)
ax_lyap.set_xlabel(r'$ r $', fontsize=15)
ax_lyap.set_ylabel(r'$ \lambda $', fontsize=15)
ax_lyap.grid(True)
plt.show()





# --------------
# Entropy as function of r (logistic map)
Neq = 1000
Niter = 1000
r_arr = np.linspace(0.2, 4.0, 2000, dtype = np.float32)
x_arr = np.arange(0, 1000, 1, dtype = np.int32)
s_data = entropy(np.float32(np.sqrt(2)/5), r_arr, Neq, Niter, logistic_map)

fig_entr, ax_entr = plt.subplots(figsize=(6.2, 4.5))
ax_entr.scatter(r_arr, s_data, s=1)
ax_entr.set_xlabel(r'$ r $', fontsize=15)
ax_entr.set_ylabel(r'$ Entropy $', fontsize=15)
ax_entr.grid(True)
plt.show()



# --------------
# Probability as a function of x for r=4 
Neq, Niter, Nstates = 1000, 1000000, 100

mimmo = iteration_logistic(np.float32(np.sqrt(2)/5), 4, Neq, Niter)
hist, bins = np.histogram(mimmo, Nstates, density=False)
bin_widths = np.diff(bins)
dens = hist / (Niter * bin_widths[0])

arr_x = np.linspace(0.001, 0.999, 1000, dtype=np.float32)
beta_arr = beta_function(arr_x)

fig_histo, ax_histo = plt.subplots(figsize=(6.2, 4.5))
ax_histo.bar(bins[:-1], dens, width=bin_widths, alpha=0.5, color='b', label='Logistic map (r = 4): Numerical distribution')
ax_histo.plot(arr_x, beta_arr, color='r', label='Logistic map (r = 4): Expected curve')
ax_histo.set_xlabel(r'$ x $', fontsize=20)
ax_histo.set_ylabel(r'$ p_{i} $', fontsize=20)
legend = ax_histo.legend()
ax_histo.grid(True)
plt.show()







# -----------------------------------------------------------------------------
# HENON MAP 2D
# -----------------------------------------------------------------------------

# -------------------
# Trajectory for fixed a and b

r_init = np.asarray([0.0,0.0], dtype=np.float32)
arr_henon = iteration_Henon(r_init, 1.4, 0.3, 10, 50)

fig_hen = plt.figure(figsize=(6.2, 4.5))
ax_hen = fig_hen.add_subplot(111)
ax_hen.scatter(arr_henon[0], arr_henon[1], marker='o', s=10, color='b', label='Chaotic trajectory')
ax_hen.set_xlabel(r'$ x_i $', fontsize=15)
ax_hen.set_ylabel(r'$ y_i $', fontsize=15)
ax_hen.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
ax_hen.grid(True)
plt.show()




# ---------------
# Estimation of Lyapunov exponents
lyap_henon = Lyapunov_spectrum_2D(r_init, 1.4, 0.3, 1000000)

