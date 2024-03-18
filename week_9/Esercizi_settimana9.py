"""
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

from Funz9 import random_spin_lattice, display_spin_lattice, initial_energy
from Funz9 import ordered_spin_lattice, Ising_conditions, accumulation
from Funz9 import Ising_conditions_open, initial_energy_open, accumulation_open
from Funz9 import animation_Ising, block_average,averaged_quantities
from Funz9 import average_error, T_variation




# -----------------------------------------------------------------------------
# STUDY OF THE EQUILIBRATION TIME
# -----------------------------------------------------------------------------

Beta = 1
neqs = 0
nmcs = 10**6
n1 = n2 = 30


results = accumulation(n1, n2, Beta, neqs, nmcs)
Nsteps_lst = np.arange(int(nmcs/(n1*n2))+1)

fig_m, ax_m = plt.subplots(figsize=(6.2, 4.5))
ax_m.plot(Nsteps_lst, results[0], marker='o' )
ax_m.set_xlabel(r'$ MCsteps/{} $'.format(n1*n2), fontsize=15)
ax_m.set_ylabel(r'$ M / N $', fontsize=15)
ax_m.grid(True)
plt.show()

fig_e, ax_e = plt.subplots(figsize=(6.2, 4.5))
ax_e.plot(Nsteps_lst, results[2], marker='o' )
ax_e.set_xlabel(r'$ MCsteps/{} $'.format(n1*n2), fontsize=15)
ax_e.set_ylabel(r'$ E / N $', fontsize=15)
ax_e.grid(True)
plt.show()






# -----------------------------------------------------------------------------
# OPEN BOUNDARY CONDITIONS
# -----------------------------------------------------------------------------

Beta = 1.5
neqs = 0
nmcs = 10**7
n1 = n2 = 30


results_open = accumulation_open(n1, n2, Beta, neqs, nmcs)
Nsteps_lst = np.arange(int(nmcs/(n1*n2))+1)

fig_mo, ax_mo = plt.subplots(figsize=(6.2, 4.5))
ax_mo.plot(Nsteps_lst, results_open[0], marker='o', label='Open Boundary Conditions')
ax_mo.set_xlabel(r'$ MCsteps/{} $'.format(n1*n2), fontsize=15)
ax_mo.set_ylabel(r'$ M / N $', fontsize=15)
ax_mo.legend()
ax_mo.grid(True)
plt.show()

fig_eo, ax_eo = plt.subplots(figsize=(6.2, 4.5))
ax_eo.plot(Nsteps_lst, results_open[2], marker='o', label='Open Boundary Conditions')
ax_eo.set_xlabel(r'$ MCsteps/{} $'.format(n1*n2), fontsize=15)
ax_eo.set_ylabel(r'$ E / N $', fontsize=15)
ax_eo.legend()
ax_eo.grid(True)
plt.show()






# -----------------------------------------------------------------------------
# PLOTTING PHYSICAL QUANTITIES: <|M|>/N, <E>/N, X/N, Cv/N
# -----------------------------------------------------------------------------

resultsT = T_variation(30, 30, 1, 4, 0.25, 10*5, 10**7, 100)
Nsteps_lst =  np.arange(1, 4, 0.25)

fig_mT, ax_mT = plt.subplots(figsize=(6.2, 4.5))
ax_mT.scatter(Nsteps_lst, resultsT[0], marker='o', s=50)
ax_mT.errorbar(Nsteps_lst, resultsT[0], yerr=resultsT[4], fmt='.',  capsize=5, color='black')
ax_mT.set_xlabel(r'$ T [K] $', fontsize=15)
ax_mT.set_ylabel(r'$ \langle |M| \rangle / N $', fontsize=15)
ax_mT.grid(True)
plt.show()

fig_eT, ax_eT = plt.subplots(figsize=(6.2, 4.5))
ax_eT.scatter(Nsteps_lst, resultsT[1], marker='o', s=50)
ax_eT.errorbar(Nsteps_lst, resultsT[1], yerr=resultsT[5], fmt='.',  capsize=5, color='black')
ax_eT.set_xlabel(r'$ T [K] $', fontsize=15)
ax_eT.set_ylabel(r'$ \langle E \rangle / N $', fontsize=15)
ax_eT.grid(True)
plt.show()

fig_cT, ax_cT = plt.subplots(figsize=(6.2, 4.5))
ax_cT.scatter(Nsteps_lst, resultsT[2], marker='o', s=50)
ax_cT.errorbar(Nsteps_lst, resultsT[2], yerr=resultsT[6], fmt='.',  capsize=5, color='black')
ax_cT.set_xlabel(r'$ T [K] $', fontsize=15)
ax_cT.set_ylabel(r'$ c_V / N $', fontsize=15)
ax_cT.grid(True)
plt.show()

fig_xT, ax_xT = plt.subplots(figsize=(6.2, 4.5))
ax_xT.scatter(Nsteps_lst, resultsT[3], marker='o', s=50)
ax_xT.errorbar(Nsteps_lst, resultsT[3], yerr=resultsT[7], fmt='.',  capsize=5, color='black')
ax_xT.set_xlabel(r'$ T [K] $', fontsize=15)
ax_xT.set_ylabel(r'$ \chi / N $', fontsize=15)
ax_xT.grid(True)
plt.show()






# -----------------------------------------------------------------------------
# CONFIGURATION SEQUENCE ANIMATION
# -----------------------------------------------------------------------------

animation_Ising(30, 30, 1.5, 200000, 'Ising_2D.gif')
plt.close('all')










