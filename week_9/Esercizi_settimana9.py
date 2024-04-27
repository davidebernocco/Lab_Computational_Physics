"""
Plots and other numerical estimations (10th week)

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from scipy.optimize import curve_fit

from Funz9 import accumulation, accumulation_open, animation_Ising
from Funz9 import T_variation, c_as_derivative, fitE, save_frames_from_gif

from data9 import n_eq4, n_eq20, n_eq30, n_eq4c, n_eq20c, n_eq30c
from data9 import m_4, e_4, c_4, x_4, sm_4, se_4, sc_4, sx_4, cT_4, scT_4
from data9 import m_10, e_10, c_10, x_10, sm_10, se_10, sc_10, sx_10, cT_10, scT_10
from data9 import m_15, e_15, c_15, x_15, sm_15, se_15, sc_15, sx_15, cT_15, scT_15
from data9 import m_20, e_20, c_20, x_20, sm_20, se_20, sc_20, sx_20, cT_20, scT_20
from data9 import m_30, e_30, c_30, x_30, sm_30, se_30, sc_30, sx_30, cT_30, scT_30
from data9 import m_4_O, e_4_O, c_4_O, x_4_O, sm_4_O, se_4_O, sc_4_O, sx_4_O, cT_4_O, scT_4_O
from data9 import m_10_O, e_10_O, c_10_O, x_10_O, sm_10_O, se_10_O, sc_10_O, sx_10_O, cT_10_O, scT_10_O
from data9 import m_15_O, e_15_O, c_15_O, x_15_O, sm_15_O, se_15_O, sc_15_O, sx_15_O, cT_15_O, scT_15_O
from data9 import m_20_O, e_20_O, c_20_O, x_20_O, sm_20_O, se_20_O, sc_20_O, sx_20_O, cT_20_O, scT_20_O
from data9 import m_30_O, e_30_O, c_30_O, x_30_O, sm_30_O, se_30_O, sc_30_O, sx_30_O, cT_30_O, scT_30_O



# -----------------------------------------------------------------------------
# ISING MODEL
# -----------------------------------------------------------------------------


# --------------------------
# e and m plots

Beta = 0.25
neqs = 0
nmcs = 10**6
n1 = n2 = 30
N = n1*n2

results = accumulation(n1, n2, Beta, neqs, nmcs)
Nsteps_lst = np.arange(int(nmcs/(n1*n2))+1)

fig_m, ax_m = plt.subplots(figsize=(6.2, 4.5))
ax_m.plot(Nsteps_lst, results[0]/N, marker='o' )
ax_m.set_xlabel(r'$ MCsteps/{} $'.format(n1*n2), fontsize=15)
ax_m.set_ylabel(r'$ M / N $', fontsize=15)
ax_m.grid(True)
plt.show()

fig_e, ax_e = plt.subplots(figsize=(6.2, 4.5))
ax_e.plot(Nsteps_lst, results[2]/N, marker='o' )
ax_e.set_xlabel(r'$ MCsteps/{} $'.format(n1*n2), fontsize=15)
ax_e.set_ylabel(r'$ E / N $', fontsize=15)
ax_e.grid(True)
plt.show()





# --------------------------
# Study on the equilibration time

# m and e are plotted for different beta and an EYE-MADE ESTIMATION is
# made in order to identify the equilibration sequence length.
# Then the procedure is repeated for different lattice size.

beta_lst = np.asarray([0.25, 0.35, 0.4, 0.425, 0.45, 0.5, 0.55, 0.65, 0.75, 0.85, 0.95])

fig_eq, ax_eq = plt.subplots(figsize=(6.2, 4.5))
ax_eq.plot(1/beta_lst, n_eq30, marker='o', label='Square lattice 30x30' )
ax_eq.plot(1/beta_lst, n_eq20, marker='o', label='Square lattice 20x20' )
ax_eq.plot(1/beta_lst, n_eq4, marker='o', label='Square lattice 4x4' )
ax_eq.set_xlabel('T [K]', fontsize=15)
ax_eq.set_ylabel('n° equil. steps', fontsize=15)
ax_eq.legend()
ax_eq.grid(True)
plt.show()





# --------------------------
# Study of equilibration time for chessboard starting configuration

fig_eq, ax_eq = plt.subplots(figsize=(6.2, 4.5))
ax_eq.plot(1/beta_lst, n_eq30c, marker='o', label='Square lattice 30x30' )
ax_eq.plot(1/beta_lst, n_eq20c, marker='o', label='Square lattice 20x20' )
ax_eq.plot(1/beta_lst, n_eq4c, marker='o', label='Square lattice 4x4' )
ax_eq.set_xlabel('T [K]', fontsize=15)
ax_eq.set_ylabel('n° equil. steps', fontsize=15)
ax_eq.legend()
ax_eq.grid(True)
plt.show()

# The estimation is UNSATISFACTORY, because with different runs the fluctuations
# are quite large. We cannot say that for all L= {30, 20, 4} the trend is
# similar to the one with random starting configuration! However, it seems
# the equilibration time here is in general lower.





# --------------------------
# Open Boundary Conditions: e and m plots

Beta = 1.5
neqs = 0
nmcs = 10**7
n1 = n2 = 30
N = n1*n2


results_open = accumulation_open(n1, n2, Beta, neqs, nmcs)
Nsteps_lst = np.arange(int(nmcs/(n1*n2))+1)

fig_mo, ax_mo = plt.subplots(figsize=(6.2, 4.5))
ax_mo.plot(Nsteps_lst, results_open[0]/N, marker='o', label='Open Boundary Conditions')
ax_mo.set_xlabel(r'$ MCsteps/{} $'.format(n1*n2), fontsize=15)
ax_mo.set_ylabel(r'$ M / N $', fontsize=15)
ax_mo.legend()
ax_mo.grid(True)
plt.show()

fig_eo, ax_eo = plt.subplots(figsize=(6.2, 4.5))
ax_eo.plot(Nsteps_lst, results_open[2]/N, marker='o', label='Open Boundary Conditions')
ax_eo.set_xlabel(r'$ MCsteps/{} $'.format(n1*n2), fontsize=15)
ax_eo.set_ylabel(r'$ E / N $', fontsize=15)
ax_eo.legend()
ax_eo.grid(True)
plt.show()





# --------------------------
# Plotting physical quantities: <|M|>/N, <E>/N, X/N, Cv/N

# All these quantities are plotted for fifferent lattice sizes: the obtained
# data are collected in the module "data9" for ease.

resultsT = T_variation(4, 4, 1, 4, 0.1, 10**4, int(1.6*10**5), 100)
Nsteps_lst =  np.arange(1, 4, 0.1)

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
ax_cT.set_ylabel(r'$ c_V $', fontsize=15)
ax_cT.grid(True)
plt.show()


lst_cT, lst_scT = c_as_derivative(4,4,resultsT[1], 0.1, resultsT[5]) # Modify dT / it is = resultsT
fig_cTder, ax_cTder = plt.subplots(figsize=(6.2, 4.5))
ax_cTder.scatter(Nsteps_lst[1:len(Nsteps_lst)-1], lst_cT, marker='o', s=50, label='c as numerical derivative')
ax_cTder.errorbar(Nsteps_lst[1:len(Nsteps_lst)-1], lst_cT, yerr=lst_scT, fmt='.',  capsize=5, color='black')
ax_cTder.set_xlabel(r'$ T [K] $', fontsize=15)
ax_cTder.set_ylabel(r'$ c_V  $', fontsize=15)
ax_cTder.legend()
ax_cTder.grid(True)
plt.show()


fig_xT, ax_xT = plt.subplots(figsize=(6.2, 4.5))
ax_xT.scatter(Nsteps_lst, resultsT[3], marker='o', s=50)
ax_xT.errorbar(Nsteps_lst, resultsT[3], yerr=resultsT[7], fmt='.',  capsize=5, color='black')
ax_xT.set_xlabel(r'$ T [K] $', fontsize=15)
ax_xT.set_ylabel(r'$ \chi  $', fontsize=15)
ax_xT.grid(True)
plt.show()





# ------------------
# Collective plots PBC

Nsteps_lst =  np.arange(1, 4, 0.1)
fig_mTOT, ax_mTOT = plt.subplots(figsize=(6.2, 4.5))
ax_mTOT.plot(Nsteps_lst, m_4, marker='o', label='L = 4' )
ax_mTOT.errorbar(Nsteps_lst, m_4, yerr=sm_4, fmt='.',  capsize=5, color='black')
ax_mTOT.plot(Nsteps_lst, m_10, marker='o', label='L = 10' )
ax_mTOT.errorbar(Nsteps_lst, m_10, yerr=sm_10, fmt='.',  capsize=5, color='black')
ax_mTOT.plot(Nsteps_lst, m_15_O, marker='o', label='L = 15' )
ax_mTOT.errorbar(Nsteps_lst, m_15, yerr=sm_15, fmt='.',  capsize=5, color='black')
ax_mTOT.plot(Nsteps_lst, m_20, marker='o', label='L = 20' )
ax_mTOT.errorbar(Nsteps_lst, m_20, yerr=sm_20, fmt='.',  capsize=5, color='black')
ax_mTOT.plot(Nsteps_lst, m_30, marker='o', label='L = 30' )
ax_mTOT.errorbar(Nsteps_lst, m_30, yerr=sm_30, fmt='.',  capsize=5, color='black')
ax_mTOT.set_xlabel('T [K]', fontsize=15)
ax_mTOT.set_ylabel(r'$ \langle |M| \rangle / N $', fontsize=20)
ax_mTOT.legend()
ax_mTOT.grid(True)
plt.show()


fig_eTOT, ax_eTOT = plt.subplots(figsize=(6.2, 4.5))
ax_eTOT.plot(Nsteps_lst, e_4, marker='o', label='L = 4' )
ax_eTOT.errorbar(Nsteps_lst, e_4, yerr=se_4, fmt='.',  capsize=5, color='black')
ax_eTOT.plot(Nsteps_lst, e_10, marker='o', label='L = 10' )
ax_eTOT.errorbar(Nsteps_lst, e_10, yerr=se_10, fmt='.',  capsize=5, color='black')
ax_eTOT.plot(Nsteps_lst, e_15, marker='o', label='L = 15' )
ax_eTOT.errorbar(Nsteps_lst, e_15, yerr=se_15, fmt='.',  capsize=5, color='black')
ax_eTOT.plot(Nsteps_lst, e_20, marker='o', label='L = 20' )
ax_eTOT.errorbar(Nsteps_lst, e_20, yerr=se_20, fmt='.',  capsize=5, color='black')
ax_eTOT.plot(Nsteps_lst, e_30, marker='o', label='L = 30' )
ax_eTOT.errorbar(Nsteps_lst, e_30, yerr=se_30, fmt='.',  capsize=5, color='black')
ax_eTOT.set_xlabel('T [K]', fontsize=15)
ax_eTOT.set_ylabel(r'$ \langle E \rangle / N $', fontsize=20)
ax_eTOT.legend()
ax_eTOT.grid(True)
plt.show()


fig_cTOT, ax_cTOT = plt.subplots(figsize=(6.2, 4.5))
ax_cTOT.plot(Nsteps_lst, c_4, marker='o', label='L = 4' )
ax_cTOT.errorbar(Nsteps_lst, c_4, yerr=sc_4, fmt='.',  capsize=5, color='black')
ax_cTOT.plot(Nsteps_lst, c_10, marker='o', label='L = 10' )
ax_cTOT.errorbar(Nsteps_lst, c_10, yerr=sc_10, fmt='.',  capsize=5, color='black')
ax_cTOT.plot(Nsteps_lst, c_15, marker='o', label='L = 15' )
ax_cTOT.errorbar(Nsteps_lst, c_15, yerr=sc_15, fmt='.',  capsize=5, color='black')
ax_cTOT.plot(Nsteps_lst, c_20, marker='o', label='L = 20' )
ax_cTOT.errorbar(Nsteps_lst, c_20, yerr=sc_20, fmt='.',  capsize=5, color='black')
ax_cTOT.plot(Nsteps_lst, c_30, marker='o', label='L = 30' )
ax_cTOT.errorbar(Nsteps_lst, c_30, yerr=sc_30, fmt='.',  capsize=5, color='black')
ax_cTOT.set_xlabel('T [K]', fontsize=15)
ax_cTOT.set_ylabel(r'$ c_V $', fontsize=20)
ax_cTOT.legend()
ax_cTOT.grid(True)
plt.show()


fig_xTOT, ax_xTOT = plt.subplots(figsize=(6.2, 4.5))
ax_xTOT.plot(Nsteps_lst, x_4, marker='o', label='L = 4' )
ax_xTOT.errorbar(Nsteps_lst, x_4, yerr=sx_4, fmt='.',  capsize=5, color='black')
ax_xTOT.plot(Nsteps_lst, x_10, marker='o', label='L = 10' )
ax_xTOT.errorbar(Nsteps_lst, x_10, yerr=sx_10, fmt='.',  capsize=5, color='black')
ax_xTOT.plot(Nsteps_lst, x_15, marker='o', label='L = 15' )
ax_xTOT.errorbar(Nsteps_lst, x_15, yerr=sx_15, fmt='.',  capsize=5, color='black')
ax_xTOT.plot(Nsteps_lst, x_20, marker='o', label='L = 20' )
ax_xTOT.errorbar(Nsteps_lst, x_20, yerr=sx_20, fmt='.',  capsize=5, color='black')
ax_xTOT.plot(Nsteps_lst, x_30, marker='o', label='L = 30' )
ax_xTOT.errorbar(Nsteps_lst, x_30, yerr=sx_30, fmt='.',  capsize=5, color='black')
ax_xTOT.set_xlabel('T [K]', fontsize=15)
ax_xTOT.set_ylabel(r'$ \chi $', fontsize=20)
ax_xTOT.legend()
ax_xTOT.grid(True)
plt.show()


fig_ctTOT, ax_ctTOT = plt.subplots(figsize=(6.2, 4.5))
ax_ctTOT.plot(Nsteps_lst[1:len(Nsteps_lst)-1], cT_4, marker='^', markersize=7, label='L = 4' )
ax_ctTOT.errorbar(Nsteps_lst[1:len(Nsteps_lst)-1], cT_4, yerr=scT_4, fmt='.',  capsize=5, color='black')
ax_ctTOT.plot(Nsteps_lst[1:len(Nsteps_lst)-1], cT_10, marker='^', markersize=7, label='L = 10' )
ax_ctTOT.errorbar(Nsteps_lst[1:len(Nsteps_lst)-1], cT_10, yerr=scT_10, fmt='.',  capsize=5, color='black')
ax_ctTOT.plot(Nsteps_lst[1:len(Nsteps_lst)-1], cT_15, marker='^', markersize=7, label='L = 15' )
ax_ctTOT.errorbar(Nsteps_lst[1:len(Nsteps_lst)-1], cT_15, yerr=scT_15, fmt='.',  capsize=5, color='black')
ax_ctTOT.plot(Nsteps_lst[1:len(Nsteps_lst)-1], cT_20, marker='^', markersize=7, label='L = 20' )
ax_ctTOT.errorbar(Nsteps_lst[1:len(Nsteps_lst)-1], cT_20, yerr=scT_20, fmt='.',  capsize=5, color='black')
ax_ctTOT.plot(Nsteps_lst[1:len(Nsteps_lst)-1], cT_30, marker='^', markersize=7, label='L = 30' )
ax_ctTOT.errorbar(Nsteps_lst[1:len(Nsteps_lst)-1], cT_30, yerr=scT_30, fmt='.',  capsize=5, color='black')
ax_ctTOT.set_xlabel('T [K]', fontsize=15)
ax_ctTOT.set_ylabel(r'$ c_V $', fontsize=20)
ax_ctTOT.legend()
ax_ctTOT.grid(True)
plt.show()



# FITTING ENERGY DATA WITH SIGMOID

Nsteps_lst =  np.arange(1, 4, 0.1) 

parE, covE = curve_fit(fitE, Nsteps_lst, m_4, sigma=sm_4, p0=[-0.4, 1, 2.3, -1.5], method='trf')
p1, p2, p3, p4 = parE

fig_E, ax_E = plt.subplots(figsize=(6.2, 4.5))
ax_E.scatter(Nsteps_lst, m_4, marker='o', s=50)
ax_E.errorbar(Nsteps_lst, m_4, yerr=sm_4, fmt='.', capsize=5, color='black')
ax_E.plot(Nsteps_lst, fitE(Nsteps_lst, p1, p2, p3, p4), label='Fit curve', color='crimson')
ax_E.set_xlabel(r'$ T [K] $', fontsize=15)
ax_E.set_ylabel(r'$ \langle |M| \rangle / N $', fontsize=20)
ax_E.legend()
ax_E.grid(True)
plt.show()







# --------------------------------
# Plotting physical quantities with OBC: <|M|>/N, <E>/N, X/N, Cv/N
# Collective plots OBC


Nsteps_lst =  np.arange(0.5, 3.5, 0.1)
fig_mTOT, ax_mTOT = plt.subplots(figsize=(6.2, 4.5))
ax_mTOT.plot(Nsteps_lst, m_4_O, marker='o', label='L = 4' )
ax_mTOT.errorbar(Nsteps_lst, m_4_O, yerr=sm_4_O, fmt='.',  capsize=5, color='black')
ax_mTOT.plot(Nsteps_lst, m_10_O, marker='o', label='L = 10' )
ax_mTOT.errorbar(Nsteps_lst, m_10_O, yerr=sm_10_O, fmt='.',  capsize=5, color='black')
ax_mTOT.plot(Nsteps_lst, m_15_O, marker='o', label='L = 15' )
ax_mTOT.errorbar(Nsteps_lst, m_15_O, yerr=sm_15_O, fmt='.',  capsize=5, color='black')
ax_mTOT.plot(Nsteps_lst, m_20_O, marker='o', label='L = 20' )
ax_mTOT.errorbar(Nsteps_lst, m_20_O, yerr=sm_20_O, fmt='.',  capsize=5, color='black')
ax_mTOT.plot(Nsteps_lst, m_30_O, marker='o', label='L = 30' )
ax_mTOT.errorbar(Nsteps_lst, m_30_O, yerr=sm_30_O, fmt='.',  capsize=5, color='black')
ax_mTOT.set_xlabel('T [K]', fontsize=15)
ax_mTOT.set_ylabel(r'$ \langle |M| \rangle / N $', fontsize=20)
ax_mTOT.legend()
ax_mTOT.grid(True)
plt.show()


fig_eTOT, ax_eTOT = plt.subplots(figsize=(6.2, 4.5))
ax_eTOT.plot(Nsteps_lst, e_4_O, marker='o', label='L = 4' )
ax_eTOT.errorbar(Nsteps_lst, e_4_O, yerr=se_4_O, fmt='.',  capsize=5, color='black')
ax_eTOT.plot(Nsteps_lst, e_10_O, marker='o', label='L = 10' )
ax_eTOT.errorbar(Nsteps_lst, e_10_O, yerr=se_10_O, fmt='.',  capsize=5, color='black')
ax_eTOT.plot(Nsteps_lst, e_15_O, marker='o', label='L = 15' )
ax_eTOT.errorbar(Nsteps_lst, e_15_O, yerr=se_15_O, fmt='.',  capsize=5, color='black')
ax_eTOT.plot(Nsteps_lst, e_20_O, marker='o', label='L = 20' )
ax_eTOT.errorbar(Nsteps_lst, e_20_O, yerr=se_20_O, fmt='.',  capsize=5, color='black')
ax_eTOT.plot(Nsteps_lst, e_30_O, marker='o', label='L = 30' )
ax_eTOT.errorbar(Nsteps_lst, e_30_O, yerr=se_30_O, fmt='.',  capsize=5, color='black')
ax_eTOT.set_xlabel('T [K]', fontsize=15)
ax_eTOT.set_ylabel(r'$ \langle E \rangle / N $', fontsize=20)
ax_eTOT.legend()
ax_eTOT.grid(True)
plt.show()


fig_cTOT, ax_cTOT = plt.subplots(figsize=(6.2, 4.5))
ax_cTOT.plot(Nsteps_lst, c_4_O, marker='o', label='L = 4' )
ax_cTOT.errorbar(Nsteps_lst, c_4_O, yerr=sc_4_O, fmt='.',  capsize=5, color='black')
ax_cTOT.plot(Nsteps_lst, c_10_O, marker='o', label='L = 10' )
ax_cTOT.errorbar(Nsteps_lst, c_10_O, yerr=sc_10_O, fmt='.',  capsize=5, color='black')
ax_cTOT.plot(Nsteps_lst, c_15_O, marker='o', label='L = 15' )
ax_cTOT.errorbar(Nsteps_lst, c_15_O, yerr=sc_15_O, fmt='.',  capsize=5, color='black')
ax_cTOT.plot(Nsteps_lst, c_20_O, marker='o', label='L = 20' )
ax_cTOT.errorbar(Nsteps_lst, c_20_O, yerr=sc_20_O, fmt='.',  capsize=5, color='black')
ax_cTOT.plot(Nsteps_lst, c_30_O, marker='o', label='L = 30' )
ax_cTOT.errorbar(Nsteps_lst, c_30_O, yerr=sc_30_O, fmt='.',  capsize=5, color='black')
ax_cTOT.set_xlabel('T [K]', fontsize=15)
ax_cTOT.set_ylabel(r'$ c_V $', fontsize=20)
ax_cTOT.legend()
ax_cTOT.grid(True)
plt.show()


fig_xTOT, ax_xTOT = plt.subplots(figsize=(6.2, 4.5))
ax_xTOT.plot(Nsteps_lst, x_4_O, marker='o', label='L = 4' )
ax_xTOT.errorbar(Nsteps_lst, x_4_O, yerr=sx_4_O, fmt='.',  capsize=5, color='black')
ax_xTOT.plot(Nsteps_lst, x_10_O, marker='o', label='L = 10' )
ax_xTOT.errorbar(Nsteps_lst, x_10_O, yerr=sx_10_O, fmt='.',  capsize=5, color='black')
ax_xTOT.plot(Nsteps_lst, x_15_O, marker='o', label='L = 15' )
ax_xTOT.errorbar(Nsteps_lst, x_15_O, yerr=sx_15_O, fmt='.',  capsize=5, color='black')
ax_xTOT.plot(Nsteps_lst, x_20_O, marker='o', label='L = 20' )
ax_xTOT.errorbar(Nsteps_lst, x_20_O, yerr=sx_20_O, fmt='.',  capsize=5, color='black')
ax_xTOT.plot(Nsteps_lst, x_30_O, marker='o', label='L = 30' )
ax_xTOT.errorbar(Nsteps_lst, x_30_O, yerr=sx_30_O, fmt='.',  capsize=5, color='black')
ax_xTOT.set_xlabel('T [K]', fontsize=15)
ax_xTOT.set_ylabel(r'$ \chi $', fontsize=20)
ax_xTOT.legend()
ax_xTOT.grid(True)
plt.show()


fig_ctTOT, ax_ctTOT = plt.subplots(figsize=(6.2, 4.5))
ax_ctTOT.plot(Nsteps_lst[1:len(Nsteps_lst)-1], cT_4_O, marker='^', markersize=7, label='L = 4' )
ax_ctTOT.errorbar(Nsteps_lst[1:len(Nsteps_lst)-1], cT_4_O, yerr=scT_4_O, fmt='.',  capsize=5, color='black')
ax_ctTOT.plot(Nsteps_lst[1:len(Nsteps_lst)-1], cT_10_O, marker='^', markersize=7, label='L = 10' )
ax_ctTOT.errorbar(Nsteps_lst[1:len(Nsteps_lst)-1], cT_10_O, yerr=scT_10_O, fmt='.',  capsize=5, color='black')
ax_ctTOT.plot(Nsteps_lst[1:len(Nsteps_lst)-1], cT_15_O, marker='^', markersize=7, label='L = 15' )
ax_ctTOT.errorbar(Nsteps_lst[1:len(Nsteps_lst)-1], cT_15_O, yerr=scT_15_O, fmt='.',  capsize=5, color='black')
ax_ctTOT.plot(Nsteps_lst[1:len(Nsteps_lst)-1], cT_20_O, marker='^', markersize=7, label='L = 20' )
ax_ctTOT.errorbar(Nsteps_lst[1:len(Nsteps_lst)-1], cT_20_O, yerr=scT_20_O, fmt='.',  capsize=5, color='black')
ax_ctTOT.plot(Nsteps_lst[1:len(Nsteps_lst)-1], cT_30_O, marker='^', markersize=7, label='L = 30' )
ax_ctTOT.errorbar(Nsteps_lst[1:len(Nsteps_lst)-1], cT_30_O, yerr=scT_30_O, fmt='.',  capsize=5, color='black')
ax_ctTOT.set_xlabel('T [K]', fontsize=15)
ax_ctTOT.set_ylabel(r'$ c_V $', fontsize=20)
ax_ctTOT.legend()
ax_ctTOT.grid(True)
plt.show()



# FITTING ENERGY DATA WITH SIGMOID

Nsteps_lst =  np.arange(0.5, 3.5, 0.1) 

parE, covE = curve_fit(fitE, Nsteps_lst, m_4_O, sigma=sm_4_O, p0=[-0.4, 1, 2.3, -1.5], method='trf')
p1, p2, p3, p4 = parE

fig_E, ax_E = plt.subplots(figsize=(6.2, 4.5))
ax_E.scatter(Nsteps_lst, m_4_O, marker='o', s=50)
ax_E.errorbar(Nsteps_lst, m_4_O, yerr=sm_4_O, fmt='.', capsize=5, color='black')
ax_E.plot(Nsteps_lst, fitE(Nsteps_lst, p1, p2, p3, p4), label='Fit curve', color='crimson')
ax_E.set_xlabel(r'$ T [K] $', fontsize=15)
ax_E.set_ylabel(r'$ \langle |M| \rangle / N $', fontsize=20)
ax_E.legend()
ax_E.grid(True)
plt.show()








# ------------------
# CONFIGURATION SEQUENCE ANIMATION

animation_Ising(30, 30, 1.5, 200000, 'Ising_2D_random.gif')
plt.close('all')


# Save gif animation frame by frame, so I can insert the animation on LateX file
gif_path = r"C:\Users\david\Desktop\Magistrale_Trieste\Primo anno\Secondo Semestre\Laboratorioi di fisica computazionale\week_9\Ising_2D_random.gif"
output_folder = r"C:\Users\david\Desktop\Magistrale_Trieste\Primo anno\Secondo Semestre\Laboratorioi di fisica computazionale\week_9\Images_9\Ising_evolution_random"
save_frames_from_gif(gif_path, output_folder)


# Save gif animation frame by frame, so I can insert the animation on LateX file
gif_path = r"C:\Users\david\Desktop\Magistrale_Trieste\Primo anno\Secondo Semestre\Laboratorioi di fisica computazionale\week_9\Ising_2D_chessboard.gif"
output_folder = r"C:\Users\david\Desktop\Magistrale_Trieste\Primo anno\Secondo Semestre\Laboratorioi di fisica computazionale\week_9\Images_9\Ising_evolution_chessboard"
save_frames_from_gif(gif_path, output_folder)







