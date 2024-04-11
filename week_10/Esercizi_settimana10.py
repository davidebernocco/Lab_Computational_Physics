"""

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from Funz10 import MC_iteration, block_average, aver_DT




# ------------------------
L = 100
MCsteps = 2500
Nsteps_lst =  np.arange(1, MCsteps + 1, 1)

fig_dR2, ax_dR2 = plt.subplots(figsize=(6.2, 4.5))
fig_D, ax_D = plt.subplots(figsize=(6.2, 4.5))

for i in [0.03, 0.1, 0.2, 0.3, 0.5, 0.7]:
    the = MC_iteration(L, L, int(i*L**2), MCsteps)
    
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


the = MC_iteration(30, 30, 27, 225)
D_aver, sD_aver = block_average(the[1], 10)
Nsteps_lst =  np.arange(1, 225 + 1, 1)

fig_D, ax_D = plt.subplots(figsize=(6.2, 4.5))
ax_D.scatter(Nsteps_lst, the[1], marker='o', s=50)
ax_D.plot([1,225], [D_aver, D_aver], label=r'$ \langle D(t) \rangle_T $')
ax_D.set_xlabel(r'$ t [a.u.] $', fontsize=15)
ax_D.set_ylabel(r'$ D(t) $', fontsize=15)
ax_D.grid(True)
plt.show()


the = MC_iteration(100, 100, 300, 2500)
D_aver, sD_aver = block_average(the[1], 10)
Nsteps_lst =  np.arange(1, 2500 + 1, 1)

fig_D, ax_D = plt.subplots(figsize=(6.2, 4.5))
ax_D.scatter(Nsteps_lst, the[1], marker='o', s=50)
ax_D.plot([1,2500], [D_aver, D_aver], label=r'$ \langle D(t) \rangle_T $')
ax_D.set_xlabel(r'$ t [a.u.] $', fontsize=15)
ax_D.set_ylabel(r'$ D(t) $', fontsize=15)
ax_D.grid(True)
plt.show()









