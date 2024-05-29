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
from FuncF import lyapunov_sine, lyapunov_logistic, entropy



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
r_arr = np.linspace(0.2, 1.0, 500, dtype = np.float32)
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




# --------------
# Lyapunov exponent as function of r (sine map)
Neq = 1000
Niter = 1000
r_arr = np.linspace(0.2, 1.0, 500, dtype = np.float32)
l_data = lyapunov_sine(r_arr,Neq, Niter)

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
r_arr = np.linspace(0.2, 4.0, 2000, dtype = np.float32)
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



# --------------
# Lyapunov exponent as function of r (logistic map)
Neq = 1000
Niter = 1000
r_arr = np.linspace(0.2, 4.0, 2000, dtype = np.float32)
l_data = lyapunov_logistic(r_arr,Neq, Niter)

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
s_data = entropy(r_arr, Neq, Niter, logistic_map)

fig_entr, ax_entr = plt.subplots(figsize=(6.2, 4.5))
ax_entr.scatter(r_arr, s_data[0], s=1)
ax_entr.set_xlabel(r'$ r $', fontsize=15)
ax_entr.set_ylabel(r'$ Entropy $', fontsize=15)
ax_entr.grid(True)
plt.show()



# --------------
# Probability as a function of x for r=4 (N.B. IT STRONGLY DEPENDS ON x0!!)
fig_entr, ax_entr = plt.subplots(figsize=(6.2, 4.5))
ax_entr.scatter(x_arr/1000, s_data[1], s=1)
ax_entr.set_xlabel(r'$ x $', fontsize=20)
ax_entr.set_ylabel(r'$ p_{i} $', fontsize=20)
ax_entr.grid(True)
plt.show()










"""
import numpy as np

def henon_map(x, a=1.4, b=0.3):
    """Compute the Hénon map for a given state x."""
    x_new = np.zeros_like(x)
    # Nonlinear Henon map equations
    x_new[0] = 1 - a * x[0]**2 + b * x[1]
    x_new[1] = x[0]
    return x_new

def linearized_henon(x, a=1.4, b=0.3):
    """Compute the linearized Hénon map for a given state x."""
    n = len(x) // 2
    x_new = np.zeros_like(x)
    # Nonlinear Henon map equations
    x_new[0] = 1 - a * x[0]**2 + b * x[1]
    x_new[1] = x[0]
    # Linearized Henon map equations
    x_new[2] = -2 * a * x[0] * x[2] + b * x[4]
    x_new[3] = -2 * a * x[0] * x[3] + b * x[5]
    x_new[4] = x[2]
    x_new[5] = x[3]
    return x_new

def gram_schmidt(vectors):
    """Apply the Gram-Schmidt process to the given set of vectors."""
    orthonormal_basis = np.zeros_like(vectors)
    for i in range(vectors.shape[1]):
        new_vector = vectors[:, i].copy()
        for j in range(i):
            new_vector -= np.dot(orthonormal_basis[:, j], vectors[:, i]) * orthonormal_basis[:, j]
        orthonormal_basis[:, i] = new_vector / np.linalg.norm(new_vector)
    return orthonormal_basis

def compute_lyapunov_exponents(a=1.4, b=0.3, irate=10, io=1e5, max_iter=1e6):
    n = 2
    nn = n * (n + 1)
    v = np.zeros(nn)
    ltot = np.zeros(n)
    znorm = np.zeros(n)

    # Initial conditions for nonlinear map
    v[:n] = np.random.uniform(-0.1, 0.1, size=n)

    # Initial conditions for linearized maps
    v[n:] = np.eye(n).flatten()

    t = 0
    m = 0

    while t < max_iter:
        # Iteration loop
        for _ in range(irate):
            x = v.copy()
            v = linearized_henon(x, a, b)
            t += 1

        # Gram-Schmidt reorthonormalization
        znorm[0] = np.linalg.norm(v[n:2*n])
        v[n:2*n] /= znorm[0]

        for j in range(1, n):
            for k in range(j):
                gsc = np.dot(v[n*k:n*(k+1)], v[n*j:n*(j+1)])
                v[n*j:n*(j+1)] -= gsc * v[n*k:n*(k+1)]
            znorm[j] = np.linalg.norm(v[n*j:n*(j+1)])
            v[n*j:n*(j+1)] /= znorm[j]

        # Update running vector magnitudes
        for k in range(n):
            if znorm[k] > 0:
                ltot[k] += np.log(znorm[k])

        m += 1

        # Output results every io iterations
        if m % io == 0:
            le = ltot / t
            lsum = np.sum(le)
            kmax = np.argmax(np.cumsum(le) > 0)
            dky = kmax + 1 - t * np.sum(le[:kmax + 1]) / ltot[kmax + 1] if ltot[0] > 0 else 0

            print(f"Time = {t:.0f}\tLEs =", end=" ")
            for exponent in le:
                print(f"{exponent: .8f}", end=" ")
            print(f"\tDky = {dky:.8f}")

if __name__ == "__main__":
    compute_lyapunov_exponents()
"""

