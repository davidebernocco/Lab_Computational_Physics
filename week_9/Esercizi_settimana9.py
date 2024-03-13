"""
@author: david
"""

import numpy as np
from PIL import Image
#from ipywidgets import interact
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



# ------------------------- ESEMPIO DA INTERNET -------------------------------
def random_spin_lattice(N, M):
    return np.random.choice([-1,1], size=(N,M))

def display_spin_lattice(lattice):
    return Image.fromarray(np.uint8((lattice + 1) * 0.5 * 255))

lattice = random_spin_lattice(100, 100)

# Define a colormap
spin_color = mcolors.ListedColormap(['blue', 'red'])

# Display the lattice using matplotlib
plt.imshow(display_spin_lattice(lattice), cmap = spin_color)
plt.axis('off')  # Turn off axis
plt.show()

"""

def Ising_update(lattice, n, m, beta):
    total = 0
    N, M = lattice.shape
    for i in range(n-1, n+2):
        for j in range(m-1, m+2):
             if i == n and j == m:
                 continue
             total += lattice[i % N, j % M]
             
    dE = 2 * lattice[n, m] * total
    
    if dE <= 0:
        lattice[n, m] *= -1
    elif np.exp(-dE * beta) > np.random.rand():
        lattice[n, m] *= -1
    


def Ising_step(lattice, beta):
    N, M = lattice.shape
    for n_offset in range(2):
        for m_offset in range(2):
            for n in range(n_offset, N, 2):
                for m in range(m_offset, M, 2):
                    Ising_update(lattice, n, m, beta)
    return lattice


def display_sequence(images):
    def _show(frame=(0, len(images) - 1)):
        return display_spin_lattice(images[frame])
    return interact(_show)

# -----------------------------------------------------------------------------





@njit
def Metropolis_Boltzmann( v0, dvmax, n, kb, T, m):
    
    acc = 0
    velocity = np.zeros(n, dtype = np.float32)
    energy = np.zeros(n, dtype = np.float32)
    velocity[0] = v0
    energy[0] = (m / 2) * v0 ** 2
    
    E_t = (m / 2) * v0 ** 2
    v_t = v0
    
    for i in range(1, n):
        v_star = np.random.uniform(v_t - dvmax, v_t + dvmax)
        E_star = (m / 2) * v_star ** 2
        
        esp1v = ( -m * v_star ** 2 / ( 2 * kb * T) )  
        esp2v = ( -m * v_t ** 2 / ( 2 * kb * T) )    
        alphav = math.e ** (esp1v - esp2v)           
        
        if alphav >= np.random.rand() :
            v_t = v_star
            acc += 1
        
        esp1E = ( - E_star / ( kb * T) )  
        esp2E = ( - E_t / ( kb * T) )    
        alphaE = math.e ** (esp1E - esp2E) 
               
        
        if alphaE >= np.random.rand() :
            E_t = E_star
            
        velocity[i] = v_t
        energy[i] = E_t
            
    return velocity, energy, acc/n
"""




