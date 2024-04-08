"""
Library of self-made functions needed for the codes implemented for the exercises of the 9th week

@author: david 
"""

import numpy as np
from PIL import Image
from numba import njit
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.colors as mcolors
from matplotlib.animation import PillowWriter
from PIL import Image, ImageSequence
import os




def random_spin_lattice(N, M):
    return np.random.choice([-1,1], size=(N,M))


def ordered_spin_lattice(N, M):
    lattice = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            if (i+j)%2 == 0:
                lattice[i,j] = 1
            else:
                lattice[i,j] = -1   
    return lattice


def display_spin_lattice(lattice):
    return Image.fromarray(np.uint8((lattice + 1) * 0.5 * 255))


@njit
def initial_energy(s):
    N, M = s.shape
    total = 0
    for i in range(N):
        for j in range(M):
            total += -s[i, j] * (s[(i-1)%N, j] + s[i, (j+1)%M])   
    return total

            


@njit
def Ising_conditions(s, beta):
    N, M = s.shape
    i = np.random.randint(0, N)
    j =  np.random.randint(0, M)
    
    NNsum = s[(i+1)%N, j] + s[i,(j+1)%M] + s[(i-1)%N, j] + s[i,(j-1)%M]
    
    dE = 2 * s[i, j] * NNsum
    
    if dE <= 0:
        s[i, j] *= -1
        deltaE = dE
        dM = 2 * s[i, j]
    elif np.exp(-dE * beta) > np.random.rand():
        s[i, j] *= -1
        deltaE = dE
        dM = 2 * s[i, j]
    
    return s, deltaE, dM




def accumulation(No, Nv, beta, eqSteps, mcSteps):
    L = No * Nv
    #config = random_spin_lattice(No, Nv)   #Starting random configuration
    config = ordered_spin_lattice(No, Nv)   #Starting chessboard configuration
    
    for i in range(eqSteps):
        config = Ising_conditions(config, beta)[0]
    
    Ener = initial_energy(config)
    Mag = np.sum(config)
    E, E2 =  np.zeros(int(mcSteps/(No*Nv))+1), np.zeros(int(mcSteps/(No*Nv))+1)
    M, M2 =  np.zeros(int(mcSteps/(No*Nv))+1), np.zeros(int(mcSteps/(No*Nv))+1)
    E[0], E2[0], M[0], M2[0] = Ener, Ener * Ener, Mag, Mag * Mag
    
    j= 1
    
    for i in range(1, mcSteps + 1):
        config, dE, dM = Ising_conditions(config, beta)
        Ener += dE
        Mag += dM
        
        if i%L == 0:
            E[j] = Ener
            E2[j] = Ener * Ener
            M[j] = Mag
            M2[j] = Mag * Mag
            j += 1
            
    """
    # Display the last lattice configuration
    cmap = { -1: [0, 0, 1], 1: [1, 0, 0] }   # blue: -1, red: +1
    def lattice_to_image(lattice):  # Convert spin lattice to colored image
        colored_pixels = [[cmap[spin] for spin in row] for row in lattice]
        return Image.fromarray(np.uint8(colored_pixels) * 255)
    fig_snapshot, ax_snapshot = plt.subplots(figsize=(6.2, 4.5))
    ax_snapshot.imshow(lattice_to_image(config))
    ax_snapshot.axis('off')
    plt.show()
    """  
    return M , M2 , E , E2 






# -----------------------------------------------------------------------------
# OPEN BOUNDARY CONDITIONS 
# -----------------------------------------------------------------------------
@njit
def initial_energy_open(s):
    N, M = s.shape
    total = 0
    for i in range(N):
        for j in range(M):
            if i==0 and j!=(M-1):
                total += -s[i,j] * s[0,j+1]
            elif i==0 and j==(M-1):
                total += 0
            elif j==(M-1) and i!=0:
                total += -s[i,j] *  s[i-1, M-1]
            else:
                total += -s[i,j] * (s[i-1, j] + s[i, j+1])  
    return total



@njit
def Ising_conditions_open(s, beta):
    N, M = s.shape
    i = np.random.randint(0, N)
    j =  np.random.randint(0, M)
    
    if i==0 and j==0:
        NNsum = s[0,1] + s[1,0]
    elif i==(N-1) and j==(M-1):
        NNsum = s[N-2, M-1] + s[N-1, M-2]
    elif i==(N-1) and j==0:
        NNsum = s[N-2, 0] + s[N-1, 1]
    elif i==0 and j==(M-1):
        NNsum = s[1, M-1] + s[0, M-2]
    elif i==0 and j!=0 and j!= (M-1):
        NNsum = s[0, j-1] + s[0, j+1] + s[1, j]
    elif i==(N-1) and j!=0 and j!= (M-1):
        NNsum = s[N-1, j-1] + s[N-1, j+1] + s[N-2, j]
    elif j==0 and i!=0 and i!= (N-1):
        NNsum = s[i-1, 0] + s[i+1, 0] + s[i, 1]
    elif j==(M-1) and i!=0 and i!= (N-1):
        NNsum = s[i-1, M-1] + s[i+1, M-1] + s[i, M-2]
    else:
        NNsum = s[i-1, j] + s[i+1, j] + s[i, j-1] + s[i, j+1]
    
    dE = 2 * s[i, j] * NNsum
    
    if dE <= 0:
        s[i, j] *= -1
        deltaE = dE
        dM = 2 * s[i, j]
    elif np.exp(-dE * beta) > np.random.rand():
        s[i, j] *= -1
        deltaE = dE
        dM = 2 * s[i, j]
    
    return s, deltaE, dM



def accumulation_open(No, Nv, beta, eqSteps, mcSteps):
    L = No * Nv
    config = random_spin_lattice(No, Nv)   #Starting random configuration
    #config = ordered_spin_lattice(No, Nv)   #Starting chessboard configuration
    
    for i in range(eqSteps):
        config = Ising_conditions_open(config, beta)[0]
    
    Ener = initial_energy_open(config)
    Mag = np.sum(config)
    E, E2 =  np.zeros(int(mcSteps/(No*Nv))+1), np.zeros(int(mcSteps/(No*Nv))+1)
    M, M2 =  np.zeros(int(mcSteps/(No*Nv))+1), np.zeros(int(mcSteps/(No*Nv))+1)
    E[0], E2[0], M[0], M2[0] = Ener, Ener * Ener, Mag, Mag * Mag
    
    j= 1
    
    for i in range(1, mcSteps + 1):
        config, dE, dM = Ising_conditions_open(config, beta)
        Ener += dE
        Mag += dM
        
        if i%L == 0:
            E[j] = Ener
            E2[j] = Ener * Ener
            M[j] = Mag
            M2[j] = Mag * Mag
            j += 1
            
    
    # Display the last lattice configuration
    """
    cmap = { -1: [0, 0, 1], 1: [1, 0, 0] }   # blue: -1, red: +1
    def lattice_to_image(lattice):  # Convert spin lattice to colored image
        colored_pixels = [[cmap[spin] for spin in row] for row in lattice]
        return Image.fromarray(np.uint8(colored_pixels) * 255)
    fig_snapshot, ax_snapshot = plt.subplots(figsize=(6.2, 4.5))
    ax_snapshot.imshow(lattice_to_image(config))
    ax_snapshot.axis('off')
    plt.show()
    """
    
    return M , M2 , E , E2 





# ----------------------------------------------------------------------------
# CONFIGURATION SEQUENCE ANIMATION
# ----------------------------------------------------------------------------


def animation_Ising(No, Nv, beta, mcSteps, plot_name):
    L = No * Nv
    config = random_spin_lattice(No, Nv)   #Starting random configuration
    #config = ordered_spin_lattice(No, Nv)   #Starting chessboard configuration
    fig = plt.figure()
    
    spin_color = mcolors.ListedColormap(['blue', 'red'])
    im = plt.imshow(display_spin_lattice(config), cmap=spin_color)

    plt.title('2D Ising model (no H)')
    plt.axis('off')
    
    metadata = dict(title='Movie', artist='codinglikened')
    writer = PillowWriter(fps=10, metadata=metadata)
    
    with writer.saving(fig, plot_name, 100):
        for i in range(1, mcSteps + 1):
            config, _, _ = Ising_conditions(config, beta)
            
            if i % L == 0:
                im.set_data(display_spin_lattice(config))
                writer.grab_frame()
                
    return




def save_frames_from_gif(gif_path, output_folder):
   
    # Open the GIF file
    with Image.open(gif_path) as gif:
        # Iterate over each frame in the GIF
        for i, frame in enumerate(ImageSequence.Iterator(gif)):
            # Construct the output file name
            output_filename = os.path.join(output_folder, f"frame_{i}.png")
            # Save the frame as PNG
            frame.save(output_filename, format="PNG")



# -----------------------------------------------------------------------------
# PLOTTING PHYSICAL QUANTITIES: <|M|>/N, <E>/N, X/N, Cv/N
# -----------------------------------------------------------------------------


@njit
def block_average(lst, s):
    
    aver = np.zeros(s, dtype = np.float32)
    block_size = int(len(lst) / s)

    for k in range(s):
        aver[k] = np.std(lst[(k * block_size):((k + 1) * block_size)])
        
    Sigma_s = np.std(aver)
        
    return Sigma_s / math.sqrt(s)



def averaged_quantities(data, No, Nv, beta, eqSteps, mcSteps):
    E_aver = np.mean(data[2])
    M_ABS_aver = np.mean(abs(data[0]))
    C = np.var(data[2]) * beta ** 2 
    X = np.var(abs(data[0])) * beta

    return E_aver, M_ABS_aver, C, X



def average_error(data, No, Nv, beta, eqSteps, mcSteps, s):
    aver = averaged_quantities(data, No, Nv, beta, eqSteps, mcSteps)
    err_E = block_average(data[2],s)
    err_M_ABS = block_average(abs(data[0]), s)
    sigma2_E = (data[2]-aver[0])**2
    sigma2_M_ABS = (data[0]-aver[1])**2
    err_C = block_average(sigma2_E, s) * (1/(No*Nv)) * beta**2   
    err_X = block_average(sigma2_M_ABS, s) * (1/(No*Nv)) * beta   
    
    return err_E, err_M_ABS, err_C, err_X
    


def T_variation(No, Nv, T_m, T_M, d_T, eqSteps, mcSteps, s):
    N = No * Nv
    arrT = np.arange(T_m, T_M, d_T)
    E, M = np.zeros(len(arrT)),  np.zeros(len(arrT))
    C, X = np.zeros(len(arrT)), np.zeros(len(arrT))
    s_E, s_M =  np.zeros(len(arrT)),  np.zeros(len(arrT))
    s_C, s_X =  np.zeros(len(arrT)),  np.zeros(len(arrT))
    
    for i in range(len(arrT)):
        data = accumulation(No, Nv, 1/arrT[i], eqSteps, mcSteps)
        E[i], M[i], C[i], X[i] = averaged_quantities(data, No, Nv, 1/arrT[i], eqSteps, mcSteps)
        s_E[i], s_M[i], s_C[i], s_X[i] = average_error(data, No, Nv, 1/arrT[i], eqSteps, mcSteps, s)
        
    return M/N, E/N, C/N, X/N, s_M/N, s_E/N, s_C/N, s_X/N




def T_variation_open(No, Nv, T_m, T_M, d_T, eqSteps, mcSteps, s):
    N = No * Nv
    arrT = np.arange(T_m, T_M, d_T)
    E, M = np.zeros(len(arrT)),  np.zeros(len(arrT))
    C, X = np.zeros(len(arrT)), np.zeros(len(arrT))
    s_E, s_M =  np.zeros(len(arrT)),  np.zeros(len(arrT))
    s_C, s_X =  np.zeros(len(arrT)),  np.zeros(len(arrT))
    
    for i in range(len(arrT)):
        data = accumulation_open(No, Nv, 1/arrT[i], eqSteps, mcSteps)
        E[i], M[i], C[i], X[i] = averaged_quantities(data, No, Nv, 1/arrT[i], eqSteps, mcSteps)
        s_E[i], s_M[i], s_C[i], s_X[i] = average_error(data, No, Nv, 1/arrT[i], eqSteps, mcSteps, s)
        
    return M/N, E/N, C/N, X/N, s_M/N, s_E/N, s_C/N, s_X/N




@njit
def c_as_derivative(No, Nv, e, dT, s_e):
    N = No*Nv
    c = np.zeros(len(e) - 2)
    s_c = np.zeros(len(e) - 2)
    E = e * N
    s_E = s_e *N
    for i in range(1, len(e) - 1):
        c[i-1] = (E[i+1] - E[i-1]) / (2*dT)
        s_c[i-1] = math.sqrt(s_E[i+1]**2 + s_E[i-1]**2) / (2*dT)
    return c/N, s_c/N



def fitE(x, L, k, x0, c):
    return L / (1 + np.exp(-k * (x - x0))) + c


