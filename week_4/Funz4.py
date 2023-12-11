"""
Library of self-made functions needed for the codes implemented for the exercises of the 4th week

@author: david
"""

import numpy as np
from numba import njit, jit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
import math
from matplotlib.animation import PillowWriter




@njit
def RW_1D(N, x0, Pl):
    
    xi = x0
    position = [x0] 
    square_pos = [x0**2]
    
    for i in range(N):
        a = np.random.rand()
        if a <= Pl:
            xi -= 1
        else:
            xi += 1
        position.append(xi)
        square_pos.append(xi**2)
        
    return np.asarray(position, dtype=np.int32), np.asarray(square_pos, dtype=np.int32)




@njit
def RW1D_average(N_w, N, x0, Pl):
    
    position = np.full((N_w, N + 1), x0, dtype=np.int32)
    square_pos = np.full((N_w, N + 1), x0**2, dtype=np.int32)
    cumul_x = np.zeros(N, dtype=np.float32)
    cumul_x2 = np.zeros(N, dtype=np.float32)
    P_N = np.zeros(2*N +1, dtype=np.int32)

    for j in range(N_w):
        xi = x0
        a = np.random.uniform(0, 1, N)
        for i in range(N):
            if a[i] <= Pl:
                xi -= 1
            else:
                xi += 1
            position[j, i + 1] = xi
            square_pos[j, i + 1] = xi**2
            cumul_x[i] += xi
            cumul_x2[i] += xi**2
        P_N[N + xi] += 1
    average_x = cumul_x / N_w
    average_x2 = cumul_x2 / N_w

    return position, square_pos, average_x, average_x2, average_x2 - average_x**2, P_N




@jit
def Accuracy(steps, acc, x0, N, Nw0, passo, Pl):
    
    N_wIdeal = 0
    
    for k in range(steps):
        
        delta = acc+1
        N_w = Nw0
        cumul_xN = 0
        cumul_x2N = 0
        average_xN = 0
        average_x2N = 0
        t = 0
        
        while delta > acc:
            
            for j in range(N_w):
                
                xi = x0
                l = np.random.uniform(0, 1, N)
                
                for i in range(N):  
                    if l[i] <= Pl:
                        xi -= 1
                    else:
                        xi += 1
                cumul_xN += xi
                cumul_x2N += xi**2
              
            average_xN =  cumul_xN / (Nw0 + passo * t)
            average_x2N = cumul_x2N /(Nw0 + passo * t)
            msd = average_x2N - average_xN**2
            delta = abs(msd/N - 1)
            N_w = passo
            t += 1
            
        Nw_fin = Nw0 + passo * (t-1)
        
        N_wIdeal += Nw_fin
    
    return N_wIdeal/steps





def iter_plot(vect, index, N, N_w, Pl, string, test):
    
    t = [i for i in range(N+1)]
    
    for i in range(N_w):    
        plt.plot(t, vect[index][i])
        
    plt.xlabel('Iteration steps i')
    plt.ylabel(string)
    #plt.title(fr'1D Random Walks $P_{{\mathrm{{left}}}} = {Pl}$, $N = {N}$')
    
    if test:
        plt.plot(t, [i*index for i in range(N+1)], color='red', label='Theoretical average')
        plt.plot(t, np.insert(vect[2+index],0,0), color='black', label='Numerical average')
        plt.legend()
        
    plt.show()    
    
    return




@jit
def line(x, m, q):
    
    return m*x + q




def graphNwalk_N():
    
    metro = [i for i in range(10, 500, 10)]
    inch = []
    for k in metro:
        inch.append(Accuracy(1000, 0.05, 0, k, 10, 10, 0.5))
    plt.plot(metro, inch)
    plt.xlabel('N')
    plt.ylabel(r'$N_{walkers}^{min}$', fontsize=12)
    plt.grid(True)
    plt.show()
    
    return




def graphMsdN():
    
    kilo = np.asarray([2**i for i in range(3, 8)], dtype=np.int32)
    pound = np.asarray([], dtype=np.float64)
    for k in kilo:
        pound = np.append( pound, RW1D_average(160, k, 0, 0.5)[4][-1])
    log_kilo = np.log(kilo)
    log_pound = np.log(pound)
    
    par, cov = curve_fit(line, log_kilo, log_pound)

    plt.scatter(log_kilo, log_pound, label='Data', color='black')
    plt.plot(log_kilo, line(log_kilo, *par), color='red', label='Linear Fit')
    plt.xlabel(r'$ln{N}$', fontsize=12)
    plt.ylabel(r'$ln{\langle (\Delta x)^2 \rangle}$', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return par, cov




def Histo_gauss():
    
    bucket = [8, 16, 32, 64]
    
    for k in bucket:
        sandwich = RW1D_average(10000, k, 0, 0.5)[5]/10000
        bin_centers = np.arange(-k, k+1, 1)
        plt.bar(bin_centers, sandwich, width=1, label=r'$P_{N}(x)^{num}$', color='blue')
        plt.xlabel('x', fontsize=12)
        plt.ylabel(r'$P_N(x)$', fontsize=12)
        plt.grid(True)
        
        mean = RW1D_average(160, k, 0, 0.5)[2][-1]
        std_dev = math.sqrt(RW1D_average(160, k, 0, 0.5)[4][-1])
        x = np.linspace(mean - k, mean + k, 1000)
        y = norm.pdf(x, mean, std_dev)
        plt.plot(x, 2*y, label=r'$P_{N}(x)^{theo}$', color='black')
        plt.legend()
        
        plt.show()
    
    return




@njit
def RW1D_average_random_l(N_w, N, x0, Pl):
    
    position = np.full((N_w, N + 1), x0, dtype=np.int32)
    square_pos = np.full((N_w, N + 1), x0**2, dtype=np.int32)
    cumul_x = np.zeros(N, dtype=np.float32)
    cumul_x2 = np.zeros(N, dtype=np.float32)
    P_N = np.empty(N_w, dtype=np.float64)

    for j in range(N_w):
        
        xi = x0
        a = np.random.uniform(0, 1, N)
        lulla = np.random.uniform(0, 1, N) # Here we can replace the distribution of the
        lst_l = -(1/3)*np.log(lulla)       # lenghts of the RW steps as we prefer: UNIFORM, EXP, GAUSS,...
        
        for i in range(N):
            if a[i] <= Pl:
                xi -= lst_l[i]
            else:
                xi += lst_l[i]
            position[j, i + 1] = xi
            square_pos[j, i + 1] = xi**2
            cumul_x[i] += xi
            cumul_x2[i] += xi**2
        P_N[j] = xi                        # We can build the histogram from P_N. It will be a gaussian as in the
                                           # case of l=1, but without empty bins: the RW can move on a continuous space now!
    average_x = cumul_x / N_w
    average_x2 = cumul_x2 / N_w

    return position, square_pos, average_x, average_x2, average_x2 - average_x**2 , P_N



# -----------------------------------------------------------------------------
# ----------- Functions for the 2d_RW part:


@njit
def RW2D_average(N_w, N, x0, y0, Pl, Pr, Pd, Theta):  # 2D analogous of the "RW1D_average" function in Funz4
    
    position_x = np.full((N_w, N + 1), x0, dtype=np.float32)
    position_y = np.full((N_w, N + 1), x0, dtype=np.float32)
    cumul_x = np.zeros(N, dtype=np.float32)
    cumul_y = np.zeros(N, dtype=np.float32)
    cumul_x2 = np.zeros(N, dtype=np.float32)
    cumul_y2 = np.zeros(N, dtype=np.float32)
    P_N = np.empty(N_w, dtype=np.float64)
    
    if not Theta:
        
        for j in range(N_w):
            
            xi = x0
            yi = y0
            a = np.random.uniform(0, 1, N)
            #lst_l = np.random.uniform(0, 1, N)   # If we want to work on a lattice of random spacing l for each step
            lst_l = np.asarray([1 for _ in range(N)], dtype=np.int32)   # If we want to work on a descrete lattice of spacing l = 1
            
            for i in range(N):
                
                if a[i] <= Pl:
                    xi -= lst_l[i] 
                    position_x[j, i + 1] = xi
                    position_y[j, i + 1] = yi
                    cumul_x[i] += xi
                    cumul_x2[i] += xi**2
                elif a[i] > Pl and a[i] <= (Pl + Pr):
                    xi += lst_l[i]
                    position_x[j, i + 1] = xi
                    position_y[j, i + 1] = yi
                    cumul_x[i] += xi
                    cumul_x2[i] += xi**2
                elif a[i] > (Pl + Pr) and a[i] <= (Pl + Pr + Pd):
                    yi -= lst_l[i] 
                    position_y[j, i + 1] = yi
                    position_x[j, i + 1] = xi
                    cumul_y[i] += yi
                    cumul_y2[i] += yi**2
                else:
                    yi += lst_l[i]  
                    position_y[j, i + 1] = yi
                    position_x[j, i + 1] = xi
                    cumul_y[i] += yi
                    cumul_y2[i] += yi**2  
        
            P_N[j] = math.sqrt(xi**2 + yi**2)   
                     
    else:  # For 2D random unit steps (on the unitary circle)
        
        for j in range(N_w):
            
            xi = x0
            yi = y0
            th = np.random.uniform(0, 2*math.pi, N)
            
            for i in range(N):
                xi += math.cos(th[i])
                yi += math.sin(th[i])
                position_x[j, i + 1] = xi
                position_y[j, i + 1] = yi
                cumul_x[i] += xi
                cumul_x2[i] += xi**2
                cumul_y[i] += yi
                cumul_y2[i] += yi**2  
            
            P_N[j] = math.sqrt(xi**2 + yi**2) 
            
        
    average_x = cumul_x / N_w
    average_y = cumul_y / N_w
    average_x2 = cumul_x2 / N_w
    average_y2 = cumul_y2 / N_w

    return position_x, position_y, average_x2 + average_y2 - average_x**2 -  average_y**2 , P_N




def RW_2D_plot(Opera, plot_name):  #Plotting the gif of all the walkers justapposed

    fig = plt.figure()
    
    l = [plt.plot([], [])[0] for _ in range(len(Opera[0]))]
    xlist = [[] for _ in range(len(Opera[0]))]
    ylist = [[] for _ in range(len(Opera[0]))]
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D RW')
    plt.xlim(np.min(Opera[0]) - 1, np.max(Opera[0]) + 1)
    plt.ylim(np.min(Opera[1]) - 1, np.max(Opera[1]) + 1)
    plt.grid(True) 
    plt.gca().set_aspect('equal', adjustable='box')
    
    metadata = dict(title='Movie', artist='codinglikened')
    writer = PillowWriter(fps=1, metadata= metadata)
    
    with writer.saving(fig, plot_name, 100):
        for k in range(len(Opera[2])+1):
            for i in range(len(Opera[0])):
                xlist[i].append(Opera[0][i][k])
                ylist[i].append(Opera[1][i][k])
              
                l[i].set_data(xlist[i], ylist[i])
               
            writer.grab_frame()
            
    return

