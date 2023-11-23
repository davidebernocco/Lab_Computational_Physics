"""
Library of self-made functions needed for the codes implemented for the exercises of the 3rd week

@author: david
"""

import numpy as np
from numba import njit, jit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



@njit
def RW_1D(N, x0, Pl):
    
    xi = x0
    position = [x0] 
    square_pos = [x0**2]
    
    for i in range(N):
        l = np.random.rand()
        if l <= Pl:
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
        l = np.random.uniform(0, 1, N)
        for i in range(N):
            if l[i] <= Pl:
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
