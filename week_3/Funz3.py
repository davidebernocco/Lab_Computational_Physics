
"""
Library of self-made functions needed for the codes implemented for the exercises of the 3rd week

@author: david
"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, int32, float64


@jit
def line(x, m, q):
    return m*x + q



@jit
def var_x(U,V,n):
    x = []
    for i in range(n):
        if U[i]**2 + V[i]**2 <= 1 :
            x.append((U[i]**2 - V[i]**2)/(U[i]**2 + V[i]**2))
    return x



@jit(float64[:](int32))
def boxmuller(fagioli):
    sacchetto = []
    for i in range(fagioli):
        gaus_stored = False
        g = 0.0
        
        if gaus_stored:
            rnd = g
            gaus_stored = False
        else:
            while True:
                x = random.uniform(-1,1) #Alternatively: x = 2.0 * random.random() - 1.0
                y = random.uniform(-1,1) #Alternatively: y = 2.0 * random.random() - 1.0
                r2 = x**2 + y**2
                if r2 > 0.0 and r2 < 1.0:
                    break
            r2 = math.sqrt(-2.0 * math.log(r2) / r2)
            rnd = x * r2
            g = y * r2
            gaus_stored = True
            
        sacchetto.append(rnd)    
    return np.asarray(sacchetto, float64)



@jit
def R(u,v,n):
    x_vet = []
    y_vet = []
    for i in range(n):
        if u[i]**2 + v[i]**2 <= 1 :
            r2 = u[i]**2 + v[i]**2
            r2 = math.sqrt(-2* math.log(r2) / r2)
            x_vet.append(r2* u[i])
            y_vet.append(r2* v[i])
    return np.asarray(x_vet, dtype=np.float64), np.asarray(y_vet, dtype=np.float64)



# The optimized function for the case 3.1; not used in the main code. Showed here merely for didactic scope
@jit(float64[:](int32))
def boxmuller_trig(ceci):
    sacco = []
    for i in range(ceci):
        gaus_stored = False
        g = 0.0
        
        if gaus_stored:
            rnd = g
            gaus_stored = False
        else:
            X = random.random() 
            Y = random.random()
            x = math.sqrt(-2 * math.log(X)) * math.cos(2 * math.pi * Y)
            y = math.sqrt(-2 * math.log(X)) * math.sin(2 * math.pi * Y)
            rnd = x
            g = y
            gaus_stored = True
            
        sacco.append(rnd)    
    return np.asarray(sacco, float64)



@jit
def decay(Ni, l):
    can = [Ni]
    Time = [0]
    t=0
    Nleft = Ni
    while Nleft > 0:
        chickpeas = 0
        for i in range(Nleft):
            r = random.random()
            if r <= l:
                chickpeas += 1
        Nleft -= chickpeas
        t += 1
        can.append(Nleft)
        Time.append(t)
    return np.asarray(can[:-1], int32), np.asarray(Time[:-1], int32)



def multiple_plot(lst, l):
    for j in range(len(lst)):
        pino = decay(lst[j], l[j])
        pigna = pino[0]
        ago = pino[1]
        plt.scatter(ago, np.log(pigna), label=f'N(0) = {lst[j]},  $\lambda$ = {round(l[j],1)}', marker='o')
        plt.xlabel('Time t')
        plt.ylabel('ln( N(t) )')
        plt.title('Simulation of radioactive decay')
        plt.legend()
        plt.grid(True)
        del pino
    plt.show()
    return

