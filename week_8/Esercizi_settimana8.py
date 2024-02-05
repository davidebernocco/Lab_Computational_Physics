"""
Now I have to do everything from the beginning again

@author: david
"""

"""
There are two distinct parts to the algorithm; an initial equilibration stage 
and an energy evaluation stage. 
- During the initial equilibration stage, the walker is moved according to the 
Metropolis algorithm, but the local energy is not accumulated along the walk. 
The required number of equilibration steps can be established by calculating 
the energy at each step from the beginning of the random walk and looking for 
the point at which there is no longer a drift in the average value of the local 
energy.
- During the energy evaluation stage, the energy of the walker is accumulated 
after each move.
"""

import math
import numpy as np
#from scipy.stats import norm
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
#from numba import njit
#import random
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from numba import njit
import time


"""
Per l'equilibration phase, stimo a parte con la funzione equil() da Funz7 il burnin period.
Una volta per tutte per ogni target e proposal function che considero..
Per il caso di proposal uniforme e target gaussiana ho già calcolato n_burnin = 2700
Rimuovo a posteriori i primi 2700 punti dalla distribuzione (o uso un if sull'indice i: se < 2700 no accumulo Eloc)
Quindi, ciò che ho fatto in Funz8 non mi serve più                                                                                           

"""
"""
def VMC_text():
    #inizializzo il parametro (qui beta = 1/(4*s**2))
    #inizializzo: last_param = ulimo parametro in cui ho generato punti
    
    #inizia un ciclo: for i in range(numero di variazioni del param)

    # aumento il parametro della quantità opportuna 
    #condizione per rigenerare i punti dopo vari reweighting (qui può essere se s'=s+0.1)
    
    #SE rigenerazione è necessaria:
        #scelgo a random il punto iniziale   (eventualmente faccio un ciclo su più RW in parallelo)
        #Equilibration phase     (per tutti eventuali RW in parallelo)
        #Accumulation phase      (per tutti eventuali RW in parallelo)
        #concludo ciclo
    
    #SE non necessario, invece:
        #SALTO equilibration phase e uso i punti generati precedentemente    (per tutti eventuali RW in parallelo)
        #calcolo i fattori di peso             (per tutti eventuali RW in parallelo)
        #Accumulation phase con reweighting    (per tutti eventuali RW in parallelo)
        #concludo ciclo
    
    #faccio medie delle quantità locali
    #calcolo errori con block average
    
    return    
"""  




@njit
def block_average(lst, s):
    
    aver = np.zeros(s, dtype = np.float32)
    block_size = int(len(lst) / s)

    for k in range(s):
        aver[k] = np.std(lst[(k * block_size):((k + 1) * block_size)])
        
    Sigma_s = np.std(aver)
        
    return Sigma_s / math.sqrt(s)



par_VMC = np.arange(0.1, 1.55, 0.05)


def VMC( reweight_d, delta, n_MC, s_blocks ):
    
    # |wave function|**2
    def trial_f(x, b):
        return np.exp( -2 * b * x ** 2 )
    
    # From the equilibration estimate fit with power low
    def burnin_f(x):
        return 5719 * x ** (0.52)
    
    # Local energy for the quantum HO with |psi|^2 = trial_f
    def Etot_l(x, b):
        return ( (1 - 4 * b**2 ) / 2 ) * x**2 + b
    
    # Metropolis kernel code, for reweighting condition
    def Metropolis( x0, delta, n, b):
        points = np.zeros(n, dtype = np.float32)
        points[0] = x0
        x_t = x0
        for i in range(1, n):
            x_star = np.random.uniform(x_t - delta, x_t + delta)        
            alpha = min(1, trial_f(x_star, b) / trial_f(x_t, b))
            if alpha >= np.random.rand() :
                x_t = x_star    
            points[i] = x_t     
        return points
        
    
    last_par = par_VMC[0] - reweight_d - 0.01
    E_loc = np.zeros((len(par_VMC), n_MC), dtype = np.float32)
    E_loc_2 = np.zeros((len(par_VMC), n_MC), dtype = np.float32)
    err1 = np.zeros(len(par_VMC), dtype = np.float32)
    err2 = np.zeros(len(par_VMC), dtype = np.float32)
    variance = np.zeros(len(par_VMC), dtype = np.float32)
    E_l = np.zeros(len(par_VMC), dtype = np.float32)
    E2_l = np.zeros(len(par_VMC), dtype = np.float32)
    err_var = np.zeros(len(par_VMC), dtype = np.float32)
    
    for i in range(len(par_VMC)):
        
        
        # Condition: use or not reweighting? Creterion end pag 18 unit 8
        pinco = Metropolis(0, 5, 50000, last_par)
        num = trial_f(pinco[0], last_par)
        den = trial_f(pinco[0], par_VMC[i])
        pesi = num / den
        N_eff = (np.sum(pesi))**2 / np.sum(pesi**2)
        
        if (N_eff/50000) < 0.95:
            
            x0 = np.random.uniform(- delta, delta)            
            x_t = x0
            
            # Equilibration phase
            equil_len = int(burnin_f(par_VMC[i]))
            for k in range(equil_len):
                x_star = np.random.uniform(x_t - delta, x_t + delta)
                
                num = trial_f(x_star, par_VMC[i]) 
                den = trial_f(x_t, par_VMC[i])    
                                                        
                alpha = num/den
                
                if alpha >= np.random.rand():
                    x_t = x_star
            
            
            
            # Accumulation phase
            sampl = np.zeros(n_MC, dtype = np.float32)
            sampl[0] = x_t
            weight_0 = np.zeros(n_MC, dtype = np.float32)
            
            for k in range(1, n_MC):
                x_star = np.random.uniform(x_t - delta, x_t + delta)
                
                num = trial_f(x_star, par_VMC[i]) 
                den = trial_f(x_t, par_VMC[i])    
                                                        
                alpha = num/den
                
                if alpha >= np.random.rand() :
                    x_t = x_star
            
                sampl[k] = x_t
                weight_0[k] = trial_f(x_t, par_VMC[i])
                E_loc[i, k] += Etot_l(x_t, par_VMC[i])
                E_loc_2[i, k] += Etot_l(x_t, par_VMC[i]) ** 2
            
            variance[i] = np.var(E_loc[i])
            err1[i] = block_average(E_loc[i], s_blocks)
            err2[i] = block_average(E_loc_2[i], s_blocks)
            E_l[i] = np.sum(E_loc[i]) / n_MC
            E2_l[i] = np.sum(E_loc_2[i]) / n_MC
            last_par = par_VMC[i]
                
            
            
        # Reweighting when possible    
        else:
            
            weight = np.zeros(n_MC, dtype = np.float32)
            
            for k in range (n_MC):
                
                weight[k] = weight_0[k] / trial_f(sampl[k], par_VMC[i])
                E_loc[i,k] = weight[k] * Etot_l(sampl[k], par_VMC[i])
                E_loc_2[i,k] = weight[k] * (Etot_l(sampl[k], par_VMC[i]) ** 2)
            
            variance[i] = np.var(E_loc[i])
            err1[i] = block_average(E_loc[i], s_blocks)
            err2[i] = block_average(E_loc_2[i], s_blocks)
            E_l[i] = np.sum(E_loc[i]) / np.sum(weight)
            E2_l[i] = np.sum(E_loc_2[i]) / np.sum(weight)
            
     
        err_var[i] = np.sqrt(err2[i]**2 + 4 * E_l[i]**2 * err1[i]**2 )

   
    return E_l, err1, variance, err_var





#Plotting the results
start_time1 = time.time()

Kazan = VMC(0.1, 5, 100000, 100)  
beta = par_VMC

fig_El, ax_El = plt.subplots(figsize=(6.2, 4.5))
ax_El.scatter(beta, Kazan[0], marker='o', s=50)
ax_El.errorbar(beta, Kazan[0], yerr=Kazan[1], fmt='.', capsize=5, color='black')
ax_El.set_xlabel(r'$ \beta = 1 / 4\sigma^2 $', fontsize=12)
ax_El.set_ylabel(r'$ \langle E \rangle $', fontsize=12)
ax_El.grid(True)
plt.show()


fig_var, ax_var = plt.subplots(figsize=(6.2, 4.5))
ax_var.scatter(beta, Kazan[2], marker='o', s=50)
ax_var.errorbar(beta, Kazan[2], yerr=Kazan[3], fmt='.',  capsize=5, color='black')
ax_var.set_xlabel(r'$ \beta = 1 / 4\sigma^2 $', fontsize=12)
ax_var.set_ylabel(r'$ \langle E^2 \rangle - \langle E \rangle^2 $', fontsize=12)
ax_var.grid(True)
plt.show()

end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print(f"CPU time 'Local energy sampling': {elapsed_time1:.4f} seconds")

#16 punti, n=10^5, s=10^2 -> t= 27sec
#31 punti, n=10^5, s=10^2 -> t= 27sec







"""
burn_in =  np.zeros(len(x), dtype = np.float32)
for j in range(len(x)):
    burn_in[j] = equil(0, 5, 10000, x[j], 100, 300)
    
---> BURNIN GIà CALCOLATO, CHE CI VUOLE TEMPO!!! 

y = np.asarray([1900, 2600, 3300, 3200, 3500, 4500, 4300, 5100, 5400, 6700, 5800, 6300,
 6800, 6600, 6900]) # Fit Pow: A=5719, a=0.52
"""
"""
def trial_f(x, b):
    return np.exp( -2 * b* x ** 2  )



def Metropolis( x0, delta, n, b):

    acc = 0
    points = np.zeros(n, dtype = np.float32)
    points[0] = x0
    
    x_t = x0
    
    for i in range(1, n):
        
        x_star = np.random.uniform(x_t - delta, x_t + delta)        
        alpha = min(1, trial_f(x_star, b) / trial_f(x_t, b))
        
        if alpha >= np.random.rand() :
            x_t = x_star
            acc += 1
            
        points[i] = x_t
            
    return points, acc/n



def equil(x0, delta, n, b, N, N_aver):
    
    theo_var = 1 / (4 * b)   #It depends on the choice of trial_f!!!

    l = int(n/N)
    Var = np.zeros(l, dtype = np.float32)
    h = np.zeros(l, dtype = np.float32)
    aver =  np.zeros(l, dtype = np.float32)

    
    for i in range(N_aver):
        
        x = Metropolis(x0, delta, n, b)[0]
        
        for j in range(l):
            Var[j] = np.var(x[ : N*(j+1)])
            h[j] = abs(Var[j] - theo_var) / theo_var
            aver[j] += h[j]
    
    D_arr = aver / N_aver
    ki = 0
    while D_arr[ki] > 0.05:
        ki += 1
        
    return (ki-1) * N


x = np.arange(0.1, 1.55, 0.1)

burn_in =  np.zeros(len(x), dtype = np.float32)
for j in range(len(x)):
    burn_in[j] = equil(0, 5, 10000, x[j], 100, 300)


from scipy.optimize import curve_fit


def fit_burnin(x, A, a):
    return A * x**a

param_b, covariance_b = curve_fit(fit_burnin, x, burn_in)

fig_bi, ax_bi = plt.subplots(figsize=(6.2, 4.5))
ax_bi.scatter(x, burn_in, label='Numerical estimation', marker='o', s=50)
ax_bi.plot(x, fit_burnin(x, *param_b), color='red', label='Power fit')

ax_bi.set_xlabel(r'$ \beta $', fontsize=12)
ax_bi.set_ylabel('Burn-in length', fontsize=12)
ax_bi.legend()
ax_bi.grid(True)
plt.show()
"""





