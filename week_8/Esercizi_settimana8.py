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

"""


@njit
def block_average(lst, s):
    
    aver = np.zeros(s, dtype = np.float32)
    block_size = int(len(lst) / s)

    for k in range(s):
        aver[k] = np.std(lst[(k * block_size):((k + 1) * block_size)])
        
    Sigma_s = np.std(aver)
        
    return Sigma_s / math.sqrt(s)



par_VMC = np.arange(0.4, 1.64, 0.08)

def VCM( reweight_d, delta, n_MC ):
    
    # |wave function|**2
    def trial_f(x, s):
        return np.exp( -x ** 2 / ( 2 * s ** 2) )
    
    # From the equilibration estimate fit with power low
    def burnin_f(x):
        return 2612 * x ** (-1.13)
    
    # Local energy for the quantum HO with |psi|^2 = trial_f
    def Etot_l(x, s):
        return ( (1 - 4 * (1/(16*s**4)) ) / 2 ) * x**2 + 1/(4*s**2)
        
    
    last_par = par_VMC[0] - reweight_d - 0.01
    E_loc = np.zeros((len(par_VMC), n_MC), dtype = np.float32)
    E_loc_2 = np.zeros((len(par_VMC), n_MC), dtype = np.float32)
    err1 = np.zeros(len(par_VMC), dtype = np.float32)
    err2 = np.zeros(len(par_VMC), dtype = np.float32)
    variance = np.zeros(len(par_VMC), dtype = np.float32)
    
    for i in range(len(par_VMC)):
        
        
        
        if (par_VMC[i] - last_par) > 0.1:
            
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
                
            E_loc[i] = E_loc[i] / n_MC
            E_loc_2[i] = E_loc_2[i] / n_MC
            last_par = par_VMC[i]
                
            
            
        # Reweighting when possible    
        else:
            for k in range (n_MC):
                
                weight[k] = weight_0[k] / trial_f(sampl[k], par_VMC[i])
                E_loc[i,k] = weight[k] * Etot_l(sampl[k], par_VMC[i])
                E_loc_2[i,k] = weight[k] * (Etot_l(sampl[k], par_VMC[i]) ** 2)
            
            E_loc[i] = E_loc[i] / np.sum(weight)
            E_loc_2[i] = E_loc_2[i] / np.sum(weight)
           
            
            
        err1[i] = block_average(E_loc[i], s)
        err2[i] = block_average(E_loc_2[i], s)
        varaince[i] = np.var(E_loc[i])
        err_var [i] = np.sqrt(err_2**2 + 4 * E_loc[i]**2 * err_1**2 )
        
    
   
    return E_loc, err1, variance, err_var





#Plotting the results
Kazan = VMC(0.1, 5, 10000)
beta = 1 / (4 * par_VMC**2)

fig_El, ax_El = plt.subplots(figsize=(6.2, 4.5))
ax_El.scatter(beta, Kazan[0], marker='o', s=50)
ax_El.errorbar(beta, Kazan[0], yerr=Kazan[1], capsize=5, color='black')
ax_El.set_xlabel(r'$ \beta = 1 / 4\sigma^2 $', fontsize=12)
ax_El.set_ylabel(r'$ \langle E \rangle $', fontsize=12)
ax_El.legend()
ax_El.grid(True)
plt.show()















"""




"""
def trial_f(x, s):
    return np.exp( -x ** 2 / ( 2 * s ** 2) )




def Metropolis( x0, delta, n, s):

    acc = 0
    points = np.zeros(n, dtype = np.float32)
    points[0] = x0
    
    x_t = x0
    
    for i in range(1, n):
        
        x_star = np.random.uniform(x_t - delta, x_t + delta)        
        alpha = min(1, trial_f(x_star, s) / trial_f(x_t, s))
        
        if alpha >= np.random.rand() :
            x_t = x_star
            acc += 1
            
        points[i] = x_t
            
    return points, acc/n
"""


def Metropolis( x0, delta, n, s):
    
    acc = 0
    points = np.zeros(n, dtype = np.float32)
    points[0] = x0
    
    x_t = x0
    
    for i in range(1, n):
        x_star = np.random.uniform(x_t - delta, x_t + delta)
        
        esp1 = ( -x_star ** 2 / ( 2 * s ** 2) ) # To modify depending on 
        esp2 = ( -x_t ** 2 / ( 2 * s ** 2) )    # choosen target function.
                                                # Here the exp are broken to
        alpha = math.e ** (esp1-esp2)           # avoid "division by 0" issue
        
        
        if alpha >= np.random.rand() :
            x_t = x_star
            acc += 1
            
        points[i] = x_t
            
    return points, acc/n


def equil(x0, delta, n, s, N, N_aver):
    
    theo_var = s**2   #It depends on the choice of trial_f!!!

    l = int(n/N)
    Var = np.zeros(l, dtype = np.float32)
    h = np.zeros(l, dtype = np.float32)
    aver =  np.zeros(l, dtype = np.float32)

    
    for i in range(N_aver):
        
        x = Metropolis(x0, delta, n,s)[0]
        
        for j in range(l):
            Var[j] = np.var(x[ : N*(j+1)])
            h[j] = abs(Var[j] - theo_var) / theo_var
            aver[j] += h[j]
    
    D_arr = aver / N_aver
    ki = 0
    while D_arr[ki] > 0.05:
        ki += 1
        
    return (ki-1) * N



   
"""
BURNIN GIà CALCOLATO, CHE CI VUOLE TEMPO!!! 
y = np.asarray([9100, 6400, 5600, 3900, 5500, 5300, 4400, 2700, 3300, 3100, 3600, 3200,
 2900, 3500, 2500, 2500, 2200, 2400, 3200, 2300, 2700, 2300, 2200, 1900,
 1700, 2000, 1800, 1800, 2300, 1900, 2300])

y = np.asarray([7700, 5700, 4800, 4400, 3500, 3700, 2700, 3200, 2600, 2100, 1900, 2000,
 1900, 1700, 1700, 1800]) # Fit Exp: A=12008, b = 1.48; Fit Pow: A=2612, b=-1.13
"""

from scipy.optimize import curve_fit

x = np.arange(0.4, 1.64, 0.08)
burn_in = np.asarray([7700, 5700, 4800, 4400, 3500, 3700, 2700, 3200, 2600, 2100, 1900, 2000,
 1900, 1700, 1700, 1800])
"""
burn_in =  np.zeros(len(x), dtype = np.float32)
for j in range(len(x)):
    burn_in[j] = equil(0, 5, 10000, x[j], 100, 300)
"""
def fit_burnin(x, A, b):
    return A * x**b

param_b, covariance_b = curve_fit(fit_burnin, x, burn_in)

fig_bi, ax_bi = plt.subplots(figsize=(6.2, 4.5))
ax_bi.scatter(x, burn_in, label='Numerical estimation', marker='o', s=50)
ax_bi.plot(x, fit_burnin(x, *param_b), color='red', label='Power fit')

ax_bi.set_xlabel('parameter', fontsize=12)
ax_bi.set_ylabel('Burn-in length', fontsize=12)
ax_bi.legend()
ax_bi.grid(True)
plt.show()






