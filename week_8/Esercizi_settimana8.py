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
from scipy.optimize import curve_fit


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






# ---------    PARABOLIC TRIAL FUNCTION HARMONIC OSCILLATOR

@njit
def block_average(lst, s):
    
    aver = np.zeros(s, dtype = np.float32)
    block_size = int(len(lst) / s)

    for k in range(s):
        aver[k] = np.std(lst[(k * block_size):((k + 1) * block_size)])
        
    Sigma_s = np.std(aver)
        
    return Sigma_s / math.sqrt(s)


"""
par_VMC = np.arange(0.8, 3, 0.05)


def VMC( n_MC, s_blocks ):
    
    # |wave function|**2
    def trial_f(x, a):
        if abs(x) < a:
            y = (a**2 - x**2) ** 2
        else:
            y = 0
        return y
    
    # From the equilibration estimate fit with power low
    def burnin_f(x):
        return 2445 * x ** (-1.25)
    
    # Local energy for the quantum HO with |psi|^2 = trial_f
    def Etot_l(x, a):
        return 1 / (a**2 - x**2) + x**2 / 2                         
    
    # Metropolis kernel code, for reweighting condition
    def Metropolis( x0, delta, n, a):
        points = np.zeros(n, dtype = np.float32)
        points[0] = x0
        x_t = x0
        for i in range(1, n):
            x_star = np.random.uniform(x_t - delta, x_t + delta)        
            alpha = min(1, trial_f(x_star, a) / trial_f(x_t, a))
            if alpha >= np.random.rand() :
                x_t = x_star    
            points[i] = x_t     
        return points
        
    
    last_par = 0.5*par_VMC[0]
    E_loc = np.zeros((len(par_VMC), n_MC), dtype = np.float32)
    E_loc_2 = np.zeros((len(par_VMC), n_MC), dtype = np.float32)
    err1 = np.zeros(len(par_VMC), dtype = np.float32)
    err2 = np.zeros(len(par_VMC), dtype = np.float32)
    variance = np.zeros(len(par_VMC), dtype = np.float32)
    E_l = np.zeros(len(par_VMC), dtype = np.float32)
    E2_l = np.zeros(len(par_VMC), dtype = np.float32)
    err_var = np.zeros(len(par_VMC), dtype = np.float32)
    
    for i in range(len(par_VMC)):
        
        delta = 2.5    #parabola case
        
        # Condition: use or not reweighting? Creterion end pag 18 unit 8
        pinco = Metropolis(0, delta, 50000, last_par)
        num_w = trial_f(pinco[0], last_par)
        den_w = trial_f(pinco[0], par_VMC[i])
        pesi = num_w / den_w
        N_eff = (np.sum(pesi))**2 / np.sum(pesi**2)
        
        if (N_eff/50000) < 0.95:
            
            x0 = np.random.uniform(- delta, delta)            
            x_t = x0
            
            # Equilibration phase
            equil_len = int(burnin_f(par_VMC[i]))
            for k in range(equil_len):
                
                x_star = np.random.uniform(x_t - delta, x_t + delta)
                
                num = trial_f(x_star, par_VMC[i]) 
                den = trial_f(x_t, par_VMC[i]) + 1e-12
                                                        
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

def fitE_parab(x, a, b):
    return a/x**2 + b*x**2

def fitVar_parab(x, a, b, c):
    return a/x**4 + b*x**4 + c


Kazan = VMC( 200000, 200)  
beta = par_VMC
xfit = np.linspace(min(beta), max(beta), 100)


#fitting E curve
parE_parab, covE_parab = curve_fit(fitE_parab, beta[3:], Kazan[0][3:], sigma=Kazan[1][3:])
a_Eparab, b_Eparab = parE_parab

#plotting E
fig_El, ax_El = plt.subplots(figsize=(6.2, 4.5))
ax_El.scatter(beta[3:], Kazan[0][3:], marker='o', s=50, label=r'$ \langle E_L(a) \rangle $')
ax_El.errorbar(beta[3:], Kazan[0][3:], yerr=Kazan[1][3:], fmt='.', capsize=5, color='black')
ax_El.plot(xfit[6:], fitE_parab(xfit[6:], a_Eparab, b_Eparab), label='Fit curve', color='crimson')
ax_El.set_xlabel(r'$ a $', fontsize=15)
ax_El.set_ylabel(r'$ \langle E \rangle $', fontsize=15)
ax_El.legend()
ax_El.grid(True)
plt.show()



#fitting Var curve
parVar_parab, covVar_parab = curve_fit(fitVar_parab, beta[:len(beta)-8], Kazan[2][:len(beta)-8], sigma=Kazan[3][:len(beta)-8])
a_Varparab, b_Varparab, c_Varparab = parVar_parab

#plotting Var
fig_var, ax_var = plt.subplots(figsize=(6.2, 4.5))
ax_var.scatter(beta[:len(beta)-8], Kazan[2][:len(beta)-8], marker='o', s=50, label=r'$ \sigma_{E_L}^2 $')
ax_var.errorbar(beta[:len(beta)-8], Kazan[2][:len(beta)-8], yerr=Kazan[3][:len(beta)-8], fmt='.',  capsize=5, color='black')
ax_var.plot(xfit[:len(xfit)-16], fitVar_parab(xfit[:len(xfit)-16], a_Varparab, b_Varparab, c_Varparab), label='Fit curve', color='limegreen')
ax_var.set_xlabel(r'$ a $', fontsize=15)
ax_var.set_ylabel(r'$ \langle E^2 \rangle - \langle E \rangle^2 $', fontsize=15)
ax_var.legend()
ax_var.grid(True)
plt.show()
end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print(f"CPU time 'Local energy sampling': {elapsed_time1:.4f} seconds")

#Parabola: 16 punti, n=10^5, s=10^2 -> t= 24sec
#Parabola: 44 punti, n=2*10^5, s=2*10^2 -> t= 118sec
"""






"""
burn_in =  np.zeros(len(x), dtype = np.float32)
for j in range(len(x)):
    burn_in[j] = equil(0, 5, 10000, x[j], 100, 300)
    
---> BURNIN GIà CALCOLATO, CHE CI VUOLE TEMPO!!! 

y_gauss = np.asarray([1900, 2600, 3300, 3200, 3500, 4500, 4300, 5100, 5400, 6700, 5800, 6300,
 6800, 6600, 6900]) # Fit Pow: A=5719, a=0.52

y_parab = np.asarray([2600, 2000, 2100, 1500, 1700, 1500, 1300, 1200, 1200, 1100,  900, 1000,
  900, 1000,  900,  800]) # Fit pow: A=2445, a= -1.25

y_hydrogen = np.asarray([6700, 5600, 6400, 5100, 6200, 7100, 7400, 6900, 6800, 8800, 9800])
   # Fit pow: A=7068, a= 0.42
"""





"""
def trial_f(x, a):
   if x > 0:
       y = np.exp(-2 * x / a)
   else:
       y = 0
   return y


def Metropolis( x0, delta, n, a):

    acc = 0
    points = np.zeros(n, dtype = np.float32)
    points[0] = x0
    
    x_t = x0
    
    for i in range(1, n):
        
        x_star =np.random.uniform(x_t - delta, x_t + delta)        
        alpha = min(1, (x_star/x_t)**2 * (trial_f(x_star, a) / trial_f(x_t, a)))
        
        if alpha >= np.random.rand() :
            x_t = x_star
            acc += 1
            
        points[i] = x_t
            
    return points, acc/n





def equil(x0, delta, n, a, N, N_aver):
    
    theo_var = (3/4) * a**2     #It depends on the choice of trial_f!!!

    l = int(n/N)
    Var = np.zeros(l, dtype = np.float32)
    h = np.zeros(l, dtype = np.float32)
    aver =  np.zeros(l, dtype = np.float32)

    
    for i in range(N_aver):
        
        x = Metropolis(x0, delta, n, a)[0]
        
        for j in range(l):
            Var[j] = np.var(x[ : N*(j+1)])
            h[j] = abs(Var[j] - theo_var) / theo_var
            aver[j] += h[j]
    
    D_arr = aver / N_aver
    
    ki = 0
    while D_arr[ki] > 0.05:
        ki += 1
       
    return (ki-1) * N



mali = np.arange(100, 10100, 100)
canarie = equil(0.2, 5, 10000, 1, 100, 100)
jasmine = np.linspace(min(mali), max(mali), 2)
fig_eq, ax_eq = plt.subplots(figsize=(6.2, 4.5))
ax_eq.plot(jasmine, [0.05 for _ in range(2)], label='Equilibration limit', color='red', linewidth=2)
ax_eq.scatter(mali, canarie, marker='o', s=50)
ax_eq.set_xlabel('n', fontsize=15)
ax_eq.set_ylabel(r'$ | \sigma_{num}^2 - \sigma_{exp}^2 | / \sigma_{exp}^2 $', fontsize=15)
ax_eq.grid(True)
plt.show()



x = np.arange(0.5, 1.6, 0.1)

burn_in =  np.zeros(len(x), dtype = np.float32)
for j in range(len(x)):
    burn_in[j] = equil(0.2, 2.7, 20000, x[j], 100, 300)


from scipy.optimize import curve_fit


def fit_burnin(x, A, a):
    return A * x**a

param_b, covariance_b = curve_fit(fit_burnin, x, burn_in)

fig_bi, ax_bi = plt.subplots(figsize=(6.2, 4.5))
ax_bi.scatter(x, burn_in, label='Numerical estimation', marker='o', s=50)
ax_bi.plot(x, fit_burnin(x, *param_b), color='red', label='Power fit')

ax_bi.set_xlabel(r'$ a_0 $', fontsize=15)
ax_bi.set_ylabel('Burn-in length', fontsize=15)
ax_bi.legend()
ax_bi.grid(True)
plt.show()
"""





"""
# STUDY ON THE ACCEPTANCE RATIO FOR psi = PARABOLA 
#DEPENDENCE ON BOTH DELTA AND A. GRAPHICALLY WE SEE:
#FOR a in [1, 2.5], WITH delta = 2.5 WE STAY BETWEEN 0.25 AND 0.56 OF ACC. RAT.

# Generate data
d_arr = np.linspace(0.5, 5, 20)
a_arr = np.linspace(0.5, 3, 20)
X, Y = np.meshgrid(d_arr, a_arr)
Z = np.zeros((len(d_arr), len(a_arr)), dtype=np.float32)
for i in range(len(d_arr)):
    for j in range(len(a_arr)):
        Z[i,j] = Metropolis(0, d_arr[j], 10000, a_arr[i])[1]


# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.set_xlabel(r'$ \delta $',fontsize=15)
ax.set_ylabel('a',fontsize=15)
ax.set_zlabel('Acceptance ratio',fontsize=15)

#Acceptance ratio ideal limit between 1/3 and 1/2
Z_plane_1_3 = np.ones_like(X) * (1/3)
Z_plane_1_2 = np.ones_like(X) * (1/2)
ax.plot_surface(X, Y, Z_plane_1_3, alpha=0.8, color='red')
ax.plot_surface(X, Y, Z_plane_1_2, alpha=0.8, color='red')

plt.show()

"""


"""
# STUDY ON THE ACCEPTANCE RATIO FOR psi = EXP (HYDROGEN ATOM) 
#DEPENDENCE ON BOTH DELTA AND A. GRAPHICALLY WE SEE:
#FOR a in [0.5, 1.5], WITH delta = 2.7 WE STAY BETWEEN 0.23 AND 0.56 OF ACC. RAT.

# Generate data
d_arr = np.linspace(0.5, 5, 20)
a_arr = np.linspace(0.5, 3, 20)
X, Y = np.meshgrid(d_arr, a_arr)
Z = np.zeros((len(d_arr), len(a_arr)), dtype=np.float32)
for i in range(len(d_arr)):
    for j in range(len(a_arr)):
        Z[i,j] = Metropolis(0.2, d_arr[j], 10000, a_arr[i])[1]


# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.set_xlabel(r'$ \delta $',fontsize=15)
ax.set_ylabel('a',fontsize=15)
ax.set_zlabel('Acceptance ratio',fontsize=15)

#Acceptance ratio ideal limit between 1/3 and 1/2
Z_plane_1_3 = np.ones_like(X) * (1/3)
Z_plane_1_2 = np.ones_like(X) * (1/2)
ax.plot_surface(X, Y, Z_plane_1_3, alpha=0.8, color='red')
ax.plot_surface(X, Y, Z_plane_1_2, alpha=0.8, color='red')

plt.show()
"""




# ---------    GAUSSIAN TRIAL FUNCTION HARMONIC OSCILLATOR
"""
par_VMC = np.arange(0.1, 1.5, 0.05)


def VMC( n_MC, s_blocks ):
    
    # |wave function|**2
    def trial_f(x, b):
        return np.exp(-2 * b * x**2)
    
    # From the equilibration estimate fit with power low
    def burnin_f(x):
        return 5719 * x ** (0.52)
    
    # Local energy for the quantum HO with |psi|^2 = trial_f
    def Etot_l(x, b):
        return ( (1 - 4 * b**2 ) / 2 ) * x**2 + b  #gauss case
    
    # Metropolis kernel code, for reweighting condition
    def Metropolis( x0, d, n, b):
        points = np.zeros(n, dtype = np.float32)
        points[0] = x0
        x_t = x0
        for i in range(1, n):
            x_star = np.random.uniform(x_t - d, x_t + d)        
            alpha = min(1, trial_f(x_star, b) / trial_f(x_t, b))
            if alpha >= np.random.rand() :
                x_t = x_star    
            points[i] = x_t     
        return points
        
    
    last_par = 0.5*par_VMC[0] 
    E_loc = np.zeros((len(par_VMC), n_MC), dtype = np.float32)
    E_loc_2 = np.zeros((len(par_VMC), n_MC), dtype = np.float32)
    err1 = np.zeros(len(par_VMC), dtype = np.float32)
    err2 = np.zeros(len(par_VMC), dtype = np.float32)
    variance = np.zeros(len(par_VMC), dtype = np.float32)
    E_l = np.zeros(len(par_VMC), dtype = np.float32)
    E2_l = np.zeros(len(par_VMC), dtype = np.float32)
    err_var = np.zeros(len(par_VMC), dtype = np.float32)
    
    for i in range(len(par_VMC)):
        
        delta = 5 / (2 * math.sqrt(par_VMC[i]))   #optimal delta = 5*sigma

        # Condition: use or not reweighting? Creterion end pag 18 unit 8
        pinco = Metropolis(0, delta, 50000, last_par)
        num_w = trial_f(pinco[0], last_par)
        den_w = trial_f(pinco[0], par_VMC[i])
        pesi = num_w / den_w
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




start_time1 = time.time()

def fitE_gauss(x, a, b):
    return a/x + b*x

def fitVar_gauss(x, a, b, c):
    return a/x**2 + b*x**2 + c


Kazan = VMC( 100000, 100)  
beta = par_VMC
xfit = np.linspace(min(beta), max(beta), 100)


#fitting E curve
parE_gauss, covE_gauss = curve_fit(fitE_gauss, beta, Kazan[0], sigma=Kazan[1])
a_Egauss, b_Egauss = parE_gauss

#plotting E
fig_El, ax_El = plt.subplots(figsize=(6.2, 4.5))
ax_El.scatter(beta, Kazan[0], marker='o', s=50, label=r'$ \langle E_L(\beta) \rangle $')
ax_El.errorbar(beta, Kazan[0], yerr=Kazan[1], fmt='.', capsize=5, color='black')
ax_El.plot(xfit, fitE_gauss(xfit, a_Egauss, b_Egauss), label='Fit curve', color='crimson')
ax_El.set_xlabel(r'$ \beta = 1 / 4\sigma^2 $', fontsize=15)
ax_El.set_ylabel(r'$ \langle E \rangle $', fontsize=15)
ax_El.legend()
ax_El.grid(True)
plt.show()



#fitting Var curve
parVar_gauss, covVar_gauss = curve_fit(fitVar_gauss, beta, Kazan[2], sigma=Kazan[3])
a_Vargauss, b_Vargauss, c_Vargauss = parVar_gauss

#plotting Var
fig_var, ax_var = plt.subplots(figsize=(6.2, 4.5))
ax_var.scatter(beta, Kazan[2], marker='o', s=50, label=r'$ \sigma_{E_L}^2 $')
ax_var.errorbar(beta, Kazan[2], yerr=Kazan[3], fmt='.',  capsize=5, color='black')
ax_var.plot(xfit, fitVar_gauss(xfit, a_Vargauss, b_Vargauss, c_Vargauss), label='Fit curve', color='limegreen')
ax_var.set_xlabel(r'$ \beta = 1 / 4\sigma^2 $', fontsize=15)
ax_var.set_ylabel(r'$ \langle E^2 \rangle - \langle E \rangle^2 $', fontsize=15)
ax_var.legend()
ax_var.grid(True)
plt.show()



end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print(f"CPU time 'Local energy sampling': {elapsed_time1:.4f} seconds")

#Gauss: 16 punti, n=10^5, s=10^2 -> t= 27sec
#Gauss: 31 punti, n=10^5, s=10^2 -> t= 53sec
"""








# ---------    GAUSSIAN TRIAL FUNCTION HARMONIC OSCILLATOR WITH PERTURBATION
"""
par_VMC = np.arange(0.2, 1.4, 0.05)


def VMC( n_MC, s_blocks ):
    
    # |wave function|**2
    def trial_f(x, b):
        return np.exp(-2 * b * x**2)
    
    # From the equilibration estimate fit with power low
    def burnin_f(x):
        return 5719 * x ** (0.52)
    
    # Local energy for the quantum HO with perturbation; |psi|^2 = trial_f
    def Etot_l(x, b):
        return ( 4*(1 - 4 * b**2 )*x**2 + x**4)/8 + b  #gauss case
    
    # Metropolis kernel code, for reweighting condition
    def Metropolis( x0, d, n, b):
        points = np.zeros(n, dtype = np.float32)
        points[0] = x0
        x_t = x0
        for i in range(1, n):
            x_star = np.random.uniform(x_t - d, x_t + d)        
            alpha = min(1, trial_f(x_star, b) / trial_f(x_t, b))
            if alpha >= np.random.rand() :
                x_t = x_star    
            points[i] = x_t     
        return points
        
    
    last_par = 0.5*par_VMC[0] 
    E_loc = np.zeros((len(par_VMC), n_MC), dtype = np.float32)
    E_loc_2 = np.zeros((len(par_VMC), n_MC), dtype = np.float32)
    err1 = np.zeros(len(par_VMC), dtype = np.float32)
    err2 = np.zeros(len(par_VMC), dtype = np.float32)
    variance = np.zeros(len(par_VMC), dtype = np.float32)
    E_l = np.zeros(len(par_VMC), dtype = np.float32)
    E2_l = np.zeros(len(par_VMC), dtype = np.float32)
    err_var = np.zeros(len(par_VMC), dtype = np.float32)
    
    for i in range(len(par_VMC)):
        
        delta = 5 / (2 * math.sqrt(par_VMC[i]))   #optimal delta = 5*sigma

        # Condition: use or not reweighting? Creterion end pag 18 unit 8
        pinco = Metropolis(0, delta, 50000, last_par)
        num_w = trial_f(pinco[0], last_par)
        den_w = trial_f(pinco[0], par_VMC[i])
        pesi = num_w / den_w
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


def fitE_Anarm(x, a, b, c):
    return a/x**2 + b/x + c*x


Kazan = VMC( 100000, 100)  
beta = par_VMC
xfit = np.linspace(min(beta), max(beta), 100)


#fitting E curve
parE_gauss, covE_gauss = curve_fit(fitE_Anarm, beta, Kazan[0], sigma=Kazan[1])
a_Anharm, b_Anharm, c_Anharm = parE_gauss

#plotting E
fig_El, ax_El = plt.subplots(figsize=(6.2, 4.5))
ax_El.scatter(beta, Kazan[0], marker='o', s=50, label=r'$ \langle E_L(\beta) \rangle $')
ax_El.errorbar(beta, Kazan[0], yerr=Kazan[1], fmt='.', capsize=5, color='black')
ax_El.plot(xfit, fitE_Anarm(xfit, a_Anharm, b_Anharm, c_Anharm), label='Fit curve', color='crimson')
ax_El.set_xlabel(r'$ \beta = 1 / 4\sigma^2 $', fontsize=15)
ax_El.set_ylabel(r'$ \langle E \rangle $', fontsize=15)
ax_El.legend()
ax_El.grid(True)
plt.show()


end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print(f"CPU time 'Local energy sampling': {elapsed_time1:.4f} seconds")

#Gauss_anharmonic: 16 punti, n=10^5, s=10^2 -> t= 28sec
"""






# ---------    EXPONENTIAL TRIAL FUNCTION: HYDROGEN ATOM (spherical wave: n=1)


par_VMC = np.arange(0.1, 2.1, 0.05)


def VMC( n_MC, s_blocks ):
    
    # |wave function|**2
    def trial_f(x, a):
        if x > 0:
            y = np.exp(-2 * x * a)
        else:
            y = 0
        return y
    
    # From the equilibration estimate fit with power low
    def burnin_f(x):
        return 7068 * x ** (0.42)
    
    # Local energy for the quantum HO with perturbation; |psi|^2 = trial_f
    def Etot_l(x, a):
        return  (2*a - 2 - x*a**2)/(2*x) #exponential case
    
    # Metropolis kernel code, for reweighting condition
    def Metropolis_H( x0, d, n, a):
        points = np.zeros(n, dtype = np.float32)
        points[0] = x0
        x_t = x0
        for i in range(1, n):
            x_star = np.random.uniform(x_t - d, x_t + d)        
            alpha = min(1, (x_star/x_t)**2 * (trial_f(x_star, a) / trial_f(x_t, a)))
            if alpha >= np.random.rand() :
                x_t = x_star    
            points[i] = x_t     
        return points
        
    
    last_par = 0.5*par_VMC[0] 
    E_loc = np.zeros((len(par_VMC), n_MC), dtype = np.float32)
    E_loc_2 = np.zeros((len(par_VMC), n_MC), dtype = np.float32)
    err1 = np.zeros(len(par_VMC), dtype = np.float32)
    err2 = np.zeros(len(par_VMC), dtype = np.float32)
    variance = np.zeros(len(par_VMC), dtype = np.float32)
    E_l = np.zeros(len(par_VMC), dtype = np.float32)
    E2_l = np.zeros(len(par_VMC), dtype = np.float32)
    err_var = np.zeros(len(par_VMC), dtype = np.float32)
    
    for i in range(len(par_VMC)):
        
        delta = 2.7     #Exponential case

        # Condition: use or not reweighting? Creterion end pag 18 unit 8
        pinco = Metropolis_H(0.2, delta, 50000, last_par)
        num_w = trial_f(pinco[0], last_par)
        den_w = trial_f(pinco[0], par_VMC[i])
        pesi = num_w / den_w
        N_eff = (np.sum(pesi))**2 / np.sum(pesi**2)
        
        if (N_eff/50000) < 0.95:
            
            x0 = np.random.uniform(- delta, delta)            
            x_t = x0
            
            
            # Equilibration phase
            equil_len = int(burnin_f(par_VMC[i]))
            for k in range(equil_len):
                
                x_star = np.random.uniform(x_t - delta, x_t + delta)
                
                num = x_star**2 * trial_f(x_star, par_VMC[i]) 
                den = x_t**2 * trial_f(x_t, par_VMC[i]) + 1e-12 
                                                        
                alpha = num/den
                
                if alpha >= np.random.rand():
                    x_t = x_star
            
            
            
            # Accumulation phase
            sampl = np.zeros(n_MC, dtype = np.float32)
            sampl[0] = x_t
            weight_0 = np.zeros(n_MC, dtype = np.float32)
            
            for k in range(1, n_MC):
                x_star = np.random.uniform(x_t - delta, x_t + delta)
                
                num = x_star**2 * trial_f(x_star, par_VMC[i]) 
                den = x_t**2 * trial_f(x_t, par_VMC[i]) + 1e-12
                                                        
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



def fitE_H(x, a, b):
    return a*x**2 + b*x


Kazan = VMC( 200000, 200)  
beta = par_VMC
xfit = np.linspace(min(beta), max(beta), 100)


#fitting E curve
parE_H, covE_H = curve_fit(fitE_H, beta, Kazan[0], sigma=Kazan[1])
a_EH, b_EH = parE_H

#plotting E
fig_El, ax_El = plt.subplots(figsize=(6.2, 4.5))
ax_El.scatter(beta, Kazan[0], marker='o', s=50, label=r'$ \langle E_L(\alpha) \rangle $')
ax_El.errorbar(beta, Kazan[0], yerr=Kazan[1], fmt='.', capsize=5, color='black')
ax_El.plot(xfit, fitE_H(xfit, a_EH, b_EH), label='Fit curve', color='crimson')
ax_El.set_xlabel(r'$ \alpha = 1/a_0 $', fontsize=15)
ax_El.set_ylabel(r'$ \langle E \rangle $', fontsize=15)
ax_El.legend()
ax_El.grid(True)
plt.show()



#plotting Var
fig_var, ax_var = plt.subplots(figsize=(6.2, 4.5))
ax_var.scatter(beta, Kazan[2], marker='o', s=50, label=r'$ \sigma_{E_L}^2 $')
ax_var.errorbar(beta, Kazan[2], yerr=Kazan[3], fmt='.',  capsize=5, color='black')
ax_var.scatter(1, 0, marker='o', s=70, color='red', label='Zero variance point')
ax_var.set_xlabel(r'$ \alpha = 1/a_0 $', fontsize=15)
ax_var.set_ylabel(r'$ \langle E^2 \rangle - \langle E \rangle^2 $', fontsize=15)
ax_var.legend()
ax_var.grid(True)
plt.show()


end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print(f"CPU time 'Local energy sampling': {elapsed_time1:.4f} seconds")

#Exponential_hydrogen: 10 punti, n=10^5, s=10^2 -> t= 20sec
#Exponential_hydrogen: 40 punti, n=2*10^5, s=2*10^2 -> t= 128sec






