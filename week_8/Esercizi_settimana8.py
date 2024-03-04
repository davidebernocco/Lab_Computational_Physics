"""
Now I have to do everything from the beginning again

@author: david
"""


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
from Funz8 import block_average, VMC_parab, fit_burnin, equil, Metropolis,trial_f
from Funz8 import VMC_H, fitE_H, VMC_anharm, fitE_Anarm, VMC_AO, fitE_gauss, fitVar_gauss


"""
Per l'equilibration phase, stimo a parte con la funzione equil() da Funz7 il burnin period.
Una volta per tutte per ogni target e proposal function che considero..
Per il caso di proposal uniforme e target gaussiana ho già calcolato n_burnin = 2700
Rimuovo a posteriori i primi 2700 punti dalla distribuzione (o uso un if sull'indice i: se < 2700 no accumulo Eloc)    
Per altre trial functions, studio a parte l'equilibration phase                                                                                     
"""


# Logical scheme for the VMC algorithm with reweighting
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





# -----------------------------------------------------------------------------
# ES 1.a)
# ---------    GAUSSIAN TRIAL FUNCTION HARMONIC OSCILLATOR

par_VMC = np.arange(0.1, 1.5, 0.05)


start_time1 = time.time()


Kazan = VMC_AO( par_VMC, 100000, 100)  
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





# ----------------------------------------------------------------------------
# ES 1.b)
# ---------    PARABOLIC TRIAL FUNCTION HARMONIC OSCILLATOR


par_VMC = np.arange(0.8, 3, 0.05)



#Plotting the results
start_time1 = time.time()

def fitE_parab(x, a, b):
    return a/x**2 + b*x**2

def fitVar_parab(x, a, b, c):
    return a/x**4 + b*x**4 + c


Kazan = VMC_parab(par_VMC, 200000, 200)  
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




# STUDY ON THE ACCEPTANCE RATIO FOR psi = PARABOLA 
#DEPENDENCE ON BOTH DELTA AND A. GRAPHICALLY WE SEE:
#FOR a in [1, 2.5], WITH delta = 2.5 WE STAY BETWEEN 0.25 AND 0.56 OF ACC. RAT.
"""
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





# -----------------------------------------------------------------------------
# ES 2)
# ---------    GAUSSIAN TRIAL FUNCTION HARMONIC OSCILLATOR WITH PERTURBATION

par_VMC = np.arange(0.2, 1.4, 0.05)


#Plotting the results
start_time1 = time.time()


Kazan = VMC_anharm( par_VMC, 100000, 100)  
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






# ----------------------------------------------------------------------------
# ES 3)
# ---------    EXPONENTIAL TRIAL FUNCTION: HYDROGEN ATOM (spherical wave: n=1)


par_VMC = np.arange(0.1, 2.1, 0.05)



#Plotting the results
start_time1 = time.time()

Kazan = VMC_H( par_VMC, 200000, 200)  
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




# STUDY ON THE ACCEPTANCE RATIO FOR psi = EXP (HYDROGEN ATOM) 
#DEPENDENCE ON BOTH DELTA AND A. GRAPHICALLY WE SEE:
#FOR a in [0.5, 1.5], WITH delta = 2.7 WE STAY BETWEEN 0.23 AND 0.56 OF ACC. RAT.
"""
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









# ----------------------------------------------------------------------------- 
# Study of the BURN-IN SEQUENCE LENGHT 
# (modify functions in Funz8 when the trial function is changed !!)
# ------
"""
#BURN-IN SEQUENCE for EXPONENTIAL FUNCTION (it takes quite a long!)

x = np.arange(0.5, 1.6, 0.1)

burn_in =  np.zeros(len(x), dtype = np.float32)
for j in range(len(x)):
    burn_in[j] = equil(0.2, 2.7, 15000, x[j], 150, 200)


param_b, covariance_b = curve_fit(fit_burnin, x, burn_in)

fig_bi, ax_bi = plt.subplots(figsize=(6.2, 4.5))
ax_bi.scatter(x, burn_in, label='Numerical estimation', marker='o', s=50)
ax_bi.plot(x, fit_burnin(x, *param_b), color='red', label='Power fit')

ax_bi.set_xlabel(r'$ \alpha = 1/a_0 $', fontsize=15)
ax_bi.set_ylabel('Burn-in length', fontsize=15)
ax_bi.legend()
ax_bi.grid(True)
plt.show()
"""




