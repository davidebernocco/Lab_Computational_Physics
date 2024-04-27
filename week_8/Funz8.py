"""
Library of self-made functions needed for the 8th week exercises

@author: david
"""


import math
import numpy as np
from numba import njit



# -----------------------------------------------------------------------------
# VARIATIONAL MONTE CARLO
# -----------------------------------------------------------------------------

"""
We start by defyining these functions, different from one another in the considered
cases (gaussian A.O., parabola A.O., gaussian Anharm. O., exponential H atom).
They will be called in the main function "VMC" later on.
"""

# -------------------
#GAUSSIAN, A.O.

# |wave function|**2
def trial_f_gauss(x, b):
    return np.exp(-2 * b * x**2)



# From the equilibration estimate fit with power low
def burnin_f_gauss(x):
    return 5719 * x ** (0.52)



# Local energy for the quantum HO with |psi|^2 = trial_f
def Etot_l_gauss(x, b):
    return ( (1 - 4 * b**2 ) / 2 ) * x**2 + b  #gauss case



#optimal delta for gaussian is = 5*sigma
def delta_gauss(x):
    return 5 / (2 * np.sqrt(x)) 





# ------------------
# PARABOLA, A.O.

# |wave function|**2
def trial_f_parab(x, a):
    if abs(x) < a:
        y = (a**2 - x**2) ** 2
    else:
        y = 0
    return y



# From the equilibration estimate fit with power low
def burnin_parab(x):
    return 2445 * x ** (-1.25)



# Local energy for the quantum HO with |psi|^2 = trial_f
def Etot_l_parab(x, a):
    return 1 / (a**2 - x**2) + x**2 / 2  



def delta_parab(x):
    return 2.5 





# --------------
# GAUSS, ANHARMONIC O.

# Local energy for the quantum HO with perturbation; |psi|^2 = trial_f
def Etot_l_anh(x, b):
    return ( 4*(1 - 4 * b**2 )*x**2 + x**4)/8 + b  #gauss case





# ------------------
# EXPONENTIAL, HYDROGEN

# |wave function|**2
def trial_f_H(x, a):
    if x > 0:
        y = np.exp(-2 * x * a)
    else:
        y = 0
    return y



# From the equilibration estimate fit with power low
def burnin_f_H(x):
    return 6586 * x ** (-0.90)



# Local energy for the quantum HO with perturbation; |psi|^2 = trial_f
def Etot_l_H(x, a):
    return  (2*a - 2 - x*a**2)/(2*x) #exponential case



def delta_H(x):
    return 2.7 





# ---------------
# OTHER FUNDAMENTAL FUNCTIONS


# Outputs the stdv of an array through block-averages
@njit
def block_average(lst, s):
    
    aver = np.zeros(s, dtype = np.float32)
    block_size = int(len(lst) / s)

    for k in range(s):
        aver[k] = np.mean(lst[(k * block_size):((k + 1) * block_size)])
        
    Sigma_s = np.std(aver)
        
    return Sigma_s / math.sqrt(s)





# Metropolis kernel code
def Metropolis( x0, d, n, b, trial_f):
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
#
#
#
# Since we are interested into the radial distribution, the trial function
# we put inside the algorithm is r*|psi(r)|^2. As a consequence the code changes:
def Metropolis_H( x0, d, n, a, trial_f):
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





# Condition: use or not reweighting? You find the creterion at end pag. 18 unit 8
def Reweighting_cond(metr, delta, last_par, par, trial_f):
    pinco = metr(0.001, delta, 50000, last_par, trial_f)
    num_w = trial_f(pinco[0], last_par)
    den_w = trial_f(pinco[0], par)
    pesi = num_w / den_w
    N_eff = (np.sum(pesi))**2 / np.sum(pesi**2)
    
    return N_eff/50000





# Equilibration phase during which no quantities are saved. Each trial function
# has its own behaviour of Neq vs parameter. The numerical trend is evaluated
# in the section dedicated to the proper study of the burn-in sequence.
def Equilibration(x_t, delta, par, burnin_f, trial_f):
    equil_len = int(burnin_f(par))
    for k in range(equil_len):
        
        x_star = np.random.uniform(x_t - delta, x_t + delta)
        
        num = trial_f(x_star, par) 
        den = trial_f(x_t, par) + 1e-12   
                                                
        alpha = num/den
        
        if alpha >= np.random.rand():
            x_t = x_star
    
    return x_t
#
#
#
# Since we are interested into the radial distribution, the trial function
# we put inside the algorithm is r*|psi(r)|^2. As a consequence the code changes:
def Equilibration_H(x_t, delta, par, burnin_f, trial_f):
    equil_len = int(burnin_f(par))
    for k in range(equil_len):
        
        x_star = np.random.uniform(x_t - delta, x_t + delta)
        
        num = x_star**2 * trial_f(x_star, par) 
        den =  x_t**2 * trial_f(x_t, par) + 1e-12   
                                                
        alpha = num/den
        
        if alpha >= np.random.rand():
            x_t = x_star
    
    return x_t





# Accumulation phase: the quantities of interest are calculated and saved for 
# each Nmc step.
def Accumulation(n_MC, x_t, delta, par, s, trial_f, Etot_l):
    sampl = np.zeros(n_MC, dtype = np.float32)
    sampl[0] = x_t
    weight_0 = np.zeros(n_MC, dtype = np.float32)
    E_loc = np.zeros(n_MC, dtype = np.float32)
    E_loc_2 = np.zeros(n_MC, dtype = np.float32)
    
    for k in range(1, n_MC):
        x_star = np.random.uniform(x_t - delta, x_t + delta)
        
        num = trial_f(x_star, par) 
        den = trial_f(x_t, par) + 1e-10   
                                                
        alpha = num/den
        
        if alpha >= np.random.rand() :
            x_t = x_star
    
        sampl[k] = x_t
        weight_0[k] = trial_f(x_t, par)
        E_loc[k] = Etot_l(x_t, par)
        E_loc_2[k] = Etot_l(x_t, par) ** 2
    
    variance = np.var(E_loc)
    err1 = block_average(E_loc, s)
    err2 = block_average(E_loc_2, s)
    E_l = np.sum(E_loc) / n_MC
    E2_l = np.sum(E_loc_2) / n_MC
    last_par = par
    
    return variance, err1, err2, E_l, E2_l, last_par, weight_0, sampl
#
#
# 
# Since we are interested into the radial distribution, the trial function
# we put inside the algorithm is r*|psi(r)|^2. As a consequence the code changes:
def Accumulation_H(n_MC, x_t, delta, par, s, trial_f, Etot_l):
    sampl = np.zeros(n_MC, dtype = np.float32)
    sampl[0] = x_t
    weight_0 = np.zeros(n_MC, dtype = np.float32)
    E_loc = np.zeros(n_MC, dtype = np.float32)
    E_loc_2 = np.zeros(n_MC, dtype = np.float32)
    
    for k in range(1, n_MC):
        x_star = np.random.uniform(x_t - delta, x_t + delta)
        
        num = x_star**2 * trial_f(x_star, par) 
        den =  x_t**2 * trial_f(x_t, par) + 1e-12    
                                                
        alpha = num/den
        
        if alpha >= np.random.rand() :
            x_t = x_star
    
        sampl[k] = x_t
        weight_0[k] = trial_f(x_t, par)
        E_loc[k] = Etot_l(x_t, par)
        E_loc_2[k] = Etot_l(x_t, par) ** 2
    
    variance = np.var(E_loc)
    err1 = block_average(E_loc, s)
    err2 = block_average(E_loc_2, s)
    E_l = np.sum(E_loc) / n_MC
    E2_l = np.sum(E_loc_2) / n_MC
    last_par = par
    
    return variance, err1, err2, E_l, E2_l, last_par, weight_0, sampl





# When reweighting is possible, we use the sample already generated
def reweight(n_MC, weight_0, sampl, par, s, trial_f, Etot_l):
    weight = np.zeros(n_MC, dtype = np.float32)
    E_loc = np.zeros(n_MC, dtype = np.float32)
    E_loc_2 = np.zeros(n_MC, dtype = np.float32)
    
    for k in range (n_MC):
        
        weight[k] = weight_0[k] / trial_f(sampl[k], par)
        E_loc[k] = weight[k] * Etot_l(sampl[k], par)
        E_loc_2[k] = weight[k] * (Etot_l(sampl[k], par) ** 2)
    
    variance = np.var(E_loc)
    err1 = block_average(E_loc, s)
    err2 = block_average(E_loc_2, s)
    E_l = np.sum(E_loc) / np.sum(weight)
    E2_l = np.sum(E_loc_2) / np.sum(weight)

    return variance, err1, err2, E_l, E2_l






# Finally the local energy and the variance (both with their numerical error)
# are calculated for different values of the variational parameter.
def VMC( lst_par, n_MC, s_blocks, metr, equil_f, accum_f, delta_f, trial_f, burnin_f, Etot_l ):
    
    last_par = 0.5*lst_par[0] 
    err1 = np.zeros(len(lst_par), dtype = np.float32)
    err2 = np.zeros(len(lst_par), dtype = np.float32)
    variance = np.zeros(len(lst_par), dtype = np.float32)
    E_l = np.zeros(len(lst_par), dtype = np.float32)
    E2_l = np.zeros(len(lst_par), dtype = np.float32)
    err_var = np.zeros(len(lst_par), dtype = np.float32)
    
    for i in range(len(lst_par)):
        
        delta = delta_f(lst_par[i]) 

        # Condition: use or not reweighting?
        rew_cond = Reweighting_cond(metr, delta, last_par, lst_par[i], trial_f)
        
        if rew_cond < 0.95:
            
            #initialization
            x0 = np.random.uniform(- delta, delta)            
            x_t = x0
            
            # Equilibration phase
            equil_f(x_t, delta, lst_par[i], burnin_f, trial_f)
                        
            # Accumulation phase
            results_i = accum_f(n_MC, x_t, delta, lst_par[i], s_blocks, trial_f, Etot_l)
            variance[i], err1[i], err2[i], E_l[i], E2_l[i], last_par, weight_0, sampl = results_i
                
        # Reweighting when possible    
        else:
            results_i = reweight(n_MC, weight_0, sampl, lst_par[i], s_blocks, trial_f, Etot_l)
            variance[i], err1[i], err2[i], E_l[i], E2_l[i]  = results_i
     
        err_var[i] = np.sqrt(err2[i]**2 + 4 * E_l[i]**2 * err1[i]**2 )
   
    return E_l, err1, variance, err_var






# ----------------------------
# Set of fitting function for Energy and Variance vs parameter; from theory.

def fitE_gauss(x, a, b):
    return a/x + b*x


def fitVar_gauss(x, a, b, c):
    return a/x**2 + b*x**2 + c


def fitE_parab(x, a, b):
    return a/x**2 + b*x**2


def fitVar_parab(x, a, b, c):
    return a/x**4 + b*x**4 + c


def fitE_Anarm(x, a, b, c):
    return a/x**2 + b/x + c*x


def fitE_H(x, a, b):
    return a*x**2 + b*x






# --------------------
# STUDY ON THE BURN-IN SEQUENCE



# Gives an estimation to the equilibration sequence, comparing the numerical 
# variance of the considered distribution with the theorethical one (from integrals)
# theo_var to be modified when changing trial function!
def equil(x0, delta, n, a, N, N_aver, theo_var, metr, trial_f):

    l = int(n/N)
    Var = np.zeros(l, dtype = np.float32)
    h = np.zeros(l, dtype = np.float32)
    aver =  np.zeros(l, dtype = np.float32)

    
    for i in range(N_aver):
        
        x = metr(x0, delta, n, a, trial_f)[0]
        
        for j in range(l):
            Var[j] = np.var(x[ : N*(j+1)])
            h[j] = abs(Var[j] - theo_var) / theo_var
            aver[j] += h[j]
    
    D_arr = aver / N_aver
    
    ki = 0
    while D_arr[ki] > 0.05:
        ki += 1
       
    return (ki-1) * N




# We use a generic power law to fit equilibration sequence length vs parameter
def fit_burnin(x, A, a):
    return A * x**a
