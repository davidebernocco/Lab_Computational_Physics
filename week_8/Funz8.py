"""
Library of self-made functions needed for the codes implemented for the exercises of the 5th week

@author: david
"""

import math
import numpy as np
from numba import njit



# -----------------------------------------------------------------------------
# Functions for VMC algorithm.

@njit
def block_average(lst, s):
    
    aver = np.zeros(s, dtype = np.float32)
    block_size = int(len(lst) / s)

    for k in range(s):
        aver[k] = np.std(lst[(k * block_size):((k + 1) * block_size)])
        
    Sigma_s = np.std(aver)
        
    return Sigma_s / math.sqrt(s)





def VMC_AO( lst_par, n_MC, s_blocks ):
    
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
        
    
    last_par = 0.5*lst_par[0] 
    E_loc = np.zeros((len(lst_par), n_MC), dtype = np.float32)
    E_loc_2 = np.zeros((len(lst_par), n_MC), dtype = np.float32)
    err1 = np.zeros(len(lst_par), dtype = np.float32)
    err2 = np.zeros(len(lst_par), dtype = np.float32)
    variance = np.zeros(len(lst_par), dtype = np.float32)
    E_l = np.zeros(len(lst_par), dtype = np.float32)
    E2_l = np.zeros(len(lst_par), dtype = np.float32)
    err_var = np.zeros(len(lst_par), dtype = np.float32)
    
    for i in range(len(lst_par)):
        
        delta = 5 / (2 * math.sqrt(lst_par[i]))   #optimal delta = 5*sigma

        # Condition: use or not reweighting? Creterion end pag 18 unit 8
        pinco = Metropolis(0, delta, 50000, last_par)
        num_w = trial_f(pinco[0], last_par)
        den_w = trial_f(pinco[0], lst_par[i])
        pesi = num_w / den_w
        N_eff = (np.sum(pesi))**2 / np.sum(pesi**2)
        
        if (N_eff/50000) < 0.95:
            
            x0 = np.random.uniform(- delta, delta)            
            x_t = x0
            
            
            # Equilibration phase
            equil_len = int(burnin_f(lst_par[i]))
            for k in range(equil_len):
                
                x_star = np.random.uniform(x_t - delta, x_t + delta)
                
                num = trial_f(x_star, lst_par[i]) 
                den = trial_f(x_t, lst_par[i])    
                                                        
                alpha = num/den
                
                if alpha >= np.random.rand():
                    x_t = x_star
            
            
            
            # Accumulation phase
            sampl = np.zeros(n_MC, dtype = np.float32)
            sampl[0] = x_t
            weight_0 = np.zeros(n_MC, dtype = np.float32)
            
            for k in range(1, n_MC):
                x_star = np.random.uniform(x_t - delta, x_t + delta)
                
                num = trial_f(x_star, lst_par[i]) 
                den = trial_f(x_t, lst_par[i])    
                                                        
                alpha = num/den
                
                if alpha >= np.random.rand() :
                    x_t = x_star
            
                sampl[k] = x_t
                weight_0[k] = trial_f(x_t, lst_par[i])
                E_loc[i, k] += Etot_l(x_t, lst_par[i])
                E_loc_2[i, k] += Etot_l(x_t, lst_par[i]) ** 2
            
            variance[i] = np.var(E_loc[i])
            err1[i] = block_average(E_loc[i], s_blocks)
            err2[i] = block_average(E_loc_2[i], s_blocks)
            E_l[i] = np.sum(E_loc[i]) / n_MC
            E2_l[i] = np.sum(E_loc_2[i]) / n_MC
            last_par = lst_par[i]
                
            
            
        # Reweighting when possible    
        else:
            
            weight = np.zeros(n_MC, dtype = np.float32)
            
            for k in range (n_MC):
                
                weight[k] = weight_0[k] / trial_f(sampl[k], lst_par[i])
                E_loc[i,k] = weight[k] * Etot_l(sampl[k], lst_par[i])
                E_loc_2[i,k] = weight[k] * (Etot_l(sampl[k], lst_par[i]) ** 2)
            
            variance[i] = np.var(E_loc[i])
            err1[i] = block_average(E_loc[i], s_blocks)
            err2[i] = block_average(E_loc_2[i], s_blocks)
            E_l[i] = np.sum(E_loc[i]) / np.sum(weight)
            E2_l[i] = np.sum(E_loc_2[i]) / np.sum(weight)
            
     
        err_var[i] = np.sqrt(err2[i]**2 + 4 * E_l[i]**2 * err1[i]**2 )

   
    return E_l, err1, variance, err_var








def VMC_parab( lst_par, n_MC, s_blocks ):
    
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
        
    
    last_par = 0.5*lst_par[0]
    E_loc = np.zeros((len(lst_par), n_MC), dtype = np.float32)
    E_loc_2 = np.zeros((len(lst_par), n_MC), dtype = np.float32)
    err1 = np.zeros(len(lst_par), dtype = np.float32)
    err2 = np.zeros(len(lst_par), dtype = np.float32)
    variance = np.zeros(len(lst_par), dtype = np.float32)
    E_l = np.zeros(len(lst_par), dtype = np.float32)
    E2_l = np.zeros(len(lst_par), dtype = np.float32)
    err_var = np.zeros(len(lst_par), dtype = np.float32)
    
    for i in range(len(lst_par)):
        
        delta = 2.5    #parabola case
        
        # Condition: use or not reweighting? Creterion end pag 18 unit 8
        pinco = Metropolis(0, delta, 50000, last_par)
        num_w = trial_f(pinco[0], last_par)
        den_w = trial_f(pinco[0], lst_par[i])
        pesi = num_w / den_w
        N_eff = (np.sum(pesi))**2 / np.sum(pesi**2)
        
        if (N_eff/50000) < 0.95:
            
            x0 = np.random.uniform(- delta, delta)            
            x_t = x0
            
            # Equilibration phase
            equil_len = int(burnin_f(lst_par[i]))
            for k in range(equil_len):
                
                x_star = np.random.uniform(x_t - delta, x_t + delta)
                
                num = trial_f(x_star, lst_par[i]) 
                den = trial_f(x_t, lst_par[i]) + 1e-12
                                                        
                alpha = num/den
                
                if alpha >= np.random.rand():
                    x_t = x_star
            
            
            # Accumulation phase
            sampl = np.zeros(n_MC, dtype = np.float32)
            sampl[0] = x_t
            weight_0 = np.zeros(n_MC, dtype = np.float32)
            
            for k in range(1, n_MC):
                x_star = np.random.uniform(x_t - delta, x_t + delta)
                
                num = trial_f(x_star, lst_par[i]) 
                den = trial_f(x_t, lst_par[i]) 
                                                        
                alpha = num/den
                
                if alpha >= np.random.rand() :
                    x_t = x_star
            
                sampl[k] = x_t
                weight_0[k] = trial_f(x_t, lst_par[i])
                E_loc[i, k] += Etot_l(x_t, lst_par[i])
                E_loc_2[i, k] += Etot_l(x_t, lst_par[i]) ** 2
            
            variance[i] = np.var(E_loc[i])
            err1[i] = block_average(E_loc[i], s_blocks)
            err2[i] = block_average(E_loc_2[i], s_blocks)
            E_l[i] = np.sum(E_loc[i]) / n_MC
            E2_l[i] = np.sum(E_loc_2[i]) / n_MC
            last_par = lst_par[i]
                
            
        # Reweighting when possible    
        else:
            
            weight = np.zeros(n_MC, dtype = np.float32)
            
            for k in range (n_MC):
                
                weight[k] = weight_0[k] / trial_f(sampl[k], lst_par[i])
                E_loc[i,k] = weight[k] * Etot_l(sampl[k], lst_par[i])
                E_loc_2[i,k] = weight[k] * (Etot_l(sampl[k], lst_par[i]) ** 2)
            
            variance[i] = np.var(E_loc[i])
            err1[i] = block_average(E_loc[i], s_blocks)
            err2[i] = block_average(E_loc_2[i], s_blocks)
            E_l[i] = np.sum(E_loc[i]) / np.sum(weight)
            E2_l[i] = np.sum(E_loc_2[i]) / np.sum(weight)
            
        err_var[i] = np.sqrt(err2[i]**2 + 4 * E_l[i]**2 * err1[i]**2 )

   
    return E_l, err1, variance, err_var








def VMC_anharm( lst_par, n_MC, s_blocks ):
    
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
        
    
    last_par = 0.5*lst_par[0] 
    E_loc = np.zeros((len(lst_par), n_MC), dtype = np.float32)
    E_loc_2 = np.zeros((len(lst_par), n_MC), dtype = np.float32)
    err1 = np.zeros(len(lst_par), dtype = np.float32)
    err2 = np.zeros(len(lst_par), dtype = np.float32)
    variance = np.zeros(len(lst_par), dtype = np.float32)
    E_l = np.zeros(len(lst_par), dtype = np.float32)
    E2_l = np.zeros(len(lst_par), dtype = np.float32)
    err_var = np.zeros(len(lst_par), dtype = np.float32)
    
    for i in range(len(lst_par)):
        
        delta = 5 / (2 * math.sqrt(lst_par[i]))   #optimal delta = 5*sigma

        # Condition: use or not reweighting? Creterion end pag 18 unit 8
        pinco = Metropolis(0, delta, 50000, last_par)
        num_w = trial_f(pinco[0], last_par)
        den_w = trial_f(pinco[0], lst_par[i])
        pesi = num_w / den_w
        N_eff = (np.sum(pesi))**2 / np.sum(pesi**2)
        
        if (N_eff/50000) < 0.95:
            
            x0 = np.random.uniform(- delta, delta)            
            x_t = x0
            
            
            # Equilibration phase
            equil_len = int(burnin_f(lst_par[i]))
            for k in range(equil_len):
                
                x_star = np.random.uniform(x_t - delta, x_t + delta)
                
                num = trial_f(x_star, lst_par[i]) 
                den = trial_f(x_t, lst_par[i])    
                                                        
                alpha = num/den
                
                if alpha >= np.random.rand():
                    x_t = x_star
            
            
            
            # Accumulation phase
            sampl = np.zeros(n_MC, dtype = np.float32)
            sampl[0] = x_t
            weight_0 = np.zeros(n_MC, dtype = np.float32)
            
            for k in range(1, n_MC):
                x_star = np.random.uniform(x_t - delta, x_t + delta)
                
                num = trial_f(x_star, lst_par[i]) 
                den = trial_f(x_t, lst_par[i])    
                                                        
                alpha = num/den
                
                if alpha >= np.random.rand() :
                    x_t = x_star
            
                sampl[k] = x_t
                weight_0[k] = trial_f(x_t, lst_par[i])
                E_loc[i, k] += Etot_l(x_t, lst_par[i])
                E_loc_2[i, k] += Etot_l(x_t, lst_par[i]) ** 2
            
            variance[i] = np.var(E_loc[i])
            err1[i] = block_average(E_loc[i], s_blocks)
            err2[i] = block_average(E_loc_2[i], s_blocks)
            E_l[i] = np.sum(E_loc[i]) / n_MC
            E2_l[i] = np.sum(E_loc_2[i]) / n_MC
            last_par = lst_par[i]
                
            
            
        # Reweighting when possible    
        else:
            
            weight = np.zeros(n_MC, dtype = np.float32)
            
            for k in range (n_MC):
                
                weight[k] = weight_0[k] / trial_f(sampl[k], lst_par[i])
                E_loc[i,k] = weight[k] * Etot_l(sampl[k], lst_par[i])
                E_loc_2[i,k] = weight[k] * (Etot_l(sampl[k], lst_par[i]) ** 2)
            
            variance[i] = np.var(E_loc[i])
            err1[i] = block_average(E_loc[i], s_blocks)
            err2[i] = block_average(E_loc_2[i], s_blocks)
            E_l[i] = np.sum(E_loc[i]) / np.sum(weight)
            E2_l[i] = np.sum(E_loc_2[i]) / np.sum(weight)
            
     
        err_var[i] = np.sqrt(err2[i]**2 + 4 * E_l[i]**2 * err1[i]**2 )

   
    return E_l, err1, variance, err_var









def VMC_H( lst_par, n_MC, s_blocks ):
    
    # |wave function|**2
    def trial_f(x, a):
        if x > 0:
            y = np.exp(-2 * x * a)
        else:
            y = 0
        return y
    
    # From the equilibration estimate fit with power low
    def burnin_f(x):
        return 6586 * x ** (-0.90)
    
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
        
    
    last_par = 0.5*lst_par[0] 
    E_loc = np.zeros((len(lst_par), n_MC), dtype = np.float32)
    E_loc_2 = np.zeros((len(lst_par), n_MC), dtype = np.float32)
    err1 = np.zeros(len(lst_par), dtype = np.float32)
    err2 = np.zeros(len(lst_par), dtype = np.float32)
    variance = np.zeros(len(lst_par), dtype = np.float32)
    E_l = np.zeros(len(lst_par), dtype = np.float32)
    E2_l = np.zeros(len(lst_par), dtype = np.float32)
    err_var = np.zeros(len(lst_par), dtype = np.float32)
    
    for i in range(len(lst_par)):
        
        delta = 2.7     #Exponential case

        # Condition: use or not reweighting? Creterion end pag 18 unit 8
        pinco = Metropolis_H(0.2, delta, 50000, last_par)
        num_w = trial_f(pinco[0], last_par)
        den_w = trial_f(pinco[0], lst_par[i])
        pesi = num_w / den_w
        N_eff = (np.sum(pesi))**2 / np.sum(pesi**2)
        
        if (N_eff/50000) < 0.95:
            
            x0 = np.random.uniform(- delta, delta)            
            x_t = x0
            
            
            # Equilibration phase
            equil_len = int(burnin_f(lst_par[i]))
            for k in range(equil_len):
                
                x_star = np.random.uniform(x_t - delta, x_t + delta)
                
                num = x_star**2 * trial_f(x_star, lst_par[i]) 
                den = x_t**2 * trial_f(x_t, lst_par[i]) + 1e-12 
                                                        
                alpha = num/den
                
                if alpha >= np.random.rand():
                    x_t = x_star
            
            
            
            # Accumulation phase
            sampl = np.zeros(n_MC, dtype = np.float32)
            sampl[0] = x_t
            weight_0 = np.zeros(n_MC, dtype = np.float32)
            
            for k in range(1, n_MC):
                x_star = np.random.uniform(x_t - delta, x_t + delta)
                
                num = x_star**2 * trial_f(x_star, lst_par[i]) 
                den = x_t**2 * trial_f(x_t, lst_par[i]) + 1e-12
                                                        
                alpha = num/den
                
                if alpha >= np.random.rand() :
                    x_t = x_star
            
                sampl[k] = x_t
                weight_0[k] = trial_f(x_t, lst_par[i])
                E_loc[i, k] += Etot_l(x_t, lst_par[i])
                E_loc_2[i, k] += Etot_l(x_t, lst_par[i]) ** 2
            
            variance[i] = np.var(E_loc[i])
            err1[i] = block_average(E_loc[i], s_blocks)
            err2[i] = block_average(E_loc_2[i], s_blocks)
            E_l[i] = np.sum(E_loc[i]) / n_MC
            E2_l[i] = np.sum(E_loc_2[i]) / n_MC
            last_par = lst_par[i]
                
            
            
        # Reweighting when possible    
        else:
            
            weight = np.zeros(n_MC, dtype = np.float32)
            
            for k in range (n_MC):
                
                weight[k] = weight_0[k] / trial_f(sampl[k], lst_par[i])
                E_loc[i,k] = weight[k] * Etot_l(sampl[k], lst_par[i])
                E_loc_2[i,k] = weight[k] * (Etot_l(sampl[k], lst_par[i]) ** 2)
            
            variance[i] = np.var(E_loc[i])
            err1[i] = block_average(E_loc[i], s_blocks)
            err2[i] = block_average(E_loc_2[i], s_blocks)
            E_l[i] = np.sum(E_loc[i]) / np.sum(weight)
            E2_l[i] = np.sum(E_loc_2[i]) / np.sum(weight)
            
     
        err_var[i] = np.sqrt(err2[i]**2 + 4 * E_l[i]**2 * err1[i]**2 )

   
    return E_l, err1, variance, err_var






# -----------------------------------------------------------------------------
# Functions for equilibration phase


# To be modified when changing trial function!
def trial_f(x, a):
   if x > 0:
       y = np.exp(-2 * x * a)
   else:
       y = 0
   return y




# To be modified when changing trial function!
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




# theo_var to be modified when changing trial function!
def equil(x0, delta, n, a, N, N_aver):
    
    theo_var = 3/(4 * a**2)     #It depends on the choice of trial_f!!!

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




def fit_burnin(x, A, a):
    return A * x**a





# -----------------------------------------------------------------------------
# Functions for plot fitting

def fitE_gauss(x, a, b):
    return a/x + b*x


def fitVar_gauss(x, a, b, c):
    return a/x**2 + b*x**2 + c


def fitE_Anarm(x, a, b, c):
    return a/x**2 + b/x + c*x


def fitE_H(x, a, b):
    return a*x**2 + b*x

