"""
Library of self-made functions needed for the 7th week exercises

@author: david
"""
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numba import njit
import random


# -----------------------------------------------------------------------------
# RANDOM NUMBERS WITH GAUSSIAN DISTRIBUTION: METROPOLIS MONTE CARLO ALGORITHM
# -----------------------------------------------------------------------------


# Proposal distribution within the Metropolis algorith.
# Usually it is a gaussian or a uniform distribution (like here)
def proposal_distr(a, b):
    
    return np.random.uniform(a, b, 1)





# Gaussian test function 
def gauss_func(x, mu, sigma):
    return norm.pdf(x, mu, sigma)





# Metropolis code for a specific gaussian trial function. Beside the sampled
# points, it outputs as well the acceptance ratio.
def Metropolis( x0, delta, n, s):
    
    acc = 0
    points = np.zeros(n, dtype = np.float32)
    points[0] = x0
    x_t = x0
    
    for i in range(1, n):
        x_star = np.random.uniform(x_t - delta, x_t + delta)
        alpha = min(1, gauss_func(x_star, 0, s) / gauss_func(x_t, 0, s))
        
        if alpha >= np.random.rand() :
            x_t = x_star
            acc += 1
            
        points[i] = x_t
            
    return points, acc/n






# Function that prints multiple histograms with gaussian fit and chi square
def plot_histo(n_arr, func, s, delta):
    
    coefficients = {}
    
    for i in range(len(n_arr)):
        
        x_lst = np.arange(n_arr[i])
        y_lst = func( 0, delta, n_arr[i], s)[0]
        
        fig = f"fig_{2*i}"
        ax = f"ax_{2*i}"
        fig, ax = plt.subplots(figsize=(6.2, 4.5))
        ax.plot(x_lst, y_lst )
        ax.set_xlabel(r'$ i $', fontsize=17)
        ax.set_ylabel(r'$ x_i $', fontsize=17)
        ax.grid(True)

        # Histo 
        IQR = np.percentile(y_lst, 75) - np.percentile(y_lst, 25)
        nbins = int((max(y_lst) - min(y_lst)) / (2 * IQR * len(y_lst)**(-1/3)))

        hist, bins = np.histogram(y_lst, nbins, density=False)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_widths = np.diff(bins)
        density = hist / (n_arr[i] * bin_widths[0])
        
        # fit
        params, covariance = curve_fit(gauss_func, bin_centers, density)
        Mu, Sigma = params
        expected_values = gauss_func(bin_centers, Mu, Sigma)
        chi_square = np.sum((density - expected_values) ** 2 / expected_values)
        
        x_range = np.linspace(min(bin_centers), max(bin_centers), 1000)
        
        fig_b = f"fig_{2*i+1}"
        ax_b = f"ax_{2*i+1}"
        fig_b, ax_b = plt.subplots(figsize=(6.2, 4.5))
        ax_b.bar(bins[:-1], density, width=bin_widths, alpha=0.5, color='b', label=r'$ PDF^{num} $')
        ax_b.plot(x_range, gauss_func(x_range, Mu, Sigma), label='Gauss fit', color='black')
        ax_b.set_xlabel(' x ', fontsize=15)
        ax_b.set_ylabel('Probability density', fontsize=15)
        ax_b.legend()
        ax_b.grid(True)
        
        plt.show()
        
        coefficients[n_arr[i]] = (params, covariance, chi_square)
    
    print(coefficients)
        
    return coefficients
        
        

 

# Gives the acceptance ratio for different values of delta (gaussian trial function)   
def acc_ratio(x0, n, s, d_arr):
    
    acc = np.zeros(len(d_arr), dtype = np.float32)
    
    for i in range(len(d_arr)):
        crauti = Metropolis(x0, d_arr[i], n, s)
        acc[i] = crauti[1]
    
    return acc
        




# It allows to estimate the equilibration length looking at the absolute difference
# between (instantaneous) numerical variance and expected variance
def equil(x0, delta, n, s, N, N_aver):
    
    l = int(n/N)
    Var = np.zeros(l, dtype = np.float32)
    h = np.zeros(l, dtype = np.float32)
    aver =  np.zeros(l, dtype = np.float32)
    
    for i in range(N_aver):
        
        x = Metropolis(x0, delta, n, s)[0]
        
        for j in range(l):
            Var[j] = np.var(x[ : N*(j+1)])
            h[j] = abs(Var[j] - s ** 2) / s ** 2
            aver[j] += h[j]
        
    return aver / N_aver





# -----------------------------------------------------------------------------
# SAMPLING PHYSICAL QUANTITIES: DIRECT SAMPLING AND METROPOLIS SAMPLING
# -----------------------------------------------------------------------------

# Typical Box-Muller code for gaussian sampling
@njit
def boxmuller(fagioli):
    
    sacchetto = np.zeros(fagioli, dtype = np.float32)
    
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
            
        sacchetto[i] = rnd   
        
    return sacchetto





# Evaluates integrals corresponding to expectation values through a direct
# sample mean algorithm
@njit
def dir_sampl_ground_state(n, s):
    
    chicco = boxmuller(n)
    norm = 0
    x_m = 0
    x2_m = 0
    
    for i in range(n):
        a =  math.e ** ( - chicco[i] ** 2 / (2 * s ** 2))
        norm += a
        x2_m += (chicco[i] ** 2) * a
        x_m += chicco[i] * a
        
    integr = norm / n
    integr1 = x_m / n
    integr2 = x2_m / n
    
    return integr2 / integr, integr1 / integr





# Evaluates integrals corresponding to expectation values through importance
# sampling, where the points are generated with Metropolis algorithm
def Metro_sampl_ground_state(n, s):
    
    x0 = 0
    d = 4*s
    chicco = Metropolis(x0, d, n, s)[0]
    x_m = 0
    x2_m = 0
    
    for i in range(n):
        x2_m += (chicco[i] ** 2)
        x_m += chicco[i]
     
    integr1 = x_m / n
    integr2 = x2_m / n
    
    return integr2, integr1





# Gives the absoulte difference between numerical value and analythical solution
def accuracy(s, lst_n, fun):
    
    E_pot_expected = s**2 / 2
    E_kin_expected = 1 / ( 8 * s ** 2)
    E_tot_expected = E_pot_expected + E_kin_expected
    
    Delta1 = np.zeros(len(lst_n), dtype = np.float32)
    Delta2 = np.zeros(len(lst_n), dtype = np.float32)
    Delta3 = np.zeros(len(lst_n), dtype = np.float32)
    Delta4 = np.zeros(len(lst_n), dtype = np.float32)
    
    for i in range(len(lst_n)):
        cachi = fun(lst_n[i], s)
        E_pot_num = cachi[0] /  2 
        E_kin_num = ( 1 / (4 * s **2)) - cachi[0] / ( 8 * s ** 4  )
        E_tot_num = E_pot_num + E_kin_num
        
        Delta1[i] = abs((cachi[0] - cachi[1] ** 2) - s ** 2)
        Delta2[i] = abs(E_pot_num - E_pot_expected)
        Delta3[i] = abs(E_kin_num - E_kin_expected)
        Delta4[i] = abs(E_tot_num - E_tot_expected)
        
    return Delta1, Delta2, Delta3, Delta4





# -----------------------------------------------------------------------------
# CORRELATIONS
# -----------------------------------------------------------------------------

# Calculates correlations over the sampled points
@njit
def corr(n, N_max, lst):
    
    corr_arr = np.zeros(N_max, dtype = np.float32)
        
    for j in range(N_max):
        
        xi = 0
        x2i = 0
        xi_xij = 0
        
        for i in range(n - j):
            xi += lst[i]
            x2i += lst[i] ** 2 
            xi_xij += lst[i] * lst[i + j] 
            
        xi = xi / (n - j)
        x2i = x2i / (n - j)
        xi_xij = xi_xij / (n - j)
        
        corr_arr[j] = (xi_xij - xi ** 2) / (x2i - xi ** 2)
        
    return corr_arr





# -----------------------------------------------------------------------------
# VERIFICATION OF THE BOLTZMANN DISTRIBUTION
# -----------------------------------------------------------------------------



# Metropolis code for a Boltzmann trial function (single particle 1D)
@njit
def Metropolis_Boltzmann( v0, dvmax, n, kb, T, m):
    
    acc = 0
    velocity = np.zeros(n, dtype = np.float32)
    energy = np.zeros(n, dtype = np.float32)
    velocity[0] = v0
    energy[0] = (m / 2) * v0 ** 2
    
    E_t = (m / 2) * v0 ** 2
    v_t = v0
    
    for i in range(1, n):
        v_star = np.random.uniform(v_t - dvmax, v_t + dvmax)
        E_star = (m / 2) * v_star ** 2
        
        esp1v = ( -m * v_star ** 2 / ( 2 * kb * T) )  
        esp2v = ( -m * v_t ** 2 / ( 2 * kb * T) )    
        alphav = math.e ** (esp1v - esp2v)           
        
        if alphav >= np.random.rand() :
            v_t = v_star
            acc += 1
        
        esp1E = ( - E_star / ( kb * T) )  
        esp2E = ( - E_t / ( kb * T) )    
        alphaE = math.e ** (esp1E - esp2E) 
               
        
        if alphaE >= np.random.rand() :
            E_t = E_star
            
        velocity[i] = v_t
        energy[i] = E_t
            
    return velocity, energy, acc/n





# Evaluates expectation values of typical physical quantities through importance
# sampling + Metropolis (See pag 59 of "Metropolis.pdf")
def Metro_sampl_Boltzmann(v0, dvmax, n, kb, T, m, s): 
    
    sesamo = Metropolis_Boltzmann(v0, dvmax, n, kb, T, m)[0]
    v_m = np.zeros(n, dtype = np.float32)
    v2_m = np.zeros(n, dtype = np.float32)
    
    for i in range(n):
        v_m[i] = sesamo[i]
        v2_m[i] = sesamo[i] ** 2
        
    integr1 = np.sum(v_m) / n
    integr2 = np.sum(v2_m) / n
    
    err1 = block_average(v_m, s)
    err2 = block_average(v2_m, s)
    
    return  integr1 , integr2, err1, err2 





# stdv of sampled points with block-averages
@njit
def block_average(lst, s):
    
    aver = np.zeros(s, dtype = np.float32)
    block_size = int(len(lst) / s)

    for k in range(s):
        aver[k] = np.mean(lst[(k * block_size):((k + 1) * block_size)])
        
    Sigma_s = np.std(aver)
        
    return Sigma_s / math.sqrt(s)





# Metropolis code for a Boltzmann trial function (1D gas of N non-interacting particles)
@njit
def Metropolis_Boltzmann_N( v0, dvmax, n, kb, T, m, N):
    
    acc = 0
    velocity = np.asarray([[0] * (N) for _ in range(n + 1)], dtype = np.float32)
    velocity[0, :] = v0
    Energy = np.zeros((n + 1), dtype = np.float32)
    Energy[0] = np.sum(velocity[0, :] ** 2)
    
    v_t = np.asarray([v0] * N, dtype = np.float32)
    
    for j in range(1, n):
    
        for i in range(N):
            
            v_star = np.random.uniform(v_t[i] - dvmax, v_t[i] + dvmax)
            
            esp1v = ( -m * v_star ** 2 / ( 2 * kb * T) )  
            esp2v = ( -m * v_t[i] ** 2 / ( 2 * kb * T) )    
            alphav = math.e ** (esp1v - esp2v)           
            
            if alphav >= np.random.rand() :
                v_t[i] = v_star
                acc += 1
                
            velocity[j, i] = v_t[i]
            
        Energy[j] = np.sum(velocity[j, :] ** 2)
            
    return velocity, (m / 2) * Energy, acc / (N * n)





# Evaluates expectation values of typical physical quantities through importance
# sampling + Metropolis (See pag 59 of "Metropolis.pdf")
def Metro_sampl_Boltzmann_N(v0, dvmax, n, kb, T, m, N, s): 
    
    sesamo = Metropolis_Boltzmann_N(v0, dvmax, n, kb, T, m, N)[0].flatten()
    v_m = np.zeros((n*N), dtype = np.float32)
    v2_m = np.zeros((n*N), dtype = np.float32)
    
    for i in range(N*n):
        v_m[i] = sesamo[i]
        v2_m[i] = sesamo[i] ** 2
        
    integr1 = np.sum(v_m) / (n * N)
    integr2 = np.sum(v2_m) / (n * N)
    
    err1 = block_average(v_m, s)
    err2 = block_average(v2_m, s)
    
    return  integr1 , integr2, err1, err2

    
