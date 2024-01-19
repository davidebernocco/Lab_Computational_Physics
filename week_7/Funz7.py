"""
Library of self-made functions needed for the codes implemented for the exercises of the 5th week

@author: david
"""
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numba import njit
import random

    

def proposal_distr(a, b): # Usually "Gaussian" or "uniform" distribution
    
    return np.random.uniform(a, b, 1)



def gauss_func(x, mu, sigma):
    return norm.pdf(x, mu, sigma)


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



def plot_histo(n_arr, func, s, delta):
    
    coefficients = {}
    
    for i in range(len(n_arr)):
        
        x_lst = np.arange(n_arr[i])
        y_lst = func( 0, delta, n_arr[i], s)[0]

        plt.plot(x_lst, y_lst )
        plt.xlabel('i step', fontsize=12)
        plt.ylabel(r'$ x_i $', fontsize=12)
        plt.grid(True)
        plt.show()

         
        IQR = np.percentile(y_lst, 75) - np.percentile(y_lst, 25)
        nbins = int((max(y_lst) - min(y_lst)) / (2 * IQR * len(y_lst)**(-1/3)))

        hist, bins = np.histogram(y_lst, nbins, density=False)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_widths = np.diff(bins)
        density = hist / (n_arr[i] * bin_widths[0])

        plt.bar(bins[:-1], density, width=bin_widths, alpha=0.5, color='b', label=r'$ PDF^{num} $')



        params, covariance = curve_fit(gauss_func, bin_centers, density)
        Mu, Sigma = params
        
        expected_values = gauss_func(bin_centers, Mu, Sigma)
        chi_square = np.sum((density - expected_values) ** 2 / expected_values)

        x_range = np.linspace(min(bin_centers), max(bin_centers), 1000)
        plt.plot(x_range, gauss_func(x_range, Mu, Sigma), label='Gauss fit', color='black')

        plt.xlabel('x', fontsize=12)
        plt.ylabel('Probability density', fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.show() 
        
        coefficients[n_arr[i]] = (params, covariance, chi_square)
    
    print(coefficients)
        
    return coefficients
        
        

    
def acc_ratio(x0, n, s, d_arr):
    
    acc = np.zeros(len(d_arr), dtype = np.float32)
    
    for i in range(len(d_arr)):
        crauti = Metropolis(x0, d_arr[i], n, s)
        acc[i] = crauti[1]
    
    return acc
        


def n_dep(x0, n_arr, s, delta):
    variance = np.zeros(len(n_arr), dtype = np.float32)
    
    for i in range(len(n_arr)):
        wurstel = Metropolis(x0, delta, n_arr[i], s)
        variance[i] = np.var(wurstel[0])
        
    return np.abs(variance - s ** 2)



def equil_time(x0, n_arr, s, delta, N_aver):
    
    aver = 0
    
    for j in range(N_aver):
        h = 1
        i = 0
    
        while h > 0.05:
            wurstel = Metropolis(x0, delta, n_arr[i], s)
            mustard = np.var(wurstel[0])
            h = abs(mustard - s ** 2) / s ** 2
            i += 1
    
        #print("Simulation completed with parameter n =",  n_arr[i], mustard)
        aver += n_arr[i]
        
    return aver / N_aver




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



@njit
def dir_sampl_ground_state(n, s):
    
    chicco = boxmuller(n)
    x = 0
    x2 = 0
    
    for i in range(n):
        a =  math.e ** ( - chicco[i] ** 2 / (2 * s ** 2))
        x += a
        x2 += a ** 2
        
    integr = x / n
    integr2 = x2 / n
    
    delta = (max(chicco) - min(chicco))
    
    return integr* delta, integr2 * delta




def Metro_sampl_ground_state(n, s):
    
    x0 = 0
    d = 5*s
    chicco = Metropolis(x0, d, n, s)[0]
    x = 0
    x2 = 0
    
    for i in range(n):
        a =  math.e ** ( - chicco[i] ** 2 / (2 * s ** 2))
        x += a
        x2 += a ** 2
        
    integr = x / n
    integr2 = x2 / n
    
    delta = (max(chicco) - min(chicco))
    
    return integr* delta, integr2 * delta



def accuracy(s, lst_n, fun):
    
    E_pot_expected = s / 2
    E_kin_expected = 1 / ( 8 * s ** 2)
    E_tot_expected = E_pot_expected + E_kin_expected
    
    Delta1 = np.zeros(len(lst_n), dtype = np.float32)
    Delta2 = np.zeros(len(lst_n), dtype = np.float32)
    Delta3 = np.zeros(len(lst_n), dtype = np.float32)
    Delta4 = np.zeros(len(lst_n), dtype = np.float32)
    
    for i in range(len(lst_n)):
        cachi = fun(lst_n[i], s)
        E_pot_num = cachi[1] / ( 2 * cachi[0] )
        E_kin_num = ( 1 / (4 * s **2)) - cachi[1] / ( 8 * s ** 4 * cachi[0] )
        E_tot_num = E_pot_num + E_kin_num
        
        Delta1[i] = abs(cachi[1] - s ** 2)
        Delta2[i] = abs(E_pot_num - E_pot_expected)
        Delta3[i] = abs(E_kin_num - E_kin_expected)
        Delta4[i] = abs(E_tot_num - E_tot_expected)
        
    return Delta1, Delta2, Delta3, Delta4

