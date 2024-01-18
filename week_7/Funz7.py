"""
Library of self-made functions needed for the codes implemented for the exercises of the 5th week

@author: david
"""
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

    

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



def plot_histo(n_arr, func, s):
    
    coefficients = {}
    
    for i in range(len(n_arr)):
        
        x_lst = np.arange(n_arr[i])
        y_lst = func( 0, 5*s, n_arr[i], s)[0]

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
        
        
        
        