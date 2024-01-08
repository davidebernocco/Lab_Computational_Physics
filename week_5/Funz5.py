"""
Library of self-made functions needed for the codes implemented for the exercises of the 5th week

@author: david
"""

from numba import njit, jit
import numpy as np
import math 




@njit
def int_trap(num, I):
    
    results = np.zeros(len(num), dtype = np.float32)
    Delta = np.zeros(len(num), dtype = np.float32)
    
    for j in range(len(num)):
        xi = np.linspace(0, 1, num[j] + 1)
        h = xi[1] - xi[0]
        somma = ( (math.e ** xi[0]) + (math.e ** xi[-1]) ) * ( 1 / 2 )
        
        for i in range(1, len(xi) - 1):
            somma += (math.e ** xi[i])
        
        results[j] = h * somma
        Delta[j] = abs(h * somma - I)
    
    return results, Delta




@njit
def int_Simpson(num, I): # N.B. Unlike the "trpazoidal method" here the number of sub intervals must be even (2^n is ok)!
    
    results = np.zeros(len(num), dtype = np.float32)
    Delta = np.zeros(len(num), dtype = np.float32)
    
    for j in range(len(num)):
        xi = np.linspace(0, 1, num[j] + 1)
        h = xi[1] - xi[0]
        somma = 0
        
        for i in range(1, num[j]/2 + 1):
            somma += math.e ** xi[2*i - 2] + 4*math.e ** xi[2*i - 1] + math.e ** xi[2*i]
        
        results[j] = (h/3) * somma
        Delta[j] = abs((h/3) * somma - I)
    
    return results, Delta





@njit
def funz_exp(x):
    
    return math.e ** (-x ** 2) 




@njit
def int_sample_mean(num, func, condition, I):
    
    results = np.zeros(len(num), dtype = np.float32)
    results_2 = np.zeros(len(num), dtype = np.float32)
    Sigma_n = np.zeros(len(num), dtype = np.float32)
    Sigma_n_su_radq_n = np.zeros(len(num), dtype = np.float32)
    Actual_err = np.zeros(len(num), dtype = np.float32)
    
    for j in range(len(num)):
        x = np.random.uniform(0, 1, num[j])
        somma = 0
        somma_2 = 0
    
        for i in range(len(x)):
            somma += func(x[i])  #(math.e ** (-x[i] ** 2) )
            somma_2 += func(x[i])**2 #(math.e ** (-x[i] ** 2) ) ** 2
        
        results[j] = somma/num[j]
        results_2[j] = somma_2/num[j]
        Sigma_n[j] = math.sqrt(somma_2/num[j] - (somma/num[j])**2)
        Sigma_n_su_radq_n[j] = Sigma_n[j] / math.sqrt(num[j])
        
    
        if condition:
            Actual_err[j] = np.abs( results[j] - I )
        
    return results, Sigma_n, Sigma_n_su_radq_n, Actual_err




@njit
def int_importance_sampl(num):
    
    results = np.zeros(len(num), dtype = np.float32)
    results_2 = np.zeros(len(num), dtype = np.float32)
    Sigma_n = np.zeros(len(num), dtype = np.float32)
    Actual_err = np.zeros(len(num), dtype = np.float32)
    
    for j in range(len(num)):
        data = np.random.uniform(1 / (math.e - 1), math.e / (math.e - 1), num[j])
        x = -np.log( ((math.e - 1) / math.e) * data )
        somma = 0
        somma_2 = 0
    
        for i in range(len(x)):
            somma += (math.e ** (-x[i] ** 2) ) / ( (math.e/(math.e - 1)) * (math.e ** (-x[i])))
            somma_2 += (math.e ** (-x[i] ** 2) / ( (math.e/(math.e - 1)) * (math.e ** (-x[i])))) ** 2
        
        results[j] = somma/num[j]
        results_2[j] = somma_2/num[j]
        Sigma_n[j] = math.sqrt(somma_2/num[j] - (somma/num[j])**2)
        Actual_err[j] = Sigma_n[j]/math.sqrt(num[j])
        
    return results, Sigma_n, Actual_err




@njit
def f_quarterPi(x):
    
    return np.sqrt(1 - x**2)




@jit
def line(x, m, q):
    
    return m*x + q




@njit
def int_acc_rejec(num, N_rep):
    
    results =np.full((N_rep, len(num)), 0, dtype = np.float32)
    cumul_res = np.zeros(len(num), dtype = np.float32)
    
    I = math.pi
    for k in range(N_rep):
        
        for j in range(len(num)):
            points = np.random.rand(2*num[j])
            n_acc = 0
            
            for i in range( int(len(points)/2) ):
                if points[2*i + 1] < math.sqrt(1 - points[2*i]**2):
                    n_acc += 1
                    
            results[k, j] = 4 * (n_acc / num[j])
            cumul_res[j] += 4 * (n_acc / num[j])
            
        delta_average = np.abs(cumul_res/N_rep - I)
    
    return results, delta_average




def average_of_averages(num, m, func):
    
    aver = 0
    aver_2 = 0
    
    for j in range(m):
        x = np.random.uniform(0, 1, num)
        somma = 0
    
        for i in range(len(x)):
            somma += func(x[i])  
        
        aver += 4 * ( somma / num )
        aver_2 += ( 4 * ( somma / num ) ) ** 2
        
    Sigma_m = math.sqrt( (aver_2 / m) - ( aver / m )**2 )        
        
    return aver, Sigma_m




def block_average(num, s, func):
    
    aver = np.zeros(s, dtype = np.float32)
    aver_2 = np.zeros(s, dtype = np.float32)
    cumul = np.zeros(s, dtype = np.float32)
    block_size = int(num / s)
    
    x = np.random.uniform(0, 1, num)
    
    for k in range(s):
        
        for i in range(block_size):
            
            cumul[k] += 4* func(x[k*block_size + i])  
        
    aver = cumul / block_size
    aver_2 = (cumul / block_size) ** 2
        
    Sigma_s = math.sqrt( np.mean(aver_2) - (np.mean(aver))**2 )      
        
    return aver, Sigma_s / math.sqrt(s)