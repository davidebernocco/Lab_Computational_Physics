"""
Library of self-made functions needed for the codes implemented for the exercises of the 5th week

@author: david
"""
import math
import numpy as np

    

def proposal_distr(a, b): # Usually "Gaussian" or "uniform" distribution
    
    return np.random.uniform(a, b, 1)



def Metropolis( x0, delta, n, s):
    
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
            
        points[i] = x_t
            
    return points


