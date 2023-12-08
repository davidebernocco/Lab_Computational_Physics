"""
Now I have to do everything from the beginning again

@author: david
"""

import math 
from numba import jit, njit
import numpy as np
from Funz5 import int_trap, int_Simpson, int_sample_mean, int_importance_sampl, int_acc_rejec, average_of_averages
import matplotlib.pyplot as plt
import time


#-- ES 1 --
#---------- Equispaced points: comparison between "trapezoidal" and "Simpson" method
"""
exact = math.e - 1

n_intervals = np.asarray([2**j for j in range(1, 11)], dtype = np.int32)

trapezoidal = int_trap(n_intervals, exact)
Simpson = int_Simpson(n_intervals, exact)


plt.scatter(np.log(n_intervals), np.log(trapezoidal[1]), label='Trapezoidal method')
plt.xlabel('log(n)')
plt.ylabel(r'$ \log(\Delta_{n}) $', fontsize=12)
plt.legend()
plt.grid(True)

plt.scatter(np.log(n_intervals), np.log(Simpson[1]), label='Simpson method')
plt.xlabel('log(n)')
plt.ylabel(r'$ \log(\Delta_{n}) $', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
"""



#-- ES 2 --
#---------- Monte Carlo Method: "generic sample mean" and "importance sampling"
"""
n_intervals = np.asarray([2**j for j in range(6, 21)], dtype = np.int32)


# Sample mean function call
start_time1 = time.time()

SampleMean = int_sample_mean(n_intervals, math.e ** (-x[i] ** 2), False, 0)

end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1


# Importance sampling function call
start_time2 = time.time()

ImportanceSampling = int_importance_sampl(n_intervals)

end_time2 = time.time()
elapsed_time2 = end_time2 - start_time2


print(f"CPU time 'Sample mean': {elapsed_time1:.4f} seconds")
print(f"CPU time 'Importance sampling': {elapsed_time2:.4f} seconds")
# The total elapsed time for the 'importance samplig' is greater. But
#If we looked at the time necessary for the 'sample mean' algorithm to obtain
#the same magnitude of actual error compared to the second algorithm, it would take a lot more time!


plt.scatter(np.log(n_intervals), np.log(SampleMean[2]), label='Sample mean')
plt.xlabel('log(n)', fontsize=12)
plt.ylabel(r'$ \log(\sigma_{n} / \sqrt{n}) $', fontsize=12)
plt.legend()
plt.grid(True)

plt.scatter(np.log(n_intervals), np.log(ImportanceSampling[2]), label='Importance sampling')
plt.xlabel('log(n)', fontsize=12)
plt.ylabel(r'$ \log(\sigma_{n} / \sqrt{n}) $', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
"""


#-- ES 3 --
#---------- Monte Carlo method: acceptance-rejection

"""
n_list = np.asarray( [10 ** k for k in range(2, 7)], dtype=np.int32 )

AcceptanceRejection = int_acc_rejec(n_list, 10**3)


plt.scatter(np.log(n_list), np.log(AcceptanceRejection[1]), label='Acceptance-rejection')
plt.xlabel('log(n)', fontsize=12)
plt.ylabel(r'$\langle \log(\Delta_{n}) \rangle$', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
"""



#-- ES 4 --
#---------- Monte Carlo method: error analysis using 'average of the averages' and 'block average'
# Here we will take as a referment the sample mean algorithm 


# ---- 4.1) and 4.2) Sample mean: Actual error, sigma_n and sigma_n / radq(n)


@njit
def f_quarterPi(x):
    return np.sqrt(1 - x**2)
"""
n_arr = np.asarray([10**2, 10**3, 10**4], dtype=np.int32)
                   
SM = int_sample_mean(n_arr, f_quarterPi, True, math.pi / 4)


plt.scatter(np.log(n_arr), np.log(SM[3]), label='SM, actual error')
plt.scatter(np.log(n_arr), np.log(SM[2]), label = r'$ SM, \sigma_{n} / \sqrt{n} $')
plt.scatter(np.log(n_arr), np.log(SM[1]), label = r'$ SM, \sigma_{n}  $')
plt.xlabel('log(n)', fontsize=12)
plt.ylabel(r'$ \log(error) $', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
"""

# ---- 4.3) AVERAGE OF THE AVERAGES 
"""
AverageOfAverages = average_of_averages(10**4, 10, f_quarterPi)

print("The error associated to each of the m=10 runs (of lenght 10000) is well estimated by the average of the averages:", AverageOfAverages[1])
"""


# ---- 4.4) BLOCK AVERAGE

def block_average(num, s, func):
    
    aver = np.zeros(s, dtype = np.float32)
    aver_2 = np.zeros(s, dtype = np.float32)
    block_size = int(num / s)
    
    x = np.random.uniform(0, 1, num)
    
    for k in range(s):
        
        for i in range(block_size):
            
            aver[k] += func(x[k*block_size + i])  
            aver_2[k] += func(x[k*block_size + i])**2 
        
        aver[k] /= block_size
        aver_2[k] /= block_size
        
    Sigma_s = math.sqrt( np.mean(aver_2) - (np.mean(aver))**2 )        
        
    return aver, Sigma_s / math.sqrt(s) 


BlockAverage = block_average(10**4, 10, f_quarterPi)

print("The error over the average of the s=10 sub-block averages (of equal lenght) built from a unique run of 10000 points is:", BlockAverage[1])





