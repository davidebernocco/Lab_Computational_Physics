"""
Now I have to do everything from the beginning again

@author: david
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from numba import jit, njit, int32, float64
import math


#-- ES 1 --
#---------- Linear congruent method and periodicity

@jit(int32[:](int32, int32, int32, int32, int32))
def lin_cong_met(x0, a, c, M, n):
    gen_lst = [x0]
    x = x0
    for i in range(n):
       x = (a*x + c)%M
       gen_lst.append(x)
    return np.asarray(gen_lst, int32)
#If I didn't use numba, it would take >10 times more to evaluate this function!! (proved)


@njit
def lin_cong_met_period(x0, a, c, M, n):
    gen_lst = [x0]
    x = x0
    for i in range(n):
       x = (a*x + c)%M
       if x != x0:
           gen_lst.append(x)
       else:
           return np.asarray(gen_lst, dtype=np.int32), np.int32(len(gen_lst))
    return np.asarray(gen_lst, dtype=np.int32), np.int32(len(gen_lst))
#It seems synthax has to be slightly changed if I want to output an array and an integer



#-- ES 2 --
#---------- Intrinsic generators: uniformity and correlation (qualitative test)

# Could be useful fixing a SEED in the prng!!!

num_rand = 1000
data = np.random.rand(num_rand)

hist, bins = np.histogram(data, bins = round(math.sqrt(num_rand)), density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

def line(x, m, q):
    return m*x + q

params, covariance = curve_fit(line, bin_centers, hist)

plt.hist(data, bins=round(math.sqrt(num_rand)), color='blue', alpha=0.7, density=True, label='Histogram')
plt.plot(bin_centers, line(bin_centers, *params), 'r', label='Linear Fit')

plt.xlabel('Values')
plt.ylabel('Frequency / Probability Density')
plt.title('Histogram with Linear Fit')
plt.legend()
plt.grid(True)
plt.show()


even_entries = data[::2]
odd_entries = data[1::2]

plt.scatter(even_entries, odd_entries, color='blue', marker='o', label='Data from Python PRNG')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Correlation test - Pairs of consecutive numbers')
plt.legend()
plt.grid(True)
plt.show()


#-- ES 3 --
#---------- Intrinsic generators: uniformity and correlation (quantitative test)

#3.1.A) Brute force method: evaluation of k-momentum using several sequences of increasing length

@njit
def momentum_oreder_k(lst, k):
    result = 0
    for x in lst:
        result += x ** k
    Delta = abs(result/len(lst) - (1/(1+k)) )
    return np.float64(result/len(lst)), np.float64(Delta)


@njit
def BruteForce(lst_n, k):
    spaghetti = []
    for i in lst_n:
        bait = np.random.rand(i)
        fish = momentum_oreder_k(bait, k)
        spaghetti.append(fish[1])
    return spaghetti

num_N = [20*i for i in range(1,5000)]
menu_k1 = BruteForce(num_N, 1)
menu_k3 = BruteForce(num_N, 3)
menu_k7 = BruteForce(num_N, 7)


plt.scatter(np.log(num_N), np.log(menu_k1), color='orange', marker='+', label=r'$\Delta_{N}(k=1)$')
plt.scatter(np.log(num_N), np.log(menu_k3), color='olive', marker='o', label=r'$\Delta_{N}(k=3)$')
plt.scatter(np.log(num_N), np.log(menu_k7), color='blue', marker='*', label=r'$\Delta_{N}(k=7)$')
plt.xlabel('log(N)')
plt.ylabel('log(Delta)')
plt.title('Uniformity test - Brute force method')
plt.legend()
plt.grid(True)
plt.show()


#3.1.B) Partial sums method: evaluation of k-moment using a single sequence

Data = np.random.rand(num_N[-1])
population = [i for i in range(1,len(Data)+1)] 

@njit
def PartialSums(lst, k):
    dinner = []
    claw = 0
    for i in range(len(lst)):
        claw = claw + lst[i]**k
        dinner.append(abs(claw/(i+1) - (1/(1+k)) ))
    return dinner
 
plt.scatter(np.log(population), np.log(PartialSums(Data, 1)), color='magenta', marker='+', label=r'$\Delta_{N}(k=1)$')
plt.scatter(np.log(population), np.log(PartialSums(Data, 3)), color='green', marker='o', label=r'$\Delta_{N}(k=3)$')
plt.scatter(np.log(population), np.log(PartialSums(Data, 7)), color='cyan', marker='*', label=r'$\Delta_{N}(k=7)$')
plt.xlabel('log(N)')
plt.ylabel('log(Delta)')
plt.title('Uniformity test - Partial sums')
plt.legend()
plt.grid(True)
plt.show()


#3.2 Correlation test using directly the Partial sums method

@njit
def Correlation_PS(lst, k):
    population = [i for i in range(1+k,len(Data)+1)] 
    dinner = []
    claw = 0
    for i in range(len(lst)-k):
        claw = claw + lst[i]*lst[i+k]
        dinner.append(abs(claw/(i+1) - 1/4 ))
    return population, dinner, claw/(i+1)

plt.scatter(np.log(Correlation_PS(Data, 1)[0]), np.log(Correlation_PS(Data, 1)[1]), color='maroon', marker='+', label=r'$\Delta_{N}(k=1)$')
plt.scatter(np.log(Correlation_PS(Data, 3)[0]), np.log(Correlation_PS(Data, 3)[1]), color='lime', marker='o', label=r'$\Delta_{N}(k=3)$')
plt.scatter(np.log(Correlation_PS(Data, 7)[0]), np.log(Correlation_PS(Data, 7)[1]), color='navy', marker='*', label=r'$\Delta_{N}(k=7)$')
plt.xlabel('log(N)')
plt.ylabel('log(Delta)')
plt.title('Correlation test - Partial sums')
plt.legend()
plt.grid(True)
plt.show()


