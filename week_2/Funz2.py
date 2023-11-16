"""
Now I have to do everything from the beginning again

@author: david
"""
from numba import jit, njit, int32, config
import numpy as np


# This function is not called anywhere, but it's here to provide a first simple example of numba usage
@jit(int32[:](int32, int32, int32, int32, int32))
def lin_cong_met(X0, A, C, m, N):
    gen_lst = [X0]
    x = X0
    for i in range(N):
       x = (A*x + C) % m
       gen_lst.append(x)
    return np.asarray(gen_lst, int32)
#If I didn't use numba, it would take >10 times more to evaluate this function!! (proved)



@njit
def lin_cong_met_period(X0, A, C, m, N):
    gen_lst = [X0]
    x = X0
    for i in range(N):
       x = (A*x + C) % m
       if x != X0:
           gen_lst.append(x)
       else:
           return np.asarray(gen_lst, dtype=np.int32), np.int32(len(gen_lst))
    return np.asarray(gen_lst, dtype=np.int32), np.int32(len(gen_lst))
#It seems synthax has to be slightly modified if I want to output an array and an integer: "njit" necessary



def line(x, m, q):
    return m*x + q



@njit
def momentum_order_k(lst, k):
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
        fish = momentum_order_k(bait, k)
        spaghetti.append(fish[1])
    return spaghetti



