"""
Now I have to do everything from the beginning again

@author: david
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import time
from Funz7 import Metropolis, gauss_func, plot_histo


sigma = 1
n_step = np.asarray([10**2, 10**3, 10**4, 10**5])

            
start_time1 = time.time()

parrot = plot_histo(n_step, Metropolis, sigma)

end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print(f"Elapsed time1: {elapsed_time1:.4f} seconds")







