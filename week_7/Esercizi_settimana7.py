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
from Funz7 import Metropolis
from scipy.stats import norm


sigma = 1
n_step = 10000

            
start_time1 = time.time()

x_lst = np.arange(n_step)
y_lst = Metropolis( 0, 2*sigma, n_step, sigma)


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
density = hist / (n_step * bin_widths[0])

plt.bar(bins[:-1], density, width=bin_widths, alpha=0.5, color='b', label=r'$ PDF^{num} $')

x = np.linspace(min(y_lst), max(y_lst), 1000)
y = norm.pdf(x, 0, sigma)
plt.plot(x, y, label=r'$PDF^{theo}$', color='black')

plt.xlabel('x', fontsize=12)
plt.ylabel('Probability density', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()  

end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print(f"Elapsed time1: {elapsed_time1:.4f} seconds")







