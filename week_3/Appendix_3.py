"""
Algorithms (less efficient) 3.1 and 3.2 for the boxmuller exercise

@author: david
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import random
import time
from numba import jit, float64, int32
from Funz3 import R


# Funvtion that outputs two list of gaussian distributed variables
@jit
def R(u,v,n):
    
    x_vet = []
    y_vet = []
    
    for i in range(n):
        if u[i]**2 + v[i]**2 <= 1 :
            r2 = u[i]**2 + v[i]**2
            r2 = math.sqrt(-2* math.log(r2) / r2)
            x_vet.append(r2* u[i])
            y_vet.append(r2* v[i])
            
    return np.asarray(x_vet, dtype=np.float64), np.asarray(y_vet, dtype=np.float64)





# The optimized function for the case 3.1; not used in the main code. Showed here merely for didactic scope
@jit(float64[:](int32))
def boxmuller_trig(ceci):
    sacco = []
    
    for i in range(ceci):
        gaus_stored = False
        g = 0.0
        
        if gaus_stored:
            rnd = g
            gaus_stored = False
        else:
            X = random.random() 
            Y = random.random()
            x = math.sqrt(-2 * math.log(X)) * math.cos(2 * math.pi * Y)
            y = math.sqrt(-2 * math.log(X)) * math.sin(2 * math.pi * Y)
            rnd = x
            g = y
            gaus_stored = True
            
        sacco.append(rnd)   
        
    return np.asarray(sacco, float64)





num_rand = 10**6
#3.1) box Muller with trigonometric functions

start_time3 = time.time()
X = np.random.rand(num_rand)
Y = np.random.rand(num_rand)
x  = np.sqrt(-2*np.log(X))*np.cos(2*math.pi*Y)
y = np.sqrt(-2*np.log(X))*np.sin(2*math.pi*Y)

IQR = np.percentile(x, 75) - np.percentile(x, 25)
bins = int((max(x) - min(x)) / (2 * IQR * len(x)**(-1/3)))

hist, bins = np.histogram(x, bins, density=False)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_widths = np.diff(bins)
density = hist / sum(hist)

plt.bar(bins[:-1], density, width=bin_widths, alpha=0.5, color='b', label=r'$ p(x) = \frac{1}{\sqrt{2\pi \sigma}} e^{-\frac{(x-\mu)^{2}}{2\sigma^{2}}} $')
plt.xlabel('Counts x')
plt.ylabel('Probability Density')
plt.title('Normalized Histogram - gaussian distributed variable (box Muller)')
plt.legend()
plt.grid()
plt.show()

print(np.mean(x), np.std(x))

IQR2 = np.percentile(y, 75) - np.percentile(y, 25)
bins2 = int((max(y) - min(y)) / (2 * IQR2 * len(y)**(-1/3)))

hist2, bins2 = np.histogram(y, bins2, density=False)
bin_centers2 = (bins2[:-1] + bins2[1:]) / 2
bin_widths2 = np.diff(bins2)
density2 = hist2 / sum(hist2)

plt.bar(bins2[:-1], density2, width=bin_widths2, alpha=0.5, color='b', label=r'$ p(y) = \frac{1}{\sqrt{2\pi \sigma}} e^{-\frac{(y-\mu)^{2}}{2\sigma^{2}}} $')
plt.xlabel('Counts y')
plt.ylabel('Probability Density')
plt.title('Normalized Histogram - gaussian distributed variable (box Muller)')
plt.legend()
plt.grid()
plt.show()

print(np.mean(y), np.std(y))

end_time3 = time.time()
elapsed_time3 = end_time3 - start_time3
print(f"Elapsed time3: {elapsed_time3:.4f} seconds")


#3.2) box Muller without trigonometric functions

start_time4 = time.time()
X1 = np.random.uniform(-1, 1, num_rand)
Y1 = np.random.uniform(-1, 1, num_rand)

IQR3 = np.percentile(R(X1,Y1,num_rand)[0], 75) - np.percentile(R(X1,Y1,num_rand)[0], 25)
bins3 = int((max(R(X1,Y1,num_rand)[0]) - min(R(X1,Y1,num_rand)[0])) / (2 * IQR3 * len(R(X1,Y1,num_rand)[0])**(-1/3)))

hist3, bins3 = np.histogram(R(X1,Y1,num_rand)[0], bins3, density=False)
bin_centers3 = (bins3[:-1] + bins3[1:]) / 2
bin_widths3 = np.diff(bins3)
density3 = hist3 / sum(hist3)

plt.bar(bins3[:-1], density3, width=bin_widths3, alpha=0.5, color='b', label=r'$ p(x) = \frac{1}{\sqrt{2\pi \sigma}} e^{-\frac{(x-\mu)^{2}}{2\sigma^{2}}} $')
plt.xlabel('Counts x')
plt.ylabel('Probability Density')
plt.title('Normalized Histogram - gaussian distributed variable (box Muller mod.)')
plt.legend()
plt.grid()
plt.show()

print(np.mean(R(X1,Y1,num_rand)[0]), np.std(R(X1,Y1,num_rand)[0]))

IQR4 = np.percentile(R(X1,Y1,num_rand)[1], 75) - np.percentile(R(X1,Y1,num_rand)[1], 25)
bins4 = int((max(R(X1,Y1,num_rand)[1]) - min(R(X1,Y1,num_rand)[1])) / (2 * IQR4 * len(R(X1,Y1,num_rand)[1])**(-1/3)))

hist4, bins4 = np.histogram(R(X1,Y1,num_rand)[1], bins4, density=False)
bin_centers4 = (bins4[:-1] + bins4[1:]) / 2
bin_widths4 = np.diff(bins4)
density4 = hist4 / sum(hist4)

plt.bar(bins4[:-1], density4, width=bin_widths4, alpha=0.5, color='b', label=r'$ p(y) = \frac{1}{\sqrt{4\pi \sigma}} e^{-\frac{(y-\mu)^{4}}{4\sigma^{4}}} $')
plt.xlabel('Counts y')
plt.ylabel('Probability Density')
plt.title('Normalized Histogram - gaussian distributed variable (box Muller mod.)')
plt.legend()
plt.grid()
plt.show()

print(np.mean(R(X1,Y1,num_rand)[1]), np.std(R(X1,Y1,num_rand)[1]))

end_time4 = time.time()
elapsed_time4 = end_time4 - start_time4
print(f"Elapsed time4: {elapsed_time4:.4f} seconds")