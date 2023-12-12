"""
2D Random Walks

"""

import numpy as np
import matplotlib.pyplot as plt
import math
from Funz4 import RW2D_average, RW_2D_plot

# -----------------------------
# ---- A) On a square lattice
"""
VonKarajan = RW2D_average(5000, 32, 0, 0, 0.25, 0.25, 0.25, False)

RW_2D_plot(VonKarajan, '2D_RW.gif')
plt.close('all')


# Create an example array with real numbers
original_array = VonKarajan[3]

# Use NumPy's unique function to get unique entries and their indices
unique_entries, indices = np.unique(original_array, return_inverse=True)

# Create a dictionary to store separated arrays
separated_arrays = {}

# Iterate through unique entries and populate the dictionary
for entry in unique_entries:
    separated_arrays[entry] = original_array[np.isclose(original_array, entry)]

# Print the separated arrays
bins = np.asarray([], dtype=np.float32)
columns = np.asarray([], dtype=np.int32)
for entry, array in separated_arrays.items():
    bins = np.append(bins, entry)
    columns = np.append(columns, len(array))
#print(bins, columns)


# Normalized Histogram - distribution of position at the end of the walkers
density = columns / (len(VonKarajan[3])*0.1)

plt.bar(bins, density, width=0.1, alpha=0.5, color='b', label=r'$P_{N}(r)^{num}$')
plt.xlabel('r(N)')
plt.ylabel('Probability Density')
plt.grid()
x = np.linspace(0, bins[-1], 1000)
Rayleigh = lambda x: (2*x/32)* math.e **(-x**2 / 32)
plt.plot(x, Rayleigh(x), label=r'$P_{N}(r)^{theo}$', color='black')
plt.legend()
plt.show()


# Plot - Mean square position over N
t = np.array([i for i in range(1,33)])

plt.scatter(t, VonKarajan[2], color='black')
plt.xlabel('i')
plt.ylabel(r'$\langle (\Delta r_{i})^2 \rangle^{num}$', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
"""
somma = 0
N=8
a=1
b=1
for i in range(a,N-b):
    somma += math.comb(i, int((i+a)/2)) * math.comb(8-i, int((8-i+b)/2))
    
print(somma/ (4**N))

# -----------------------------
# ---- B) With unit random steps (on the unitary circle theta belonging to [0, 2PI[ )

"""
Mozart = RW2D_average(5000, 64, 0, 0, 0, 0, 0, True)

RW_2D_plot(Mozart, '2D_RW_continuous.gif')
plt.close('all')

# Normalized Histogram - distribution of position at the end of the walkers
IQR = np.percentile(Mozart[3], 75) - np.percentile(Mozart[3], 25)
nbins = int((max(Mozart[3]) - min(Mozart[3])) / (2 * IQR * len(Mozart[3])**(-1/3)))

hist, bins = np.histogram(Mozart[3], nbins, density=False)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_widths = np.diff(bins)
density = hist / (len(Mozart[3]) * bin_widths[0])

plt.bar(bins[:-1], density, width=bin_widths, alpha=0.5, color='b', label=r'$P_{N}(r)^{num}$')
plt.xlabel('r(N)')
plt.ylabel('Probability Density')
x = np.linspace(0, bins[-1], 1000)
Rayleigh = lambda x: (2*x/64)* math.e **(-x**2 / 64)
plt.plot(x, Rayleigh(x), label=r'$P_{N}(r)^{theo}$', color='black')
plt.legend()
plt.grid()
plt.show()


# Plot - Mean square position over N
t = np.array([i for i in range(1,65)])

plt.scatter(t, Mozart[2], color='black')
plt.xlabel('i')
plt.ylabel(r'$\langle (\Delta r_{i})^2 \rangle^{num}$', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
"""




# -------------------------------------------------------------------------------
# ----- Easy example of gif animation with python:
# -------------------------------------------------------------------------------

"""
fig = plt.figure()
l, = plt.plot([], [], 'k-')
l2, = plt.plot([], [], 'm--')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Title')

plt.xlim(-5, 5)
plt.ylim(-5, 5)

def func(x):
    return np.sin(x)*3

def func2(x):
    return np.cos(x)*3

metadata = dict(title='Movie', artist='codinglikened')
writer = PillowWriter(fps=5, metadata= metadata)

xlist = []
ylist = []
ylist2 = []

with writer.saving(fig, 'SinCos_wave.gif', 100):
    for xval in np.linspace(-5, 5, 100):
        xlist.append(xval)
        ylist.append(func(xval))
        ylist2.append(func2(xval))
        
        l.set_data(xlist, ylist)
        l2.set_data(xlist, ylist2)
        
        writer.grab_frame()
"""