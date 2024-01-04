"""
2D Random Walks

"""

import numpy as np
import matplotlib.pyplot as plt
import math
from Funz4 import RW2D_average, RW_2D_plot, line
from lattice_PN import Prob_distr_lattice
from scipy.optimize import curve_fit


# -----------------------------
# ---- A) On a square lattice


VonKarajan = RW2D_average(1000, 64, 0, 0, 0.25, 0.25, 0.25, False)


"""
RW_2D_plot(VonKarajan, '2D_RW.gif')
plt.close('all')
"""


original_array = VonKarajan[3]  # Define an array

unique_entries, indices = np.unique(original_array, return_inverse=True)  # Use NumPy's unique function to get unique entries and their indices

separated_arrays = {}  # Create a dictionary to store separated arrays

for entry in unique_entries:   # Iterate through unique entries and populate the dictionary
    separated_arrays[entry] = original_array[np.isclose(original_array, entry)]

Bins = np.asarray([], dtype=np.float32)
columns = np.asarray([], dtype=np.int32)
for entry, array in separated_arrays.items():   # Print the separated arrays
    Bins = np.append(Bins, entry)
    columns = np.append(columns, len(array))


# Normalized distribution of position at the end of the walkers
density = columns / (len(VonKarajan[3]))
Clessidra = Prob_distr_lattice(64)

plt.scatter(Bins, density, color='magenta', label=r'$P_{N}(r)^{theo}$', marker='^', s=50)
plt.xlabel('r(N)')
plt.ylabel('Probability Density')
plt.grid()

plt.scatter(Clessidra[0], Clessidra[1], label=r'$P_{N}(r)^{theo}$', color='blue', marker='o', s=50)
plt.legend()
plt.show()



#Histogram for the square lattice case
positions = Bins
column_heights = columns/10000

bin_edges = np.arange(0, max(positions) + 2, 1)
hist, _ = np.histogram(positions, bins=bin_edges, weights=column_heights)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  

plt.bar(bin_centers, hist, width=1, edgecolor='black')
plt.xlabel('r(N)')
plt.ylabel('Probability distribution')

x = np.linspace(0, Bins[-1], 1000)
Rayleigh = lambda x: (2*x/32)* math.e **(-x**2 / 32)
plt.plot(x, Rayleigh(x), label=r'$P_{N}(r)^{theo}$', color='black')

plt.show()




# Plot - Mean square position over N
t = np.array([i for i in range(1,65)])
params, covariance = curve_fit(line,np.log(t),np.log(VonKarajan[2]))

plt.scatter(np.log(t),np.log(VonKarajan[2]), label='Square lattice', color='black', marker="s")
plt.plot(np.log(t), line(np.log(t), *params), color='blue', label='Linear Fit')
plt.xlabel('ln(i)', fontsize=12)
plt.ylabel(r'ln($\langle (\Delta r_{i})^2 \rangle^{num}$)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()





# -----------------------------
# ---- B) With unit random steps (on the unitary circle theta belonging to [0, 2PI[ )

"""
Mozart = RW2D_average(160, 64, 0, 0, 0, 0, 0, True)


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
params2, covariance2 = curve_fit(line, np.log(t),np.log(Mozart[2]))

plt.scatter(np.log(t),np.log(Mozart[2]), label=r'direction in $[0, 2\pi[ $', color='black', marker="o")
plt.plot(np.log(t), line(np.log(t), *params2), color='red', label='Linear Fit')
plt.xlabel('ln(i)', fontsize=12)
plt.ylabel(r'ln($\langle (\Delta r_{i})^2 \rangle^{num})$', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
"""



#Square lattice wirh relevant points
"""
x_values = [0,2,4,6,8,1,3,5,7,2,4,6,3,5,4]
y_values = [0,0,0,0,0,1,1,1,1,2,2,2,3,3,4]
labels = ['{0,2,4,6,8}', '{2,4,6,8}', '{4,6,8}', '{6,8}', '{8}', '{1,3,5,7}', '{3,5,7}', '{5,7}', '{7}', '{2,4,6}', '{4,6}', '{6}', '{3,5}', '{5}', '{4}']

# Create a scatter plot
plt.scatter(x_values, y_values, s=150)

# Add labels to each point
for i, label in enumerate(labels):
    plt.text(x_values[i], y_values[i], label, fontsize=30, ha='left', va='bottom')

# Add labels, title, etc.
plt.xlabel('x', fontsize=25)
plt.ylabel('y', fontsize=25)
plt.yticks(np.arange(0, 5, 1), fontsize=30)
plt.xticks(fontsize=30)
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