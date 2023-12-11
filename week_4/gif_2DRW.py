"""
2D Random Walks

"""

import numpy as np
import matplotlib.pyplot as plt
import math
from Funz4 import RW2D_average, RW_2D_plot

# -----------------------------
# ---- A) On a square lattice

VonKarajan = RW2D_average(10000, 64, 0, 0, 0.25, 0.25, 0.25, False)

RW_2D_plot(VonKarajan, '2D_RW.gif')
plt.close('all')


# Normalized Histogram - distribution of position at the end of the walkers
IQR = np.percentile(VonKarajan[3], 75) - np.percentile(VonKarajan[3], 25)
nbins = int((max(VonKarajan[3]) - min(VonKarajan[3])) / (2 * IQR * len(VonKarajan[3])**(-1/3)))

hist, bins = np.histogram(VonKarajan[3], nbins, density=False)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_widths = np.diff(bins)
density = hist / (len(VonKarajan[3]) * bin_widths[0])

plt.bar(bins[:-1], density, width=bin_widths, alpha=0.5, color='b', label=r'$P_{N}(r)^{num}$')
plt.xlabel('r(N)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid()
plt.show()


# Plot - Mean square position over N
t = np.array([i for i in range(1,65)])

plt.scatter(t, VonKarajan[2], color='black')
plt.xlabel('i')
plt.ylabel(r'$\langle (\Delta r_{i})^2 \rangle^{num}$', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()



# -----------------------------
# ---- B) With unit random steps (on the unitary circle theta belonging to [0, 2PI[ )

Mozart = RW2D_average(10000, 64, 0, 0, 0, 0, 0, True)

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