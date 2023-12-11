import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from numba import njit
import math




@njit
def RW2D_average(N_w, N, x0, y0, Pl, Pr, Pd, Theta):  # 2D analogous of the "RW1D_average" function in Funz4
    
    position_x = np.full((N_w, N + 1), x0, dtype=np.float32)
    position_y = np.full((N_w, N + 1), x0, dtype=np.float32)
    cumul_x = np.zeros(N, dtype=np.float32)
    cumul_y = np.zeros(N, dtype=np.float32)
    cumul_x2 = np.zeros(N, dtype=np.float32)
    cumul_y2 = np.zeros(N, dtype=np.float32)
    P_N = np.empty(N_w, dtype=np.float64)
    
    if not Theta:
        
        for j in range(N_w):
            
            xi = x0
            yi = y0
            a = np.random.uniform(0, 1, N)
            #lst_l = np.random.uniform(0, 1, N)   # If we want to work on a lattice of random spacing l for each step
            lst_l = np.asarray([1 for _ in range(N)], dtype=np.int32)   # If we want to work on a descrete lattice of spacing l = 1
            
            for i in range(N):
                
                if a[i] <= Pl:
                    xi -= lst_l[i] 
                    position_x[j, i + 1] = xi
                    position_y[j, i + 1] = yi
                    cumul_x[i] += xi
                    cumul_x2[i] += xi**2
                elif a[i] > Pl and a[i] <= (Pl + Pr):
                    xi += lst_l[i]
                    position_x[j, i + 1] = xi
                    position_y[j, i + 1] = yi
                    cumul_x[i] += xi
                    cumul_x2[i] += xi**2
                elif a[i] > (Pl + Pr) and a[i] <= (Pl + Pr + Pd):
                    yi -= lst_l[i] 
                    position_y[j, i + 1] = yi
                    position_x[j, i + 1] = xi
                    cumul_y[i] += yi
                    cumul_y2[i] += yi**2
                else:
                    yi += lst_l[i]  
                    position_y[j, i + 1] = yi
                    position_x[j, i + 1] = xi
                    cumul_y[i] += yi
                    cumul_y2[i] += yi**2  
        
            P_N[j] = math.sqrt(xi**2 + yi**2)   
                     
    else:    # For 2D random unit steps (on the unitary circle)
        
        for j in range(N_w):
            
            xi = x0
            yi = y0
            th = np.random.uniform(0, 2*math.pi, N)
            
            for i in range(N):
                xi += math.cos(th[i])
                yi += math.sin(th[i])
                position_x[j, i + 1] = xi
                position_y[j, i + 1] = yi
                cumul_x[i] += xi
                cumul_x2[i] += xi**2
                cumul_y[i] += yi
                cumul_y2[i] += yi**2  
            
            P_N[j] = math.sqrt(xi**2 + yi**2) 
            
        
    average_x = cumul_x / N_w
    average_y = cumul_y / N_w
    average_x2 = cumul_x2 / N_w
    average_y2 = cumul_y2 / N_w

    return position_x, position_y, average_x2 + average_y2 - average_x**2 -  average_y**2 , P_N




def RW_2D_plot(Opera, plot_name):  #Plotting the gif of all the walkers justapposed

    fig = plt.figure()
    
    l = [plt.plot([], [])[0] for _ in range(len(Opera[0]))]
    xlist = [[] for _ in range(len(Opera[0]))]
    ylist = [[] for _ in range(len(Opera[0]))]
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D RW')
    plt.xlim(np.min(Opera[0]) - 1, np.max(Opera[0]) + 1)
    plt.ylim(np.min(Opera[1]) - 1, np.max(Opera[1]) + 1)
    plt.grid(True) 
    plt.gca().set_aspect('equal', adjustable='box')
    
    metadata = dict(title='Movie', artist='codinglikened')
    writer = PillowWriter(fps=1, metadata= metadata)
    
    with writer.saving(fig, plot_name, 100):
        for k in range(len(Opera[2])+1):
            for i in range(len(Opera[0])):
                xlist[i].append(Opera[0][i][k])
                ylist[i].append(Opera[1][i][k])
              
                l[i].set_data(xlist[i], ylist[i])
               
            writer.grab_frame()
            
    return
     
           

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
