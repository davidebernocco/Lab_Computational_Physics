import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from numba import njit
import math




@njit
def RW2D_average(N_w, N, x0, y0, Pl, Pr, Pd):  # 2D analogous of the "RW1D_average" function in Funz4
    
    position_x = np.full((N_w, N + 1), x0, dtype=np.float32)
    position_y = np.full((N_w, N + 1), x0, dtype=np.float32)
    cumul_x = np.zeros(N, dtype=np.float32)
    cumul_y = np.zeros(N, dtype=np.float32)
    cumul_x2 = np.zeros(N, dtype=np.float32)
    cumul_y2 = np.zeros(N, dtype=np.float32)
    P_N = np.empty(N_w, dtype=np.float64)

    for j in range(N_w):
        
        xi = x0
        yi = y0
        a = np.random.uniform(0, 1, N)
        lst_l = np.random.uniform(0, 1, N) # Line to be deleted if we want to work on a descrete lattice
        
        for i in range(N):
            
            if a[i] <= Pl:
                xi -= lst_l[i] # If I want the walkers to move on a descrete lattice of dimension 1, here write "-= 1" instead
                position_x[j, i + 1] = xi
                position_y[j, i + 1] = yi
                cumul_x[i] += xi
                cumul_x2[i] += xi**2
            elif a[i] > Pl and a[i] <= (Pl + Pr):
                xi += lst_l[i] # If I want the walkers to move on a descrete lattice of dimension 1, here write "+= 1" instead
                position_x[j, i + 1] = xi
                position_y[j, i + 1] = yi
                cumul_x[i] += xi
                cumul_x2[i] += xi**2
            elif a[i] > (Pl + Pr) and a[i] <= (Pl + Pr + Pd):
                yi -= lst_l[i] # If I want the walkers to move on a descrete lattice of dimension 1, here write "-= 1" instead
                position_y[j, i + 1] = yi
                position_x[j, i + 1] = xi
                cumul_y[i] += yi
                cumul_y2[i] += yi**2
            else:
                yi += lst_l[i]  # If I want the walkers to move on a descrete lattice of dimension 1, here write "+= 1" instead
                position_y[j, i + 1] = yi
                position_x[j, i + 1] = xi
                cumul_y[i] += yi
                cumul_y2[i] += yi**2  
    
        P_N[j] = math.sqrt(xi**2 + yi**2)                        
        
    average_x = cumul_x / N_w
    average_y = cumul_y / N_w
    average_x2 = cumul_x2 / N_w
    average_y2 = cumul_y2 / N_w

    return position_x, position_y, average_x, average_x2, average_x2 + average_y2 - average_x**2 -  average_y**2 , P_N, a, lst_l





def RW_2D_plot(Opera):  #Plotting the gif of all the walkers justapposed

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
    
    with writer.saving(fig, '2D_RW.gif', 100):
        for k in range(len(Opera[2])+1):
            for i in range(len(Opera[0])):
                xlist[i].append(Opera[0][i][k])
                ylist[i].append(Opera[1][i][k])
              
                l[i].set_data(xlist[i], ylist[i])
               
            writer.grab_frame()
            
    return
     
           

VonKarajan = RW2D_average(100, 64, 0, 0, 0.25, 0.25, 0.25)
RW_2D_plot(VonKarajan)




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