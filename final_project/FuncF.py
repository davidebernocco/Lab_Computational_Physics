"""
Library of self-made functions needed for the final project exercises

@author: david
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt



x_min, x_max = np.float32(0.0), np.float32(1.0)



@njit
def tent_map(x, r):
    if x < 0.5:
        return r * x
    else:
        return r * (1 - x)



@njit
def sine_map(x, r):
    return r * np.sin(np.pi * x)



@njit
def logistic_map(x, r):
    return r * x * (1 - x)




@njit
def d_sine_map(x, r):
    return np.pi * r * np.cos(np.pi * x)



@njit
def d_logistic_map(x, r):
    return r * (1 - 2 * x)





@njit
def iteration_tent(r, n0, n):
    trajectory =  np.zeros((n0 + n), dtype = np.float32)
    x0 = np.random.rand()
    trajectory[0] = x0
    x = x0
    
    for i in range(1, n0):
        x = tent_map(x, r)
        trajectory[i] = x
        
    for i in range(n):
        x = tent_map(x, r)
        trajectory[i + n0] = x 
        
    return trajectory



@njit
def iteration_sine(r, n0, n):
    trajectory =  np.zeros((n0 + n), dtype = np.float32)
    x0 = np.random.rand()
    trajectory[0] = x0
    x = x0
    
    for i in range(1, n0):
        x = sine_map(x, r)
        trajectory[i] = x
        
    for i in range(n):
        x = sine_map(x, r)
        trajectory[i + n0] = x 
        
    return trajectory




@njit
def iteration_logistic(r, n0, n):
    trajectory =  np.zeros((n0 + n), dtype = np.float32)
    x0 = np.random.rand()
    trajectory[0] = x0
    x = x0
    
    for i in range(1, n0):
        x = logistic_map(x, r)
        trajectory[i] = x
        
    for i in range(n):
        x = logistic_map(x, r)
        trajectory[i + n0] = x 
        
    return trajectory




@njit
def bifurcation_tent(r, n0, n):
    accum = np.zeros((len(r), n), dtype = np.float32)
    
    
    for i in range(len(r)):
        
        x = np.float32(0.5)
        
        for _ in range(1, n0):
            x = np.float32(tent_map(x, r[i]))

        for k in range(n):
            x = np.float32(tent_map(x, r[i]))
            accum[i][k] = x
            
    return accum



@njit
def bifurcation_sine(r, n0, n):
    accum = np.zeros((len(r), n), dtype = np.float32)
    
    
    for i in range(len(r)):
        
        x = np.float32(0.5)
        
        for _ in range(1, n0):
            x = np.float32(sine_map(x, r[i]))

        for k in range(n):
            x = np.float32(sine_map(x, r[i]))
            accum[i][k] = x
            
    return accum



@njit
def bifurcation_logistic(r, n0, n):
    accum = np.zeros((len(r), n), dtype = np.float32)
    
    
    for i in range(len(r)):
        
        x = np.float32(0.5)
        
        for _ in range(1, n0):
            x = np.float32(logistic_map(x, r[i]))

        for k in range(n):
            x = np.float32(logistic_map(x, r[i]))
            accum[i][k] = x
            
    return accum






def bif_iter(r, n0, n, h, w, Map):
    image = np.zeros((h, w), dtype = np.float32)
    
    for i in range(w):
        
        x = np.float32(np.sqrt(2)/5)
        
        for _ in range(1, n0):
            x = np.float32(Map(x, r[i]))

        for k in range(n):
            x = np.float32(Map(x, r[i]))
            x_idx = int((x - x_min) / (x_max - x_min) * (h - 1))
            image[x_idx][i] += 1
    return image




def non_zero_counting(matrix):
    N, M = matrix.shape
    nzc = np.count_nonzero(matrix, axis=0)
    result_matrix = matrix * nzc
    result_matrix = result_matrix.astype(np.float32)
    return result_matrix





# scatter plot image
def bifurcation_image(r, n0, n, Map):
    width, height = len(r), 1000
    
    # Calculate (r,x) points frequencies
    image = bif_iter(r, n0, n, height, width, Map)
            
    # Normalize frequencies
    image= non_zero_counting(image)
    
    # Cap values above 180,000
    image[image > 1.8*n] = int(1.8*n)

    # Normalize the entire image to 0-255
    max_value = np.max(image)
    image = (image / max_value) * 255
    
    return image


    


# pixel image
def bifurcation_diagram(arr_r, n0, n, Map):
    width, height = len(arr_r), 1000

    # Initialize the image array
    image = np.zeros((height, width), dtype=np.int32)

    for r_idx, r in enumerate(arr_r):
        
        x = np.float32(0.25)
        # Stabilize x
        for _ in range(n0):
            x = np.float32(Map(x, r))

        # Collect 100,000 x-values
        for _ in range(n):
            x = np.float32(Map(x, r))
            if x_min <= x < x_max:
                x_idx = int((x - x_min) / (x_max - x_min) * (height - 1))
                image[x_idx, r_idx] += 1

    # Normalize intensities
    for r_idx in range(width):
        col = image[:, r_idx]
        non_zero_pixels = np.count_nonzero(col)
        if non_zero_pixels > 0:
            image[:, r_idx] = col * non_zero_pixels

    # Cap values above 180,000
    image[image > 1.8*n] = 1.8*n

    # Normalize the entire image to 0-255
    max_value = np.max(image)
    image = (image / max_value) * 255
    image = image.astype(np.uint8)

    return image





@njit
def lyapunov_sine(r, n0, n):
    l = np.zeros(len(r), dtype = np.float32)
    
    
    for i in range(len(r)):
        
        x = np.float32(0.5)
        
        for _ in range(1, n0):
            x = np.float32(sine_map(x, r[i]))

        for k in range(n):
            x = np.float32(sine_map(x, r[i]))
            derivative = d_sine_map(x, r[i])
            logarithm = np.log(np.abs(derivative))
            l[i] += logarithm
            
    return l / n



@njit
def lyapunov_logistic(r, n0, n):
    l = np.zeros(len(r), dtype = np.float32)
    
    
    for i in range(len(r)):
        
        x = np.float32(0.5)
        
        for _ in range(1, n0):
            x = np.float32(logistic_map(x, r[i]))

        for k in range(n):
            x = np.float32(logistic_map(x, r[i]))
            derivative = d_logistic_map(x, r[i])
            logarithm = np.log(np.abs(derivative))
            l[i] += logarithm
            
    return l / n






def normalize_freq(matrix):
    N, M = matrix.shape
    norm = np.sum(matrix, axis=0)
    result_matrix = matrix / norm
    result_matrix = result_matrix.astype(np.float32)
    return result_matrix



@njit
def log_freq(matrix):
    N, M = matrix.shape
    m_log_m = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        for j in range(M):
            m = matrix[i][j]
            if m != 0:
                m_log_m[i][j] = m * np.log(m)
            else:
                m_log_m[i][j] = 0
    
    return m_log_m

    


# entropy of a map
def entropy(r, n0, n, Map):
    width, height = len(r), 1000
    
    # Calculate (r,x) points frequencies
    image = bif_iter(r, n0, n, height, width, Map)
            
    # Normalize frequencies
    p_i = normalize_freq(image)
    
    # Evaluate  -Sum_{i}(p_i * ln(p_i))  for all sampled values of r
    s_i = log_freq(p_i)
    S = - np.sum(s_i, axis=0)
    
    return S, p_i[:, -1]






def entropy_histogram(x, p, bin_width):
    # Create bin edges based on the bin width
    bins = np.arange(0, 1 + bin_width, bin_width)
    
    # Create an array to hold the histogram counts
    histogram = np.zeros(len(bins) - 1, dtype = np.float32)
    
    # Populate the histogram bins with probabilities
    for coord, prob in zip(x, p):
        # Find the appropriate bin for each coordinate
        bin_index = np.searchsorted(bins, coord, side='right') - 1
        if 0 <= bin_index < len(histogram):
            histogram[bin_index] += prob
    
    # Plot the histogram
    fig_histo, ax_histo = plt.subplots(figsize=(6.2, 4.5))
    ax_histo.bar(bins[:-1], histogram, width=bin_width, alpha=0.5, color='b')
    ax_histo.set_xlabel(r'$ x $', fontsize=20)
    ax_histo.set_ylabel(r'$ p_{i} $', fontsize=20)
    ax_histo.grid(True)
    plt.show()




