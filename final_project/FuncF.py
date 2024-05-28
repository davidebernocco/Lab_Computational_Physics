"""
Library of self-made functions needed for the final project exercises

@author: david
"""

import numpy as np
from numba import njit


@njit
def tent_map(x, r):
    if x < 1/2:
        x = r*x   
    else:
        x = r - r*x
    
    return x



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
def bifurcation(r, n0, n):
    accum = np.zeros((len(r), n), dtype = np.float32)
    
    
    for i in range(len(r)):
        
        x = np.float32(0.5)
        
        for _ in range(1, n0):
            x = np.float32(tent_map(x, r[i]))

        for k in range(n):
            x = np.float32(tent_map(x, r[i]))
            accum[i][k] = x
            
    return accum





def bif_iter(n0, n, r, h, w, x_m, x_M):
    image = np.zeros((h, w), dtype = np.float32)
    
    for i in range(w):
        
        x = np.float32(0.5)
        
        for _ in range(1, n0):
            x = np.float32(tent_map(x, r[i]))

        for k in range(n):
            x = np.float32(tent_map(x, r[i]))
            x_idx = int((x - x_m) / (x_M - x_m) * (h - 1))
            image[x_idx][i] += 1
    return image




def non_zero_counting(matrix):
    N, M = matrix.shape
    nzc = np.count_nonzero(matrix, axis=0)
    result_matrix = matrix * nzc
    result_matrix = result_matrix.astype(np.float32)
    return result_matrix





# scatter plot image
def bifurcation_image(r, n0, n):
    width, height = len(r), 1000
    x_min, x_max = np.float32(0.0), np.float32(1.0)
    
    # Calculate (r,x) points frequencies
    image = bif_iter(n0, n, r, height, width, x_min, x_max)
            
    # Normalize frequencies
    image= non_zero_counting(image)
    
    # Cap values above 180,000
    image[image > 3600] = 3600

    # Normalize the entire image to 0-255
    max_value = np.max(image)
    image = (image / max_value) * 255
    
    return image


    


# pixel image
def bifurcation_diagram():
    width, height = 1000, 1000
    r_min, r_max = 0.2, 2.0
    x_min, x_max = 0.0, 1.0
    r_values = np.linspace(r_min, r_max, width)

    # Initialize the image array
    image = np.zeros((height, width), dtype=np.int32)

    for r_idx, r in enumerate(r_values):
        x = np.float32(0.25)
        # Stabilize x
        for _ in range(1000):
            x = np.float32(tent_map(x, r))

        # Collect 100,000 x-values
        for _ in range(100000):
            x = np.float32(tent_map(x, r))
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
    image[image > 180000] = 180000

    # Normalize the entire image to 0-255
    max_value = np.max(image)
    image = (image / max_value) * 255
    image = image.astype(np.uint8)

    return image













