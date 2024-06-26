"""
Library of self-made functions needed for the final project exercises

@author: david
"""

import numpy as np
from numba import njit, jit
import matplotlib.pyplot as plt




# -----------------------------------------------------------------------------
# STUDY OF THE TENT MAP
# -----------------------------------------------------------------------------


@njit
def tent_map(x, r):
    if x < 0.5:
        return r * x
    else:
        return r * (1 - x)




@njit
def iteration_tent(x0, r, n0, n):
    trajectory =  np.zeros((n0 + n), dtype = np.float32)
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
def bifurcation_tent(x0, r, n0, n):
    accum = np.zeros((len(r), n), dtype = np.float32)
    
    
    for i in range(len(r)):
        
        x = x0
        
        for _ in range(1, n0):
            x = np.float32(tent_map(x, r[i]))

        for k in range(n):
            x = np.float32(tent_map(x, r[i]))
            accum[i][k] = x
            
    return accum








# -----------------------------------------------------------------------------
# STUDY OF THE SINE MAP
# -----------------------------------------------------------------------------


@njit
def sine_map(x, r):
    return r * np.sin(np.pi * x)



@njit
def d_sine_map(x, r):
    return np.pi * r * np.cos(np.pi * x)




@njit
def iteration_sine(x0, r, n0, n):
    trajectory =  np.zeros((n0 + n), dtype = np.float32)
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
def bifurcation_sine(x0, r, n0, n):
    accum = np.zeros((len(r), n), dtype = np.float32)
    
    
    for i in range(len(r)):
        
        x = x0
        
        for _ in range(1, n0):
            x = np.float32(sine_map(x, r[i]))

        for k in range(n):
            x = np.float32(sine_map(x, r[i]))
            accum[i][k] = x
            
    return accum








# -----------------------------------------------------------------------------
# STUDY OF THE LOGISTIC MAP
# -----------------------------------------------------------------------------


@njit
def logistic_map(x, r):
    return r * x * (1 - x)




@njit
def d_logistic_map(x, r):
    return r * (1 - 2 * x)




@njit
def iteration_logistic(x0, r, n0, n):
    trajectory =  np.zeros((n0 + n), dtype = np.float32)
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
def non_predicibility_vs_chaos(r, n):
    m = int(n/3)
    x_chaos =  np.zeros((3, m), dtype = np.float32)
    x_random =  np.zeros((3, m), dtype = np.float32)
    x0_c = np.random.rand()
    x_c = x0_c
    
    for i in range(n):
        x_c = logistic_map(x_c, r)
        x_r = np.random.rand()
        j = i % 3
        k = i // 3
        
        x_chaos[j][k] = x_c
        x_random[j][k] = x_r
       
    return x_chaos, x_random




@njit
def bifurcation_logistic(x0, r, n0, n):
    accum = np.zeros((len(r), n), dtype = np.float32)
    
    
    for i in range(len(r)):
        
        x = x0
        
        for _ in range(1, n0):
            x = np.float32(logistic_map(x, r[i]))

        for k in range(n):
            x = np.float32(logistic_map(x, r[i]))
            accum[i][k] = x
            
    return accum








# -----------------------------------------------------------------------------
# IMAGING 
# -----------------------------------------------------------------------------


@jit(nopython=True)
def bif_iter(x0, r, n0, n, h, w, Map):
    image = np.zeros((h, w), dtype = np.float32)
    x_min, x_max = np.float32(0.0), np.float32(1.0)
    
    for i in range(w):
        
        x = x0
        
        for _ in range(1, n0):
            x = np.float32(Map(x, r[i]))

        for k in range(n):
            x = np.float32(Map(x, r[i]))
            x_idx = int((x - x_min) / (x_max - x_min) * (h - 1))
            image[x_idx][i] += 1
    return image




@jit(nopython=True)
def non_zero_counting(matrix):
    N, M = matrix.shape
    nzc = np.count_nonzero(matrix, axis=0)
    result_matrix = matrix * nzc
    result_matrix = result_matrix.astype(np.float32)
    return result_matrix




# scatter plot image
def bifurcation_image(x0, r, n0, n, Map):
    width, height = len(r), 1000
    
    # Calculate (r,x) points frequencies
    image = bif_iter(x0, r, n0, n, height, width, Map)
            
    # Normalize frequencies
    image= non_zero_counting(image)
    
    # Cap values above 180,000
    image[image > 1.8*n] = int(1.8*n)

    # Normalize the entire image to 0-255
    max_value = np.max(image)
    image = (image / max_value) * 255
    
    return image

    


# pixel image
def bifurcation_diagram(x0, arr_r, n0, n, Map):
    x_min, x_max = np.float32(0.0), np.float32(1.0)
    width, height = len(arr_r), 1000

    # Initialize the image array
    image = np.zeros((height, width), dtype=np.int32)

    for r_idx, r in enumerate(arr_r):
        
        x = x0
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








# -----------------------------------------------------------------------------
# LYAPUNOV EXPONENTS
# -----------------------------------------------------------------------------


@njit
def lyapunov_sine(x0, r, n0, n):
    l = np.zeros(len(r), dtype = np.float32)
    
    
    for i in range(len(r)):
        
        x = x0
        
        for _ in range(1, n0):
            x = np.float32(sine_map(x, r[i]))

        for k in range(n):
            x = np.float32(sine_map(x, r[i]))
            derivative = d_sine_map(x, r[i])
            logarithm = np.log(np.abs(derivative))
            l[i] += logarithm
            
    return l / n




@njit
def lyapunov_logistic(x0, r, n0, n):
    l = np.zeros(len(r), dtype = np.float32)
    
    
    for i in range(len(r)):
        
        x = x0
        
        for _ in range(1, n0):
            x = np.float32(logistic_map(x, r[i]))

        for k in range(n):
            x = np.float32(logistic_map(x, r[i]))
            derivative = d_logistic_map(x, r[i])
            logarithm = np.log(np.abs(derivative))
            l[i] += logarithm
            
    return l / n








# -----------------------------------------------------------------------------
# ENTROPY
# -----------------------------------------------------------------------------


def normalize_freq(matrix):
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
#(For some unknown reason the entropy for r=4 is wrong: always concentrates 
# around 1. And this because "bif_iter" at r=4 gives a trajectory that collapse
# towards zero no matter how we choose x0)
def entropy(x0, r, n0, n, Map, Nstates):
    width, height = len(r), Nstates
    
    # Calculate (r,x) points frequencies
    image = bif_iter(x0, r, n0, n, height, width, Map)
    #print(image)
            
    # Normalize frequencies
    p_i = normalize_freq(image)
    #print(p_i)
    
    # Evaluate  -Sum_{i}(p_i * ln(p_i))  for all sampled values of r
    s_i = log_freq(p_i)
    #print(s_i)
    S = - np.sum(s_i, axis=0)
    
    return S




@njit
def beta_function(x):
    den = np.pi * np.sqrt(x * (1-x))
    return 1/den








# -----------------------------------------------------------------------------
# 2D HENON MAP
# -----------------------------------------------------------------------------


@njit
def Henon_map(r0, a, b):
    r = np.zeros(2, dtype=np.float32)
    r[0] = r0[1] + 1 - a*r0[0]**2
    r[1] = b*r0[0]
    
    return r




@njit
def iteration_Henon(r0, a, b, n0, n):
    trajectory =  np.zeros((2, n), dtype = np.float32)
    trajectory[0][0], trajectory[1][0] = r0[0], r0[1]
    r = r0.copy()
    
    for i in range(1, n0):
        r = Henon_map(r, a, b)
        
    for i in range(n):
        r = Henon_map(r, a, b)
        trajectory[0][i] = r[0]
        trajectory[1][i] = r[1]
        
    return trajectory




@njit
def jacobian(r, a, b):
   J = np.zeros((2, 2), dtype=np.float32)
   J[0, 0] = -2 * a * r[0]
   J[0, 1] = 1.0
   J[1, 0] = b
   J[1, 1] = 0.0
   
   return J




@njit
def orthonormalize(v1, v2):
    U1 = v1
    U2 = v2 - (np.dot(v2, U1) / np.dot(U1, U1)) * U1
    norm1 = np.float32(np.linalg.norm(U1))
    norm2 = np.float32(np.linalg.norm(U2))
    u1 = U1 / norm1
    u2 = U2 / norm2
    return u1, u2, norm1, norm2




@njit
def Lyapunov_spectrum_2D(r0, a, b, n):
    w1_0 = np.asarray([1.0, 0.0], dtype=np.float32)
    w2_0 = np.asarray([0.0, 1.0], dtype=np.float32)
    lambda1, lambda2 = np.float32(0), np.float32(0)
    w1, w2 = w1_0, w2_0
    r = r0

    for _ in range(n):
        Jf = jacobian(r, a, b)
        z1 = np.dot(Jf, w1)
        z2 = np.dot(Jf, w2)
        u1, u2, norm1, norm2 = orthonormalize(z1, z2)
        lambda1 += np.log(norm1)
        lambda2 += np.log(norm2)
        w1, w2 = u1, u2
        r = Henon_map(r, a, b)
    
    return lambda1 / n, lambda2 / n
        


