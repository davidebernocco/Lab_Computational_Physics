
"""
Probability distribution

@author: david
"""
import math
import matplotlib.pyplot as plt
from numba import njit
import numpy as np


def Prob_distr_lattice(N):
    points = []
    radius = []
    s = []  # Number of doable horizontal steps to reach the selected point. It has values a <= s <= (N-b) for each P(a,b)
    for j in range(N+1):
        for i in range(j, N+1):
            if (i+j) % 2 == (N % 2) and (i+j) <= N:
                points.append([i, j])
                radius.append(math.sqrt(i**2 + j**2))
                s.append([k for k in range(i, (N-j) + 1, 2)])
    
    
    num_walks = [0] * len(points)
    P_N = [0] * len(points)
    merda = 0
    for i in range(len(points)):
        print(s[i])       
        for k in (s[i]):
            
            f1 =  math.factorial(N) / ( math.factorial( int((points[i][0] + k)/2) ) * math.factorial( int((-points[i][0] + k)/2) ) * math.factorial(N-k) )
            f2 =  math.factorial(N - k) / ( math.factorial( int((N - k + points[i][1])/2) ) * math.factorial( int((N - k - points[i][1])/2) ) )
            num_walks[i] += int(f1 * f2)
        
        if sum(points[i]) == 0:
            print('a',points[i])
            P_N[i] +=  num_walks[i] / 4**N
            merda += num_walks[i]
            
        elif points[i][0] == points[i][1] or points[i][1] == 0:
            print('\n', 'b',points[i])
            P_N[i] +=  (4 * num_walks[i]) / 4**N
            merda += 4*num_walks[i]
        else:
            print('\n', 'c', points[i])
            P_N[i] +=  (8 * num_walks[i]) / 4**N
            merda += 8*num_walks[i]
                
    return np.asarray(radius, dtype=np.float32), np.asarray(P_N, dtype=np.float32), np.sum(np.asarray(P_N, dtype=np.float32)), np.asarray(num_walks, dtype=np.float32) ,np.sum(np.asarray(num_walks, dtype=np.int64)), merda

"""
Clessidra = Prob_distr_lattice(16)

plt.scatter(Clessidra[0], Clessidra[1], color='blue', marker='o')
plt.xlabel('r(N)')
plt.ylabel(r'P_{N}(x)')
plt.title('Theroretical probability distribution on a square lattice')
plt.legend()
plt.show()
"""         
"""

# Initialize an empty dictionary
points_dict = {}

# Iterate over the lists and populate the dictionary
for i in range(len(points)):
    coordinates = tuple(points[i])
    distance = radius[i]
    horiz_displ = s[i]

    # Use the coordinates as keys and store associated information as values
    points_dict[coordinates] = {'distance': distance, 'horiz_displ': horiz_displ}

# Print the resulting dictionary
for key, value in points_dict.items():
    print(f"Point: {key}, Distance: {value['distance']}, Horizontal displacement: {value['horiz_displ']}")
"""    


