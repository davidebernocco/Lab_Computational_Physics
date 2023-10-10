"""
First trial of Numba Just-In-Time Compiler

@author: david
"""
import numpy as np
from numba import jit, float64
import random
import timeit

def sum_of_squares(lst):
    result = 0
    for x in lst:
        result += x ** 2
    return result


@jit(float64(float64[:]))
def sum_of_squares_numba(lst):
    result = 0
    for x in lst:
        result += x ** 2
    return result




# Generate a list of 1 million random integers
lst = [random.randint(1, 100) for _ in range(1000000)]
arr = np.array(lst, dtype=np.float64)

# Measure the time it takes to run the original Python function
python_time = timeit.timeit(lambda: sum_of_squares(arr), number=100)

# Measure the time it takes to run the Numba-accelerated function
numba_time = timeit.timeit(lambda: sum_of_squares_numba(arr), number=100)

print(f"Python Time: {python_time:.6f} seconds")
print(f"Numba Time: {numba_time:.6f} seconds")

