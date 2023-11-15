"""
Now I have to do everything from the beginning again

@author: david
"""

from Funz4 import  RW1D_average, iter_plot
#import matplotlib.pyplot as plt

#-- ES 1 --
#---------- 1D Random Walks (RW)

# 1.1) Properties

ocean = RW1D_average(100, 64, 0, 0.5)

iter_plot(ocean[0], 64, 100, 0.5, 'Istantaneous position $x_i$')

iter_plot(ocean[1], 64, 100, 0.5, 'Istantaneous squared position $x_i ^2$')








   