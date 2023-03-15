import math
from cmath import inf
#import tabulate
#from tabulate import tabulate
import numpy as np

# Overflow and Underflow
numover = 1.
nover=1.
for i in range(10000):
    nover=numover
    numover = numover*2
    if numover == inf :
        numover=nover
        print(numover)
        print(i)
        break

numunder = 1.
nunder = 1.
for i in range(10000):
    nunder = numunder
    numunder = numunder/2
    if numunder == 0. :
        numunder = nunder
        print(numunder)
        print(i)
        break

# Machine precision
epsilon = 1
for j in range(10000):
    epsilon = epsilon/2
    if (1 + epsilon) == 1.:
        print(j)
        break

#Roundoff: derivative
"""
tabella = []
x = 1.
desatta = math.cos(x)
h = [0.5, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.0001, 0.00001, 10**(-6), 10**(-7), 10**(-8), 10**(-9), 10**(-10)]
for i in range(len(h)):
    tabella[print(h[i], abs((math.sin(x+h[i])-math.sin(x-h[i]))/(2*h[i])-desatta), abs((math.sin(x+h[i])-math.sin(x))/h[i]-desatta), abs((math.sin(x)-math.sin(x-h[i]))/h[i]-desatta))]
    print(tabulate(tabella,headers=['incremento','derivata simmetrica','derivata destra','derivata sinistra']))
"""

#Truncation and roundoff: exponential
"""def fattoriale(num):
    fact = 1
    for i in range(1,num+1):
        fact = fact * i
    return int(fact)

y = [0.1, 1, 5, 7, 8, 200]
min = (10**(-4))

def badesp(x):
    somma = 1
    for j in range(1,10000):
        somma += ((-x)**j)/fattoriale(j)
        if abs(somma-(math.e**(-x)))/(math.e**(-x))<10**(-10):
            return somma

def goodesp(x):
    somma = 1.
    elemento =1.
    for j in range(1,1000):
        elemento = elemento*((-x)/j)
        somma = somma + elemento
        if abs(somma-(math.e**(-x)))/(math.e**(-x))<min:
            return somma
            break

for i in range(6):
    #print(badesp(y[i]))
    print(goodesp(y[i]))"""
