"""
Now I have to do everything from the beginning again

@author: david
"""
import math
from cmath import inf
import numpy as np
import tabulate
from tabulate import tabulate


# Overflow 
numover = np.float32(1)
nover = np.float32(1)
for i in range(10000):
    nover = np.float32(numover)
    numover = np.float32(numover*2)
    if np.float32(numover) == inf :
        print('\n','Overflow test using float32 type numbers:')
        print(nover)
        print(i, '\n')
        break

numover = np.float64(1)
nover = np.float64(1)
for i in range(10000):
    nover = np.float64(numover)
    numover = np.float64(numover*2)
    if np.float64(numover) == inf :
        print('Overflow test using float64 type numbers: ')
        print(nover)
        print(i, '\n')
        break


# Underflow
numunder = np.float32(1)
nunder = np.float32(1)
for i in range(10000):
    nunder = np.float32(numunder)
    numunder = np.float32(numunder/2)
    if numunder == np.float32(0) :
        print('Underflow test using float32 type numbers:')
        print(nunder)
        print(i, '\n')
        break

numunder = np.float64(1)
nunder = np.float64(1)
for i in range(10000):
    nunder = np.float64(numunder)
    numunder = np.float64(numunder/2)
    if numunder == np.float64(0) :
        print('Underflow test using float64 type numbers:')
        print(nunder)
        print(i, '\n')
        break


# Machine precision
epsilon = np.float32(1)
for j in range(10000):
    epsilon = np.float32(epsilon/2)
    if (np.float32(1) + np.float32(epsilon)) == np.float32(1):
        print('Machine precision test using float32 type numbers:')
        print(epsilon)
        print(j, '\n')
        break

epsilon = np.float64(1)
for j in range(10000):
    epsilon = np.float64(epsilon/2)
    if (np.float64(1) + np.float64(epsilon)) == np.float64(1):
        print('Machine precision test using float64 type numbers:')
        print(epsilon)
        print(j, '\n')
        break

#Roundoff: derivative
dati = []
x = 1
desatta = math.cos(x)
h = np.float32([0.5, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.0001, 0.00001, 10**(-6)])
for i in range(len(h)):
    dati.append({h[i], abs((math.sin(x+h[i])-math.sin(x-h[i]))/(2*h[i])-desatta), abs((math.sin(x+h[i])-math.sin(x))/h[i]-desatta), abs((math.sin(x)-math.sin(x-h[i]))/h[i]-desatta)})
headers=['Increment','Delta simm. deriv.','Delta right deriv.','Delta left deriv.']
tabella = tabulate(dati,headers)
print(tabella, '\n')


#Truncation and roundoff: exponential
def fattoriale(num):
    fact = 1
    for i in range(1,num+1):
        fact = fact * i
    return int(fact)

y = [0.1, 1, 5, 7, 8, 15]
min = (10**(-4))

def badesp(x):
    somma = 1
    for j in range(1,10000):
        somma += ((-x)**j)/fattoriale(j)
        if abs(somma-(math.e**(-x)))/(math.e**(-x))<10**(-10):
            return somma

def goodesp(x):
    somma = np.float64(1)
    elemento =np.float(1)
    for j in range(1,1000):
        elemento = np.float64(elemento*((-x)/j))
        somma = np.float64(somma + elemento)
        if abs(somma-(math.e**(-x)))/(math.e**(-x))<min:
            return np.float64(somma), np.float64(math.e**(-x)),j
            break

print('Truncation test for the negative exopnential Tayolr expansion with different x values:')
for i in range(6):
    print(goodesp(y[i]))
print('\n')
