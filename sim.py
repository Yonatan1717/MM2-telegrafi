import matplotlib.pyplot as plt
import numpy as np  


R = 1
L = 1
C = 1
G = 1

l = 100

alpha = G/C
beta = R/L
c = 1/np.sqrt(L*C)
a = l*.01
t = np.linspace(0, 1e-6, 2000)

def sum_u(x,t,n):
    d = (alpha + beta)/2
    w_n = np.sqrt(((n*np.pi*c)/l)**2 - (1/4)*(alpha - beta)**2)
    phi_n = np.arctan(d/w_n)
    A_n = (2/(np.cos(phi_n)*l))*np.sin((n*np.pi*x)/l)
    return A_n*np.exp(-d*t)*np.cos(w_n*t - phi_n)*np.sin((n*np.pi*a)/l)

