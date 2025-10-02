import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Kabelparametre (Cat5e typisk) 
R = 0.085        # ohm/m
L = 5e-7         # H/m
C = 5e-11        # F/m
G = 1e-10        # S/m
l = 1000         # kabelens lengde (m)

alpha = G/C
beta = R/L
c = 1/np.sqrt(L*C)   
a = l*0.01         
space_alpha = 0.002  # Dempingsparameter for avstand 

n_terms = 20
t_vals = np.linspace(0, 10e-6, 400)   
x_vals = np.linspace(0, l, 500)       

def sum_u(x, t, n):
    d = (alpha + beta)/2
    w_n = np.sqrt(((n*np.pi*c)/l)**2 - (0.25)*(alpha - beta)**2)
    phi_n = np.arctan(d/w_n)
    A_n = (2/(np.cos(phi_n)*l))
    return A_n*np.exp(-d*t) * np.exp(-space_alpha*x) * np.cos(w_n*t - phi_n) * np.sin((n*np.pi*x)/l)

def u(x, t, n_terms):
    u_val = np.zeros_like(x)
    for n in range(1, n_terms+1):
        u_val += sum_u(x, t, n)
    return u_val

# --- Plot og animasjon ---
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2, label="Signal")
front_line = ax.axvline(0, color='r', linestyle='--', label="Bølgefront (v*t)")

ax.set_xlim(0, l)             
ax.set_ylim(-0.02, 0.02)      
ax.set_xlabel("Posisjon langs kabel (m)")
ax.set_ylabel("Spenning (V)")
ax.set_title("Bølgeforplantning i 1000 m kabel (Fourier-sum)")
ax.legend()

def init():
    line.set_data([], [])
    front_line.set_xdata(0)
    return line, front_line

def animate(i):
    t = t_vals[i]
    y = u(x_vals, t, n_terms)
    line.set_data(x_vals, y)
    
    # Marker bølgefrontens posisjon (starter ved a)
    x_front = a + c*t
    if x_front <= l:
        front_line.set_xdata(x_front)
    else:
        front_line.set_xdata(np.nan) 
    return line, front_line

ani = FuncAnimation(fig, animate, init_func=init, frames=len(t_vals), interval=50, blit=True)
plt.show()
