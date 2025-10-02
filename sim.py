import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Kabelparametre (Cat5e typisk) 
R = 0.085        # ohm/m
L = 5e-7         # H/m
C = 5e-11        # F/m
G = 1e-10        # S/m
l = 150          # kabelens lengde (m)

alpha = G/C
beta = R/L
c = 1/np.sqrt(L*C)   
a = l*0.01         
# Romlig demping: sett ønsket relativ amplitude ved kabelenden (0<amp_at_end<=1)
amp_at_end = 0.1  # f.eks. 0.6 betyr ~60% amplitude igjen ved x = l
space_alpha = -np.log(amp_at_end)/l  # skaleres automatisk med l

n_terms = 20
# Tidsvindu: ~2 ganger tiden bølgefronten bruker til enden
t_max = 2*l/c
t_vals = np.linspace(0, t_max, 400)   
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

# Sett aksegrenser
ax.set_xlim(0, l)             
# Estimer startamplitude for å sette y-akse dynamisk (symmetrisk rundt 0)
y0 = u(x_vals, 0.0, n_terms)
ymax = max(1e-6, float(np.max(np.abs(y0))))
ax.set_ylim(-1.2*ymax, 1.2*ymax)
ax.set_xlabel("Posisjon langs kabel (m)")
ax.set_ylabel("Spenning (V)")
ax.set_title(f"Bølgeforplantning i {l} m kabel (Fourier-sum)")
ax.legend()

def init():
    line.set_data([], [])
    front_line.set_xdata(a)
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

ani = FuncAnimation(fig, animate, init_func=init, frames=len(t_vals), interval=25, blit=True)
plt.show()
