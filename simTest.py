import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parametre (juster fritt)
# -----------------------------
# Pulstog
T = 100e-9          # periode (s)  -> f0 = 10 MHz
duty = 0.5          # 0..1
N_harm = 200        # antall harmoniske på hver side (totalt 2N+1)

# Kabellengder (m)
lengths_m = [10, 50, 100]

# RLGC ~ typisk tvunnet parkabel (approx—tilpass datasheet)
L = 525e-9          # H/m
C = 52e-12          # F/m
tan_delta = 2e-3    # dielektrisk loss tangent
V0 = 5              # V, amplitudeskala (brukes ikke i normalisering)
# Frekvensakse til H-plot
f_max = 25e6
Nf = 4000

# Tidsprøver for rekonstruksjon
Nt = 5000
t_cycles = 1.0      # tegn en periode




j = 1j
pi = np.pi

def R_f(f, rho=1.72e-8, mu=4*np.pi*1e-7, r=0.255e-3, conductors=2):
    """
    R(f) for skinneffekt.
    r: lederradius (m). conductors: antall seriekoblede ledere (2 for tvinnet par/coax retur).
    """
    a = r
    Rdc_per = rho/(np.pi*a*a)          # ohm/m for en leder (DC)
    fs = rho/(np.pi*mu*a*a)            # Hz
    f = np.asarray(f, dtype=float)
    Rac_one = np.where(f < fs, Rdc_per, Rdc_per*np.sqrt(f/fs))
    return conductors*Rac_one          # total serie-R per meter


def G_f(f):
    """Dielektrisk tap: G = ω C tanδ (G(0)=0)."""
    omega = 2*pi*np.asarray(f, dtype=float)
    return omega * C * tan_delta

def gamma_f(f):
    """gamma(f) = sqrt((R + jωL)(G + jωC))."""
    f = np.asarray(f, dtype=float)
    R = R_f(f)
    G = G_f(f)
    return np.sqrt((R + j*2*pi*f*L) * (G + j*2*pi*f*C))

def H_f(f, length):
    """Overføringsfunksjon H(f, l) = exp(-gamma*l) med H(0)=1."""
    f = np.asarray(f, dtype=float)
    H = np.exp(-gamma_f(f) * length)
    H[f == 0.0] = 1.0
    return H

alpha = np.real(gamma_f(np.linspace(0, f_max, Nf)))
beta = np.imag(gamma_f(np.linspace(0, f_max, Nf)))

beta_prime_f = np.gradient(beta, f_max/(Nf-1))
T_g = {}
T_g['10m'] = beta_prime_f * lengths_m[0]
T_g['50m'] = beta_prime_f * lengths_m[1]
T_g['100m'] = beta_prime_f * lengths_m[2]


def fourierRekkeCoeffs(T, duty, N):
    """ 
    Fourierkoeffisienter c_n for 0/1-puls med varighet tau=duty*T per periode.
    f(t)=sum_{n} c_n e^{j n ω0 t},  ω0=2π/T.
    """
    n = np.arange(-N, N+1)
    w0 = 2*pi/T
    tau = duty*T
    c = np.zeros_like(n, dtype=complex)
    c[n==0] = duty * V0  # DC-komponent
    nz = n != 0
    c[nz] = V0 * (1.0/T) * (1 - np.exp(-j*n[nz]*w0*tau)) / (j*n[nz]*w0)
    return n, c


def rekonTidsSignal(T, n, c, Hn=None, Nt=4000, t_cycles=1.0):
    """f(t)=Re{ sum c_n H_n e^{j n ω0 t} }."""
    t = np.linspace(0, t_cycles*T, Nt, endpoint=False)
    w0 = 2*pi/T
    if Hn is None:
        Hn = np.ones_like(c, dtype=complex)
    expo = np.exp(j*np.outer(n*w0, t))   
    f_t = np.real(np.sum((c*Hn)[:,None] * expo, axis=0))
    return t, f_t


n, c = fourierRekkeCoeffs(T, duty, N_harm)

f0 = 1.0/T
harm_freqs = n * f0


#frekvensdommene: Spekter (amplituder etter kabel)
harm_freqs = n * f0
odd_pos_idx = np.where((harm_freqs > 0) & (np.abs(n) % 2 == 1))[0]
freqs_odd_MHz = harm_freqs[odd_pos_idx] / 1e6

amps_in = np.abs(c)
Hn_by_len = {}
for Lm in lengths_m:
    # behold full lengde (brukes senere i tidsdomenet), men evaluer på |f|
    Hn_by_len[Lm] = H_f(np.abs(harm_freqs), Lm)

fig1, ax1 = plt.subplots(figsize=(9,5))
ax1.stem(freqs_odd_MHz, amps_in[odd_pos_idx],
         linefmt='k-', markerfmt='ko', basefmt=' ', label='Inn (odd n)')

for Lm in lengths_m:
    amps_out = np.abs(c * Hn_by_len[Lm])
    ax1.stem(freqs_odd_MHz, amps_out[odd_pos_idx],
             linefmt='-', markerfmt='o', basefmt=' ', label=f'{Lm} m')

ax1.set_xlabel('Frekvens (MHz)')
ax1.set_ylabel('Amplitude |c_n·H| (V)')
ax1.set_title('Harmoniske (kun odd n) og amplituder etter kabelen')
ax1.legend()
ax1.grid(True, alpha=0.3)



#Tidsdomene: en periode pr. lengde
fig2, ax2 = plt.subplots(figsize=(9,5))
t_in, x_in = rekonTidsSignal(T, n, c, Hn=None, Nt=Nt, t_cycles=t_cycles)
ax2.plot(t_in*1e9, x_in, 'k', lw=1.5, label='Inn (referanse)')

for Lm in lengths_m:
    Hn = Hn_by_len[Lm]
    t_out, x_out = rekonTidsSignal(T, n, c, Hn=Hn, Nt=Nt, t_cycles=t_cycles)
    t_out_shifted = t_out - np.mean(T_g[f'{Lm}m'])  # juster for forsinkelse
    ax2.plot(t_out_shifted*1e9, x_out, lw=1.6, label=f'{Lm} m')

ax2.set_xlabel('Tid (ns)')
ax2.set_ylabel('Spenning (norm.)')
ax2.set_title('en periode av pulstoget etter ulike kabellengder')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.show()
