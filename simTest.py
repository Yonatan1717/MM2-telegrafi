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
lengths_m = [10, 50, 1000]

# RLGC ~ typisk tvunnet parkabel (approx—tilpass datasheet)
L = 0.5e-6          # H/m
C = 50e-12          # F/m
R0 = 0.08           # ohm/m @ f_ref (effektiv pr. par)
tan_delta = 2e-3    # dielektrisk loss tangent
f_ref = 1e6         # Hz, referanse for R(f) skalering

# Frekvensakse til H-plot
f_max = 200e6
Nf = 4000

# Tidsprøver for rekonstruksjon
Nt = 5000
t_cycles = 1.0      # tegn en periode




j = 1j
pi = np.pi

def R_f(f):
    """Skinneffekt: R ~ R0 * sqrt(f/f_ref) for f>0, og R(0)=0."""
    f = np.asarray(f, dtype=float)
    R = np.zeros_like(f)
    pos = f > 0
    R[pos] = R0 * np.sqrt(f[pos] / f_ref)
    return R

def G_f(f):
    """Dielektrisk tap: G = ω C tanδ (G(0)=0)."""
    omega = 2*pi*np.asarray(f, dtype=float)
    return omega * C * tan_delta

def gamma_f(f):
    """γ(f) = sqrt((R + jωL)(G + jωC))."""
    f = np.asarray(f, dtype=float)
    omega = 2*pi*f
    R = R_f(f)
    G = G_f(f)
    return np.sqrt((R + j*omega*L) * (G + j*omega*C))

def H_f(f, length):
    """Overføringsfunksjon H(f, l) = exp(-γ l) med H(0)=1."""
    f = np.asarray(f, dtype=float)
    H = np.exp(-gamma_f(f) * length)
    H[f == 0.0] = 1.0
    return H

def fourierRekkeCoeffs(T, duty, N):
    """
    Fourierkoeffisienter c_n for 0/1-puls med varighet tau=duty*T per periode.
    f(t)=sum_{n} c_n e^{j n ω0 t},  ω0=2π/T.
    """
    n = np.arange(-N, N+1)
    w0 = 2*pi/T
    tau = duty*T
    c = np.zeros_like(n, dtype=complex)
    c[n==0] = duty
    nz = n != 0
    c[nz] = (1.0/T) * (1 - np.exp(-j*n[nz]*w0*tau)) / (j*n[nz]*w0)
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


#Spekter (amplituder etter kabel)
amps_in = np.abs(c)
Hn_by_len = {}
for Lm in lengths_m:
    Hn_by_len[Lm] = H_f(np.abs(harm_freqs), Lm)

mask_pos = n > 0
fig1, ax1 = plt.subplots(figsize=(9,5))
ax1.stem(harm_freqs[mask_pos]/1e6, amps_in[mask_pos],
         linefmt='k-', markerfmt='ko', basefmt=' ')
for Lm in lengths_m:
    amps_out = np.abs(c * Hn_by_len[Lm])
    ax1.stem(harm_freqs[mask_pos]/1e6, amps_out[mask_pos],
             linefmt='-', markerfmt='o', basefmt=' ', label=f'{Lm} m')
ax1.set_xlabel('Frekvens (MHz)')
ax1.set_ylabel('Amplitude |c_n·H|')
ax1.set_title('Harmoniske og amplituder etter kabelen')
ax1.legend()
ax1.grid(True, alpha=0.3)

#Tidsdomene: en periode pr. lengde
fig2, ax2 = plt.subplots(figsize=(9,5))
t_in, x_in = rekonTidsSignal(T, n, c, Hn=None, Nt=Nt, t_cycles=t_cycles)
ax2.plot(t_in*1e9, x_in, 'k', lw=1.5, label='Inn (referanse)')

for Lm in lengths_m:
    Hn = Hn_by_len[Lm]
    t_out, x_out = rekonTidsSignal(T, n, c, Hn=Hn, Nt=Nt, t_cycles=t_cycles)
    ax2.plot(t_out*1e9, x_out, lw=1.6, label=f'{Lm} m')

ax2.set_xlabel('Tid (ns)')
ax2.set_ylabel('Spenning (norm.)')
ax2.set_title('Én periode av pulstoget etter ulike kabellengder')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.show()
