import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parametre (juster fritt)
# -----------------------------
# Pulstog
T = 2e-6          # periode (s)  -> f0 = 10 MHz
duty = 0.5          # 0..1
N_harm = 200      # antall harmoniske på hver side (totalt 2N+1)

# Kabellengder (m)
lengths_m = [10, 50, 100, 1000]

# RLGC ~ typisk tvunnet parkabel (approx—tilpass datasheet)
L = 525e-9          # H/m
C = 52e-12          # F/m
tan_delta = 2e-3    # dielektrisk loss tangent
V0 = 5              # V, amplitudeskala (brukes ikke i normalisering)
# Frekvensakse til H-plot
f_max = 100e6
Nf = 4000

# Tidsprøver for rekonstruksjon
Nt = 5000
t_cycles = 1.0      # tegn en periode
mode = "kunn_fase"  # "full", "kunn_fase", "kunn_forsinkelse", "kunn_tap"




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
    gamma = np.sqrt((R + j*2*pi*f*L) * (G + j*2*pi*f*C))
    return gamma

def H_f(f, length, mode="full"):
    f = np.asarray(f, dtype=float)
    if mode == "full":
        gamma = gamma_f(f)                      # α + jβ (med tap)
        gamma_eff = gamma
    elif mode == "kunn_fase":                  # bruk bare β fra faktisk gamma
        beta = np.imag(gamma_f(f))              # β(f)
        gamma_eff = 1j*beta
    elif mode == "kunn_forsinkelse":              # ideell, tapsfri og ikke-dispersiv
        omega = 2*np.pi*f
        beta0 = omega*np.sqrt(L*C)              # β0 = ω√(LC)
        gamma_eff = 1j*beta0
    elif mode == "kunn_tap":                  # bruk bare α fra faktisk gamma
        alpha = np.real(gamma_f(f))             # α(f)
        gamma_eff = alpha
    else:
        raise ValueError("Ukjent mode")

    H = np.exp(-gamma_eff*length)
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
    Hn_by_len[Lm] = H_f(harm_freqs, Lm, mode=mode)

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
# en referanseberegning
# --- valg av forsinkelsesmodell ---
df_rel = 0.05                 # for group_at_f0: +/-5% rundt f0

def tau_per_meter(mode="ideal"):
    if mode == "ideal":
        return np.sqrt(L*C)   # τ/L = √(LC)
    elif mode == "group_at_f0":
        f0 = 1.0 / T
        df = df_rel * f0
        f_minus, f_plus = f0 - df, f0 + df
        beta_minus, beta_plus = np.imag(gamma_f([f_minus, f_plus]))
        omega_minus, omega_plus = 2*np.pi*np.array([f_minus, f_plus])
        d_beta_d_omega = (beta_plus - beta_minus) / (omega_plus - omega_minus)  # s/m
        return d_beta_d_omega              # τ/L = dβ/dω
    else:
        raise ValueError("Ukjent delay_mode")

# referanse
cmap = plt.cm.get_cmap('tab10', len(lengths_m))   # evt. 'tab20'
color_of = {Lm: cmap(i) for i, Lm in enumerate(lengths_m)}
ref_style = dict(color='0.45', alpha=0.35, lw=1.0)  # grå referanse

# --- Referanse ---
t_in, x_in = rekonTidsSignal(T, n, c, Hn=None, Nt=Nt, t_cycles=t_cycles)

fig, axes = plt.subplots(
    nrows=len(lengths_m), ncols=1, sharex=True, sharey=True,
    figsize=(9, 2.2*len(lengths_m))
)
axes = np.atleast_1d(axes)

if mode in ("kunn_forsinkelse", "kunn_fase"):
    # forsinkelse pr. meter
    tau_per_m = tau_per_meter()  # s/m

    for ax, Lm in zip(axes, lengths_m):
        Hn = Hn_by_len[Lm]
        t_out, x_out = rekonTidsSignal(T, n, c, Hn=Hn, Nt=Nt, t_cycles=t_cycles)

        # signaler
        ax.plot(t_in*1e9, x_in, **ref_style, label='Inn (ref.)')
        ax.plot(t_out*1e9, x_out, lw=1.6, color=color_of[Lm], label=f'{Lm} m')

        # forsinkelsesmarkør (mod T så den havner i vinduet)
        tau = tau_per_m * Lm                       # s
        tmark_ns = (tau % (t_cycles*T)) * 1e9      # ns
        ax.axvline(tmark_ns, ls='--', lw=1.0, color=color_of[Lm])

        ymax = ax.get_ylim()[1]
        ax.text(
            tmark_ns, ymax*0.92, fr'$\tau \approx {tau*1e9:.1f}\,$ns',
            rotation=0, ha='left', va='baseline', fontsize=10,
            color=color_of[Lm]
        )

        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Spenning (norm.)')
        ax.legend(loc='upper right')
else:
    for ax, Lm in zip(axes, lengths_m):
        Hn = Hn_by_len[Lm]
        t_out, x_out = rekonTidsSignal(T, n, c, Hn=Hn, Nt=Nt, t_cycles=t_cycles)

        ax.plot(t_in*1e9, x_in, **ref_style, label='Inn (referanse)')
        ax.plot(t_out*1e9, x_out, lw=1.6, color=color_of[Lm], label=f'{Lm} m')

        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Spenning (norm.)')
        ax.legend(loc='upper right')

axes[-1].set_xlabel('Tid (ns)')
fig.suptitle('En periode av pulstoget etter ulike kabellengder', y=0.98)
fig.align_ylabels(axes)
fig.tight_layout()




from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------
# FIGUR 3 – 3D waterfall: partielle summer (tid vs. maks frekvens)
# -----------------------------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Felles precompute
Lm_sel = 50
Hn_sel = Hn_by_len[Lm_sel]
cH = c * Hn_sel
t3 = np.linspace(0, T, 800, endpoint=False)
w0 = 2*np.pi/T
expo3 = np.exp(1j * np.outer(n*w0, t3))
f0 = 1.0/T
cuts = 13  # antall kutt i waterfall

mask_odd = (n % 2 != 0) & (n != 0)

# Finn en felles z-skala (uten normalisering)
x_full = np.real(np.sum((cH[:, None] * expo3)[mask_odd, :], axis=0))
zmax = 1.05 * np.max(np.abs(x_full))  # felles z-limit

# ===== LP: opp-til f_cut =====
fig_lp = plt.figure(figsize=(10,6))
ax_lp = fig_lp.add_subplot(121, projection='3d')
ax_hp = fig_lp.add_subplot(122, projection='3d')

f_top = min(N_harm * f0, 100e6)
n_max_list = np.linspace(1, int(f_top/f0), cuts, dtype=int)
n_max_list = np.where(n_max_list % 2 == 0, n_max_list+1, n_max_list)  # odd

for n_max in n_max_list:
    mask_lp = mask_odd & (np.abs(n) <= n_max)
    x_lp = np.real(np.sum((cH[:, None] * expo3)[mask_lp, :], axis=0))
    f_cut_MHz = (n_max * f0) / 1e6
    ax_lp.plot(t3*1e9, np.full_like(t3, f_cut_MHz), x_lp, lw=1.3)

ax_lp.set_title(f'LP waterfall: summer |f|<=f_cut  (L={Lm_sel} m)')
ax_lp.set_xlabel('Tid [ns]'); ax_lp.set_ylabel('Toppfrekvens f_cut [MHz]'); ax_lp.set_zlabel('Amplitude [V]')
ax_lp.set_ylim(0, f_top/1e6); ax_lp.set_zlim(-zmax, zmax)
ax_lp.view_init(elev=28, azim=-60); ax_lp.grid(True, alpha=0.2)
ax_lp.set_box_aspect([2.2,1.0,0.8])

# ===== HP: fra-og-med f_low =====
f_low_list = np.linspace(f_top, f0, cuts)  # fra 100 MHz ned mot fundamentalen

for f_low in f_low_list:
    mask_hp = mask_odd & ((np.abs(n)*f0) >= f_low) & ((np.abs(n)*f0) <= f_top)
    x_hp = np.real(np.sum((cH[:, None] * expo3)[mask_hp, :], axis=0))
    ax_hp.plot(t3*1e9, np.full_like(t3, f_low/1e6), x_hp, lw=1.3)

ax_hp.set_title(f'HP waterfall: summer |f|≥f_low  (L={Lm_sel} m)')
ax_hp.set_xlabel('Tid [ns]'); ax_hp.set_ylabel('Nedre grense f_low [MHz]'); ax_hp.set_zlabel('Amplitude [V]')
ax_hp.set_ylim(0, f_top/1e6); ax_hp.set_zlim(-zmax, zmax)
ax_hp.view_init(elev=28, azim=-60); ax_hp.grid(True, alpha=0.2)
ax_hp.set_box_aspect([2.2,1.0,0.8])

plt.tight_layout()

plt.show()