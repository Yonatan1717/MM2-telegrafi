import numpy as np
import matplotlib.pyplot as plt


def read_waveform_csv(filepath):
    """Les bølgeform CSV file fra Lab.
    """
    header = {}
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_start = None
    for i, raw in enumerate(lines):
        line = raw.strip()
        if line.lower().startswith('waveform data'):
            data_start = i + 1
            break
        # header key,value pairs
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2 and parts[0] != '' and parts[1] != '':
            header[parts[0]] = parts[1]

    if data_start is None:
        for i, raw in enumerate(lines):
            line = raw.strip()
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    float(parts[0]); float(parts[1])
                    data_start = i
                    break
                except Exception:
                    continue

    for raw in lines[data_start:]:
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except Exception:
            continue
        data.append((x, y))

    return header, data

def generate_log_ticks(fmin, fmax):
    """Generate nice log-scale ticks at 1,2,5 within each decade between fmin and fmax."""
    if fmin <= 0:
        fmin = 1e-12
    ticks = []
    start_dec = int(np.floor(np.log10(fmin)))
    end_dec = int(np.ceil(np.log10(max(fmax, fmin*10))))
    for p in range(start_dec, end_dec + 1):
        base = 10.0 ** p
        for m in (1.0, 2.0, 5.0):
            val = m * base
            if fmin <= val <= fmax:
                ticks.append(val)
    return ticks

vInn_csv_file = "C:\\Users\\yonat\\Documents\\Tesla-main\\MM2-telegrafi\\data\\shortCableData5m\\in\\vIn.CSV"
hdr_in, rows_in = read_waveform_csv(vInn_csv_file)

vpos = 0.0
try:
    if 'Vertical Position' in hdr_in:
        vpos = float(hdr_in['Vertical Position'])
except Exception:
    vpos = 0.0

t_in = np.array([r[0] for r in rows_in])
v_in = np.array([r[1] + vpos for r in rows_in])  # shift baseline like original (+2.4 V)

vOut_csv_file = "C:\\Users\\yonat\\Documents\\Tesla-main\\MM2-telegrafi\\data\\shortCableData5m\\out\\vOut.CSV"
hdr_out, rows_out = read_waveform_csv(vOut_csv_file)

vpos_out = 0.0
try:
    if 'Vertical Position' in hdr_out:
        vpos_out = float(hdr_out['Vertical Position'])
except Exception:
    vpos_out = 0.0

t_out = np.array([r[0] for r in rows_out])
v_out = np.array([r[1] + vpos_out for r in rows_out])

TIME_WINDOW_S = 2e-6

t_min_common = max(float(t_in.min()), float(t_out.min())) if t_out.size else float(t_in.min())
t_max_common = min(float(t_in.max()), float(t_out.max())) if t_out.size else float(t_in.max())

if (t_min_common <= 0.0) and (0.0 + TIME_WINDOW_S <= t_max_common):
    t0 = 0.0
else:
    t0 = t_min_common
    if t0 + TIME_WINDOW_S > t_max_common:
        t0 = max(t_min_common, t_max_common - TIME_WINDOW_S)

t1 = t0 + TIME_WINDOW_S

mask_in = (t_in >= t0) & (t_in <= t1)
t_in_w = t_in[mask_in]
v_in_w = v_in[mask_in]

if t_out.size:
    mask_out_t = (t_out >= t0) & (t_out <= t1)
    t_out_w = t_out[mask_out_t]
    v_out_w = v_out[mask_out_t]
else:
    t_out_w, v_out_w = t_out, v_out

fig1, ax1 = plt.subplots()
ax1.plot(t_in_w, v_in_w, label='vIn', color='blue')
ax1.plot(t_out_w, v_out_w, label='vOut', color='orange', alpha=0.85)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Voltage (V)')
ax1.set_title('Vin vs Vout (time domain) — 2 µs window')
ax1.grid(True)
ax1.legend()
ax1.set_xlim(t0, t1)


freq_csv_file = "C:\\Users\\yonat\\Documents\\Tesla-main\\MM2-telegrafi\\data\\shortCableData5m\\in\\freq.CSV"
hdr_f, rows_f = read_waveform_csv(freq_csv_file)

f_hz = np.array([r[0] for r in rows_f])
amp_vrms = np.array([r[1] for r in rows_f])

f_mhz = f_hz / 1e6

mask = f_mhz >= 1.0
f_mhz = f_mhz[mask]
amp_vrms = amp_vrms[mask]
    
if f_mhz.size == 0:
    print("No frequency points >= 1 MHz to plot.")
    plt.show()
    raise SystemExit(0)


freq_csv_file_out = "C:\\Users\\yonat\\Documents\\Tesla-main\\MM2-telegrafi\\data\\shortCableData5m\\out\\freq.CSV"
hdr_f_out, rows_f_out = read_waveform_csv(freq_csv_file_out)
f_hz_out = np.array([r[0] for r in rows_f_out])
amp_vrms_out = np.array([r[1] for r in rows_f_out])

f_mhz_out = f_hz_out / 1e6
mask_out = f_mhz_out >= 1.0
f_mhz_out = f_mhz_out[mask_out]
amp_vrms_out = amp_vrms_out[mask_out]

fig2, ax2 = plt.subplots()
ax2.semilogx(f_mhz + 1e-12, amp_vrms, label='Vin FFT rms (>= 1 MHz)', color='green')
ax2.semilogx(f_mhz_out + 1e-12, amp_vrms_out, label='Vout FFT rms (>= 1 MHz)', color='red', alpha=0.85)
ax2.set_xlabel('Frequency (MHz)')
ax2.set_ylabel('Amplitude (V rms)')
ax2.set_title('Vin vs Vout (FFT rms spectrum)')
ax2.grid(True, which='both', linestyle=':')
ax2.legend()

fmin = float(min(f_mhz.min(), f_mhz_out.min()))
fmax = float(max(f_mhz.max(), f_mhz_out.max()))
xticks = generate_log_ticks(fmin, fmax)
ax2.set_xticks(xticks)
from matplotlib.ticker import FuncFormatter
ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))


vOut_csv_file = "C:\\Users\\yonat\\Documents\\Tesla-main\\MM2-telegrafi\\data\\shortCableData5m\\out\\vOut.CSV"
hdr_out, rows_out = read_waveform_csv(vOut_csv_file)

t_out = np.array([r[0] for r in rows_out])
v_out = np.array([r[1] for r in rows_out])

fig3, ax3 = plt.subplots()
ax3.plot(t_out, v_out, label='vOut', color='red')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Voltage (V)')
ax3.set_title('Output Voltage Over Time')
ax3.grid(True)
ax3.legend()

freq_csv_file = "C:\\Users\\yonat\\Documents\\Tesla-main\\MM2-telegrafi\\data\\shortCableData5m\\out\\freq.CSV"
hdr_f_out, rows_f_out = read_waveform_csv(freq_csv_file)
f_hz_out = np.array([r[0] for r in rows_f_out])
amp_vrms_out = np.array([r[1] for r in rows_f_out])
f_mhz_out = f_hz_out / 1e6
mask_out = f_mhz_out >= 1.0
f_mhz_out = f_mhz_out[mask_out]
amp_vrms_out = amp_vrms_out[mask_out]   
fig4, ax4 = plt.subplots()
ax4.set_xticks(xticks)
ax4.semilogx(f_mhz_out + 1e-12, amp_vrms_out, label='FFT rms amplitude (>= 1 MHz)', color='orange')
ax4.set_xlabel('Frequency (MHz)')
ax4.set_ylabel('Amplitude (V rms)')
ax4.grid(True, which='both', linestyle=':')
ax4.legend()

plt.show()
