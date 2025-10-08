import numpy as np
import matplotlib.pyplot as plt


def read_waveform_csv(filepath):
	"""les bølgeform CSV file fra Lab.
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
			x = float(parts[0]); y = float(parts[1])
		except Exception:
			continue
		data.append((x, y))

	return header, data


def generate_log_ticks(fmin, fmax):
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


base = "C:\\Users\\yonat\\Documents\\Tesla-main\\MM2-telegrafi\\data\\longCableData30m"
vin_path = base + "\\in\\vIn.CSV"
vout_path = base + "\\out\\vOut.CSV"
fin_path = base + "\\in\\freq.CSV"
fout_path = base + "\\out\\freq.CSV"


hdr_in, rows_in = read_waveform_csv(vin_path)
hdr_out, rows_out = read_waveform_csv(vout_path)

vpos_in = float(hdr_in.get('Vertical Position', '0') or 0)
vpos_out = float(hdr_out.get('Vertical Position', '0') or 0)

t_in = np.array([r[0] for r in rows_in])
v_in = np.array([r[1] + vpos_in for r in rows_in])

t_out = np.array([r[0] for r in rows_out])
v_out = np.array([r[1] + vpos_out for r in rows_out])

TIME_WINDOW_S = 2e-6
PAD_RATIO = 0.1

t_min_common = max(float(t_in.min()), float(t_out.min()))
t_max_common = min(float(t_in.max()), float(t_out.max()))

if (t_min_common <= 0.0) and (0.0 + TIME_WINDOW_S <= t_max_common):
	t0 = 0.0
else:
	t0 = t_min_common
	if t0 + TIME_WINDOW_S > t_max_common:
		t0 = max(t_min_common, t_max_common - TIME_WINDOW_S)
t1 = t0 + TIME_WINDOW_S

mask_in = (t_in >= t0) & (t_in <= t1)
mask_out = (t_out >= t0) & (t_out <= t1)

t_in_w, v_in_w = t_in[mask_in], v_in[mask_in]
t_out_w, v_out_w = t_out[mask_out], v_out[mask_out]

fig1, ax1 = plt.subplots()
ax1.plot(t_in_w, v_in_w, label='Vin (in)', color='blue')
ax1.plot(t_out_w, v_out_w, label='Vout (out)', color='orange', alpha=0.85)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Voltage (V)')
ax1.set_title('Vin vs Vout (time domain) — 2 µs window (30 m)')
ax1.grid(True)
ax1.legend()
tp = PAD_RATIO * TIME_WINDOW_S
ax1.set_xlim(t0 - tp, t1 + tp)


hdr_fin, rows_fin = read_waveform_csv(fin_path)
hdr_fout, rows_fout = read_waveform_csv(fout_path)

f_hz_in = np.array([r[0] for r in rows_fin])
amp_vrms_in = np.array([r[1] for r in rows_fin])
f_hz_out = np.array([r[0] for r in rows_fout])
amp_vrms_out = np.array([r[1] for r in rows_fout])

f_mhz_in = f_hz_in / 1e6
f_mhz_out = f_hz_out / 1e6

mask_in_f = f_mhz_in >= 1.0
mask_out_f = f_mhz_out >= 1.0
f_mhz_in, amp_vrms_in = f_mhz_in[mask_in_f], amp_vrms_in[mask_in_f]
f_mhz_out, amp_vrms_out = f_mhz_out[mask_out_f], amp_vrms_out[mask_out_f]

fig2, ax2 = plt.subplots()
ax2.semilogx(f_mhz_in + 1e-12, amp_vrms_in, label='Vin FFT rms (>= 1 MHz)', color='red')
ax2.semilogx(f_mhz_out + 1e-12, amp_vrms_out, label='Vout FFT rms (>= 1 MHz)', color='black', alpha=0.85)
ax2.set_xlabel('Frequency (MHz)')
ax2.set_ylabel('Amplitude (V rms)')
ax2.set_title('Vin vs Vout (FFT rms spectrum) — 30 m')
ax2.grid(True, which='both', linestyle=':')
ax2.legend()

fmin = float(min(f_mhz_in.min(), f_mhz_out.min()))
fmax = float(max(f_mhz_in.max(), f_mhz_out.max()))
xticks = generate_log_ticks(fmin, fmax)
ax2.set_xticks(xticks)
from matplotlib.ticker import FuncFormatter
ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))

plt.show()
