#!/usr/bin/python
from __future__ import division, print_function

import sys

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

# for p in 4K-20uA 2K-20uA 2K-100uA; do ./plot-hall.py $p & done

param = sys.argv[1]
loadparams = {'unpack': True, 'skiprows': 3, 'delimiter': ';'}

_, B_hall = np.loadtxt(f'src/{param}-hall-B.dat', **loadparams)
_, U_hall = np.loadtxt(f'src/{param}-hall-U.dat', **loadparams)
_, B_res = np.loadtxt(f'src/{param}-res-B.dat', **loadparams)
_, U_res = np.loadtxt(f'src/{param}-res-U.dat', **loadparams)

if len(B_hall) != len(U_hall) or len(B_res) != len(U_res):
	print('lengths don\'t match')
	exit(1)

# make figure
fig = plt.figure(1)

axL = fig.add_subplot(111)
axR = axL.twinx()

fig_R_hall = plt.figure(2)
ax_R_hall = fig_R_hall.add_subplot(111)

fig_B_periodicity = plt.figure(3)
ax_B_periodicity = fig_B_periodicity.add_subplot(111)

# find index where shit goes haywire
I_hall = B_hall + 1j * U_hall
I_res = B_res + 1j * U_res

I_hall_diff = np.abs(np.diff(I_hall))
I_res_diff = np.abs(np.diff(I_res))

cutoff_hall = np.argmax(I_hall_diff > 1) - 1
cutoff_res = np.argmax(I_res_diff > 1) - 1

# cut off haywire shit and scale voltages to mV
B_hall = B_hall[:cutoff_hall]
U_hall = U_hall[:cutoff_hall] * 1000
B_res = B_res[:cutoff_res]
U_res = U_res[:cutoff_res] * 1000

# choose filter
if False:
	# apply weiner filter, to filter weiners
	weiner_size = 11
	B_hall = scipy.signal.wiener(B_hall, weiner_size)
	U_hall = scipy.signal.wiener(U_hall, weiner_size)
	B_res = scipy.signal.wiener(B_res, weiner_size)
	U_res = scipy.signal.wiener(U_res, weiner_size)
if True:
	# apply savgol filter
	savgol_size = 51
	savgol_order = 2
	B_hall = scipy.signal.savgol_filter(B_hall, savgol_size, savgol_order)
	U_hall = scipy.signal.savgol_filter(U_hall, savgol_size, savgol_order)
	B_res = scipy.signal.savgol_filter(B_res, savgol_size, savgol_order)
	U_res = scipy.signal.savgol_filter(U_res, savgol_size, savgol_order)

# scale magnetic field
max_B = np.max(np.append(B_hall, B_res))
min_B = np.min(np.append(B_hall, B_res))

factor_B = 6 / (max_B - min_B)
print(f'magnetic field scaling factor: {factor_B:0.4f}; offset: {min_B:0.2f}T')
B_hall = (B_hall - min_B) * factor_B
B_res = (B_res - min_B) * factor_B

# get U_res peak indexes
peak_dict = {
	'4K-20uA': [1298, 2029, 2295, 2442],
	'2K-20uA': [710, 1417, 1671, 1817, 1900],
	'2K-100uA': [657, 1451, 1712]
}  # (I'm sorry)
indexes_peaks_U_res = peak_dict[sys.argv[1]]
U_res_peak = U_res[indexes_peaks_U_res]
B_res_peak = B_res[indexes_peaks_U_res]

i_B_periodicity = np.arange(len(B_res_peak)) * 2
one_over_B_periodicity = 1 / B_res_peak

one_over_B_period, _ = np.polyfit(i_B_periodicity, one_over_B_periodicity, 1)

ax_B_periodicity.plot(i_B_periodicity, one_over_B_periodicity, 'xk')
ax_B_periodicity.set_xlabel('$i$')
ax_B_periodicity.set_ylabel('$\\frac{1}{B}$ (T$^{-1}$)')
ax_B_periodicity.set_title('absolute value of i neither correct nor relevant')

print(f'Delta (1/B) = {one_over_B_period:0.2f} T^-1')
electron_charge = 1.602e-19
planck = 6.626e-34
charge_carrier_density = 2 * electron_charge / (planck * one_over_B_period)
print(f'charge carrier density: {charge_carrier_density * 1e-4:0.3e} 1/m²')

# print U_res peaks
print('peaks of logitudinal voltage:')
print('B (T)\tU (mV)')
for (B, U) in zip(B_res_peak, U_res_peak):
	print(f'{B:0.2f}\t{U:0.0f}')

# get U_h plateau voltages
current_dict = {
	'4K-20uA': 20e-6,
	'2K-20uA': 20e-6,
	'2K-100uA': 100e-6
}
current = current_dict[sys.argv[1]]
hall_resistance = 25.81281e3
# dict contents: (index_min, index_max, current uA, hall index)
plateau_dict = {
	'4K-20uA': [
		(165, 190),
		(490, 620),
		(1334, 2060)
	],
	'2K-20uA': [
		(113, 165),
		(424, 630),
		(1252, 2160)
	],
	'2K-100uA': [
		(1135, 1202),
		(1972, 2615)
	]
}
indexes_plateau_U_hall = plateau_dict[sys.argv[1]]

print()
print('hall plateaus:')
print('B_min (T)\tB_max (T)\tU_h_mean (mV)\tU_h_std (mV)\t(i)\ti\tR_h (ohm)')

data_hall_regression_i = []
data_hall_regression_1_over_R_hall = []

for (plat_start, plat_end) in indexes_plateau_U_hall:
	U_hall_plat = U_hall[plat_start:plat_end]
	B_hall_plat = B_hall[plat_start:plat_end]

	B_min = np.min(B_hall_plat)
	B_max = np.max(B_hall_plat)
	U_mean = np.mean(U_hall_plat)
	U_std = np.std(U_hall_plat)
	R_hall = U_mean * 1e-3 / current
	i = hall_resistance / R_hall
	i_actual = int(2 * round(i / 2))

	print(f'{B_min:0.2f}\t\t{B_max:0.2f}\t\t{U_mean:0.0f}\t\t{U_std:0.1f}\t\t{i:0.2f}\t{i_actual:d}\t{R_hall:0.1f}')

	data_hall_regression_i.append(i_actual)
	data_hall_regression_1_over_R_hall.append(1 / R_hall)

	# draw ellipse
	# el = matplotlib.patches.Ellipse(xy=(np.mean(B_hall_plat), np.mean(
	# 	U_hall_plat)), width=(0.5 + np.max(B_hall_plat) - np.min(B_hall_plat)), height=10, fill=False, linewidth=2)
	# axL.add_artist(el)

	# mark start + end
	axL.vlines(np.min(B_hall_plat), np.mean(
		U_hall_plat) - 10, np.mean(U_hall_plat) + 10)
	axL.vlines(np.max(B_hall_plat), np.mean(
		U_hall_plat) - 10, np.mean(U_hall_plat) + 10)
	# TODO: prettify that

# do linear regression of hall voltages

slope_1_over_R_hall_over_i, _ = np.polyfit(
	data_hall_regression_i, data_hall_regression_1_over_R_hall, 1)

R_k_measured = 1 / slope_1_over_R_hall_over_i
alpha_measured = 299792458 * 4 * np.pi * 1e-7 / (2 * R_k_measured)

print(
	f'1/(R_h * i) = {slope_1_over_R_hall_over_i*1e6:0.2f} uS ^= {R_k_measured:0.1f}ohm')
print(f'1/alpha = {1/alpha_measured:0.2f}')

ax_R_hall.plot(data_hall_regression_i,
               np.array(data_hall_regression_1_over_R_hall) * 1e3, 'xk')
ax_R_hall.set_xlabel('$i$')
ax_R_hall.set_ylabel('$\\frac{1}{R_h}$ (mS)')

# align y axes
Uh_max = np.max(U_hall)
Uh_max = 100 * np.ceil(Uh_max / 100)

axL.set_ylim(0, Uh_max)
axR.set_ylim(0, Uh_max / 4)

ticks = np.linspace(0, Uh_max, 6)

axL.set_yticks(ticks)
axR.set_yticks(ticks / 4)

# plot
axL.plot(B_hall, U_hall, '.', label='U_h', markersize=1)
axR.plot(0, 0)  # skip first color
axR.plot(B_res, U_res, '.', label='U_res', markersize=1)
axR.plot(B_res_peak, U_res_peak, '+k', markersize=20)

axL.set_xlabel('B (T)')
axL.set_ylabel('$U_h$ (mV)')
axR.set_ylabel('$U_{res}$ (mV)')

fig.legend()

plt.grid()

if len(sys.argv) == 2:
	plt.show()
elif len(sys.argv) >= 3:
	plt.savefig(sys.argv[2])
