#!/usr/bin/python
from __future__ import division, print_function

import argparse
import sys

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import kafe
from kafe.function_library import linear_2par

parser = argparse.ArgumentParser()

parser.add_argument('curve', metavar='C', type=str, choices=[
					'4K-20uA', '2K-20uA', '2K-100uA'])
parser.add_argument('--output-main', type=str,
					help='output file for main U over B plot')
parser.add_argument('--output-1oB', type=str,
					help='output file for (one over B) over i plot')
parser.add_argument('--output-plateaus', type=str,
					help='output file for Hall voltage over i plot')
parser.add_argument('--filter', type=str,
					choices=['savgol', 'weiner'])
parser.add_argument('--table', action='store_true',
					help='show table')

args = parser.parse_args()

loadparams = {'unpack': True, 'skiprows': 3, 'delimiter': ';'}

_, B_hall = np.loadtxt(f'src/{args.curve}-hall-B.dat', **loadparams)
_, U_hall = np.loadtxt(f'src/{args.curve}-hall-U.dat', **loadparams)
_, B_res = np.loadtxt(f'src/{args.curve}-res-B.dat', **loadparams)
_, U_res = np.loadtxt(f'src/{args.curve}-res-U.dat', **loadparams)

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

# save unfiltered values for standard deviation and maximum flavor
B_hall_unfiltered = B_hall
U_hall_unfiltered = U_hall
B_res_unfiltered = B_res
U_res_unfiltered = U_res

# choose filter
if args.filter == 'weiner':
	# apply weiner filter, to filter weiners
	weiner_size = 11
	B_hall = scipy.signal.wiener(B_hall, weiner_size)
	U_hall = scipy.signal.wiener(U_hall, weiner_size)
	B_res = scipy.signal.wiener(B_res, weiner_size)
	U_res = scipy.signal.wiener(U_res, weiner_size)
elif args.filter == 'savgol':
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

B_hall = (B_hall - min_B) * factor_B
B_res = (B_res - min_B) * factor_B

# get U_res peak indexes
peak_dict = {
	'4K-20uA': [1298, 2029, 2295, 2442],
	'2K-20uA': [710, 1417, 1671, 1817, 1900],
	'2K-100uA': [657, 1451, 1712]
}  # (I'm sorry)
indexes_peaks_U_res = peak_dict[args.curve]
U_res_peak = U_res[indexes_peaks_U_res]
B_res_peak = B_res[indexes_peaks_U_res]

i_B_periodicity = np.arange(len(B_res_peak)) * 2
one_over_B_periodicity = 1 / B_res_peak

one_over_B_period, _ = np.polyfit(i_B_periodicity, one_over_B_periodicity, 1)

ax_B_periodicity.plot(i_B_periodicity, one_over_B_periodicity, 'xk')
ax_B_periodicity.set_xlabel('$i$')
ax_B_periodicity.set_ylabel('$\\frac{1}{B}$ (T$^{-1}$)')
ax_B_periodicity.set_title('absolute value of i neither correct nor relevant')


electron_charge = 1.602e-19
planck = 6.626e-34
charge_carrier_density = 2 * electron_charge / (planck * one_over_B_period)

# print U_res peaks

# get U_h plateau voltages
current_dict = {
	'4K-20uA': 20e-6,
	'2K-20uA': 20e-6,
	'2K-100uA': 100e-6
}
current = current_dict[args.curve]
hall_resistance = 25.81281e3
# dict contents: (index_min, index_max)
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
indexes_plateau_U_hall = plateau_dict[args.curve]


data_hall_regression_i = []
data_hall_regression_1_over_R_hall = []
data_hall_regression_1_over_R_hall_err = []

#I don't even care anymore at this point
B_min_arr = []
B_max_arr = []
U_mean_arr = []
U_std_arr = []
R_hall_arr = []
R_hall_err_arr = []
i_arr = []
i_actual_arr = []

for (plat_start, plat_end) in indexes_plateau_U_hall:
	U_hall_plat = U_hall_unfiltered[plat_start:plat_end]
	B_hall_plat = B_hall_unfiltered[plat_start:plat_end]
	U_hall_plat_f = U_hall[plat_start:plat_end]
	B_hall_plat_f = B_hall[plat_start:plat_end]

	B_min = np.min(B_hall_plat)
	B_min_arr.append(B_min)
	B_max = np.max(B_hall_plat)
	B_max_arr.append(B_max)
	U_mean = np.mean(U_hall_plat)
	U_mean_arr.append(U_mean)
	U_std = np.std(U_hall_plat)
	U_std_arr.append(U_std)
	R_hall = U_mean * 1e-3 / current
	R_hall_arr.append(R_hall)
	R_hall_err = U_std * 1e-3 / current
	R_hall_err_arr.append(R_hall_err)
	i = hall_resistance / R_hall
	i_arr.append(i)
	i_actual = int(2 * round(i / 2))
	i_actual_arr.append(i_actual)

	axL.annotate(i_actual, xy=(np.mean([np.min(B_hall_plat_f), np.max(B_hall_plat_f)]), np.mean(U_hall_plat_f) + 10),ha="center",va="bottom")

	data_hall_regression_i.append(i_actual)
	data_hall_regression_1_over_R_hall.append(1 / R_hall)
	data_hall_regression_1_over_R_hall_err.append(R_hall_err / (R_hall**2))

	# draw ellipse
	# el = matplotlib.patches.Ellipse(xy=(np.mean(B_hall_plat), np.mean(
	# 	U_hall_plat)), width=(0.5 + np.max(B_hall_plat) - np.min(B_hall_plat)), height=10, fill=False, linewidth=2)
	# axL.add_artist(el)

	# mark start + end
	axL.vlines(np.min(B_hall_plat_f), np.mean(
		U_hall_plat) - 10, np.mean(U_hall_plat_f) + 10)
	axL.vlines(np.max(B_hall_plat_f), np.mean(
		U_hall_plat) - 10, np.mean(U_hall_plat_f) + 10)
	# TODO: prettify that

# do linear regression of hall voltages
data_hall_regression_i = np.array(data_hall_regression_i)
data_hall_regression_1_over_R_hall = np.array(
	data_hall_regression_1_over_R_hall)

slope_1_over_R_hall_over_i, offset_1_over_R_hall_over_i = np.polyfit(
	data_hall_regression_i, data_hall_regression_1_over_R_hall, 1)


dataset_1_over_R_hall_over_i = kafe.Dataset(data=[data_hall_regression_i, data_hall_regression_1_over_R_hall])
dataset_1_over_R_hall_over_i.add_error_source('y', 'simple', data_hall_regression_1_over_R_hall_err)
fit_1_over_R_hall_over_i = kafe.Fit(dataset_1_over_R_hall_over_i, linear_2par)
fit_1_over_R_hall_over_i.do_fit(quiet=True)
# my_p = kafe.Plot(fit_1_over_R_hall_over_i)
# my_p.plot_all()
# my_p.show()
slope_1_over_R_hall_over_i_err = fit_1_over_R_hall_over_i.get_results()[0][1]

R_k_measured = 1 / slope_1_over_R_hall_over_i

# alpha_measured = 299792458 * 4 * np.pi * 1e-7 / (2 * R_k_measured)
alpha_measured = slope_1_over_R_hall_over_i * 299792458 * 4 * np.pi * 1e-7 / 2
alpha_measured_err = slope_1_over_R_hall_over_i_err * 299792458 * 4 * np.pi * 1e-7 / 2

ax_R_hall.plot(data_hall_regression_i,
			   data_hall_regression_1_over_R_hall * 1e3, 'xk')
ax_R_hall.plot(data_hall_regression_i, (offset_1_over_R_hall_over_i +
			   slope_1_over_R_hall_over_i * data_hall_regression_i)*1e3)
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


def genTables():

	dick = '\\begin{tabular}{S|S}\n\\toprule\nB (\\si{\\tesla})\t&\tU (\\si{\\mV})\t\\\\\n'
	dick += '\\midrule\n'
	for (B, U) in zip(B_res_peak, U_res_peak):
		dick += (f'{B:0.2f}\t&\t{U:0.0f} \\\\\n')
	dick += '\\bottomrule\n\\end{tabular}'

	print(dick)

	dick = '\\begin{tabular}{SS|SS|S|S}\n\\toprule\nB_\\text{min} (\\si{\\tesla})\t&\tB_\\text{max} (\\si{\\tesla})\t&\t\\bar{U_\\text{h}} (\\si{\\mV})\t&\t\\Delta U_\\text{h} (\\si{\\mV})\t&\ti\t&\tR_\\text{h} (\\si{\\ohm})\t\\\\\n'
	dick += '\\midrule\n'
	for (B_min, B_max, U_mean, U_std, i, i_actual, R_hall) in zip(B_min_arr, B_max_arr, U_mean_arr, U_std_arr, i_arr, i_actual_arr, R_hall_arr):
		dick += (f'{B_min:0.2f}\t&\t{B_max:0.2f}\t&\t{U_mean:0.0f}\t&\t{U_std:0.1f}\t&\t{i:0.2f}\t&\t{R_hall:0.1f} \\\\\n')
	dick += '\\bottomrule\n\\end{tabular}'

	print(dick)


showPlots = True

if args.table:
	genTables()
	showPlots = False

if args.output_main:
	fig.savefig(args.output_main)
	showPlots = False
if args.output_plateaus:
	fig_R_hall.savefig(args.output_plateaus)
	showPlots = False
if args.output_1oB:
	fig_B_periodicity.savefig(args.output_1oB)
	showPlots = False

if showPlots:
	print(
		f'magnetic field scaling factor: {factor_B:0.4f}; offset: {min_B:0.2f}T')
	print(f'Delta (1/B) = {one_over_B_period:0.2f} T^-1')
	print(f'charge carrier density: {charge_carrier_density * 1e-4:0.3e} 1/m²')
	print('peaks of logitudinal voltage:')
	print('B (T)\tU (mV)')
	for (B, U) in zip(B_res_peak, U_res_peak):
		print(f'{B:0.2f}\t{U:0.0f}')

	print('\nhall plateaus:')
	print('B_min (T)\tB_max (T)\tU_h_mean (mV)\tU_h_std (mV)\t(i)\ti\tR_h (ohm)')
	for (B_min, B_max, U_mean, U_std, i, i_actual, R_hall) in zip(B_min_arr, B_max_arr, U_mean_arr, U_std_arr, i_arr, i_actual_arr, R_hall_arr):
		print(f'{B_min:0.2f}\t\t{B_max:0.2f}\t\t{U_mean:0.0f}\t\t{U_std:0.1f}\t\t{i:0.2f}\t{i_actual:d}\t{R_hall:0.1f}')
	print(
		f'1/(R_h * i) = {slope_1_over_R_hall_over_i*1e6:0.2f} uS ^= {R_k_measured:0.1f}ohm')
	print(f'1/alpha = {1/alpha_measured:0.2f}')
	print(f'1/alpha err = {alpha_measured_err/(alpha_measured**2):0.2f}')
	plt.show()
