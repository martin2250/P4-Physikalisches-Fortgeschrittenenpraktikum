#!/usr/bin/python
import argparse
import bisect

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

import constants

parser = argparse.ArgumentParser()

parser.add_argument('plot', type=str, choices=['2.1', '2.2', '4', '5'],
					help='which plot to draw')
parser.add_argument('--output', type=str,
					help='output file')
parser.add_argument('--verbose', action="store_true",
					help='enable verbose output')

args = parser.parse_args()


temperature, current, voltage_long, voltage_hall_pos, voltage_hall_neg = np.loadtxt(
	f'src/data-A.dat', unpack=True)
################################################################################
temperature += 273.15
current *= 1e-3
voltage_long *= 1
voltage_hall_pos *= 1e-3
voltage_hall_neg *= 1e-3

magnetic_field = 0.5
################################################################################
resistance = voltage_long / current
resistance_hall = (voltage_hall_pos - voltage_hall_neg) / (2 * current)

hall_coefficient = - resistance_hall * 1e-3 / (magnetic_field)
sheet_resistance = resistance * 10 / 19
specific_resistance = sheet_resistance * 1e-3
conductivity = 1 / specific_resistance

temperature_extrinsic_max = 160
temperature_intrinsic_min = 288
################################################################################


def b(T):
	return 1.24553 + 0.00107 * T


def n(R, T):
	return 1 / (constants.e * R) * (1 - b(T)) / (1 + b(T))


################################################################################
# limit to intrinsic region
limit = bisect.bisect(temperature, temperature_intrinsic_min)
limited_temp = temperature[limit:]
limited_hall_coeff = hall_coefficient[limit:]
################################################################################
if args.plot == '2.1':
	plt.plot(temperature, conductivity, 'x', label='conductivity $\\sigma$')
	plt.plot(temperature, 1 / hall_coefficient, 'o',
			label='hall coefficient $\\frac{1}{|R_\\mathrm{H}|}$')
	plt.ylabel('conductivity (S/m)')
	plt.xlabel('temperature (K)')

	plt.vlines(temperature_intrinsic_min, 1, 2e1)
	plt.vlines(temperature_extrinsic_max, 1.3,
			5, colors='C0', linestyles='dotted')

	plt.text(temperature_intrinsic_min + 3, 1.3, 'purely intrinsic')
	plt.text(temperature_intrinsic_min - 3, 1.3e1,
			'transition region', horizontalalignment='right')

	ax = plt.gca()
	ax.set_yscale("log", nonposy='clip')
	plt.legend()
################################################################################
elif args.plot == '2.2':
	plt.plot(temperature, conductivity * hall_coefficient, 'x')

	plt.vlines(temperature_extrinsic_max, 0.5, 1)
	plt.vlines(temperature_intrinsic_min, 0.14, 0.3,
			colors='C0', linestyles='dotted')

	plt.text(temperature_extrinsic_max - 3, 0.55,
			'purely extrinsic', horizontalalignment='right')
	plt.text(temperature_extrinsic_max + 3, 0.93, 'transition region')

	ax = plt.gca()
	ax.set_xscale("log", nonposx='clip')
	ax.set_yscale("log", nonposy='clip')
	ax.set_xlabel('$\\log\\ T$ (K)')
	ax.set_ylabel('$\\log(\\sigma\\times |R_H|)$ ($\\frac{m^{2}}{Vs}$)')
	xformatter = matplotlib.ticker.FormatStrFormatter('%0.0f')
	ax.get_xaxis().set_minor_formatter(xformatter)
	ax.get_xaxis().set_major_formatter(xformatter)
################################################################################
elif args.plot == '4':
	plt.plot(limited_temp, - n(limited_hall_coeff, limited_temp), 'x')

	if args.verbose:
		print(f'n_i(300 K) = {-n(hall_coefficient[21], temperature[21]):0.3e}')

	ax = plt.gca()
	#ax.set_xscale("log", nonposx='clip')
	ax.set_yscale("log", nonposy='clip')
	ax.set_xlabel('temperature (K)')
	ax.set_ylabel('$\\log\\ n_{i}\\ (m^{-3})$')

	xformatter = matplotlib.ticker.FormatStrFormatter('%0.0f')
	ax.get_xaxis().set_minor_formatter(xformatter)
	ax.get_xaxis().set_major_formatter(xformatter)
################################################################################
elif args.plot == '5':
	# Arrhenius representation for band gap
	slp, inter = np.polyfit(
			1 / limited_temp, np.log(-n(limited_hall_coeff, limited_temp) / limited_temp**(3. / 2)), 1)
	plt.plot(1 / limited_temp, np.log(-n(limited_hall_coeff,
									  limited_temp) / limited_temp**(3. / 2)), '+')
	plt.plot(1 / limited_temp, slp * 1 /
		  limited_temp + inter, '--', color='red', )
	plt.xlabel('$\\frac{1}{T} \\left(\\frac{1}{K}\\right)$')
	plt.ylabel('$\\log\\ \\frac{n_i}{T^{\\frac{3}{2}}}$')

	if args.verbose:
		Eg = -slp * 2 * constants.k
		print(Eg)
		print(f'Eg_0 = {Eg * 6.2415091e18:.2f} eV, offset = {inter:.2f}')
	plt.plot()
################################################################################
plt.grid(which='both')

if args.output:
	plt.savefig(args.output)
else:
	plt.show()
