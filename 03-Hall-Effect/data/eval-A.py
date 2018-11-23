#!/usr/bin/python
import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('plot', type=str, choices=['2.1', '2.2'],
                    help='which plot to draw')
parser.add_argument('--output', type=str,
                    help='output file')

args = parser.parse_args()


temperature, current, voltage_long, voltage_hall_pos, voltage_hall_neg = np.loadtxt(
    f'src/data-A.dat', unpack=True)

temperature += 273.15
current *= 1e-3
voltage_long *= 1
voltage_hall_pos *= 1e-3
voltage_hall_neg *= 1e-3

magnetic_field = 0.5

resistance = voltage_long / current
resistance_hall = (voltage_hall_pos - voltage_hall_neg) / (2 * current)

hall_coefficient = - resistance_hall / (1e-3 * magnetic_field)
sheet_resistance = resistance * 10 / 19
specific_resistance = sheet_resistance / 1e-3
conductivity = 1 / specific_resistance

temperature_extrinsic_max = 160
temperature_intrinsic_min = 288

if args.plot == '2.1':
	plt.plot(temperature, conductivity, 'x', label='conductivity $\\sigma$')
	plt.plot(temperature, 1 / hall_coefficient, 'o',
	         label='hall coefficient $\\frac{1}{|R_\\mathrm{H}|}$')
	plt.ylabel('conductivity (S/m)')
	plt.xlabel('temperature (K)')

	plt.vlines(temperature_intrinsic_min, 1e-6, 2e-5)
	plt.vlines(temperature_extrinsic_max, 1.3e-6,
	           5e-6, colors='C0', linestyles='dotted')

	plt.text(temperature_intrinsic_min + 3, 1.3e-6, 'purely intrinsic')
	plt.text(temperature_intrinsic_min - 3, 1.3e-5,
	         'transition region', horizontalalignment='right')

	ax = plt.gca()
	ax.set_yscale("log", nonposy='clip')

elif args.plot == '2.2':
	plt.plot(temperature, conductivity
	         * hall_coefficient, 'x', label='$\\sigma\\times R_{H}$')

	plt.vlines(temperature_extrinsic_max, 0.5, 1)
	plt.vlines(temperature_intrinsic_min, 0.14, 0.3,
	           colors='C0', linestyles='dotted')

	plt.text(temperature_extrinsic_max - 3, 0.55,
	         'purely extrinsic', horizontalalignment='right')
	plt.text(temperature_extrinsic_max + 3, 0.93, 'transition region')

	ax = plt.gca()
	ax.set_xscale("log", nonposx='clip')
	ax.set_yscale("log", nonposy='clip')
	xformatter = matplotlib.ticker.FormatStrFormatter('%0.0f')
	ax.get_xaxis().set_minor_formatter(xformatter)
	ax.get_xaxis().set_major_formatter(xformatter)

plt.grid(which='both')
plt.legend()
if args.output:
	plt.savefig(args.output)
else:
	plt.show()
