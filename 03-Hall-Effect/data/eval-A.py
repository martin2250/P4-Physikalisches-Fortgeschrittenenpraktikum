#!/usr/bin/python
import argparse

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--output', type=str,
                    help='output file')

args = parser.parse_args()


temperature, current, voltage_long, voltage_hall_pos, voltage_hall_neg = np.loadtxt(
    f'src/data-A.dat', unpack=True)

current *= 1e-3
voltage_long *= 1
voltage_hall_pos *= 1e-3
voltage_hall_neg *= 1e-3

magnetic_field = 0.5

resistance = voltage_long / current
resistance_hall = (voltage_hall_pos - voltage_hall_neg) / (2 * current)

hall_coefficient = - resistance_hall * 1e-3 / magnetic_field
sheet_resistance = resistance * 10 / 19
specific_resistance = sheet_resistance / 1e-3
conductivity = 1 / specific_resistance

if False:
	plt.plot(temperature, conductivity, 'x', label='conductivity')
	plt.plot(temperature, 1 / hall_coefficient, 'o', label='hall coefficient')

	ax = plt.gca()
	ax.set_yscale("log", nonposy='clip')
else:
	plt.plot(temperature, conductivity *
	         hall_coefficient, 'x', label='conductivity')

	ax = plt.gca()
	ax.set_xscale("log", nonposx='clip')
	ax.set_yscale("log", nonposy='clip')

plt.grid(which='both')
plt.legend()
if args.output:
	plt.savefig(args.output)
else:
	plt.show()
