#!/usr/bin/python
import argparse

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('sample', type=str, choices=['A', 'B'],
                    help='sample')
parser.add_argument('--output', type=str,
                    help='output file')

args = parser.parse_args()


temperature, current, voltage_long, voltage_hall_pos, voltage_hall_neg = np.loadtxt(
    f'src/data-{args.sample}.dat', unpack=True)

current *= {'A': 1e-3, 'B': 1e-6}[args.sample]
voltage_long *= {'A': 1, 'B': 1e-3}[args.sample]
voltage_hall_pos *= 1e-3
voltage_hall_neg *= 1e-3

magnetic_field = 0.5

resistance = voltage_long / current
resistance_hall = (voltage_hall_pos - voltage_hall_neg) / (2 * current)

# only works form sample A
hall_coefficient = resistance_hall * {'A': 1e-3} / B

# sheet_resistance = resistance * {'A': 10 / 19, 'B': 1}[args.sample]

plt.plot(temperature, resistance)
plt.plot(temperature, resistance_hall)

if args.output:
	plt.savefig(args.output)
else:
	plt.show()
