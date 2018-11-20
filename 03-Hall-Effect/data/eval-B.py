#!/usr/bin/python
import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('plot', type=int, choices=[1, 2],
                    help='which plot to draw')
parser.add_argument('--output', type=str,
                    help='output file')

args = parser.parse_args()


temperature, current, voltage_long, voltage_hall_pos, voltage_hall_neg = np.loadtxt(
    f'src/data-B.dat', unpack=True)

temperature += 273.15
current *= 1e-6
voltage_long *= 1e-3
voltage_hall_pos *= 1e-3
voltage_hall_neg *= 1e-3

magnetic_field = 0.5

delta_voltage_BD = (voltage_hall_pos - voltage_hall_neg) / 2

hall_coefficient = np.abs((delta_voltage_BD / current) * (1 / magnetic_field))
conductivity = (np.log(2) / np.pi) * (current / voltage_long)

if args.plot == 1:

    plt.plot(temperature, conductivity * hall_coefficient, 'x')

    ax = plt.gca()
    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    ax.set_xlabel('$\\log\\ T$ (K)')
    ax.set_ylabel('$\\log\\ \\mu$ ($\\frac{m^2}{Vs}$)')
    xformatter = matplotlib.ticker.FormatStrFormatter('%0.0f')
    ax.get_xaxis().set_minor_formatter(xformatter)
    ax.get_xaxis().set_major_formatter(xformatter)

elif args.plot == 2:

    pass

plt.grid(which='both')
plt.legend()

if args.output:
    plt.savefig(args.output)
else:
    plt.show()
