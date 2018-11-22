#!/usr/bin/python
import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('plot', type=int, choices=[1, 2],
                    help='which plot to draw')
parser.add_argument('--output', type=str,
                    help='output file')

args = parser.parse_args()


temperature, current, voltage_long, voltage_hall_pos, voltage_hall_neg = np.loadtxt(
    f'src/data-B.dat', unpack=True)
temperatureA, currentA, voltage_longA, voltage_hall_posA, voltage_hall_negA = np.loadtxt(
    f'src/data-A.dat', unpack=True)

# mapping B
temperature += 273.15
current *= 1e-6
voltage_long *= 1e-3
voltage_hall_pos *= 1e-3
voltage_hall_neg *= 1e-3


# mapping A
temperatureA += 273.15
currentA *= 1e-3
voltage_longA *= 1
voltage_hall_posA *= 1e-3
voltage_hall_negA *= 1e-3

# constants
magnetic_field = 0.5

# computations sample A
resistanceA = voltage_longA / currentA
resistance_hallA = (voltage_hall_posA - voltage_hall_negA) / (2 * currentA)

hall_coefficientA = - resistance_hallA / (1e-3 * magnetic_field)
sheet_resistanceA = resistanceA * 10 / 19
specific_resistanceA = sheet_resistanceA / 1e-3
conductivityA = 1 / specific_resistanceA

# computations sample B
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

    mobilityA = hall_coefficientA[temperatureA
                                  <= 273.15] * conductivityA[temperatureA <= 273.15]
    mobility = conductivity * hall_coefficient

    fig, ax = plt.subplots()
    ax.plot(temperature, mobilityA,
            'x', label='mobility sample A')
    ax.plot(temperature, mobility, 'o', label='mobility sample B')

    ax.set_yscale("log", nonposy='clip')
    ax.set_xlabel('T (K)')
    ax.set_ylabel('$\\log\\ \\mu$ ($\\frac{m^2}{Vs}$)')
    xformatter = matplotlib.ticker.FormatStrFormatter('%0.0f')
    ax.get_xaxis().set_major_formatter(xformatter)
    plt.legend()

    # # created zoomed up inset plot
    # # zoom-factor: 2.5, location: upper-right
    # axins = ax.inset_axes([0.65, 0.5, 0.3, 0.3])
    # axins.plot(temperature, mobilityA, 'x')
    # axins.plot(temperature, mobility, 'o')
    # axins.set_xlim(90, 160)
    # axins.set_ylim(mobilityA[0]-0.8, mobility[0])
    # ax.indicate_inset_zoom(axins)
    # axins.set_xticklabels('')
    # axins.set_yticklabels('')

plt.grid(which='both', linestyle='--')

if args.output:
    plt.savefig(args.output)
else:
    plt.show()
