#!/usr/bin/python
import argparse
import bisect

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import scipy
import scipy.stats

import constants

parser = argparse.ArgumentParser()

parser.add_argument('plot', type=str, choices=['2.1', '2.2', '4', '5+6'],
                    help='which plot to draw')
parser.add_argument('--output', type=str,
                    help='output file')
parser.add_argument('--size', type=float, nargs=2,
                    help='size X Y (inch)')
parser.add_argument('--verbose', action="store_true",
                    help='enable verbose output')

args = parser.parse_args()

if args.size:
	fig = matplotlib.pyplot.gcf()
	fig.set_size_inches(*args.size)


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

temperature_extrinsic_max = 268
temperature_intrinsic_min = 320
alpha = 4e-4  # eV/K
################################################################################


def b(T):
	return 1.24553 + 0.00107 * T


def n(R, T):
	return 1 / (constants.e * R) * (1 - b(T)) / (1 + b(T))


def bandgap(Eg, T):
	return Eg - alpha * T


################################################################################
# limit to intrinsic region
limit = bisect.bisect(temperature, temperature_intrinsic_min)
limited_temp = temperature[limit:]
limited_hall_coeff = hall_coefficient[limit:]
################################################################################
if args.plot == '2.1':
	plt.plot(temperature, conductivity, 'x',
	         label='conductivity $\\sigma$ in S/m')
	plt.plot(temperature, 1 / hall_coefficient, 'o',
          label='inverse hall coefficient $\\frac{1}{|R_\\mathrm{H}|}$ in m$^3$/C')
	plt.ylabel('')
	plt.xlabel('temperature $T$ (K)')

	plt.vlines(temperature_intrinsic_min, 3, 90)
	plt.vlines(temperature_extrinsic_max, 1, 7, colors='C0', linestyles='dotted')

	plt.text(temperature_intrinsic_min + 3, 3.6, 'purely intrinsic')
	plt.text(temperature_intrinsic_min - 3, 75,
          'transition region', horizontalalignment='right')

	ax = plt.gca()
	ax.set_yscale("log", nonposy='clip')
	plt.legend()
################################################################################
elif args.plot == '2.2':
	plt.plot(temperature, conductivity * hall_coefficient, 'x')

	plt.vlines(temperature_extrinsic_max, 0.2, 0.4)
	plt.vlines(temperature_intrinsic_min, 0.1, 0.2,
            colors='C0', linestyles='dotted')

	plt.text(temperature_extrinsic_max - 3, 0.2,
          'purely extrinsic', horizontalalignment='right')
	plt.text(temperature_extrinsic_max + 3, 0.4, 'transition region')

	plt.plot([145, 264], [0.94, 0.315], 'C1')
	plt.plot([322, 427], [0.141, 0.079], 'C1')

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
		n_at_300K = np.interp(300, temperature, -n(hall_coefficient, temperature))
		print(f'n_i(300 K) = {n_at_300K:0.3e}')

	ax = plt.gca()
	# ax.set_xscale("log", nonposx='clip')
	ax.set_yscale("log", nonposy='clip')
	ax.set_xlabel('temperature (K)')
	ax.set_ylabel('$\\log\\ n_{i}\\ (m^{-3})$')

	xformatter = matplotlib.ticker.FormatStrFormatter('%0.0f')
	ax.get_xaxis().set_minor_formatter(xformatter)
	ax.get_xaxis().set_major_formatter(xformatter)
################################################################################
elif args.plot == '5+6':
	# Arrhenius representation for band gap
	def linear(p, x):
		return p[0] * x + p[1]

	import scipy.odr  # too lazy to scroll up

	data = scipy.odr.RealData(
            1 / limited_temp, np.log(-n(limited_hall_coeff, limited_temp) / limited_temp**(3 / 2)))  # , sx=(1 / limited_temp**2))

	odr = scipy.odr.ODR(data, scipy.odr.Model(
            linear), beta0=(-4458.922336972104, 50.90372804322858))

	result = odr.run()

	slp, inter = result.beta
	slp_err, inter_err = result.sd_beta

	slpb, interb, r, __, __ = scipy.stats.linregress(
            1 / limited_temp, np.log(-n(limited_hall_coeff, limited_temp) / limited_temp**(3 / 2)))

	plt.plot(1 / limited_temp, np.log(-n(limited_hall_coeff,
                                      limited_temp) / limited_temp**(3. / 2)), '+', markersize=12, label='data')
	plt.plot(1 / limited_temp, slp * 1
          / limited_temp + inter, '--', label=f'linear fit\n $R^2={r**2:.4f}$', color='red')
	plt.xlabel('$\\frac{1}{T} \\left(\\frac{1}{K}\\right)$')
	plt.ylabel('$\\log\\ \\frac{n_\\mathrm{i}}{T^{\\frac{3}{2}}}$')

	if args.verbose:
		Eg = -slp * 2 * constants.k / constants.e  # in eV
		Eg_err = slp_err * 2 * constants.k / constants.e  # in eV

		print(
                    f'Eg_0 = ({Eg:.3f} +- {Eg_err:.3f}) eV, offset = {inter:.3f} +- {inter_err:.3f}')
		print(f'band gap energy (300K)= {bandgap(Eg, 300):.3f}')
		n_i_300 = np.exp(inter + slp * (1 / 300)) * 300**(3 / 2)
		n_i_300_err = (slp_err / 300 + inter_err) * n_i_300
		print(f'n_i (300K) = ({n_i_300:0.3e} +- {n_i_300_err:0.3e}) 1/m^3')
	plt.plot()
	plt.legend()
################################################################################
plt.grid(which='both')
plt.tight_layout()

if args.output:
	plt.savefig(args.output, bbox_inches="tight", pad_inches=0.02)
else:
	plt.show()
