#!/usr/bin/python
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.odr

from calibration import channel, channel_to_energy, gaussian, linear
from datasets import datasets

################################################################################


def center_estimate(angle):
	index = int((angle - 25) / 10)
	return [158, 143, 127, 111, 97, 87, 75, 67, 60][index]


for dataset in datasets:
	dataset.center = center_estimate(dataset.angle)

################################################################################

fitradius = 16


def fit_gaussian():
	for dataset in datasets:
		guess_channel = int(dataset.center)
		guess = (guess_channel, dataset.count_diff[guess_channel], fitradius)

		fitrange = slice(guess_channel - fitradius,
	                  guess_channel + fitradius)

		dataset.channel_fit = channel[fitrange]
		dataset.count_fit = dataset.count_diff[fitrange]

		data = scipy.odr.RealData(
			dataset.channel_fit, dataset.count_fit, sy=np.sqrt(dataset.count_fit))

		odr = scipy.odr.ODR(data, scipy.odr.Model(
			lambda p, x: gaussian(x, *p)), beta0=guess)

		result = odr.run()

		dataset.popt = result.beta
		dataset.center = result.beta[0]
		dataset.center_error = result.sd_beta[0]
		dataset.center_energy, dataset.center_energy_error = channel_to_energy(
			dataset.center, dataset.center_error)


# run this two times to improve center guess as it is used for selecting the fit range
fit_gaussian()
fit_gaussian()

################################################################################

# plot individual fits
if False:
	for dataset in datasets:
		print(f'{dataset.angle}°: ({center_energy/1e3:0.3e} +- {center_energy_error/1e3:0.3e}) keV')
		plt.plot(dataset.count_diff, label=f'{dataset.angle}')
		plt.plot(dataset.channel_fit, gaussian(dataset.channel_fit, *dataset.popt))
		plt.legend()
		plt.show()

################################################################################

angles = np.zeros(len(datasets))
energies = np.zeros(len(datasets))
energy_errors = np.zeros(len(datasets))

for (i, dataset) in enumerate(datasets):
	angles[i] = dataset.angle
	energies[i] = dataset.center_energy
	energy_errors[i] = dataset.center_energy_error

################################################################################
# linearize data

X = (1 - np.cos(angles * np.pi / 180))
Y = 1 / energies
Y_err = energy_errors / energies**2

################################################################################


def fit_final():
	data = scipy.odr.RealData(X, Y, sx=Y_err)

	odr = scipy.odr.ODR(data, scipy.odr.Model(
		lambda p, x: linear(x, *p)), beta0=[2e-3, 1e-3])

	out = odr.run()

	return out.beta, out.sd_beta


(slope, intercept), (slope_error, intercept_error) = fit_final()

################################################################################

electron_mass_eV = 1 / slope
electron_mass_eV_err = slope_error / slope**2

if len(sys.argv) < 2:
	print(
		f'fit coefficients: a={slope:.4e} +/- {slope_error:.4e} 1/eV, b={intercept:.4e} +/- {intercept_error:.4e} 1/eV')
	print(f'electron mass: {electron_mass_eV*1e-3:0.2f} keV')
	print(f'electron mass error: {electron_mass_eV_err*1e-3:0.2f} keV')

################################################################################

fig, ax = plt.subplots(constrained_layout=True)

ax.errorbar(X, Y * 1e6, Y_err * 1e6, fmt='none', label='experimental data')
ax.plot(X, (X * slope + intercept) * 1e6, label='linear fit')

ax.legend()
ax.set_ylabel('$E^{-1}$ ($E$ in MeV)')
ax.set_xlabel(r'$(1 - \cos(\Theta))$')
plt.grid()

if len(sys.argv) < 2:
	plt.show()
else:
	plt.savefig(sys.argv[1])
