#!/usr/bin/python
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.odr

_verbose = False
if __name__ == "__main__":
	_verbose = True

################################################################################


@dataclass
class Peak:
	energy: float
	fitradius: int = 20
	popt: np.ndarray = None
	center: float = None
	error: float = None
	channel_fit: np.ndarray = None
	count_fit: np.ndarray = None


@dataclass
class Isotope:
	name: str
	peaks: list
	count: np.ndarray = None


isotopes = [
	# http://www.spectrumtechniques.com/products/sources/cobalt-57/
	Isotope('Co-57', [Peak(122.0614e3, 7)]),
	# https://www.gammaspectacular.com/Co-60-gamma-spectrum
	Isotope('Co-60', [Peak(1173.2e3), Peak(1332.5e3)]),
	# http://www.spectrumtechniques.com/products/sources/cesium-137/
	# (283.53e3)
	Isotope('Cs-137', [Peak(661.657e3)]),
	# https://ehs.umich.edu/wp-content/uploads/2016/04/Sodium-22.pdf
	Isotope('Na-22', [Peak(511e3), Peak(1274.5e3)])
]

# load spectrum of each isotope
for isotope in isotopes:
	with np.load(f'src/{isotope.name}.npz') as data:
		isotope.count = data['count']
	del data

channel = np.arange(len(isotopes[0].count))

################################################################################
# model functions


def linear(x, slope, intercept):
	return intercept + x * slope


def gaussian(energy, energy_center, height, width):
	return height * np.exp(-0.5 * (energy - energy_center)**2 / width**2)


################################################################################

# ch 28: 122keV
# ch 362: 1132keV
etc_slope_guess = (362 - 28) / (1332e3 - 122e3)
etc_intercept_guess = 28 - 122e3 * (362 - 28) / (1332e3 - 122e3)


def energy_to_channel_estimation(energy):
	return linear(energy, etc_slope_guess, etc_intercept_guess)

################################################################################
# fit gaussian to every known peak


def fit_gaussians_to_isotopes():
	for isotope in isotopes:
		# except for Na-22, because Na-22 sucks
		if isotope.name == 'Na-22':
			continue

		for peak in isotope.peaks:
			guess_channel = int(energy_to_channel_estimation(peak.energy))
			guess = (guess_channel, isotope.count[guess_channel], peak.fitradius)

			fitrange = slice(guess_channel - peak.fitradius,
	                    guess_channel + peak.fitradius)

			peak.channel_fit = channel[fitrange]
			peak.count_fit = isotope.count[fitrange]

			data = scipy.odr.RealData(
				peak.channel_fit, peak.count_fit, sy=np.sqrt(peak.count_fit))

			odr = scipy.odr.ODR(data, scipy.odr.Model(
				lambda p, x: gaussian(x, *p)), beta0=guess)

			result = odr.run()

			peak.popt = result.beta
			peak.center = result.beta[0]
			peak.error = result.sd_beta[0]

			if _verbose:
				print(
					f'{isotope.name},\t{peak.energy/1e6:0.3f} MeV,\tchannel guess: {guess_channel}\tfit: {peak.center:0.2f}\tuncertainty: {peak.error:0.2f}')


fit_gaussians_to_isotopes()

################################################################################
# perform linear regression to get energy scale


def fit_energy_scale():
	energies = []
	channels = []
	channel_uncertainties = []

	for isotope in isotopes:
		for peak in isotope.peaks:
			if peak.center:
				energies.append(peak.energy)
				channels.append(peak.center)
				channel_uncertainties.append(peak.error)

	data = scipy.odr.RealData(channels, energies, sx=channel_uncertainties)

	odr = scipy.odr.ODR(data, scipy.odr.Model(
		lambda p, x: linear(x, *p)), beta0=[1 / etc_slope_guess, 0])

	out = odr.run()

	return out.beta, out.sd_beta


(cte_slope, cte_intercept), (cte_slope_error,
                             cte_intercept_error) = fit_energy_scale()

if _verbose:
	print('energy = channel * slope + offset')
	print(
		f'slope = ({cte_slope*1e-3:0.3e} +- {cte_slope_error*1e-3:0.3e}) keV/ch')
	print(
		f'offset = ({cte_intercept*1e-3:0.3e} +- {cte_intercept_error*1e-3:0.3e}) keV')

################################################################################
# this function can be imported from other scripts


def channel_to_energy(channel, channel_error=0):
	energy = channel * cte_slope + cte_intercept
	error = np.sqrt((channel * cte_slope_error)**2
	                + (channel_error * cte_slope)**2 + (cte_intercept_error)**2)
	return energy, error


################################################################################


def main():
	fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 3))

	for isotope in isotopes:
		# if isotope.name == 'Co-57':
			#isotope.count = isotope.count / 3
		line = ax.plot(isotope.count, label=isotope.name)[0]
		ax.vlines([energy_to_channel_estimation(peak.energy)
                    for peak in isotope.peaks], 0, 5000)
		for peak in isotope.peaks:
			ax.annotate(f'{peak.energy/1e3:0.1f} keV',
                            (energy_to_channel_estimation(peak.energy) + 1, 10), color=line.get_color(), rotation=90, fontsize=9)
			if peak.center:
				ax.plot(peak.channel_fit, gaussian(peak.channel_fit, *peak.popt), '-k')

	################################################################################
	ax.set_yscale("log", nonposy='clip')
	ax.set_ylim(10, 12000)
	ax.legend()
	ax.grid()
	ax.set_xlabel('channel')
	ax.set_ylabel('count')

	if len(sys.argv) < 2:
		plt.show()
	else:
		fig.savefig(sys.argv[1])


if _verbose:
	main()
