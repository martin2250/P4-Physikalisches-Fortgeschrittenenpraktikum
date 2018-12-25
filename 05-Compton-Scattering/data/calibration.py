#!/usr/bin/python
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize


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


def energy_to_channel_estimation(energy):
	# ch 28: 122keV
	# ch 362: 1132keV
	return (energy - 122e3) * (362 - 28) / (1332e3 - 122e3) + 28


verbose = False
if __name__ == "__main__":
	verbose = True

################################################################################

# load spectrum of each isotope
for isotope in isotopes:
	data = np.load(f'src/{isotope.name}.npz')
	isotope.count = data['count']
	data.close()
	del data

channel = np.arange(len(isotopes[0].count))

################################################################################


def gaussian(energy, energy_center, height, width):
	return height / np.sqrt(2 * np.pi) / width * np.exp(-0.5 * (energy - energy_center)**2 / width**2)


# fit gaussian to every known peak
for isotope in isotopes:
	# except for Na-22, because Na-22 sucks
	if isotope.name == 'Na-22':
		continue

	for peak in isotope.peaks:
		guess_channel = int(energy_to_channel_estimation(peak.energy))
		guess = (guess_channel, isotope.count[guess_channel], 2 * peak.fitradius)

		fitrange = slice(guess_channel - peak.fitradius,
		                 guess_channel + peak.fitradius)

		peak.channel_fit = channel[fitrange]
		peak.count_fit = isotope.count[fitrange]

		peak.popt, pcov = scipy.optimize.curve_fit(
			gaussian, peak.channel_fit, peak.count_fit, guess, sigma=np.sqrt(peak.count_fit), absolute_sigma=True)

		peak.center = peak.popt[0]
		peak.error = np.sqrt(pcov[0, 0])

		if verbose:
			print(
				f'{isotope.name}, {peak.energy/1e6:0.3f} MeV, channel guess: {guess_channel} fit: {peak.center:0.2f} uncertainty: {peak.error:0.2f}')

################################################################################


def main():
	for isotope in isotopes:
		# if isotope.name == 'Co-57':
			#isotope.count = isotope.count / 3
		line = plt.plot(isotope.count, label=isotope.name)[0]
		plt.vlines([energy_to_channel_estimation(peak.energy)
	             for peak in isotope.peaks], 0, 5000)
		for peak in isotope.peaks:
			plt.annotate(f'{peak.energy/1e3:0.1f} keV',
			             (energy_to_channel_estimation(peak.energy) + 1, 20), color=line.get_color(), rotation='vertical', fontsize=13)
			if peak.center:
				plt.plot(peak.channel_fit, gaussian(peak.channel_fit, *peak.popt))

	################################################################################
	plt.gca().set_yscale("log", nonposy='clip')
	plt.ylim(10, 12000)
	plt.legend()
	plt.xlabel('channel')
	plt.ylabel('count')
	plt.show()


if __name__ == "__main__":
	main()
