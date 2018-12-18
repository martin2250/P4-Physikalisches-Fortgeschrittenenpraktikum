#!/usr/bin/python
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize


@dataclass
class Isotope:
	name: str
	energies: np.ndarray  # in eV
	count: np.ndarray = None


isotopes = [
	# http://www.spectrumtechniques.com/products/sources/cobalt-57/
	Isotope('Co-57', np.array([122.0614e3])),
	# https://www.gammaspectacular.com/Co-60-gamma-spectrum
	Isotope('Co-60', np.array([1173.2e3, 1332.5e3])),
	# http://www.spectrumtechniques.com/products/sources/cesium-137/
	# (283.53e3)
	Isotope('Cs-137', np.array([661.657e3])),
	# https://ehs.umich.edu/wp-content/uploads/2016/04/Sodium-22.pdf
	Isotope('Na-22', np.array([511e3, 1274.5e3]))
]


def energy_to_channel_estimation(energy):
	# ch 28: 122keV
	# ch 362: 1132keV
	return (energy - 122e3) * (362 - 28) / (1332e3 - 122e3) + 28

################################################################################


for isotope in isotopes:
	data = np.load(f'src/{isotope.name}.npz')
	isotope.count = data['count']
	data.close()
	del data

for isotope in isotopes:
	if isotope.name == 'Co-57':
		isotope.count = isotope.count / 3
	line = plt.plot(isotope.count, label=isotope.name)[0]
	plt.vlines([energy_to_channel_estimation(e)
             for e in isotope.energies], 0, 5000)
	for e in isotope.energies:
		plt.annotate(f'{e/1e3:0.1f} keV',
		             (energy_to_channel_estimation(e) + 1, 20), color=line.get_color(), rotation='vertical', fontsize=13)

################################################################################
plt.gca().set_yscale("log", nonposy='clip')
plt.ylim(10, 6000)
plt.legend()
plt.xlabel('channel')
plt.ylabel('count')
plt.show()
