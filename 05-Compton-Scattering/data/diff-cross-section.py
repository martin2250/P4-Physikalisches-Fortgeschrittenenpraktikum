#!/usr/bin/python
import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

import constants
from calibration import channel, channel_to_energy
from datasets import datasets
from detector import detector_inv_efficiency

bb_angles, bb_diff_cross = np.loadtxt(
	'src/bb-diff-cross.dat', usecols=[0, 2], unpack=True)

angles = []
counts = []

for dataset in datasets:
	angles.append(dataset.angle)
	channel = np.argmax(dataset.count_diff)
	energy, _ = channel_to_energy(channel)
	count = np.sum(dataset.count_diff) * detector_inv_efficiency(energy)
	counts.append(count)

angles = np.array(angles)
counts = np.array(counts)
counts_err = np.sqrt(counts)
time = 300  # s
rate = counts / time
rate_err = rate / counts * counts_err

# day not given, assume middle of july
date_inital = datetime.date(1971, 7, 15)
date_experiment = datetime.date(2018, 12, 17)
ratio_remaining = np.power(
	0.5, (date_experiment - date_inital).days / (30.17 * 365.25))

gamma_flux_at_target_initial = 1.54e6 * 1e4  # per (cm2*s), converted to m2s
gamma_flux_at_target_initial_err = 0.09e6 * \
	1e4  # per (cm2*s), converted to m2s
gamma_flux_at_target = gamma_flux_at_target_initial * ratio_remaining
gamma_flux_at_target_err = gamma_flux_at_target_initial_err * ratio_remaining

print(
	f'gamma flux at target: {gamma_flux_at_target:0.3e} +/- {gamma_flux_at_target_err:0.3e} 1/(s*m^2)')

length_target = 1  # cm
diameter_target = 1  # cm
size_target_err = 0.1  # cm
volume_target = length_target * np.pi * (diameter_target / 2)**2  # in cm3
volume_target_err = np.sqrt((volume_target / length_target) **
                            2 + (2 * volume_target / diameter_target)**2) * size_target_err
print(f'target volume: {volume_target:0.2f} +/- {volume_target_err:.2f} cm³')
atomic_weight_target = 26.981539  # Al, gram per mole
atomic_number_target = 13  # Al
density_target = 2.70  # Al, gram per cm3
number_electrons_target = constants.avogadro_constant / \
	atomic_weight_target * atomic_number_target * density_target * volume_target
number_electrons_target_err = number_electrons_target / \
	volume_target * volume_target_err

print(
	f'number of electrons: {number_electrons_target:0.3e} +/- {number_electrons_target_err:.3e}')

area_scintillator = np.pi * (2.55e-2 / 2)**2
distance_scintillator = 21.5e-2
solid_angle_scintillator = area_scintillator / \
	(distance_scintillator**2)

print(f'solid angle: {solid_angle_scintillator:0.3e}')

differential_cross_section = rate / \
	(solid_angle_scintillator * gamma_flux_at_target *
	 number_electrons_target)

differential_cross_section_err = np.sqrt(
	(differential_cross_section / number_electrons_target * number_electrons_target_err)**2 +
	(differential_cross_section / rate * rate_err)**2 +
	(differential_cross_section / gamma_flux_at_target * gamma_flux_at_target_err)**2
)

print('angle (°)	rate (1/s)	crosssec (10^-28 m^2)')
for (a, r, d) in zip(angles, rate, differential_cross_section):
	print(f'{a}	{r:0.2e}	{d*1e28:0.2e}')

fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 3))

ax.errorbar(angles, differential_cross_section * 1e28,
            yerr=differential_cross_section_err * 1e28, fmt='x', label='experimental data')
# solid line for theoretical prediction, see https://en.wikipedia.org/wiki/Klein%E2%80%93Nishina_formula
ax.plot(bb_angles, bb_diff_cross / 10,
        label='theoretical prediction (Klein-Nishina)')

ax.set_ylabel(
	r'differential cross section $\frac{\mathrm{d} \sigma}{\mathrm{d} \Omega}$ (barn)')
ax.set_xlabel(r'scattering angle $\Theta$ (°)')
ax.legend()
ax.grid()


if len(sys.argv) < 2:
	plt.show()
else:
	fig.savefig(sys.argv[1])
