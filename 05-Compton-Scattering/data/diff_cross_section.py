#!/usr/bin/python
import datetime

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

import constants
from calibration import channel, channel_to_energy
from datasets import datasets

detector_cal_energy, detector_cal_inv_efficiency = np.loadtxt(
	'src/detector-efficiency.dat', unpack=True)
bb_angles, bb_diff_cross = np.loadtxt(
	'src/bb-diff-cross.dat', usecols=[0, 2], unpack=True)


def detector_cal_fitfunc(E, a, b, c):
	return a + b * np.exp(-E / c)


popt, _ = scipy.optimize.curve_fit(
	detector_cal_fitfunc, detector_cal_energy, detector_cal_inv_efficiency)

if False:
	X_fit = np.linspace(0, 1.25, 20)
	plt.plot(detector_cal_energy, detector_cal_inv_efficiency, 'x')
	plt.plot(X_fit, detector_cal_fitfunc(
		X_fit, *popt), '-')

	plt.show()

angles = []
counts = []

for dataset in datasets:
	angles.append(dataset.angle)
	energy, _ = channel_to_energy(channel)
	inverse_quantum_efficiency = detector_cal_fitfunc(energy, *popt)
	count = np.sum(dataset.count_diff * inverse_quantum_efficiency)
	counts.append(count)

angles = np.array(angles)
counts = np.array(counts)
time = 300  # s
rate = counts / time

# day not given, assume middle of july
date_inital = datetime.date(1970, 7, 15)
date_experiment = datetime.date(2018, 12, 17)
ratio_remaining = np.exp(
	(date_inital - date_experiment).days / (30.17 * 365.25))

gamma_flux_at_target_initial = 1.54e6 * 1e4  # per (cm2*s), converted to m2s
gamma_flux_at_target = gamma_flux_at_target_initial * ratio_remaining

volume_target = 1 * np.pi * 0.5**2  # in cm3
atomic_weight_target = 26.981539  # Al, gram per mole
atomic_number_target = 13  # Al
density_target = 2.70  # Al, gram per cm3
number_electrons_target = constants.avogadro_constant / \
	atomic_weight_target * atomic_number_target * density_target * volume_target

area_scintillator = np.pi * (2.55e-2 / 2)**2
distance_scintillator = 21.5e-2
solid_angle_scintillator = area_scintillator / \
	(4 * np.pi * distance_scintillator**2)

differential_cross_section = rate / \
	(solid_angle_scintillator * gamma_flux_at_target
	 * number_electrons_target)

plt.plot(angles, differential_cross_section * 1e28)
#plt.scatter(bb_angles, bb_diff_cross)
plt.ylabel(r'$\frac{\mathrm{d} \sigma}{\mathrm{d} \Omega}$ (barn)')
plt.show()
