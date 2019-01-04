#!/usr/bin/python
import numpy as np
import scipy.optimize

detector_cal_energy, detector_cal_inv_efficiency = np.loadtxt(
	'src/detector-efficiency.dat', unpack=True)


def _get_detector_cal():
	def detector_cal_fitfunc(E, a, b, c):
		return a + b * np.exp(-E / c)

	popt, _ = scipy.optimize.curve_fit(
		detector_cal_fitfunc, detector_cal_energy, detector_cal_inv_efficiency)

	return lambda E: detector_cal_fitfunc(E, *popt)


detector_inv_efficiency = _get_detector_cal()

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	X_fit = np.linspace(0, 1.25, 20)
	plt.plot(detector_cal_energy, detector_cal_inv_efficiency, 'x')
	plt.plot(X_fit, detector_inv_efficiency(X_fit), '-')

	plt.show()
