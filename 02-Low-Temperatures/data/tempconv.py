#!/usr/bin/python
from __future__ import division

import numpy as np


def pt100_linear(R):
	return 31.95 + 2.353 * R


def pt100_nonlinear(R):
	coeffs = [
		16.61,
		6.262,
		-3.695,
		0.01245
	]
	return coeffs * R**np.arange(len(coeffs))


def pt100(R):
	""" calculate temperature from pt100 resistance """
	return pt100_linear(R) if (pt100_linear(R) > 60) else pt100_nonlinear(R)


def carbon(R):
	""" calculate temperature from carbon resistor resistance """
	return np.exp(1.116 * np.log(R) / (-4.374 + np.log(R)) - 1231 * np.log(R) / (9947 + np.log(R)))
