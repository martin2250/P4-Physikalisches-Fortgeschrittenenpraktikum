#!/usr/bin/python
from __future__ import division

import numpy as np

ptT, ptR = np.loadtxt(f'src/RT-pt100.dat', unpack=True)
CT, CR = np.loadtxt(f'src/RT-carbon.dat', unpack=True)

# sort lists for interpolation
CR = np.flip(CR)
CT = np.flip(CT)

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

def int_pt100(R):
	""" interpolate temperature from pt100 resistance (-1 kelvin output for troubleshoot ranges)"""
	return np.interp(R, ptR, ptT, -1, -1)

def int_carbon(R):
	""" interpolate temperature from carbon resistor resistance (-1 kelvin output for troubleshoot ranges)"""
	return np.interp(R, CR, CT, -1, -1)
