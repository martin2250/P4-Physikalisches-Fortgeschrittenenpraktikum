#!/usr/bin/python
from __future__ import division, print_function

import argparse
import importlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

import constants
import tempconv as tc

parser = argparse.ArgumentParser()

parser.add_argument('--output', type=str,
                    help='output file')

args = parser.parse_args()

I, R = np.loadtxt(f'src/Tc-over-B.dat', unpack=True)

T = tc.carbon(R)


def coil(I):
	""" calculate magnetic field in superconducting coil from current"""
	l = 0.1
	n = 4019
	r = 0.01925
	mu_0 = 4 * np.pi * 1e-7

	return mu_0 * n / 2 * I / np.sqrt(r**2 + (l / 2)**2)


B = coil(I)

plt.plot(T, B, '+k', label='$B_{C2}(T)$ of Nb sample', markersize=15)

################################################################################

slope, offset = np.polyfit(T, B, 1)

# slope * T + offset = magnetic_flux_quantum / (2 * pi * XiGL0) * (1 - T/Tc)
Tc = - offset / slope
XiGL0 = np.sqrt(constants.magnetic_flux_quantum / (2 * np.pi * offset))
mean_free_path_Nb = XiGL0**2 / 39e-9

print(f'XiGL0: {XiGL0:0.2e}m')
print(f'mean free path in Nb: {mean_free_path_Nb:0.2e}m')
print(f'critical temperature: {Tc:0.2f}K')

################################################################################

T_fit_plot = np.array([T[-1], Tc])
plt.plot(T_fit_plot, slope * T_fit_plot + offset,
         label=f'$B(T) = a + b \\cdot T$\na = {offset:0.2f} T\nb = {slope:0.3f} T/K')

plt.xlabel('$T$ (K)')
plt.ylabel('$B$ (T)')
plt.grid()
plt.legend()

if args.output:
	plt.savefig(args.output)
else:
	plt.show()
