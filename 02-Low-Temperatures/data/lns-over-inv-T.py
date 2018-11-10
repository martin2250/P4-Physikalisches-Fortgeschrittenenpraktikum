#!/usr/bin/python
from __future__ import division, print_function

import argparse
import importlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

import constants as const
import loadRT as RT

parser = argparse.ArgumentParser()

parser.add_argument('--output', type=str,
                    help='output file')

args = parser.parse_args()

################################################################################

T, R = RT.T, RT.R_si

i_wa = len(RT.T_cd)  # index where warmup starts

################################################################################

sigma = (3.7e-3 / (3e-3 * 2.5e-3)) * 1 / R
sigma_ref = 1  # S/m


iT = 1 / T

################################################################################

# not final, just for testing
T_fit_min = 8.4
T_fit_max = 45

T_fit, sigma_fit = np.array(
    [(t, s) for (t, s) in zip(T, sigma) if T_fit_min < t < T_fit_max]).T


slope, offset = np.polyfit(1 / T_fit, np.log(sigma_fit), 1)

A = np.exp(offset)
E2 = - 2 * slope * const.k

print(f'activation energy: {E2/const.e:e} eV')
print(f'factor: {A:g} S/m')

################################################################################

plt.plot(iT[:i_wa], sigma[:i_wa], 'o', color='C0', label='cooldown')
plt.plot(iT[i_wa:], sigma[i_wa:], 'o', color='C1', label='warmup')
plt.plot(1 / T_fit, A * np.exp(-E2 / (2 * const.k * T_fit)),
         color='C3', label='linear fit')


plt.xlabel(r'$\frac{1}{T} (\mathrm{K}^{-1})$')
plt.ylabel(r'$\sigma (\mathrm{\frac{S}{m}})$')

ax = plt.gca()
ax.set_yscale("log", nonposy='clip')
plt.grid(which='both')

plt.legend()

if args.output:
	plt.savefig(args.output)
else:
	plt.show()
