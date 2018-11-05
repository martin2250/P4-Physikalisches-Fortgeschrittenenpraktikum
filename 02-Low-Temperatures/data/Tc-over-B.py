#!/usr/bin/python
from __future__ import division, print_function

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

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

plt.plot(B, T, label='$T_c$ of Nb sample')

plt.xlabel('$B$ (T)')
plt.ylabel('$T_c$ (K)')
plt.grid()
plt.legend()

if args.output:
	plt.savefig(args.output)
else:
	plt.show()
