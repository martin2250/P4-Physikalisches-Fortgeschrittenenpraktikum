#!/usr/bin/python
from __future__ import division, print_function

import argparse
import os
import sys

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np

import loadRT as RT
import tempconv as tc

################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('plot', choices=['overview', 'separate', 'reduced'],
                    help='query either overview/separate/reduced plots')
parser.add_argument('--output', type=str,
                    help='output file')
parser.add_argument('--magnify', action='store_true',
                    help='magnify on low temperature regimes')
parser.add_argument('--verbose', action='store_true',
                    help='print verbose output')
args = parser.parse_args()

################################################################################

######################## function defs #########################################

# debye temp and resistance


def debye(slope, off, resid):
	""" calculates debye temperature and resistance for a
	given linear fit and residual resistance """
	R_thet = (resid - off) / 0.17
	thet = 1.17 * R_thet / slope
	return thet, R_thet

################################################################################

################################################################################


# import needed vars from loadRT
T_cd = RT.T_cd
T_wa = RT.T_wa

R_cd_cu = RT.R_cd_cu

R = RT.R

################################################################################

# constants
# to do this properly one has to detect the critical slope and take the last value before it
resid = {
    'Cu': np.min(R_cd_cu),
   	'Nb': 23.58
}

theta = {
    'Cu': (0.0, 0.0),
   	'Nb': (0.0, 0.0)
}

markers = ['o', 'x', '+']
labels = ['Cu', 'Si', 'Nb']
sz_o = 25
sz_s = 45
################################################################################

# linear regression
T_lin = T_cd[T_cd > 60]
T = np.linspace(0, np.max(T_cd), 5)

slps = {
    'Cu': 0,
   	'Nb': 0
}

inters = {
    'Cu': 0,
   	'Nb': 0
}

for r, l in zip(R, labels):

	if not l == 'Si':
		r_lin = r[0][T_cd > 60]
		slps[l], inters[l] = np.polyfit(T_lin, r_lin, 1)
		theta[l] = debye(slps[l], inters[l], resid[l])

		if args.verbose:
			print(f'{l} -- slope = {slps[l]:0.5f} ohm/K, intercept = {inters[l]:0.5f}')
			print(
                            f'{l} -- theta = {theta[l][0]:0.2f} K, rtheta = {theta[l][1]:0.2f} Ohm')

################################################################################

# overview over all samples
if not args.plot == 'separate':

	figs, ax = plt.subplots()
	figs = [figs]

	if args.plot == 'overview':

		for r, l, m in zip(R, labels, markers):
			plt.scatter(np.append(T_cd, T_wa), np.append(r[0], r[1]),
                            label='$R_{%s}$' % l, marker=m, s=sz_o)

		ax.set_xlabel('$T$ (K)')
		ax.set_ylabel('$R$ ($\\Omega$)')

		# zooming in on low temperature regime
		if args.magnify:
			plt.xlim(np.min(T_cd) - 2, 60)

	else:

		for r, l, m in zip(R, labels, markers):

			if not l == 'Si':
				plt.scatter(np.append(T_cd, T_wa) / theta[l][0], (np.append(r[0], r[1]) - resid[l]) / theta[l][1],
                                    label=l, marker=m, s=sz_o)
				ax.set_xlabel('$\\frac{T}{\\theta}$')
				ax.set_ylabel('$\\frac{R-R_{res}}{R_{\\theta}}$')

				# zooming in on low temperature regime (although this isn't interesting in this case)
				if args.magnify:
					plt.xlim(0, 0.5)
	ax.grid()
	ax.legend()

else:

	ind = np.arange(3)
	figs = [plt.figure(n + 1) for n in ind]
	axs = [figs[n].add_subplot(111) for n in ind]

	for fig, ax, r, l in zip(figs, axs, R, labels):

		if not args.magnify and not l == 'Si':

			ax.plot(T, slps[l] * T + inters[l], '-',
                            label='linear regression', color='red')

		ax.scatter(T_cd, r[0], marker='x', label='cooldown', s=sz_s)
		ax.scatter(T_wa, r[1], marker='x', label='warmup', s=sz_s)
		ax.set_xlabel('$T$ (K)')
		ax.set_ylabel('$R_{%s}$ ($\\Omega$)' % l)
		ax.grid()
		ax.legend()

		# zooming in on low temperature regime
		if args.magnify:
			ax.set_xlim(np.min(T_cd) - 2, 60)
			ax.set_ylim(0 - np.max(r[1]) / 10, np.max(r[1]))

# if output secified, save figures
if args.output:

	pdf = matplotlib.backends.backend_pdf.PdfPages(args.output)
	for fig in figs:
		pdf.savefig(fig)
	pdf.close()

else:
	plt.show()
