#!/usr/bin/python
from __future__ import division, print_function

import argparse
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
import tempconv as tc

parser = argparse.ArgumentParser()

parser.add_argument('plot', choices=['overview', 'separate'],
					help='query either overview or separate plots')
parser.add_argument('--output', type=str,
					help='output file')
parser.add_argument('--magnify', action='store_true',
					help='magnify on low temperature regimes')
args = parser.parse_args()

# debye temp and resistance
def debye(slope, off, resid):

	R_thet = ( resid - off ) / 0.17
	thet = 1.17 * R_thet / slope
	return thet, R_thet

# load data
Rtcdpt, Rcucdpt, Rsicdpt, Rnbcdpt = np.loadtxt(f'src/RoverRT-cooldown-pt.dat', unpack=True)
Rtwapt, Rcuwapt, Rsiwapt, Rnbwapt = np.loadtxt(f'src/RoverRT-warmup-pt.dat', unpack=True)
Rtcdc, Rcucdc, Rsicdc, Rnbcdc = np.loadtxt(f'src/RoverRT-cooldown-C.dat', unpack=True)
Rtwac, Rcuwac, Rsiwac, Rnbwac = np.loadtxt(f'src/RoverRT-warmup-C.dat', unpack=True)

# convert kOhms to Ohms
Rtcdc = Rtcdc*1e3
Rtwac = Rtwac*1e3

Tcdpt = tc.int_pt100(Rtcdpt)

# replace out of bounds values with regression
Tcdpt = np.where(Tcdpt == -1, tc.pt100_linear(Rtcdpt), Tcdpt)

# append temperatures
T_cd = np.append(Tcdpt, tc.int_carbon(Rtcdc))
T_wa = np.append(tc.int_carbon(Rtwac), tc.int_pt100(Rtwapt))

# append resistances
R_cd_cu = np.append(Rcucdpt, Rcucdc)
R_cd_si = np.append(Rsicdpt, Rsicdc)
R_cd_nb = np.append(Rnbcdpt, Rnbcdc)
R_wa_cu = np.append(Rcuwac, Rcuwapt)
R_wa_si = np.append(Rsiwac, Rsiwapt)
R_wa_nb = np.append(Rnbwac, Rnbwapt)

R = np.array([np.array([R_cd_cu, R_wa_cu]), np.array([R_cd_si, R_wa_si]), np.array([R_cd_nb, R_wa_nb])])
R_c_nb = 23.58

# draw plots
markers = ['o', '^', 's']
labels = ['Cu', 'Si', 'Nb']

# overview over all samples
if args.plot == 'overview':

	figs, ax = plt.subplots()
	figs = [figs]

	for r, l, m in zip(R, labels, markers):
		plt.scatter(np.append(T_cd, T_wa), np.append(r[0], r[1]), label='$R_{%s}$'%l, marker=m, s=12)

	ax.set_xlabel('$T$ (K)')
	ax.set_ylabel('$R$ ($\\Omega$)')
	ax.grid()
	ax.legend()

	# zooming in on low temperature regime
	if args.magnify:
		plt.xlim(np.min(T_cd)-2, 60)

else:

	ind = np.arange(3)
	figs = [plt.figure(n+1) for n in ind]
	axs = [figs[n].add_subplot(111) for n in ind]

	T_lin = T_cd[T_cd>60]

	for fig, ax, r, l in zip(figs, axs, R, labels):

		#linear regression
		if not l == 'Si' and not args.magnify:
			r_lin = r[0][T_cd>60]
			T = np.linspace(0, np.max(T_cd), 5)
			slp, inter = np.polyfit(T_lin, r_lin, 1)
			ax.plot(T, slp*T + inter, '--', label='linear regression', color='red')
			print('Nb -- theta = %.2f K, rtheta = %.2f Ohm'%debye(slp, inter, R_c_nb)) if l == 'Nb' else print('Cu -- theta = %.2f K, rtheta = %.2f Ohm'%debye(slp, inter, np.min(R_cd_cu)))

		ax.scatter(T_cd, r[0], marker='o', label='cooldown', s=12)
		ax.scatter(T_wa, r[1], marker='s', label='warmup', s=12)
		ax.set_xlabel('$T$ (K)')
		ax.set_ylabel('$R_{%s}$'%l)
		#ax.grid()
		ax.legend()
		if args.magnify:
			ax.set_xlim(np.min(T_cd)-2, 60)
			ax.set_ylim(0, np.max(r[1]))

if args.output:

	pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join('plots', args.output))
	for fig in figs:
		pdf.savefig(fig)
	pdf.close()

else:
	plt.show()
