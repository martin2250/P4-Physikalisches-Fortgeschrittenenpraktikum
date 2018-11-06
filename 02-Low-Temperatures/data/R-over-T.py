#!/usr/bin/python
from __future__ import division, print_function

import argparse
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tempconv as tc

parser = argparse.ArgumentParser()

parser.add_argument('plot', choices=['overview', 'separate'],
					help='query either overview or separate plots')
parser.add_argument('--magnify', action='store_true',
					help='magnify on low temperature regimes')
args = parser.parse_args()

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
R_wa_cu = np.append(Rcuwapt, Rcuwac)
R_wa_si = np.append(Rsiwapt, Rsiwac)
R_wa_nb = np.append(Rnbwapt, Rnbwac)

# draw plots
if args.plot == 'overview':
	plt.scatter(T_cd, R_cd_cu, label='$R_{Cu}$ cooldown', marker='x')
	plt.scatter(T_cd, R_cd_si, label='$R_{Si}$ cooldown', marker='x')
	plt.scatter(T_cd, R_cd_nb, label='$R_{Nb}$ cooldown', marker='x')
	plt.xlabel('$T$ (K)')
	plt.ylabel('$R$ ($\\Omega$)')
	plt.grid()
	plt.legend()
	plt.savefig(os.path.join('plots', f'{args.plot}.pdf'))
else:
	print('Entered separate thing.')
