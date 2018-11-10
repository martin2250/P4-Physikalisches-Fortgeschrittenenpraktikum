#!/usr/bin/python
from __future__ import division, print_function

import argparse
import os
import sys

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np

import tempconv as tc

# load data
Rtcdpt, Rcucdpt, Rsicdpt, Rnbcdpt = np.loadtxt(
	f'src/RoverRT-cooldown-pt.dat', unpack=True)
Rtwapt, Rcuwapt, Rsiwapt, Rnbwapt = np.loadtxt(
	f'src/RoverRT-warmup-pt.dat', unpack=True)
Rtcdc, Rcucdc, Rsicdc, Rnbcdc = np.loadtxt(
	f'src/RoverRT-cooldown-C.dat', unpack=True)
Rtwac, Rcuwac, Rsiwac, Rnbwac = np.loadtxt(
	f'src/RoverRT-warmup-C.dat', unpack=True)

# convert kOhms to Ohms
Rtcdc = Rtcdc * 1e3
Rtwac = Rtwac * 1e3

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

R = np.array([np.array([R_cd_cu, R_wa_cu]),
              np.array([R_cd_si, R_wa_si]),
              np.array([R_cd_nb, R_wa_nb])])

T = np.append(T_cd, T_wa)
R_si = np.append(R_cd_si, R_wa_si)
