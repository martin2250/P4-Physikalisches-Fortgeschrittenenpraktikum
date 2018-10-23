#!/usr/bin/python
from __future__ import division, print_function

import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

# for p in 4K-20uA 2K-20uA 2K-100uA; do ./plot-hall.py $p & done

param = sys.argv[1]
loadparams = {'unpack': True, 'skiprows': 3, 'delimiter': ';'}

_, B_hall = np.loadtxt(f'src/{param}-hall-B.dat', **loadparams)
_, U_hall = np.loadtxt(f'src/{param}-hall-U.dat', **loadparams)
_, B_res = np.loadtxt(f'src/{param}-res-B.dat', **loadparams)
_, U_res = np.loadtxt(f'src/{param}-res-U.dat', **loadparams)

if len(B_hall) != len(U_hall) or len(B_res) != len(U_res):
	print('lengths don\'t match')
	exit(1)

# find index where shit goes haywire
I_hall = B_hall + 1j * U_hall
I_res = B_res + 1j * U_res

I_hall_diff = np.abs(np.diff(I_hall))
I_res_diff = np.abs(np.diff(I_res))

cutoff_hall = np.argmax(I_hall_diff > 1) - 1
cutoff_res = np.argmax(I_res_diff > 1) - 1

# cut off bad parts and simultaneously apply weiner filter, to filter weiners
weiner_size = 11
B_hall = scipy.signal.wiener(B_hall[:cutoff_hall], weiner_size)
U_hall = scipy.signal.wiener(U_hall[:cutoff_hall], weiner_size) * 1000
B_res = scipy.signal.wiener(B_res[:cutoff_res], weiner_size)
U_res = scipy.signal.wiener(U_res[:cutoff_res], weiner_size) * 1000

# make figure
fig = plt.figure()

axL = fig.add_subplot(111)
axR = axL.twinx()

# align y axes
Uh_max = np.max(U_hall)
Uh_max = 100 * np.ceil(Uh_max / 100)

axL.set_ylim(0, Uh_max)
axR.set_ylim(0, Uh_max / 4)

ticks = np.linspace(0, Uh_max, 6)

axL.set_yticks(ticks)
axR.set_yticks(ticks / 4)

# plot
axL.plot(B_hall, U_hall, '.', label='U_h')
axR.plot(0, 0)  # skip first color
axR.plot(B_res, U_res, '.', label='U_res')

axL.set_xlabel('B (T)')
axL.set_ylabel('$U_h$ (mV)')
axR.set_ylabel('$U_{res}$ (mV)')

fig.legend()

plt.grid()

if len(sys.argv) == 2:
	plt.show()
elif len(sys.argv) >= 3:
	plt.savefig(sys.argv[2])
