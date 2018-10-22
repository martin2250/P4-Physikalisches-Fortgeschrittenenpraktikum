#!/usr/bin/python
from __future__ import division, print_function

import sys

import matplotlib.pyplot as plt
import numpy as np

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

B_hall = B_hall[:cutoff_hall]
U_hall = U_hall[:cutoff_hall]
B_res = B_res[:cutoff_res]
U_res = U_res[:cutoff_res]

# plot remaining data
plt.plot(B_hall, U_hall, label='U_h')
plt.plot(B_res, U_res, label='U_res')

plt.legend()
plt.grid()
plt.show()
