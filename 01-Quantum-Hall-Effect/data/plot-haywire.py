#!/usr/bin/python
from __future__ import division, print_function

import argparse
import sys

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

parser = argparse.ArgumentParser()
parser.add_argument('curve', metavar='C', type=str, choices=[
                    '4K-20uA', '2K-20uA', '2K-100uA'])
parser.add_argument('-o', '--output', metavar='F', type=str,
                    help='output file, show plot if omitted')
args = parser.parse_args()

loadparams = {'unpack': True, 'skiprows': 3, 'delimiter': ';'}
# B is not to scale, but who gives a damn
T, B_hall = np.loadtxt(f'src/{args.curve}-hall-B.dat', **loadparams)
_, U_hall = np.loadtxt(f'src/{args.curve}-hall-U.dat', **loadparams)

axL = plt.gca()
axR = axL.twinx()

axL.set_xlabel('time (s)')
axL.set_ylabel('$B$ (T)')
axR.set_ylabel('$U_{hall}$ (mV)')

axL.plot(T, B_hall, label='$B$')
axR.plot(0, label='$B$')
axR.plot(T, U_hall, label='$U_{hall}$')

plt.legend()

if args.output:
    plt.savefig(args.output)
else:
    plt.show()
