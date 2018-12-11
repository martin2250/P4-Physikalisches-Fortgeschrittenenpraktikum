#!/usr/bin/env python

import argparse

################################################################################

parser = argparse.ArgumentParser(
	description='plots stuff')


parser_group_common = parser.add_argument_group('common')
parser_group_common.add_argument('--title', help='plot title')
parser_group_common.add_argument('--grid', action='store_true')
parser_group_common.add_argument('--size', metavar=['width', 'height'],
                                 type=float, nargs=2,
                                 help='plot size (mm)')
parser_group_common.add_argument('--xlabel', metavar=['name', 'unit'], nargs=2,
                                 help='x axis label')
parser_group_common.add_argument('--ylabel', metavar=['name', 'unit'], nargs=2,
                                 help='y axis label')
parser_group_common.add_argument('--logx', action='store_true',
                                 help='scale x axis logarithmically')
parser_group_common.add_argument('--logy', action='store_true',
                                 help='scale y axis logarithmically')
parser_group_common.add_argument('--output', help='output file')
parser_group_common.add_argument('-v', '--verbose', action='store_true',
                                 help='output extra information (eg. fit parameters)')

parser_group_trace = parser.add_argument_group('trace')
parser_group_trace.add_argument('file', metavar='file', help='input files')
parser_group_trace.add_argument('--label', default='', help='trace label')
parser_group_trace.add_argument('--fit', action='store_true',
                                help='perform linear regression')
parser_group_trace.add_argument('--fit-freq', action='store_true',
                                help='fit frequency dependence to data')
parser_group_trace.add_argument('--fit-freq-shitty', nargs=2, type=int, metavar=['index_1', 'index_2'],
                                help='fit frequency dependence to data ')
parser_group_trace.add_argument('--fit-ignore', type=int,
                                help='ignore last X points for linear fit')
parser_group_trace.add_argument(
	'--fit-label', action='store_true', help='add linear fit to legend')

################################################################################
# split arguments into groups separated by dashes '-', treat first group as both general args and first trace
if True:
	import sys

args_raw = sys.argv[1:]
args_split = []

while '-' in args_raw:
	index_split = args_raw.index('-')
	args_split.append(args_raw[0:index_split])
	args_raw = args_raw[index_split + 1:]
if args_raw:
	args_split.append(args_raw)

args = parser.parse_args(args_split[0])

traces = [parser.parse_args(args_trace_raw)
          for args_trace_raw in args_split]

################################################################################
# prevent autopep8 from moving this to the front (speeds up argcomplete and parser)
if True:
	import os

	import matplotlib.pyplot as plt
	import numpy as np
	import scipy.optimize

################################################################################
# plot stuff

if args.title:
	plt.title(args.title)

if args.logx:
	plt.gca().set_xscale("log", nonposx='clip')

if args.logy:
	plt.gca().set_yscale("log", nonposy='clip')

if args.grid:
	plt.grid()

if args.xlabel:
	xlabel = f'${args.xlabel[0]}$'
	if args.xlabel[1]:
		xlabel += f' ({args.xlabel[1]})'
	plt.xlabel(xlabel)

if args.ylabel:
	ylabel = f'${args.ylabel[0]}$'
	if args.ylabel[1]:
		ylabel += f' ({args.ylabel[1]})'
	plt.ylabel(ylabel)

legend_used = False


def atof_comma(s):
	return float(s.decode().replace(',', '.'))


for trace in traces:
	if trace.fit or trace.fit_freq or trace.fit_freq_shitty:
		X, Y = np.loadtxt(trace.file, unpack=True, usecols=(
			0, 1), converters={0: atof_comma, 1: atof_comma})

		if args.fit_ignore is not None:
			X = X[0:-args.fit_ignore]
			Y = Y[0:-args.fit_ignore]

		xunit = args.xlabel[1] if args.xlabel else None
		yunit = args.ylabel[1] if args.ylabel else None

		if trace.fit:
			slope, intercept = np.polyfit(X, Y, 1)
			if args.verbose:
				print(f'fit parameters for {trace.label or trace.file}')
				print(f'  > slope: {slope:0.3e} {xunit}/{yunit}')
				print(f'  > intercept: {intercept:0.3e} {yunit}')
			fitlabel = f'linear fit\n${args.ylabel[0] if args.ylabel else "Y"} = {slope:0.3e} {xunit}/{yunit} \\cdot {args.xlabel[0] if args.xlabel else "X"} + {intercept:0.3e} {yunit}$'
			plt.plot(X, slope * X + intercept, 'r-',
                            label=fitlabel if args.fit_label else None)
			if args.fit_label:
				legend_used = True
		if trace.fit_freq:
			def fitfunc_freq(f, A, mean_lifetime):
				return A / np.sqrt(1 + (2 * np.pi * f * mean_lifetime)**2)
			(A_fit, mean_lifetime_fit), _ = scipy.optimize.curve_fit(
				fitfunc_freq, X, Y, (1000, 1e-4))
			if args.verbose:
				print(f'fit parameters for {trace.label or trace.file}')
				print(f'  > A: {A_fit:0.3e}')
				print(f'  > mean lifetime: {mean_lifetime_fit * 1e6:0.3e} us')
			X_fit = np.logspace(np.log10(np.min(X)), np.log10(np.max(X)), 50)
			plt.plot(X_fit, fitfunc_freq(X_fit, A_fit, mean_lifetime_fit), 'r-')
		if trace.fit_freq_shitty:
			X_fit = np.logspace(np.log10(np.min(X)), np.log10(np.max(X)), 50)
			X_1 = X[:trace.fit_freq_shitty[0]]
			Y_1 = Y[:trace.fit_freq_shitty[0]]
			X_2 = X[trace.fit_freq_shitty[1]:]
			Y_2 = Y[trace.fit_freq_shitty[1]:]
			intercept_1 = np.mean(Y_1)
			slope_2, intercept_2 = np.polyfit(np.log(X_2), np.log(Y_2), 1)

			plt.plot(X_fit, np.ones(len(X_fit)) * intercept_1)
			plt.plot(X_fit,  np.exp(slope_2 * np.log(X_fit) + intercept_2))

			X_intersect = np.exp((np.log(intercept_1) - intercept_2) / slope_2)
			plt.axvline(X_intersect, linestyle='--')

			if args.verbose:
				print(f'intersect X: {X_intersect}')

			plt.ylim(np.min(Y) / 1.1, np.max(Y) * 1.1)  # shitty solution, as promised


for trace in traces:
	X, Y = np.loadtxt(trace.file, unpack=True, usecols=(0, 1),
                   converters={0: atof_comma, 1: atof_comma})
	plt.plot(X, Y, 'o', label=trace.label)
	if trace.label:
		legend_used = True

if legend_used:
	plt.legend()

if args.output:
	if args.size:
		fig = matplotlib.pyplot.gcf()
		fig.set_size_inches(args.size[0] / 25.4, args.size[1] / 25.4)
	plt.savefig(args.output)
else:
	plt.show()
