#!/usr/bin/env python

import argparse

################################################################################

parser = argparse.ArgumentParser(
	description='plots stuff')


parser_group_common = parser.add_argument_group('common')
parser_group_common.add_argument('--title', help='plot title')
parser_group_common.add_argument('--grid', action='store_true')
parser_group_common.add_argument('--size', metavar=['width', 'height'], type=float, nargs=2,
                                 help='plot size (mm)')
parser_group_common.add_argument('--xlabel', metavar=['name', 'unit'], nargs=2,
                                 help='x axis label')
parser_group_common.add_argument('--ylabel', metavar=['name', 'unit'], nargs=2,
                                 help='y axis label')
parser_group_common.add_argument('--output', help='output file')

parser_group_trace = parser.add_argument_group('trace')
parser_group_trace.add_argument('file', metavar='file', help='input files')
parser_group_trace.add_argument('--label', default='', help='trace label')

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

################################################################################
# plot stuff

if args.title:
	plt.title(args.title)

if args.grid:
	plt.grid()

if args.xlabel:
	plt.xlabel(f'${args.xlabel[0]}$ ({args.xlabel[1]})')

if args.ylabel:
	plt.ylabel(f'${args.ylabel[0]}$ ({args.ylabel[1]})')

legend_used = False


def atof_comma(s):
	return float(s.decode().replace(',', '.'))


for trace in traces:
	X, Y = np.loadtxt(trace.file, unpack=True, converters={
	                  0: atof_comma, 1: atof_comma})
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
