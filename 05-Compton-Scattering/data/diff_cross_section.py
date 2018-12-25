#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np

from datasets import datasets

angles = []
counts = []

for dataset in datasets:
	angles.append(dataset.angle)
	counts.append(dataset.count_diff_total)


plt.plot(angles, counts)
plt.show()
