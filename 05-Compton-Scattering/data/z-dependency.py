#!/usr/bin/python
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.odr

from calibration import channel, channel_to_energy
from datasets import load_count
from detector import detector_inv_efficiency


@dataclass
class Element:
	symbol: str
	name: str
	atomic_number: int
	atomic_mass: float
	density: float
	count_total: float = None
	cross_section_relative: float = None


elements = [
	Element('al', 'aluminum (or aluminium?)', 13, 26.981539, 2.7),
	Element('fe', 'iron',                     26, 55.845,    7.874),
	Element('cu', 'copper',                   29, 63.546,    8.96),
	Element('pb', 'lead',                     82, 207.2,     11.34)
]

# keep this?
count_baseline = load_count(f'src/45-wo-target.npz')

for element in elements:
	count = load_count(f'src/compare-{element.symbol}.npz') - count_baseline
	energy, _ = channel_to_energy(channel)
	element.count_total = np.sum(count * detector_inv_efficiency(energy))

	element.cross_section_relative = element.count_total * \
		element.atomic_mass / (element.density * element.atomic_number)

	print(f'{element.symbol}&\t{element.atomic_number}&\t{element.atomic_mass:0.1f}&\t{element.density:0.2f}&\t{element.count_total:0.0f}&\t{element.cross_section_relative:0.0f}\\\\')
