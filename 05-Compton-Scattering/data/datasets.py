from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
	angle: float

	count_target: np.ndarray = None
	count_baseline: np.ndarray = None

	count_diff: np.ndarray = None
	count_diff_error: np.ndarray = None
	count_diff_total: float = None


datasets = [Dataset(angle) for angle in range(25, 106, 10)]


def load_count(path):
	with np.load(path) as data:
		count = data['count']

	return count.astype(float)


for dataset in datasets:
	dataset.count_target = load_count(f'src/{dataset.angle}-target.npz')
	dataset.count_baseline = load_count(f'src/{dataset.angle}-wo-target.npz')
	dataset.count_diff = dataset.count_target - dataset.count_baseline
	dataset.count_diff_error = np.sqrt(
		dataset.count_target + dataset.count_baseline)
	dataset.count_diff_total = np.sum(dataset.count_diff)
