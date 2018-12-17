#!/usr/bin/python
import sys

import numpy as np

channel, count = np.loadtxt(sys.argv[1], unpack=True)
#np.savetxt(sys.argv[2], count, fmt='%d')
np.savez_compressed(sys.argv[2], count=count.astype(np.uint16))
