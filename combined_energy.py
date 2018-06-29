import numpy as np
from skimage import filters, color
from scipy import signal

import part1_energy
import network_energy

# filter for energy map
scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy

def part1_tail(img):
    part1 = part1_energy.combine(img)
    tail = network_energy.tail_map(img)
    part1 = part1_energy.range_normalize(part1, 0, 1)
    tail = part1_energy.range_normalize(tail, 0, 1)
    return part1 + tail

