import numpy as np
from skimage import filters, color
from scipy import signal

import part1_energy
import network_energy

def part1_tail(img):
    part1 = part1_energy.combine(img)
    tail = network_energy.tail_map(img)
    part1 = part1_energy.range_normalize(part1, 0, 1)
    tail = part1_energy.range_normalize(tail, 0, 1)
    return part1 + tail

