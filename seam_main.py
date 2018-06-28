from skimage import io
from matplotlib import pyplot as plt
import sys
import random

import sc
import network_energy
import forward_energy
import part1_energy

image_dir = sys.argv[1]
print(sys.argv)

out_c, out_r = map(int, [sys.argv[2], sys.argv[3]])
energy_type = int(sys.argv[4])
output_dir = sys.argv[5]

img = io.imread(image_dir)

def get_setting(type):
    if type == 0:
        return {'func':part1_energy.RGBdifference, 'forward':False}
    if type == 1:
        return {'func':part1_energy.combine, 'forward':False}
    if type == 2:
        return {'func':forward_energy.energy_map, 'forward':True}
    if type == 3:
        return {'func':network_energy.energy_map, 'forward':False}

setting = get_setting(energy_type)

def determine_directions(img, out_r, out_c, setting):
    r, c = img.shape[0], img.shape[1]
    dr = abs(r - out_r)
    dc = abs(c - out_c)
    dirs = ['vertical']*dc + ['honrizontal']*dr
    random.shuffle(dirs)
    return dirs


directions = determine_directions(img, out_r, out_c, setting)

out = img
for d in directions:
    em = setting['func'](out)
    out = sc.carve(out, em, d, num=1, border=1, forward=setting['forward'])

io.imsave(output_dir, out)

def show_res(img, out):
    plt.figure()
    plt.imshow(img)
    plt.figure()
    plt.imshow(out)
    plt.show()

show_res(img, out)