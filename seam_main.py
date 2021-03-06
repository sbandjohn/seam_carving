from skimage import io
from matplotlib import pyplot as plt
import sys

import sc
import network_energy
import forward_conv_energy
import part1_energy
import combined_energy

image_dir = sys.argv[1]
print(sys.argv)

out_c, out_r = map(int, [sys.argv[2], sys.argv[3]])
energy_type = int(sys.argv[4])
output_dir = sys.argv[5]

Len = len(sys.argv)
if Len >= 7 and sys.argv[6]=='opt':
    opt = True
else:
    opt = False

img = io.imread(image_dir)

def get_setting(type):
    if type == 0:
        return {'func':part1_energy.RGBdifference, 'forward':False}
    if type == 1:
        return {'func':part1_energy.combine, 'forward':False}
    if type == 2:
        return {'func':forward_conv_energy.energy_map, 'forward':True}
    if type == 3:
        return {'func':network_energy.energy_map, 'forward':False}
    if type == 4:
        return {'func':combined_energy.part1_tail, 'forward':False}

setting = get_setting(energy_type)

out = sc.resize_multi(img, setting['func'], setting['forward'], out_r, out_c, opt)

io.imsave(output_dir, out)

def show_res(img, out):
    plt.figure()
    plt.imshow(img)
    plt.figure()
    plt.imshow(out)
    plt.show()

#show_res(img, out)