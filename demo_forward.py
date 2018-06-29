from skimage import io
from matplotlib import pyplot as plt
import random

import sc
import network_energy
import forward_conv_energy
import forward_energy
import part1_energy

image_dir = "cutecat.jpg"
# sb: 500*375 c*r
# cat: 360*263
# cute cat: 333*220 c*r
# timg: 325*186
# 

cuts = 20

# seam carve and show result
img = io.imread(image_dir)

def draw_seam(seam):
    L = len(seam)
    xs = [seam[i][1] for i in range(L)]
    ys = [seam[i][0] for i in range(L)]
    plt.plot(xs, ys, '--')

def determine_directions(dr, dc):
    dirs = ['vertical']*dc + ['horizontal']*dr
    random.shuffle(dirs)
    return dirs

def compare_resize_with_ori(ori, func, forward, cut_r, cut_c, name, func2, forward2, name2):
    r, c = ori.shape[0], ori.shape[1]
    out_r = r + cut_r
    out_c = c + cut_c

    img = sc.resize_multi(ori, func, forward, out_r, out_c)
    img2 = sc.resize_multi(ori, func2, forward2, out_r, out_c)

    plt.figure()
    plt.subplot(131)
    plt.title('original')
    plt.imshow(ori)
    plt.subplot(132)
    plt.title(name)
    plt.imshow(img)
    #for seam in seams:
    #    draw_seam(seam.coor)
    plt.subplot(133)
    plt.title(name2)
    plt.imshow(img2)
    print("saving...")
    plt.savefig(name+" vs "+name2)
    print("saved")
    #for seam in carved2:
    #    draw_seam(seam.coor)

compare_resize_with_ori(img, part1_energy.combine, False, 0, 70, "RGB+entropy",
                         forward_conv_energy.energy_map, True, "forward")
                    

#try_resize(img, part1_energy.combine, -40, 0, forward=False)
#try_resize(img, network_energy.energy_map, -50, 0, forward=False)

plt.figure()
plt.title('Original Image')
plt.imshow(img)

plt.show()
