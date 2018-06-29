from skimage import io
from matplotlib import pyplot as plt
import random

import sc
import network_energy
import scharr_energy
import forward_energy
import part1_energy

image_dir = "dolphin.jpg"
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

def try_resize(ori, func, cut_r, cut_c, forward=False):
    r, c = ori.shape[0], ori.shape[1]
    out_r = r + cut_r
    out_c = c + cut_c

    img = ori
    dirs = sc.determine_directions(img, func, forward, out_r, out_c)
    carved_seams = []
    for d in dirs:
        em = func(img)
        img, tmp_seams = sc.carve(img, em, d, num=1, border=1, forward=forward, need_seam=True)
        carved_seams += tmp_seams
    seams = sc.transform_seams(carved_seams)

    img2, carved2 = sc.resize(ori, func, forward, out_r, out_c, need_seam=True, dirs=dirs)

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    for seam in carved_seams:
        draw_seam(seam.coor)
    plt.subplot(122)
    plt.imshow(img2)
    for seam in carved2:
        draw_seam(seam.coor)
    
    plt.figure()
    plt.title("resized")
    plt.imshow(img2)

#try_resize(img, forward_energy.energy_map, 1, 0, forward=True)
#try_resize(img, part1_energy.combine, 10, 10, forward=False)
try_resize(img, network_energy.energy_map, 0, 100, forward=False)

plt.figure()
plt.title('Original Image')
plt.imshow(img)

plt.show()
