from skimage import io
from matplotlib import pyplot as plt
import random

import sc
import network_energy
import forward_conv_energy
import forward_energy
import part1_energy

image_dir = "smallcar.jpg"
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

def try_resize(ori, func, forward, cut_r, cut_c, filename):
    r, c = ori.shape[0], ori.shape[1]
    out_r = r + cut_r
    out_c = c + cut_c

    img, seams = sc.resize_once(ori, func, forward, out_r, out_c, need_seam=True, opt=True)

    img2, seams2 = sc.resize_once(ori, func, forward, out_r, out_c, need_seam=True, opt=False)
    """for i in range(len(carved2)):
        print(i,':', carved2[i].dir, len(carved2[i].coor))
        #carved2[i].coor)
        print("\n===========\n")
    """

    plt.figure()
    plt.subplot(221)
    plt.title("opt seams")
    plt.imshow(ori)
    for seam in seams:
        draw_seam(seam.coor)
    plt.subplot(222)
    plt.title("opt result")
    plt.imshow(img)

    plt.subplot(223)
    plt.title("random seams")
    plt.imshow(ori)
    for seam in seams2:
        draw_seam(seam.coor)
    plt.subplot(224)
    plt.title("random result")
    plt.imshow(img2)

    plt.savefig(filename)
    #for seam in carved2:
    #    draw_seam(seam.coor)

try_resize(img, part1_energy.combine, False, -30, -60, 'cut 30 60 smallcar: opt vs random')

plt.figure()
plt.title('Original Image')
plt.imshow(img)

plt.show()
