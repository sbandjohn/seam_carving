from skimage import io
from matplotlib import pyplot as plt
import random

import sc
import network_energy
import scharr_energy
import forward_energy
import part1_energy

image_dir = "dolphin.jpg"
# sb: 500*375 c*r
# cat: 360*263
# cute cat: 500*330 c*r
# timg: 325*186

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

def compare_resize(ori, func, forward, cut_r, cut_c, name, func2, forward2, name2):
    r, c = ori.shape[0], ori.shape[1]
    out_r = r + cut_r
    out_c = c + cut_c

    """
    img = ori
    dirs = sc.determine_directions(img, func, forward, out_r, out_c)
    carved_seams = []
    cnt = 0
    len_d = len(dirs)
    for d in dirs:
        #em = func(img)
        print("process {}/{}".format(cnt, len_d))
        cnt += 1
        img, tmp_seams = sc.carve(img, func, forward, d, num=1, border=1, need_seam=True)
        carved_seams += tmp_seams
    seams = sc.transform_seams(carved_seams)
    """

    #img = sc.resize_once(ori, func, forward, out_r, out_c)
    img = sc.resize_multi(ori, func, forward, out_r, out_c)
    img2 = sc.resize_multi(ori, func2, forward2, out_r, out_c)

    plt.figure()
    plt.subplot(121)
    plt.title(name)
    plt.imshow(img)
    #for seam in seams:
    #    draw_seam(seam.coor)
    plt.subplot(122)
    plt.title(name2)
    plt.imshow(img2)
    plt.savefig(name+" vs "+name2)
    #for seam in carved2:
    #    draw_seam(seam.coor)

def try_resize(ori, func, forward, cut_r, cut_c, name):
    r, c = ori.shape[0], ori.shape[1]
    out_r = r + cut_r
    out_c = c + cut_c

    img, carved_seams = sc.resize_once(ori, func, forward, out_r, out_c, need_seam=True)
    seams = sc.transform_seams(carved_seams)

    plt.figure()
    plt.subplot(121)
    plt.title(name+' seams')
    plt.imshow(ori)
    for seam in seams:
        draw_seam(seam.coor)
    plt.subplot(122)
    plt.title(name+" result")
    plt.imshow(img)
    plt.savefig(name)
    #for seam in carved2:
    #    draw_seam(seam.coor)

try_resize(img, part1_energy.RGBdifference, False, -15, -55, "cut:RGB")
#try_resize(img, part1_energy.combine, False, +10, +20, "enlarge:RGB+entropy")
#try_resize(img, forward_energy.energy_map, True, -15, -55, "cut:forward")
#compare_resize(img, part1_energy.combine, False, -15, -55, "resize: RGB+entropy",
#                forward_energy.energy_map, True, "resize: forward")

#try_resize(img, part1_energy.combine, -40, 0, forward=False)
#try_resize(img, network_energy.energy_map, -50, 0, forward=False)

plt.figure()
plt.title('Original Image')
plt.imshow(img)

plt.show()
