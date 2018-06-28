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

def try_resize(img, func, cut_r, cut_c, forward=False):
    ori = img
    dirs = determine_directions(cut_r, cut_c)
    carved_seams = []
    for d in dirs:
        em = func(img)
        img, tmp_seams = sc.carve(img, em, d, num=1, border=1, forward=forward, need_seam=True)
        carved_seams += tmp_seams
    seams = sc.transform_seams(carved_seams)
    """
    for s in carved_seams:
        print(s.coor)
    for i in range(5): print()
    for s in seams:
        print(s.coor)
    """
    plt.figure()
    plt.imshow(img)
    plt.figure()
    plt.imshow(ori)
    for seam in seams:
        draw_seam(seam.coor)


def try_energy_funciton(img, func, orientation, num, border, name, forward = False):
    # input
    #   func: function to calculate energy map of 'img'
    #   orientation: 'horizontal' or 'vertical'
    #   num: number of cuts
    print("trying "+name+" ===================")

    ori = img
    # get energy map
    em = func(img)
    # carve once: cut 'num' seams by one energy map
    once, carved_seams = sc.carve(img, em, orientation, num=num, border=2, forward=forward, need_seam=True)
    seams = sc.transform_seams(carved_seams)

    plt.figure()
    plt.title(name+' once')
    plt.subplot(121)
    plt.imshow(ori)
    for seam in seams:
        draw_seam(seam.coor)

    plt.subplot(122)
    plt.imshow(once)

    # iteratively calculate energy map and cut 'percuts' seams each time
    times = 5
    percuts = num//times
    em = func(img)
    it = img
    carved_seams = []
    for i in range(times):
        print("iter:", i, " {}/{}".format(i*percuts, num))
        em = func(it)
        it, tmp_seams = sc.carve(it, em, orientation, num=percuts, border=2, forward=forward, need_seam=True)
        carved_seams += tmp_seams
    seams = sc.transform_seams(carved_seams)

    plt.figure()
    plt.title(name+' iter')
    plt.subplot(121)
    plt.imshow(ori)
    for seam in seams:
        draw_seam(seam.coor)
    plt.subplot(122)
    plt.imshow(it)

#try_energy_funciton(img, forward_energy.energy_map, 'vertical', cuts, 'forward', forward=True)
#try_energy_funciton(img, part1_energy.combine, 'horizontal', cuts, 'combine')
#try_energy_funciton(img, network_energy.energy_map, 'vertical', cuts, 'network')

#try_resize(img, forward_energy.energy_map, 1, 0, forward=True)
#try_resize(img, part1_energy.combine, 10, 10, forward=False)
try_resize(img, network_energy.energy_map, 20, 20, forward=False)

plt.figure()
plt.title('Original Image')
plt.imshow(img)

plt.show()
