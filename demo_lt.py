from skimage import io
from matplotlib import pyplot as plt
import random

import sc
import network_energy
import forward_conv_energy
import forward_energy
import part1_energy
import combined_energy

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

def compare_resize(ori, func, forward, cut_r, cut_c, name, func2, forward2, name2, filename):
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
    plt.savefig(filename)
    #for seam in carved2:
    #    draw_seam(seam.coor)

def compare_resize_3(ori, func1, forward1, cut_r, cut_c, name1, func2, forward2, name2, func3, forward3, name3, filename):
    r, c = ori.shape[0], ori.shape[1]
    out_r = r + cut_r
    out_c = c + cut_c

    img1 = sc.resize_multi(ori, func1, forward1, out_r, out_c)
    img2 = sc.resize_multi(ori, func2, forward2, out_r, out_c)
    img3 = sc.resize_multi(ori, func3, forward3, out_r, out_c)

    plt.figure()
    plt.subplot(131)
    plt.title(name1)
    plt.imshow(img1)
    #for seam in seams:
    #    draw_seam(seam.coor)
    plt.subplot(132)
    plt.title(name2)
    plt.imshow(img2)
    plt.subplot(133)
    plt.title(name3)
    plt.imshow(img3)
    plt.savefig(filename)
    #for seam in carved2:
    #    draw_seam(seam.coor)

def try_resize(ori, func, forward, cut_r, cut_c, name, filename):
    r, c = ori.shape[0], ori.shape[1]
    out_r = r + cut_r
    out_c = c + cut_c

    img = sc.resize_multi(ori, func, forward, out_r, out_c)

    plt.figure()
    plt.subplot(121)
    plt.title("original")
    plt.imshow(ori)
    #for seam in seams:
    #    draw_seam(seam.coor)
    plt.subplot(122)
    plt.title(name+" result")
    plt.imshow(img)
    plt.savefig(filename)
    #for seam in carved2:
    #    draw_seam(seam.coor)

#try_resize(img, forward_conv_energy.energy_map, True, -15, -55, "cut:forward")
#try_resize(img, part1_energy.combine, False, +10, +20, "enlarge:RGB+entropy")
#try_resize(img, network_energy.energy_map, False, 0, 150, "network", '3 150 network')
#compare_resize(img, network_energy.energy_map, False, 0, 100, "network",
#                    network_energy.tail_map, False, "tail", "3 100 network vs tail")

#compare_resize(img, part1_energy.combine, False, 0, 150, "RGB+entropy",
#                    forward_conv_energy.energy_map, True, " forward", "3 150 RGB_entopy vs forward")

#compare_resize_3(img, part1_energy.combine, False, -50, 60, "RGB+entropy",
#                    network_energy.tail_map, False, "tail", 
#                    combined_energy.part1_tail, False, "combined", "cutecat -+ RGB tail combined")

def test():
    img = io.imread('dolphin.jpg')
    RGB = part1_energy.RGBdifference(img)
    H = part1_energy.minus_entropy(img)
    RGBH = part1_energy.combine(img)
    N = network_energy.energy_map(img)
    plt.figure()
    plt.subplot(221)
    plt.title("RGB energy map")
    plt.imshow(RGB)
    plt.subplot(222)
    plt.title("entropy energy map")
    plt.imshow(H)
    plt.subplot(223)
    plt.title("RGBH energy map")
    plt.imshow(RGBH)
    plt.subplot(224)
    plt.title("NETWORK energy map")
    plt.imshow(N)
    plt.savefig("energy_map_compare")
    plt.show()
    #H = combine(img, show=True)
    #plt.imshow(H)
    #plt.show()

test()

"""
plt.figure()
plt.title('Original Image')
plt.imshow(img)
"""
plt.show()
