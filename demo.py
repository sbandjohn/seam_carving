from skimage import io
from matplotlib import pyplot as plt

import sc
import network_energy
import scharr_energy

image_dir = "dolphin.jpg"
cuts = 50

# seam carve and show result
img = io.imread(image_dir)

def draw_seam(seam):
    L = len(seam)
    xs = [seam[i][1] for i in range(L)]
    ys = [seam[i][0] for i in range(L)]
    plt.plot(xs, ys, '--')

def try_energy_funciton(img, func, orientation, num, name):
    # input
    #   func: function to calculate energy map of 'img'
    #   orientation: 'horizontal' or 'vertical'
    #   num: number of cuts

    print("trying "+name+" ===================")

    # show first k seams
    em = func(img)
    seams = sc.first_k_seams(img, em, orientation, k = 20, border=2)

    plt.figure()
    plt.title("seams by "+name)
    plt.imshow(img)
    for seam in seams:
        draw_seam(seam)


    # carve once: cut 'num' seams by one energy map
    once = sc.carve(img, em, orientation, num=num, border=2)

    plt.figure()
    plt.title("once by"+name)
    plt.imshow(once)

    # iteratively calculate energy map and cut 'percuts' seams each time
    times = 10
    percuts = num//times
    em = func(img)
    it = img
    for i in range(times):
        print("iter:", i, " {}/{}".format(i*percuts, num))
        em = func(it)
        it = sc.carve(it, em, orientation, num=percuts, border=2)
    plt.figure()
    plt.title("iter by "+name)
    plt.imshow(it)

try_energy_funciton(img, scharr_energy.energy_map, 'vertical', cuts, 'scharr')
try_energy_funciton(img, network_energy.energy_map, 'vertical', cuts, 'network')


plt.figure()
plt.title('Original Image')
plt.imshow(img)

plt.show()
