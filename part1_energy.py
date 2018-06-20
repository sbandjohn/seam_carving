import scipy.signal as signal
import numpy as np
import torchvision.transforms as transforms
from skimage import filters, color, io
import matplotlib.pyplot as plt

"""
kernels = []
for i in range(3):
    for j in range(3):
        if (i, j) != (1, 1):
            tmp = np.zeros([3, 3])
            tmp[i][j] = -1
            tmp[1][1] = 1
            kernels.append(tmp)
"""

dxy = [(i-1, j-1) for j in range(3) for i in range(3) if (i, j)!=(1, 1)]

EPSILON = 1e-3
ones = np.ones([9, 9])

def entropy(img, show=False):
    f = color.rgb2gray(img)
    sf = signal.convolve2d(f, ones, mode='same', boundary='fill', fillvalue=0)
    p = f / sf
    log_p = np.log(p+EPSILON)
    plog_p = p * log_p
    minusH = signal.convolve2d(plog_p, ones, mode='same', boundary='fill', fillvalue=0)
    # greater entropy, more uniform the distrbution, less energy
    return minusH

def RGBdiffernece(img, show=False):
    # img: h*w*c array. byte array: 0~255
    # convert it to normalized torch tensor c*h*w
    # then to numpy array
    tensor = transforms.ToTensor()(img)
    aha = [x.detach().numpy() for x in tensor]
    if show:
        print(aha)
    h, w, c = img.shape
    #rgb = [img[..., k] for k in range(c)]
    values = np.zeros([h, w])
    for i in range(h):
        for j in range(w):
            cnt = 0
            for (dx, dy) in dxy:
                x = i - dx
                y = j - dy
                if x>=0 and x<h and y>=0 and y<w:
                    cnt += 1
                    tmp = sum(abs(aha[k][x][y]-aha[k][i][j]) for k in range(c))
                    values[i][j] += tmp
            values[i][j] /= cnt
    values = values / 3
    """
        for kernel in kernels:
            tmp = signal.convolve2d(rgb[k], kernel, mode='same', boundary='fill', fillvalue=0)
            values += np.absolute(tmp)
            if show:
                print("convolve")
                print(rgb[k])
                print("  with")
                print(kernel)
                print("  gets")
                print(tmp)
    if show:
        print(values)
    values = values / 3
    neighbors = np.ones([h, w], dtype=np.int)*8
    for (i,j) in ((0,0), (0, w-1), (h-1, 0), (h-1, w-1)):
        neighbors[i][j] = 3
    for i in range(1, h-1):
        neighbors[i][0] = neighbors[i][w-1] = 5
    for j in range(1, w-1):
        neighbors[0][j] = neighbors[h-1][j] = 5
    if show:
        print(neighbors)
    values = values / neighbors
    """
    if show:
        print(values)
    return values

def mean_std_normalize(v, mean, std):
    s = v.std()
    m = v.mean()
    return v*mean/m * std/s

def combine(img, show=True):
    H = mean_std_normalize(entropy(img), mean=1.0, std=1.0)
    RGB = mean_std_normalize(RGBdiffernece(img), mean=1.0, std=1.0)
    if show:
        plt.figure()
        plt.title("H")
        plt.show(H)
        plt.figure()
        plt.title("RGB")
        plt.show(RGB)
        print(H.max(), H.min())
        print(RGB.max(), RGB.min())
    return H + RGB


def test():
    img = io.imread('dolphin.jpg')
    H = combine(img, show=True)
    plt.imshow(H)
    plt.show()

if __name__=="__main__":
    test()
