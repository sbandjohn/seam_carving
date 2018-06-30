import scipy.signal as signal
import numpy as np
import torchvision.transforms as transforms
from skimage import filters, color, io
import matplotlib.pyplot as plt

dxy = [(i-1, j-1) for j in range(3) for i in range(3) if (i, j)!=(1, 1)]

EPSILON = 1e-3
ones = np.ones([9, 9])

def minus_entropy(img, show=False):
    f = color.rgb2gray(img)
    sf = signal.convolve2d(f, ones, mode='same', boundary='symm')
    p = f / sf + EPSILON
    plog_p = p * np.log(p)
    minusH = signal.convolve2d(plog_p, ones, mode='same', boundary='symm')
    # greater entropy, more uniform the distrbution, less energy
    if show:
        print(minusH)
        print(minusH.min(), minusH.max())
        plt.figure()
        plt.title('minusH')
        plt.imshow(minusH)
    return minusH

kernels = [np.zeros([3, 3]) for i in range(8)]
for i in range(8):
    kernels[i][1+dxy[i][0]][1+dxy[i][1]] = -1
    kernels[i][1][1] = 1


def RGBdifference(img, show=False):
    img = np.array(img, dtype=np.float64)
    h, w, c = img.shape
    aha = [img[..., k] for k in range(c)]
    values = np.zeros([h, w])

    def forpixel(i, j):
        values[i][j] = 0
        cnt = 0
        for (dx, dy) in dxy:
            x = i + dx
            y = j + dy
            if x>=0 and x<h and y>=0 and y<w:
                cnt += 1
                tmp = sum(abs(aha[k][x][y]-aha[k][i][j]) for k in range(c))
                values[i][j] += tmp
        values[i][j] /= cnt

    for k in range(c):
        for i in range(8):
            values += np.abs(signal.convolve2d(aha[k], kernels[i], mode='same', boundary='fill'))
    values /= 8

    for j in range(w):
        forpixel(0, j)
        forpixel(h-1, j)
    for i in range(1, h-1):
        forpixel(i, 0)
        forpixel(i, w-1)

    values /= c
    return values

def range_normalize(v, a, b):
    mi = v.min()
    ma = v.max()
    return (v - mi)/(ma-mi)*(b-a) + a

def combine(img, show=False):
    # combine minus entropy and RGB difference, ratio= 1:1
    mH = range_normalize(minus_entropy(img, show), 0, 1)
    RGB = range_normalize(RGBdifference(img, show), 0, 1)
    
    res = 0.5*mH + RGB

    if False:
        plt.figure()
        plt.title('combine')
        plt.imshow(RGB)
        print(mH.max(), mH.min())
        print(RGB.max(), RGB.min())
        
    return res


def test():
    img = io.imread('dolphin.jpg')
    RGB = range_normalize(RGBdifference(img),0, 1)
    H = range_normalize(minus_entropy(img),0, 1)
    plt.figure()
    plt.subplot(121)
    plt.title("RGB energy map")
    plt.imshow(RGB)
    plt.subplot(122)
    plt.title("entropy energy map")
    plt.imshow(H)
    plt.show()
    #H = combine(img, show=True)
    #plt.imshow(H)
    #plt.show()

if __name__=="__main__":
    test()
