import scipy.signal as signal
import numpy as np
import torchvision.transforms as transforms
from skimage import filters, color, io
import matplotlib.pyplot as plt

dxy = [(i-1, j-1) for j in range(3) for i in range(3) if (i, j)!=(1, 1)]

EPSILON = 1e-3
ones = np.ones([9, 9])

# problem: boundary
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
    if show:
        print(values)
        print(values.min())
        print(values.max())
        plt.figure()
        plt.title("RGB")
        plt.imshow(values)
    return values

def range_normalize(v, a, b):
    mi = v.min()
    ma = v.max()
    return (v - mi)/(ma-mi)*(b-a) + a

def combine(img, show=False):
    # combine minus entropy and RGB difference, ratio= 1:1
    mH = range_normalize(minus_entropy(img, show), 0, 1)
    RGB = range_normalize(RGBdiffernece(img, show), 0, 1)
    res = mH + RGB

    if show:
        plt.figure()
        plt.title('combine')
        plt.imshow(RGB)
        print(mH.max(), mH.min())
        print(RGB.max(), RGB.min())
        
    return res


def test():
    img = io.imread('dolphin.jpg')
    H = combine(img, show=True)
    plt.imshow(H)
    plt.show()

if __name__=="__main__":
    test()
