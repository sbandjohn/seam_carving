import numpy as np
from skimage import filters, color
from scipy import signal

def dist(pixel_1, pixel_2) : 
    ret = 0
    c = pixel.size
    for x in range(c): 
        ret += np.abs(pixel_1[x] - pixel_2[x])
    return ret

def energy_map(img):
    # input: image in rgb of size h*w*c, c=3
    # output: energy map: numpy array of size h*w

    h, w, c = img.shape
    E_l = np.zeros(img.shape, dtype=np.float32)
    E_u = np.zeros(img.shape, dtype=np.float32)
    E_r = np.zeros(img.shape, dtype=np.float32)

    for i in range(1, h): 
        for j in range(w): 
            if (j > 0) : 
                E_l[i][j] = dist(img[i][j + 1], img[i][j - 1]) + dist(img[i - 1][j], img[i][j - 1])
            else :
                E_l[i][j] = 10000000

            E_u[i][j] = dist(img[i][j + 1], img[i][j - 1])

            if (j < h - 1) : 
                E_r[i][j] = dist(img[i][j + 1], img[i][j - 1]) + dist(img[i - 1][j], img[i][j + 1])
            else : 
                E_r[i][j] = 10000000
            
    return E_l, E_u, E_r
