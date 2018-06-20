import numpy as np
from skimage import filters, color
from scipy import signal

def dist(pixel_1, pixel_2) : 
    ret = 0.0
    c = pixel_1.size
    p_1 = pixel_1.astype(np.int32)
    p_2 = pixel_2.astype(np.int32)
    for x in range(c):
        ret += np.abs(p_1[x] - p_2[x])
        if (ret > 1000000) or (ret < 0) : 
            print(ret)
    return ret

def energy_map(img):
    # input: image in rgb of size h*w*c, c=3
    # output: energy map: numpy array of size h*w

    h, w, c = img.shape
    E_l = np.zeros([h, w], dtype=np.float32)
    E_u = np.zeros([h, w], dtype=np.float32)
    E_r = np.zeros([h, w], dtype=np.float32)

    for i in range(1, h): 
        for j in range(w): 
            if (j > 0) and (j < h - 1): 
                E_l[i][j] = dist(img[i][j + 1], img[i][j - 1]) + dist(img[i - 1][j], img[i][j - 1])
            else :
                E_l[i][j] = 10000000

            if (j > 0) and (j < h - 1)  : 
                E_u[i][j] = dist(img[i][j + 1], img[i][j - 1])
            else : 
                E_u[i][j] = 10000000

            if (j > 0) and (j < h - 1) : 
                E_r[i][j] = dist(img[i][j + 1], img[i][j - 1]) + dist(img[i - 1][j], img[i][j + 1])
            else : 
                E_r[i][j] = 10000000
            
    return np.array([E_l, E_u, E_r])
