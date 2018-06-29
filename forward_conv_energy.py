import numpy as np
from skimage import filters, color
from scipy import signal

# filter for energy map
scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy

kernel_u = np.array([[0, 0, 0],
                      [-1, 0, +1],
                      [0, 0, 0]])

kernel_l = np.array([[0, +1, 0],
                      [-1, 0, 0],
                      [0, 0, 0]])

kernel_r = np.array([[0, +1, 0],
                      [0, 0, -1],
                      [0, 0, 0]])

def energy_map(img):
    # input: image in rgb of size h*w*c, c=3
    # output: energy map: numpy array of size h*w

    gray = color.rgb2gray(img)
    E_u = signal.convolve2d(gray, kernel_u, boundary='symm', mode='same')
    E_u = np.absolute(E_u)

    E_l = signal.convolve2d(gray, kernel_l, boundary='symm', mode='same')
    E_l = np.absolute(E_l)
    E_l = E_l + E_u

    E_r = signal.convolve2d(gray, kernel_r, boundary='symm', mode='same')
    E_r = np.absolute(E_r)
    E_r = E_r + E_u

    return np.array([E_l, E_u, E_r])

