import numpy as np
from skimage import filters, color
from scipy import signal

# filter for energy map
scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy

def energy_map(img):
    # input: image in rgb of size h*w*c, c=3
    # output: energy map: numpy array of size h*w
    gray = color.rgb2gray(img)
    grad = signal.convolve2d(gray, scharr, boundary='symm', mode='same')
    energy = np.absolute(grad)
    return energy
