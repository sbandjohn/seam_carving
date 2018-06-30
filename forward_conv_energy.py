import numpy as np
from skimage import filters, color, io
from scipy import signal

kernel_u = np.array([[0, 0, 0],
                      [-1, 0, +1],
                      [0, 0, 0]])

kernel_l1 = np.array([[0, 0, 0],
                      [0, 0, -1],
                      [0, +1, 0]])

kernel_l2 = np.array([[0, 0, 0],
                      [-1, 0, +1],
                      [0, 0, 0]])

kernel_r1 = np.array([[0, 0, 0],
                      [-1, 0, 0],
                      [0, +1, 0]])

kernel_r2 = np.array([[0, 0, 0],
                      [-1, 0, 1],
                      [0, 0, 0]])

def energy_map(img):
    # input: image in rgb of size h*w*c, c=3
    # output: energy map: numpy array of size h*w

    #gray = color.rgb2gray(img)
    h, w, c = img.shape
    aha = [img[..., k] for k in range(c)]
    E_u = np.zeros([h, w])
    E_l = np.zeros([h, w])
    E_r = np.zeros([h, w])

    for k in range(c):
        E_u += np.absolute(signal.convolve2d(aha[k], kernel_u, boundary='symm', mode='same'))
        E_l += np.absolute(signal.convolve2d(aha[k], kernel_l1, boundary='symm', mode='same'))
        E_l += np.absolute(signal.convolve2d(aha[k], kernel_l2, boundary='symm', mode='same'))
        E_r += np.absolute(signal.convolve2d(aha[k], kernel_r1, boundary='symm', mode='same'))
        E_r += np.absolute(signal.convolve2d(aha[k], kernel_r2, boundary='symm', mode='same'))

    return np.array([E_l, E_u, E_r], dtype=np.float64)

def test():
    img = io.imread('dolphin.jpg')
    import forward_energy as fe
    m2 = fe.energy_map(img)
    m2 = energy_map(img)
    print(m2[0])
    for i in range(5):
        print()
    print(m2[0])
    #plt.figure()
    #plt.imshow(m1[0])

    #plt.show()

if __name__=="__main__":
    test()