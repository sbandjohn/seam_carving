import numpy as np
import random

class Seam:
    def __init__(self, one_coor=None, dir=None):
        if one_coor:
            self.make_from_one_coor(one_coor, dir)
        else:
            self.one_coor = one_coor
            self.dir = dir
    
    def make_from_coor(self, coor, dir):
        self.coor = coor
        self.dir = dir
    
    def make_from_one_coor(self, one_coor, dir):
        n = len(one_coor)
        if dir=='vertical':
            coor = [[i, one_coor[i]] for i in range(n)]
        else:
            coor = [[one_coor[i], i] for i in range(n)]
        self.dir = dir
        self.coor = coor
    
    def get_one_coor(self):
        if self.dir == 'vertical':
            return self.get_y_coor()
        else:
            return self.get_x_coor()
    
    def get_x_coor(self):
        return [xy[0] for xy in self.coor]
    
    def get_y_coor(self):
        return [xy[1] for xy in self.coor]

    def transform_same_dir(self, other):
        s = self.get_one_coor()
        t = other.get_one_coor()
        n = len(s)
        assert n == len(t)
        f = [-1]*n
        for i in range(n):
            if s[i]<t[i]:
                f[i] = s[i]
            else:
                f[i] = s[i] + 1
        return Seam(f, self.dir)
    
    def transform_different_dir(self, other):
        s = self.get_one_coor()
        t = other.get_one_coor()
        n = len(s) + 1
        f = [-1]*n
        for i in range(n-1):
            if i < t[s[i]]:
                f[i] = s[i]
            else:
                f[i+1] = s[i]
        for i in range(n):
            if f[i] == -1:
                if i==0:
                    f[i] = f[i+1]
                if i==(n-1):
                    f[i] = f[i-1]
                else:
                    f[i] = f[random.choice((i-1, i+1))]
        return Seam(f, self.dir)
    
    def transform(self, other):
        if self.dir == other.dir:
            return self.transform_same_dir(other)
        else:
            return self.transform_different_dir(other)
    
    def flip(self):
        one_coor = self.get_one_coor()
        if self.dir == 'vertical':
            dir = 'horizontal'
        else:
            dir = 'vertical'
        return Seam(one_coor, dir)

def transform_seams(old_seams):
    n = len(old_seams)
    seams = [x for x in old_seams]
    for i in range(n-2, -1, -1):
        for j in range(i+1, n):
            seams[j] = seams[j].transform(seams[i])
    return seams

def dp(E, border):
    h, w = E.shape
    leftmost = border
    rightmost = w-1-border
    dir = np.zeros(E.shape, dtype=np.int64)
    value = np.array(E)
    for i in range(1, h):
        for j in range(leftmost, rightmost+1):
            if j ==leftmost:
                l = j; r = j+1
            elif j == rightmost:
                l = j-1; r = j
            else:
                l = j-1; r = j+1
            t = l
            for k in range(l+1, r+1):
                if value[i-1][k] < value[i-1][t]:
                    t = k
            value[i][j] = value[i-1][t] + E[i][j]
            dir[i][j] = t - j
    return value, dir

def dp_forward(E_l, E_u, E_r, border):

    h, w = E_l.shape
    print("shape:", h, w)
    leftmost = border
    rightmost = w-1-border
    dir = np.zeros(E_l.shape, dtype=np.int64)
    value = np.array(E_l)

    for i in range(1, h):
        for j in range(leftmost, rightmost+1):

            temp = 10000000

            if j > leftmost and value[i - 1][j - 1] + E_l[i][j] < temp : 
                temp = value[i - 1][j - 1] + E_l[i][j]
                t = j - 1

            if value[i - 1][j] + E_u[i][j] < temp : 
                temp = value[i - 1][j] + E_u[i][j]
                t = j

            if j < rightmost and value[i - 1][j + 1] + E_r[i][j] < temp : 
                temp = value[i - 1][j + 1] + E_r[i][j]
                t = j + 1

            dir[i][j] = t - j
            value[i][j] = temp

    return value, dir

def backtrace(dir, h, pos):
    positions = [-1 for i in range(h)]
    for i in range(h-1, 0, -1):
        positions[i] = pos
        pos += dir[i][pos]
    positions[0] = pos
    return Seam(positions, 'vertical')

def cut_by_seam(img, seam):
    positions = seam.get_one_coor()
    if len(img.shape) == 3:
        h, w, c = img.shape
        new = np.zeros((h, w-1, c), dtype=img[0][0][0].dtype)
    else:
        h, w = img.shape
        new = np.zeros((h, w-1), dtype=img[0][0].dtype)
    for i in range(h):
        pos = positions[i]
        new[i][0:pos] = img[i][0:pos]
        new[i][pos:w-1] = img[i][pos+1:w]
    return new

def carve_vertical_once(img, energy, border = 1, forward = False, need_seam=False):
    #direction = "vertical" or "horizontal"
    #border: prevent border from being cut
    #output: carved img, energy

    h, w, c = img.shape
    print("dp...")
    if forward == False : 
        value, dir = dp(energy, border)
    else : 
        value, dir = dp_forward(E_l = energy[0], E_u = energy[1], E_r = energy[2], border = border)

    choice = -1
    for i in range(border, w-border):
        if choice==-1 or value[h-1][i] < value[h-1][choice]:
            choice = i
    
    seam = backtrace(dir, h, choice)
    print("cut by seam...")
    img = cut_by_seam(img, seam)

    if forward == False : 
        energy = cut_by_seam(energy, seam)
    else : 
        sb = []
        for i in range(3) :
            sb.append(cut_by_seam(energy[i], seam))
        energy = np.array(sb)
    
    if need_seam:
        return img, energy, seam
    else:
        return img, energy

def carve(img, energy, direction, num = 1, border = 1, forward = False, need_seam=False):
    #input
    #   num: number of seams to cut
    #   direction = "vertical" or "horizontal"
    #   border: prevent border from being cut
    #output: carved img

    if direction == "horizontal":
        img = np.swapaxes(img, 0, 1)
        if forward == False : 
            energy = np.swapaxes(energy, 0, 1)
        else : 
            for i in range(3) :
                print(energy[i].shape)
                energy[i] = np.swapaxes(energy[i], 0, 1)
            #energy = np.swapaxes(energy, 1, 2)
    
    seams = []
    for i in range(num):
        if need_seam:
            img, energy, seam = carve_vertical_once(img, energy, border, forward, need_seam)
            if direction == "horizontal":
                seam = seam.flip()
            seams.append(seam)
        else:
            img, energy = carve_vertical_once(img, energy, border, forward, need_seam)

    if direction == "horizontal":
        img = np.swapaxes(img, 0, 1)
    if need_seam:
        return img, seams
    else:
        return img

"""
def first_k_seams(img, energy, direction, k=1, border = 1, forward = False):
    if direction == "horizontal":
        img = np.swapaxes(img, 0, 1)
        if forward == False : 
            energy = np.swapaxes(energy, 0, 1)
        else : 
            for i in range(3) : 
                energy[i] = np.swapaxes(energy[i], 0, 1)
      
    if forward == False : 
        value, dir = dp(energy, border)
    else : 
        value, dir = dp_forward(E_l = energy[0], E_u = energy[1], E_r = energy[2], border = border)

    h, w = img.shape[0], img.shape[1]
    leftmost = border; rightmost = w-1-border
    ps = [ (value[h-1][i], i) for i in range(leftmost, rightmost+1) ]
    ps.sort(key = lambda p: p[0])
    traces = [ backtrace(dir, h, ps[i][1]) for i in range(k) ]

    if direction == "vertical":
        seams = []
        for trace in traces:
            seam = [(i, trace[i]) for i in range(h)]
            seams.append(seam)
    else:
        seams = []
        for trace in traces:
            seam = [(trace[i], i) for i in range(h)]
            seams.append(seam)
    return seams
"""