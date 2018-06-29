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
                else:
                    f[i] = f[i-1]
        return Seam(f, self.dir)
    
    def transform(self, other):
        if self.dir == other.dir:
            return self.transform_same_dir(other)
        else:
            return self.transform_different_dir(other)
    
    def cut_by(self, other):
        if self.dir == other.dir:
            return self.cut_same_dir(other)
        else:
            return self.cut_different_dir(other)
    
    def cut_same_dir(self, other):
        s = self.get_one_coor()
        t = other.get_one_coor()
        n = len(s)
        f = [-1]*n
        for i in range(n):
            if s[i]<t[i]: f[i] = s[i]
            else: f[i] = s[i] - 1
        return Seam(f, self.dir)
    
    def cut_different_dir(self, other):
        s = self.get_one_coor()
        t = other.get_one_coor()
        n = len(s)-1
        f = [-1]*n
        for i in range(n+1):
            if i<t[s[i]]: f[i] = s[i]
            else: f[i-1] = s[i]
        return Seam(f, self.dir)
    
    def enlarge_by(self, other):
        if self.dir == other.dir:
            return self.enlarge_same_dir(other)
        else:
            return self.enlarge_different_dir(other)
    
    def enlarge_same_dir(self, other):
        s = self.get_one_coor()
        t = other.get_one_coor()
        n = len(s)
        f = [-1]*n
        for i in range(n):
            if s[i]<t[i]: f[i] = s[i]
            else: f[i] = s[i] + 1
        return Seam(f, self.dir)
    
    def enlarge_different_dir(self, other):
        s = self.get_one_coor()
        t = other.get_one_coor()
        n = len(s)+1
        f = [-1]*n
        for i in range(n-1):
            if i<=t[s[i]]: f[i] = s[i]
            else: f[i+1] = s[i]
        for i in range(n):
            if f[i] == -1:
                if i==0:
                    f[i] = f[i+1]
                else:
                    f[i] = f[i-1]
        return Seam(f, self.dir)

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

def horizontal_trans(img, seam):
    if seam.dir == 'horizontal':
        return np.swapaxes(img, 0, 1)
    else:
        return img

def cut_by_seam(img, seam):
    img = horizontal_trans(img, seam)
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
    return horizontal_trans(new, seam)

def enlarge_image_by_seam(img, seam):
    img = horizontal_trans(img, seam)
    positions = seam.get_one_coor()
    assert len(img.shape) == 3
    h, w, c = img.shape
    new = np.zeros((h, w+1, c), dtype=img[0][0][0].dtype)
    
    for i in range(h):
        pos = positions[i]
        new[i][0:pos+1] = img[i][0:pos+1]
        new[i][pos+2:w+1] = img[i][pos+1:w]
        for k in range(c):
            new[i][pos+1][k] = (int(img[i][pos][k]) + int(img[i][pos+1][k])) // 2
    return horizontal_trans(new, seam)


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

def determine_directions(img, func, forward, out_r, out_c):
    r, c = img.shape[0], img.shape[1]
    dr = abs(r - out_r)
    dc = abs(c - out_c)
    dirs = ['vertical']*dc + ['horizontal']*dr
    random.shuffle(dirs)
    return dirs

def resize(ori, func, forward, out_r, out_c, need_seam=False, dirs = None):
    if dirs == None:
        dirs = determine_directions(ori, func, forward, out_r, out_c)

    # cut and get the seams
    img = ori
    carved_seams = []
    len_dir = len(dirs)
    for i in range(len_dir):
        d = dirs[i]
        em = func(img)
        print("process {}/{}".format(i, len_dir))
        img, tmp_seams = carve(img, em, d, num=1, border=1, forward=forward, need_seam=True)
        carved_seams += tmp_seams
    
    get_op = lambda input, output: 'cut' if input>output else 'enlarge'
    row_op = get_op(ori.shape[0], out_r)
    col_op = get_op(ori.shape[1], out_c)
    #if row_op == 'cut' and col_op == 'cut':
    #    return img

    # cut or enlarge by seams
    img = ori
    seams = transform_seams(carved_seams)
    n = len(seams)

    for i in range(n):
        if seams[i].dir == 'vertical':
            op = col_op
        else:
            op = row_op
        if op=='cut':
            img = cut_by_seam(img, seams[i])
            for j in range(i+1, n):
                seams[j] = seams[j].cut_by(seams[i])
        else:
            #print("enlarge", seams[i].dir)
            img = enlarge_image_by_seam(img, seams[i])
            for j in range(i+1, n):
                seams[j] = seams[j].enlarge_by(seams[i])
    if need_seam:
        return img, seams
    else:
        return img

