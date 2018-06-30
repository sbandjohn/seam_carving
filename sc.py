import numpy as np
import random
import math
from numba import jit

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

#@jit
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

#@jit
def dp_forward(E_l, E_u, E_r, border):

    h, w = E_l.shape
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


def carve_vertical_once(img, energy, border = 1, forward = False, need_seam=False, abort_energy=False):
    #direction = "vertical" or "horizontal"
    #border: prevent border from being cut
    #output: carved img, energy, seam (if needed)

    h, w, c = img.shape
    if forward == False : 
        #print("     dp...")
        value, dir = dp(energy, border)
    else : 
        #print("       dp forward...")
        value, dir = dp_forward(E_l = energy[0], E_u = energy[1], E_r = energy[2], border = border)

    choice = -1
    for i in range(border, w-border):
        if choice==-1 or value[h-1][i] < value[h-1][choice]:
            choice = i
    minv = value[h-1][choice]

    seam = backtrace(dir, h, choice)
    #print("     cut by seam...")
    img = cut_by_seam(img, seam)

    if not abort_energy:
        if forward == False : 
            energy = cut_by_seam(energy, seam)
        else : 
            sb = []
            for i in range(3) :
                sb.append(cut_by_seam(energy[i], seam))
            energy = np.array(sb)
    else:
        energy = None
    
    if need_seam:
        return img, energy, minv, seam
    else:
        return img, energy, minv

def carve(img, func, forward, direction, num = 1, border = 1, need_seam=False, need_v=False):
    #input
    #   num: number of seams to cut
    #   direction = "vertical" or "horizontal"
    #   border: prevent border from being cut
    #output: carved img

    #print("calculating energy...")
    if direction == "horizontal":
        if forward == False:
            energy = np.swapaxes(func(img), 0, 1)
        img = np.swapaxes(img, 0, 1)
        if forward == True:
            energy = func(img)
    else:
        energy = func(img)
    
    seams = []
    v = 0
    for i in range(num):
        final = (i==num-1)
        if need_seam:
            img, energy, minv, seam = carve_vertical_once(img, energy, border, forward, 
                                                    need_seam=need_seam, abort_energy=final)
            if direction == "horizontal":
                seam = seam.flip()
            seams.append(seam)
        else:
            img, minv, energy = carve_vertical_once(img, energy, border, forward, 
                                                need_seam, abort_energy=final)
        v += minv

    if direction == "horizontal":
        img = np.swapaxes(img, 0, 1)
    if need_seam:
        if not need_v:
            return img, seams
        else:
            return img, seams, v
    else:
        return img
    return None

def opt_multi_carve(ori, func, forward, dr, dc, need_seam=True, need_v=False):
    print(" optimal carving....")
    arr = [ [None for j in range(dc+1)] for i in range(dr+1) ]
    arr[0][0] = (ori, None, 0, 'end')
    for i in range(dr+1):
        for j in range(dc+1):
            if (i==0 and j==0):
                continue
            if i>0:
                pre1 = arr[i-1][j]
                img1, seams1, v1 = carve(pre1[0], func, forward, 'horizontal', 
                                        num=1, border=1, need_seam=True, need_v=True)
                tot1 = pre1[2] + v1
            if j>0:
                pre2 = arr[i][j-1]
                img2, seams2, v2 = carve(pre2[0], func, forward, 'vertical', 
                                        num=1, border=1, need_seam=True, need_v=True)
                tot2 = pre2[2] + v2
            if j==0 or (i>0 and tot1 < tot2):
                arr[i][j] = (img1, seams1, tot1, 'horizontal')
            else:
                arr[i][j] = (img2, seams2, tot2, 'vertical')
            print(i, j, dr, dc, arr[i][j][2], arr[i][j][3])
    img = arr[dr][dc][0]
    if need_seam:
        v = arr[dr][dc][2]
        seams = []
        x, y = dr, dc
        while (x>0 or y>0):
            assert arr[x][y][1][0].dir == arr[x][y][3]
            seams = arr[x][y][1] + seams
            if arr[x][y][3] == 'horizontal':
                x -= 1
            else:
                y -= 1
        if need_v:
            return img, seams, v
        else:
            return img, seams
    else:
        return img

def random_directions(dr, dc):
    dirs = ['vertical']*dc + ['horizontal']*dr
    random.shuffle(dirs)
    return dirs

min_size = 20
max_step = 3

def one_sequence(a, b, step_size):
    res = [a]
    t = a
    while (a<b and t<b) or (a>b and t>b):
        if a<b:
            t += step_size
            if t>b: t = b
        else:
            t -= step_size
            if t<b: t = b
        res.append(t)
    return res

def resize_sequence(r, c, out_r, out_c):
    tot = abs(r-out_r) + abs(c-out_c)
    step_size = max(min_size, math.ceil(tot/max_step))

    rs = one_sequence(r, out_r, step_size)
    cs = one_sequence(c, out_c, step_size)
    n1 = len(rs)
    n2 = len(cs)
    if n1<n2:
        rs += [rs[-1]]*(n2-n1)
    if n2<n1:
        cs += [cs[-1]]*(n1-n2)
    return [[rs[i], cs[i]] for i in range(1, max(n1, n2))]

def resize_once(ori, func, forward, out_r, out_c, need_seam=False, opt=False):
    dr = ori.shape[0] - out_r
    dc = ori.shape[1] - out_c
    if opt == True:
        # carve multiple times in opt directions
        img, carved_seams, v = opt_multi_carve(ori, func, forward, abs(dr), abs(dc), need_seam=True, need_v=True)
        print("   cost of opt directions:", v)
    else:
        #randomly choose directions to carve
        dirs = random_directions(abs(dr), abs(dc))
        img = ori
        carved_seams = []
        len_dir = len(dirs)
        v = 0
        for i in range(len_dir):
            d = dirs[i]
            print("  resizing: {}/{}".format(i, len_dir))
            img, tmp_seams, minv = carve(img, func, forward, d, num=1, border=1, need_seam=True, need_v=True)
            carved_seams += tmp_seams
            v += minv
        print("   cost of random directions:", v)
    
    get_op = lambda dx: 'cut' if dx>0 else 'enlarge'
    row_op = get_op(dr)
    col_op = get_op(dc)
    if row_op == 'cut' and col_op == 'cut':
        if need_seam:
            return img, carved_seams
        else:
            return img

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

def resize_multi(ori, func, forward, out_r, out_c, opt=False):
    seq = resize_sequence(ori.shape[0], ori.shape[1], out_r, out_c)
    len_s = len(seq)
    cnt = 0
    for s in seq:
        print("total step: {}/{}".format(cnt, len_s))
        cnt += 1
        ori = resize_once(ori, func, forward, s[0], s[1], need_seam=False, opt=opt)
    return ori
