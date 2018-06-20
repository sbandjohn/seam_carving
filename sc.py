import numpy as np

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
                temp = value[i - 1][j] + E_l[i][j]
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
    return positions

def cut_by_seam(img, seam):
    if len(img.shape) == 3:
        h, w, c = img.shape
        new = np.zeros((h, w-1, c), dtype=img[0][0][0].dtype)
    else:
        h, w = img.shape
        new = np.zeros((h, w-1), dtype=img[0][0].dtype)
    for i in range(h):
        pos = seam[i]
        new[i][0:pos] = img[i][0:pos]
        new[i][pos:w-1] = img[i][pos+1:w]
    return new

def carve_vertical_once(img, energy, border = 1, forward = False):
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

    return img, energy

def carve(img, energy, direction, num = 1, border = 1, forward = False):
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
    
    for i in range(num):
        img, energy = carve_vertical_once(img, energy, border, forward)

    if direction == "horizontal":
        img = np.swapaxes(img, 0, 1)
    return img

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
