import numpy as np
import copy

class yeilder():
    def __init__(self, size):
        self.full_set = {i for i in range(size)}
        self.size = size
        self.broken_set = set()
    
    def update_broken_set(self, addition):
        self.broken_set.add(addition)

    def get_next_column(self):
        goflag = True
        while goflag:
            if len(self.broken_set) > 0:
                self.full_set = {i for i in range(self.size)}
                # print("1")
                yield self.broken_set.pop()
            else:
                yield self.full_set.pop()
                if len(self.full_set) == 0:
                    goflag = False

class conflictcalculator():
    def __init__(self, size):
        self.size = size
        self.filt = lambda x: (y:=np.unique(np.where([0<=i<size for i in x], x, -1)))[y>=0]
        self.r_h = lambda x: [set(self.filt(i)) for i in x]
        self.r_calc = lambda r,c,s : self.r_h(np.array([[r-(i:= np.abs(c-x)), r+i]for x in range(s)]))

def check_valid(playfield, cc, size):
    conflict_flag = 0
    for c, r in enumerate(playfield):
        if r >= 0:
            x = cc.r_calc(r, c, size)
            conflict_flag += len([j for j in [idx for idx,i in enumerate(playfield) if ((i in x[idx]) or (i in x[c]))] if j!= c])
    return conflict_flag == 0

def link_maker(playfield):
    # playfield must be a numpy array, python list, or some other iterable of length 8.
    # Each item of value j at index i represents a queen in column i at row j
    # Columns are counted 0->7 left->right, and rows are counted 0->7 top->bottom
    # Example input : [4, 0, 3, 5, 7, 1, 6, 2]
    pr = lambda x: x if x > 0 else ""
    lx = list(playfield)
    liformula = np.zeros(8, dtype="object")
    for i in range(0,8):
        idx = lx.index(i)
        liformula[i] = f"{pr(idx)}Q{pr(7-idx)}"
    return "https://lichess.org/editor/" + "/".join(list(liformula)) + "_w_-_-_0_1?color=white"

# Quick function to check if the distance between two 2D points is sqrt(5) - knight opposition boolean
dist_check = lambda a,b : (a[0]-b[0])**2 + (a[1]-b[1])**2 == 5


def list_replace_copy(listi, idx, val):
    """
    Function to replace one index in a list and return a copy
    
    listi (list) -> List of at least len (idx)
    idx (int) -> The index of the node to replace
    val (int) -> The value to replace at index (idx)

    Returns: 
    nl (list) -> Copied (listi) with replaced index value
    """
    nl = copy.copy(listi)
    nl[idx] = val
    return nl