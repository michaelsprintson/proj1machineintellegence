import numpy as np

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
    def generate_link(self, x):
        pr = lambda x: x if x > 0 else ""
        lx = list(x)
        liformula = np.zeros(self.size, dtype="object")
        for i in range(0,self.size):
            idx = lx.index(i)
            liformula[i] = f"{pr(idx)}Q{pr(self.size-1-idx)}"
        return "https://lichess.org/editor/" + "/".join(list(liformula)) + "_w_-_-_0_1?color=white", x