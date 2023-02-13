import numpy as np

class conflictcalculator():
    def __init__(self, size):
        self.filt = lambda x: (y:=np.unique(np.where([0<=i<size for i in x], x, -1)))[y>=0]
        self.r_h = lambda x: [set(self.filt(i)) for i in x]
        self.r_calc = lambda r,c,s : self.r_h(np.array([[r-(i:= np.abs(c-x)), r+i]for x in range(s)]))