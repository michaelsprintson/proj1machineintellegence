import numpy as np
from func import conflictcalculator, yeilder
from itertools import product, combinations
from collections import Counter
import copy

def min_conflict(playfield, size, comp_limit, printflag = False):
    cc = conflictcalculator(size)
    y = yeilder(size)

    breaker = 0
    for check_i in y.get_next_column():
        breaker += 1
        if breaker > comp_limit:
            if printflag:
                print("no solution found in time")
            break
            
        x = cc.r_calc(playfield[check_i],check_i,size) #calculate if its broken
        broken_flag = False
        for cidx,ridx in enumerate(playfield):
            if cidx != check_i:
                if((ridx in x[check_i]) or (ridx in x[cidx])):
                    broken_flag = True
        # print(check_i,broken_flag)
        if broken_flag: # if its broken, calculate which row it would have least conflicts in
            conflicts = {}
            for k in range(size):
                x = cc.r_calc(k, check_i, size)
                conflicts[k] = len([j for j in [idx for idx,i in enumerate(playfield) if ((i in x[idx]) or (i in x[check_i]))] if j!= check_i])
            # print(conflicts)
            # choose new pos with min conflict
            new_min_c = conflicts[sorted(conflicts, key = lambda k: conflicts[k])[0]]
            new_pos = np.random.choice([i for i,k in conflicts.items() if k == new_min_c])
            # print(playfield[check_i], new_pos)

            # calculate conflicts with new pos
            x = cc.r_calc(new_pos, check_i,size)
            new_conflicts = [j for j in [idx for idx,i in enumerate(playfield) if ((i in x[idx]) or (i in x[check_i]))] if j!= check_i]
            # add conflicts to broken set
            for nc in new_conflicts:
                y.broken_set.add(nc)
            # update new item in field
            # print(check_i, "changed",playfield[check_i], "to", new_pos)
            playfield[check_i] = new_pos
    if printflag:
        print("mc found", playfield, breaker)
    return playfield, breaker

#------------------------------------------------------------------------------------------------------------------

dist_check = lambda a,b : np.around(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2), decimals=2) == 2.24


def list_replace_copy(list, idx, val):
    nl = copy.copy(list)
    nl[idx] = val
    return nl

class game_node():
    def __init__(self, playfield, g = 0, f = 0, h = 0, rc = False, ccol = None, ccolval = 0) -> None:
        l = locals()
        [self.__setattr__(name, l[name]) for name in self.__init__.__code__.co_varnames if name != "self"]
    
    def calc_score(self):
        return self.g+self.f-self.h

    def __eq__(self, other):
        if (self.playfield == other.playfield) and (self.calc_score() == other.calc_score()) and (self.rc == other.rc) and (self.ccol == other.ccol):
            return True
        return False

    def __lt__(self, other): #is self less than other
        o1, o2 = self.calc_score(), other.calc_score()
        if o1 == o2:
            if self.g == other.g:
                if self.f == other.f:
                    if self.h == other.h:
                        if self.rc == other.rc:
                            if self.ccolval == other.ccolval:
                                return True
                            else:
                                return False # objects are equal
                        else:
                            return self.rc
                    else:
                        return self.h > other.h
                else:
                    return self.f < other.f
            else:
                return self.g < other.g
        else:
            return o1 < o2

    def __repr__(self):
        return str(self.playfield) + f" g = {self.g}, f = {self.f}, h = {self.h}, rc = {self.rc}, cc = {self.ccol},{self.ccolval} \n"

def h_calc(pot_playspace, cc, size):
    full_set = {i for i in range(size)}
    open_columns = [idx for idx,i in enumerate(pot_playspace) if i == -1]
    closed_columns = full_set.difference(set(open_columns))
    h = 0
    conflicted_columns = []
    for c1, c2 in combinations(closed_columns,2):
        x = cc.r_calc(pot_playspace[c1], c1, size)
        if((pot_playspace[c2] in x[c1]) or (pot_playspace[c2] in x[c2])):
            conflicted_columns.append(c1)
            conflicted_columns.append(c2)
            h += 1
    return h, dict(Counter(conflicted_columns))

def makegfhrc(pot_playspace, cc, size, kflag = False):
    full_set = {i for i in range(size)}
    g = (sum([True for i in pot_playspace if i != -1]) - 1) * size
    open_columns = [idx for idx,i in enumerate(pot_playspace) if i == -1]
    closed_columns = full_set.difference(set(open_columns))
    h, colsplit = h_calc(pot_playspace, cc, size)
    
    full_spaces = set(product(full_set, open_columns))
    for full_col_idx in closed_columns:
        x = cc.r_calc(pot_playspace[full_col_idx], full_col_idx, size)
        broken_spaces = set()
        for r, c in full_spaces:
            if ((r in x[full_col_idx]) or (r in x[c])):
                broken_spaces.add((r,c))
        full_spaces = full_spaces.difference(broken_spaces)

    num_knights = 0
    if kflag:
        for i in combinations(closed_columns,2):
            a, b = (i[0],pot_playspace[i[0]]),(i[1], pot_playspace[i[1]])
            if dist_check(a,b):
                num_knights += 1
    f = len(full_spaces) + num_knights
    
    rc = len(x:=([pot_playspace[i] for i in closed_columns])) != len(np.unique(x))
    return g, f, h, rc, colsplit

def astar(size, init_playfield, max_iter, printflag = False, kflag = False):
    cc = conflictcalculator(size)
    if not -1 in init_playfield:
        g, f, h, rc, colsplit = makegfhrc(init_playfield, cc, size, kflag=kflag)
        fringe = [game_node(init_playfield, g, f, h, rc, ccol=k, ccolval=v) for k,v in colsplit.items()] #make self-sorting list
    else:
        fringe = [game_node(init_playfield)]
    explored_nodes = list()
    breaker = 0
    while len(fringe) > 0:
        
        op = fringe.pop()
        explored_nodes.append(op)
        if len([idx for idx,i in enumerate(op.playfield) if i == -1]) > 0:
            new_pos =  [idx for idx, i in enumerate(op.playfield) if i == -1][0]# in the future, order by num conflicts per column, consider row conflicts as more important
            # new_pos = np.random.choice([i for i,k in conflicts.items() if k == new_min_c])
            # print(op, "operating on", [idx for idx, i in enumerate(op.playfield) if i == -1], new_pos)
        else:
            new_pos = op.ccol
            # print(op)
        if new_pos is None:
            if printflag:
                print("a found", op.playfield, breaker)
            return op.playfield, breaker
        else:
            breaker += 1
            if breaker > max_iter:
                break
        for pot_col in range(size):
            pot_playspace = list_replace_copy(op.playfield, new_pos, pot_col)
            g, f, h, rc, colsplit = makegfhrc(pot_playspace, cc, size, kflag=kflag)
            
            # print(pot_playspace, g, f, h, "\n")
            if len(colsplit)>0:
                for k,v in colsplit.items():
                    new_node = game_node(pot_playspace, g, f, h, rc, ccol = k, ccolval=v)
                    fringe.append(new_node) if ((new_node not in fringe) and (new_node not in explored_nodes)) else None
            else:
                new_node = game_node(pot_playspace, g, f, h, rc)
                fringe.append(new_node) if ((new_node not in fringe) and (new_node not in explored_nodes)) else None
            # print(x, "\n")
            #calculate g, f, h
        fringe = sorted(fringe)
        # fringe = sorted(fringe)
        # print(fringe)
    if printflag:
        print("nothing")
    return None, breaker