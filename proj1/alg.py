import numpy as np
from func import conflictcalculator, yeilder
from itertools import product, combinations
from collections import Counter
import copy

def min_conflict(playfield, size, comp_limit, printflag = False):
    """
    Function to compute 8-queens solution given initial starting position with 8 queens placed. 
    
    playfield (list) -> list of len (size) with integers [0-size)
    size (int) -> The size of one side of the square chessboard
    comp_limit (int) -> The amount of nodes to search in the tree
    printflag (boolean) -> Whether or not to print out results

    Returns: 
    If solution not found:
        None
    If solution found
        playfield (list) -> computed 8-queens solution
        breaker (int) -> the amount of nodes checked
    """
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

class game_node():
    """
    Class to define one search state. holds information about problem state and next search path
    """

    def __init__(self, playfield, g = 0, f = 0, h = 0, rc = False, ccol = None, ccolval = 0) -> None:
        """
        Initialization function
    
        playfield (list) -> list of len (size) with integers [0-size)
        g (int) -> The number of fixed column positions in this search state
        f (int) -> The number of avaliable positions left in unfixed columns
        h (int) -> The number of conflicts in fixed columns
        rc (int) -> A boolean to denote whether a row column is present
        ccol (int) -> The next column to search through
        ccolval (int) -> The number of conflicts present at (ccol)
        """
        l = locals()
        [self.__setattr__(name, l[name]) for name in self.__init__.__code__.co_varnames if name != "self"]
    
    def calc_score(self):
        """
        Function to calculate the score of a particular search state

        Returns:
        (int) -> The score
        """
        return self.g+self.f-self.h

    def __eq__(self, other):
        """
        A function to check equality between any two search state nodes

        other (game_node) -> The other node to check equality against

        Returns:
        (boolean) -> Whether or not the nodes are equal
        """
        if (self.playfield == other.playfield) and (self.calc_score() == other.calc_score()) and (self.rc == other.rc) and (self.ccol == other.ccol):
            return True
        return False

    def __lt__(self, other): #is self less than other
        """
        A function to check if this instance of game_node is less than (other)

        other (game_node) -> The other node to check less than property against

        Returns:
        (boolean) -> Whether or not the current instance of game_node is less than (other)
        """
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
        """
        A function to print the current game_node

        Returns:
        (str) -> str representation of game_node
        """
        return str(self.playfield) + f" g = {self.g}, f = {self.f}, h = {self.h}, rc = {self.rc}, cc = {self.ccol},{self.ccolval} \n"

def h_calc(pot_playspace, cc, size):
    """
    Function to compute conflicts in current 8-queens solution
    
    pot_playspace (list) -> list of len (size) with integers [0-size)
    cc (int) -> An instance of conflictcalculator at size (size)
    size (int) -> The size of the chess board

    h (int) -> The number of conflicts found
    (dict) -> A dictionary of columns with the number of conflicts per column
    """
    full_set = {i for i in range(size)}
    open_columns = [idx for idx,i in enumerate(pot_playspace) if i == -1] #find which columns are unfixed
    closed_columns = full_set.difference(set(open_columns)) #find which columns are fixed
    h = 0
    conflicted_columns = []
    for c1, c2 in combinations(closed_columns,2): #For each combination of columns, check if they conflict
        x = cc.r_calc(pot_playspace[c1], c1, size)
        if((pot_playspace[c2] in x[c1]) or (pot_playspace[c2] in x[c2])):
            conflicted_columns.append(c1) #if they do, update the counter and which columns are under conflict
            conflicted_columns.append(c2)
            h += 1
    return h, dict(Counter(conflicted_columns))

def makegfhrc(pot_playspace, cc, size, kflag = False):
    """
    Function to compute 8-queens solution given initial starting position
    
    pot_playspace (list) -> list of len (size) with integers [0-size)
    cc (int) -> An instance of conflictcalculator at size (size)
    size (int) -> The size of the chess board
    kflag (boolean) -> Whether or not to add in knight opposition metric

    g (int) -> The number of fixed column positions in this search state
    f (int) -> The number of avaliable positions left in unfixed columns
    h (int) -> The number of conflicts in fixed columns
    rc (boolean) -> The number of conflicts found
    colsplit (dict) -> A dictionary of columns with the number of conflicts per column
    """
    full_set = {i for i in range(size)}
    g = (sum([True for i in pot_playspace if i != -1]) - 1) * size #calculate g by seeing how many columns are fixed
    open_columns = [idx for idx,i in enumerate(pot_playspace) if i == -1] #find which columns are unfixed
    closed_columns = full_set.difference(set(open_columns)) #find which columns are fixed
    h, colsplit = h_calc(pot_playspace, cc, size) #calculate h
    
    full_spaces = set(product(full_set, open_columns)) #calculate f by checking all spaces in unfixed columns 
    for full_col_idx in closed_columns:
        x = cc.r_calc(pot_playspace[full_col_idx], full_col_idx, size)
        broken_spaces = set()
        for r, c in full_spaces: #if space conflicts, remove from number of spaces remaining
            if ((r in x[full_col_idx]) or (r in x[c])):
                broken_spaces.add((r,c))
        full_spaces = full_spaces.difference(broken_spaces)

    num_knights = 0
    if kflag:
        for i in combinations(closed_columns,2): #for each combination of queens in fixed columns, update counter if those queens are in knight opposition
            a, b = (i[0],pot_playspace[i[0]]),(i[1], pot_playspace[i[1]])
            if dist_check(a,b):
                num_knights += 1
    f = len(full_spaces) + num_knights
    
    rc = len(x:=([pot_playspace[i] for i in closed_columns])) != len(np.unique(x)) # calculate rc by checking for row conflicts
    return g, f, h, rc, colsplit

def astar(size, init_playfield, max_iter, printflag = False, kflag = False):
    """
    Function to compute 8-queens solution given initial starting position with anywhere between 1-8 queens placed. 

    size (int) -> The size of one side of the square chessboard
    init_playfield (list) -> list of len (size) with integers [0-size)
    max_iter (int) -> The amount of nodes to search in the tree
    printflag (boolean) -> Whether or not to print out results
    kflag (boolean) -> Whether or not to include knight opposition metric

    Returns: 
    If solution not found:
        None
        breaker (int) -> the amount of nodes checked
    If solution found
        playfield (list) -> computed 8-queens solution
        breaker (int) -> the amount of nodes checked
    """
    cc = conflictcalculator(size)
    if not -1 in init_playfield: #if full initial starting position, calculate conflicts
        g, f, h, rc, colsplit = makegfhrc(init_playfield, cc, size, kflag=kflag)
        fringe = [game_node(init_playfield, g, f, h, rc, ccol=k, ccolval=v) for k,v in colsplit.items()]
    else: #if partial position, start at this node in fringe
        fringe = [game_node(init_playfield)]
    explored_nodes = list()
    breaker = 0
    while len(fringe) > 0: #while there is a node in the search space left to explore
        
        op = fringe.pop() #select node in the search space
        explored_nodes.append(op)
        if len([idx for idx,i in enumerate(op.playfield) if i == -1]) > 0: #calculate column to explore in this computation
            new_pos =  [idx for idx, i in enumerate(op.playfield) if i == -1][0]# in the future, order by num conflicts per column, consider row conflicts as more important
            # new_pos = np.random.choice([i for i,k in conflicts.items() if k == new_min_c])
        else:
            new_pos = op.ccol
        if new_pos is None: #if we have found a solution, return solution
            if printflag:
                print("a found", op.playfield, breaker)
            return op.playfield, breaker
        else: #if no solution, update iterator
            breaker += 1
            if breaker > max_iter:
                break
        for pot_col in range(size): #expand search space from that node
            pot_playspace = list_replace_copy(op.playfield, new_pos, pot_col) #for each child node, find playfield and parameters
            g, f, h, rc, colsplit = makegfhrc(pot_playspace, cc, size, kflag=kflag)
            
            if len(colsplit)>0: #if there are conflicts, add a copy of this node with unique columns to explore next
                for k,v in colsplit.items():
                    new_node = game_node(pot_playspace, g, f, h, rc, ccol = k, ccolval=v)
                    fringe.append(new_node) if ((new_node not in fringe) and (new_node not in explored_nodes)) else None
            else: #if no conflicts but not all columns are explored, add each search node to fringe
                new_node = game_node(pot_playspace, g, f, h, rc)
                fringe.append(new_node) if ((new_node not in fringe) and (new_node not in explored_nodes)) else None
        
        fringe = sorted(fringe) #sort the fringe to ensure pop function works
    if printflag: 
        print("nothing")
    return None, breaker #if no solution found, return None