import numpy as np
from func import conflictcalculator, yeilder

def min_conflict(playfield, size, comp_limit):
    cc = conflictcalculator(size)
    y = yeilder(size)

    breaker = 0
    for check_i in y.get_next_column():
        breaker += 1
        if breaker > comp_limit:
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
    return playfield, breaker