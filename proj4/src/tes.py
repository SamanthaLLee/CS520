import numpy as np
import copy

# one = [[1,0,0],[0,0,0],[0,0,0]]
# two = [[2,0,0],[0,0,0],[0,0,0]]
# three = [[[3,0],[0,0],[0,0]],[[3,0],[0,0],[0,0]],[[3,0],[0,0],[0,0]]]
# mat = [one,two,three]

# print(mat[1][1][1])

dim = 3

currstate = np.full((2, dim, dim), -1)
curr_knowledge = np.full((2, dim, dim), 0)

print(currstate)
print(curr_knowledge)

currstate = np.concatenate([currstate, curr_knowledge])

print(currstate)

# master = []
# dim = 5
# currstate = np.full((dim, dim), 2)

# master.append(copy.deepcopy(currstate))
# currstate[1][1] = 6
# master.append(copy.deepcopy(currstate))

# print(master[0])
# print(master[1])