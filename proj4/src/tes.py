import numpy as np
import copy

# one = [[1,0,0],[0,0,0],[0,0,0]]
# two = [[2,0,0],[0,0,0],[0,0,0]]
# three = [[[3,0],[0,0],[0,0]],[[3,0],[0,0],[0,0]],[[3,0],[0,0],[0,0]]]
# mat = [one,two,three]

# print(mat[1][1][1])

# dim = 3

# currstate = np.full((2, dim, dim), -1)
# curr_knowledge = np.full((2, dim, dim), 0)

# print(currstate)
# print(curr_knowledge)

# currstate = np.concatenate([currstate, curr_knowledge])

# print(currstate)

# master = []
# dim = 5
# currstate = np.full((dim, dim), 2)

# master.append(copy.deepcopy(currstate))
# currstate[1][1] = 6
# master.append(copy.deepcopy(currstate))

# print(master[0])
# print(master[1])

def append_local(currstate, x, y, r):

    local_view = np.full((7, 2*r+1, 2*r+1), 0)
    
    for i in range(2*r+1):
        for j in range(2*r+1):
            if 0<=x-r+i<len(gridworld) and 0<=y-r+j<len(gridworld):
                for k in range(7):
                    local_view[k][i][j] = currstate[k][x-r+i][y-r+j]
            else:
                local_view[0][i][j] = 1
                local_view[1][i][j] = -1
    
    print(local_view)
    input_states.append(local_view)

input_states = []

gridworld = [
[1,1,1,1,1],
[2,2,2,2,2],
[3,3,3,3,3],
[1,2,3,4,5],
[1,2,3,4,5]]


# 7x5x5
currstate = [
[[1,1,1,1,1],
[2,2,2,2,2],
[3,3,3,3,3],
[1,2,3,4,5],
[1,2,3,4,5]],

[[1,2,3,4,5],
[1,2,3,4,5],
[4,2,3,4,5],
[1,2,3,4,5],
[1,2,3,4,5]],

[[1,2,3,4,5],
[1,2,3,4,5],
[5,2,3,4,5],
[1,2,3,4,5],
[1,2,3,4,5]],

[[1,2,3,4,5],
[1,2,3,4,5],
[1,2,3,4,5],
[1,2,3,4,5],
[1,2,3,4,5]],

[[1,2,3,4,5],
[1,7,3,4,5],
[1,2,3,4,5],
[1,2,3,4,5],
[1,2,3,4,5]],

[[1,2,3,4,5],
[1,2,3,4,5],
[1,2,3,4,5],
[1,2,3,4,5],
[1,2,3,4,5]],

[[1,2,3,4,5],
[1,7,3,4,5],
[1,2,6,4,5],
[1,2,3,4,5],
[1,2,3,4,5]],
]
# append_local(currstate, 1, 1, 1)
# append_local(currstate, 0, 0, 1)

# append_local(currstate, 2, 2, 2)
append_local(currstate, 1, 1, 2)

# print(input_states)