import numpy as np

master = []
dim = 5
currstate = np.full((dim, dim), 2)

master.append(currstate)

currstate[1][1] = 6

master.append(currstate)

print(master)