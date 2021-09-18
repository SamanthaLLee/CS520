from array import *
import random

gridworld = []
visited = [] # bool
fringe = []
g = 0 # length of the shortest path
h = 0 # heuristic (manhattan distance)
goal = []
curr = []


def main():
    print("do thing")
    # prompt user to enter dimensions
    dim = input("What is the length of your gridworld? ")
  	p = input("With what probability will a cell be blocked? ")
	while p > 1 or p < 0:
      p = input("Enter a valid probability. ")
    generateGridworld(dim, p)
    solve()
    

# Generates a random gridworld 
def generateGridworld(dim, p):
    # Let each cell independently be blocked with probability p, and empty with probability 1−p.
	for i in range(dim):
    	for j in range(dim):
          rand = random.random()
          if rand < p:
            gridworld[i][j] = 1
            
	# Exclude the upper left corner (chosen to be the start position) and the lower right corner (chosen to be the end position) from being blocked.
    gridworld[0][0] = 0
    gridworld[dim-1][dim-1] = 0
    
          
  	
def solve():
    print("do thing")
    parent = [] # last node
    
    # add start node
    
    # plan shortest presumed path from its current position to the goal.
    path[0] = goal[0] - curr[0]
    path[1] = goal[1] - curr[0]
    
    # attempt to follow this path plan, observing cells in its field of view as it moves
    
    
    
    # update the agent’s knowledge of the environment as ut observes blocked an unblocked cells
    # if the agent discovers a block in its planned path, it re-plans, based on its current knowledge of the environment.
    # the cycle repeats until the agent either a) reaches the target or b) determines that there is no unblocked pathto the target.
  
if __name__ == "__main__":
  	main()

# hi