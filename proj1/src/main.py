from array import *
from queue import PriorityQueue
import random

# remove globals
gridworld = []
visited = []  # bool
fringe = PriorityQueue()
g = 0  # length of the shortest path
h = 0  # heuristic (manhattan distance)
goal = []
curr = []
directions = [[1, 0],
              [-1, 0],
              [0, 1],
              [0, -1]]


def generateGridworld(dim, p):
    """Generates a random gridworld based on user inputs"""
    global curr, goal

    # Let each cell independently be blocked with probability p, and empty with probability 1−p.
    for i in range(dim):
        for j in range(dim):
            rand = random.random()
            if rand < p:
                gridworld[i][j] = 1

    # Exclude the upper left corner (chosen to be the start position) and the lower right corner (chosen to be the end position) from being blocked.
    gridworld[0][0]
    gridworld[dim-1][dim-1] = 0
    curr = [0, 0]
    goal = [dim-1, dim-1]


def solve():
    global curr, goal
    parent = []  # how will this be used?

    # plan shortest presumed path from its current position to the goal.
    # is this needed? or is this part caputured with fringe.get()?
    path[0] = goal[0] - curr[0]
    path[1] = goal[1] - curr[0]

    # attempt to follow this path plan, observing cells in its field of view as it moves

    # this loop examines all possible directions and adds them to PQ
    while curr != goal or not fringe.isEmpty():

        # test all directions (generate children)
        for i in range(len(directions)):
            x = curr[0] + directions[0]
            y = curr[1] + directions[1]
            if isInBounds():
                # case: see unblocked path
                if visited[x][y] == 0 and gridworld[x][y] == 0:
                    g = g + 1
                    f = g + getHeuristic(x, y)
                    fringe.put([x, y], f)

                # case: see blocked path
                # if the agent discovers a block in its planned path, it re-plans, based on its current knowledge of the environment.
                elif gridworld[x][y] == 1:
                    # update the agent’s knowledge of the environment as it observes blocked an unblocked cells
                    visited[x][y] = 1

        # the cycle repeats until the agent either a) reaches the target or b) determines that there is no unblocked pathto the target.

        parent = curr
        curr = fringe.get()


def isInBounds():
    """Determines whether next move is within bounds"""
    return 0 <= curr[0] < len(gridworld) and 0 <= curr[1] < len(gridworld[0])


def getHeuristic(x, y):
    """Calculates Manhattan distance"""
    return abs(x-goal[0]) + abs(y-goal[1])


if __name__ == "__main__":
    print("do thing")
    # prompt user to enter dimensions
    dim = input("What is the length of your gridworld? ")
    p = input("With what probability will a cell be blocked? ")
    while p > 1 or p < 0:
        p = input("Enter a valid probability. ")
    generateGridworld(dim, p)
    solve()
