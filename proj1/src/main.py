from array import *
from queue import PriorityQueue
import numpy as np
import random

from .cell import Cell

# remove globals
gridworld = []
goal = []
path = []


def generateGridworld(dim, p):
    """Generates a random gridworld based on user inputs"""
    global goal, gridworld

    # Cell(g, h, f, blocked, seen, parent)
    gridworld = [[Cell(x, y) for x in range(dim)] for y in range(dim)]

    # Let each cell independently be blocked with probability p, and empty with probability 1−p.
    for i in range(dim):
        for j in range(dim):
            rand = random.random()
            if rand < p:
                gridworld[i][j].blocked = 1

    # Exclude the upper left corner(chosen to be the start position) and
    # the lower right corner(chosen to be the end position) from being blocked.
    # Initialize starting cell values
    gridworld[0][0].g = 0
    gridworld[0][0].h = getHeuristic(0, 0)
    gridworld[0][0].f = gridworld[0][0].g + gridworld[0][0].h
    gridworld[0][0].seen = True

    gridworld[0][0].blocked = 0
    gridworld[dim-1][dim-1].blocked = 0
    goal = [dim-1, dim-1]

    print(np.matrix(gridworld))


def AStar():
    global goal, path, gridworld
    g = 0  # Length of the shortest path
    fringe = PriorityQueue()

    directions = [[1, 0],
                  [-1, 0],
                  [0, 1],
                  [0, -1]]

    path = Cell(-1, -1)  # Dummy
    ptr = path

    # Add start to fringe
    fringe.put(gridworld[0][0].f, gridworld[0, 0])

    # Generate all children and add to fringe
    while not fringe.isEmpty() or curr != goal:
        curr = fringe.get()

        for i in range(len(directions)):
            x = curr[0] + directions[i][0]
            y = curr[1] + directions[i][1]
            if isInBounds(curr):
                if gridworld[x][y].seen == False:
                    g = g + 1
                    f = g + getHeuristic(x, y)
                    fringe.put(f, gridworld[x, y])

        ptr.child = curr
        prevCell = ptr
        ptr = ptr.child
        ptr.parent = prevCell

    return path.child


def solve():
    global goal, path, gridworld
    print("do thing")

    # plan shortest presumed path from its current position to the goal.
    # attempt to follow this path plan, observing cells in its field of view as it moves

    # if the agent discovers a block in its planned path, it re-plans, based on its current knowledge of the environment.
    # update the agent’s knowledge of the environment as it observes blocked an unblocked cells
    # elif gridworld[x][y].blocked == True:
    #     gridworld[x][y].seen == True


def isInBounds(curr):
    global gridworld
    """Determines whether next move is within bounds"""
    return 0 <= curr[0] < len(gridworld) and 0 <= curr[1] < len(gridworld[0])


def getHeuristic(x, y):
    global goal
    """Calculates Manhattan distance"""
    return abs(x-goal[0]) + abs(y-goal[1])


if __name__ == "__main__":
    # prompt user to enter dimensions
    dim = input("What is the length of your gridworld? ")
    p = input("With what probability will a cell be blocked? ")
    while float(p) > 1 or float(p) < 0:
        p = input("Enter a valid probability. ")
    generateGridworld(int(dim), float(p))
    solve()
