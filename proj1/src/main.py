from array import *
from queue import PriorityQueue
import numpy as np
import random
from cell import Cell

# Global gridworld of Cell objects
gridworld = []

# Global goal cell
goal = []


def generategridworld(dim, p):
    """Generates a random gridworld based on user inputs"""
    global goal, gridworld

    # Cells are constructed in the following way:
    # Cell(g, h, f, blocked, seen, parent)
    gridworld = [[Cell(x, y) for x in range(dim)] for y in range(dim)]

    id = 0

    # Let each cell independently be blocked with probability p, and empty with probability 1−p.
    for i in range(dim):
        for j in range(dim):
            gridworld[i][j].id = id
            id = id + 1
            rand = random.random()
            if rand < p:
                gridworld[i][j].blocked = 1

    # Set the goal node
    goal = gridworld[dim-1][dim-1]

    # Ensure that the start and end positions are unblocked
    gridworld[0][0].blocked = 0
    goal.blocked = 0

    # Initialize starting cell values
    gridworld[0][0].g = 0
    gridworld[0][0].h = getheuristic(0, 0)
    gridworld[0][0].f = gridworld[0][0].g + gridworld[0][0].h
    gridworld[0][0].seen = True

    print(np.matrix(gridworld))


def astar(start):
    """Performs the A* algorithm on the gridworld

    Args:
        start (Cell): The cell from which A* will find a path to the goal

    Returns:
        Cell: The head of a Cell linked list containing the shortest path
    """
    global goal, gridworld
    fringe = PriorityQueue()
    fringeSet = set()

    # Vectors that represent the four cardinal directions
    directions = [(1, 0),
                  (-1, 0),
                  (0, 1),
                  (0, -1)]

    # Dummy cell to anchor final path
    path = Cell(-1, -1)

    # Pointer to move along path
    ptr = path

    # Add start to fringe
    curr = start
    fringe.put((curr.f, curr))
    fringeSet.add(curr.id)

    # Generate all valid children and add to fringe
    # Terminate loop if fringe is empty or if path has reached goal
    while len(fringeSet) != 0 and curr != goal:
        f, curr = fringe.get()
        fringeSet.remove(curr.id)
        for x, y in directions:
            xx = curr.x + x
            yy = curr.y + y

            if isinbounds([xx, yy]):
                nextCell = gridworld[xx][yy]
                # Add children to fringe if inbounds AND unblocked and unseen
                if not (nextCell.blocked and nextCell.seen):
                    # Add child if not already in fringe
                    # If in fringe, update child in fringe if old g value > new g value
                    if(not nextCell.id in fringeSet or nextCell.g > curr.g + 1):
                        nextCell.g = curr.g + 1
                        nextCell.h = getheuristic(xx, yy)
                        nextCell.f = nextCell.g + nextCell.h
                        fringe.put((nextCell.f, nextCell))
                        fringeSet.add(nextCell.id)

        # Adds curr cell to return doubly linked list
        ptr.child = curr
        prevCell = ptr
        ptr = ptr.child
        ptr.parent = prevCell

    return path.child


# def exists(self, item):
#     return item in (x[1] for x in self)


def solve():
    """
    Solves the gridworld using Repeated Forward A*.
    """
    global goal, gridworld

    path = astar(gridworld[0][0])

    curr = path

    while(curr is not None):
        print(curr.x, curr.y)
        curr = curr.child

    # print(path)

    # curr = path

    # while(True):
    #     if(curr.child is None):
    #         # Goal found
    #         return path

    #     # Run into blocked cell
    #     if curr.blocked == True:
    #         curr.seen == True
    #         path = astar(curr.parent)
    #         curr = path

    #     # Continue along A* path
    #     else:
    #         curr.seen = True
    #         curr = curr.child

    # plan shortest presumed path from its current position to the goal.
    # attempt to follow this path plan, observing cells in its field of view as it moves

    # if the agent discovers a block in its planned path, it re-plans, based on its current knowledge of the environment.
    # update the agent’s knowledge of the environment as it observes blocked an unblocked cells
    # elif gridworld[x][y].blocked == True:
    #     gridworld[x][y].seen == True


def isinbounds(curr):
    """Determines whether next move is within bounds"""
    global gridworld
    return 0 <= curr[0] < len(gridworld) and 0 <= curr[1] < len(gridworld[0])


def getheuristic(x, y):
    """Calculates Manhattan distance"""
    global goal
    return abs(x-goal.x) + abs(y-goal.y)


def isfloat(str):
    """Determines whether a given string can be converted to float"""
    try:
        float(str)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    dim = input("What is the length of your gridworld? ")
    while not dim.isdigit() or int(dim) < 0:
        dim = input("Enter a valid length. ")
    p = input("With what probability will a cell be blocked? ")
    while not isfloat(p) or float(p) > 1 or float(p) < 0:
        p = input("Enter a valid probability. ")
    generategridworld(int(dim), float(p))
    solve()

# new priority queue
# have dict alongside q
