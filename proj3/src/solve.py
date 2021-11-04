import random
from array import *
from queue import PriorityQueue
from cell import Cell
from terrain import Terrain
import time

# Global gridworld of Cell objects
gridworld = []

# Global goal cell
goal = None

# Vectors that represent directions
cardinaldirections = [(1, 0), (-1, 0), (0, 1), (0, -1)]
alldirections = [(1, 0), (-1, 0), (0, 1), (0, -1),
                 (-1, -1), (1, 1), (1, -1), (-1, 1)]

numcellsprocessed = 0
trajectorylen = 0
dim = 0
totalplanningtime = 0
finaldiscovered = False
fullgridworld = False


def generategridworld(d, p):
    """Generates a random gridworld based on user inputs"""
    global goal, gridworld, dim
    dim = d
    # Cells are constructed in the following way:
    # Cell(g, h, f, blocked, seen, parent)
    gridworld = [[Cell(x, y) for y in range(dim)] for x in range(dim)]
    id = 0

    terrains = [Terrain.FLAT, Terrain.HILLY, Terrain.FOREST]

    # Let each cell independently be blocked with probability p, and empty with probability 1−p.
    for i in range(dim):
        for j in range(dim):
            curr = gridworld[i][j]

            # Determine block status, randomly decide terrain
            rand = random.random()
            if rand < p:
                curr.blocked = 1
            else:
                curr.terrain = random.choice(terrains)

            # Assign ID
            curr.id = id
            id += 1

    # Set the goal node
    goal = gridworld[dim-1][dim-1]

    # Ensure that the start and end positions are unblocked
    gridworld[0][0].blocked = 0
    goal.blocked = 0

    # Initialize starting cell values
    gridworld[0][0].g = 1
    gridworld[0][0].h = get_weighted_manhattan_distance(0, 0, goal.x, goal.y)
    gridworld[0][0].f = gridworld[0][0].g + gridworld[0][0].h
    gridworld[0][0].seen = True


def printGridworld():
    """Prints out the current state of the gridworld.
    Key:
            B: blocked and seen
            b: blocked and unseen
            *: current path
            ' ': free space
    """
    global gridworld
    leng = len(gridworld)

    string = ''
    for i in range(leng):
        string += ('-'*(leng*2+1) + '\n')
        for j in range(leng):
            string += '|'
            curr = gridworld[i][j]
            if curr.blocked and curr.seen:
                string += 'B'
            elif curr.blocked and not curr.seen:
                string += 'b'
            elif not curr.blocked and curr.seen:
                string += '*'
            else:
                string += ' '
        string += '|\n'
    string += ('-'*(leng*2+1))
    print(string)


def astar(start, agent):
    """Performs the A* algorithm on the gridworld
    Args:
        start (Cell): The cell from which A* will find a path to the goal
    Returns:
        Cell: The head of a Cell linked list containing the shortest path
        int: Length of returned final path
    """
    global goal, gridworld, finaldiscovered, fullgridworld, cardinaldirections, numcellsprocessed, totalplanningtime
    starttime = time.time()
    fringe = PriorityQueue()
    fringeSet = set()
    seenSet = set()
    astarlen = 0
    goalfound = False

    curr = start
    fringe.put((curr.f, curr))
    fringeSet.add(curr.id)

    # Generate all valid children and add to fringe
    # Terminate loop if fringe is empty or if path has reached goal
    while len(fringeSet) != 0:
        f, curr = fringe.get()
        if curr is goal:
            goalfound = True
            break
        if curr.id not in fringeSet:
            continue
        fringeSet.remove(curr.id)
        seenSet.add(curr.id)
        numcellsprocessed += 1
        for x, y in cardinaldirections:
            xx = curr.x + x
            yy = curr.y + y

            if is_in_bounds([xx, yy]):
                nextCell = gridworld[xx][yy]
                validchild = ((agent == 1 or agent == 2) and not (nextCell.blocked and nextCell.seen)) or (
                    (agent == 3 or agent == 4) and not (nextCell.blocked and nextCell.confirmed))
                # Add children to fringe if inbounds AND unblocked and unseen
                if (finaldiscovered and nextCell.seen) or not finaldiscovered:
                    if validchild or (fullgridworld and not nextCell.blocked):
                        # Add child if not already in fringe
                        # If in fringe, update child in fringe if old g value > new g value
                        if(((not nextCell.id in fringeSet) or (nextCell.g > curr.g + 1)) and nextCell.id not in seenSet):
                            nextCell.parent = curr
                            nextCell.g = curr.g + 1
                            nextCell.h = get_weighted_manhattan_distance(
                                xx, yy, goal.x, goal.y)
                            nextCell.f = nextCell.g + nextCell.h
                            fringe.put((nextCell.f, nextCell))
                            fringeSet.add(nextCell.id)

    # Return None if no solution exists
    if len(fringeSet) == 0:
        endtime = time.time()
        totalplanningtime += endtime - starttime
        return None, 0

    # Starting from goal cell, work backwards and reassign child attributes correctly
    if goalfound:
        parentPtr = goal
        childPtr = None
        oldParent = start.parent
        start.parent = None
        while(parentPtr is not None):
            astarlen += 1
            parentPtr.child = childPtr
            childPtr = parentPtr
            parentPtr = parentPtr.parent
        start.parent = oldParent
        endtime = time.time()
        totalplanningtime += endtime - starttime

        return start, astarlen
    else:
        endtime = time.time()
        totalplanningtime += endtime - starttime
        return None, 0


def istarget(curr):
    rand = random.random()
    if curr.terrain == Terrain.FLAT and rand < repr(Terrain.FLAT):
        return False
    elif curr.terrain == Terrain.HILLY and rand < repr(Terrain.HILLY):
        return False
    elif curr.terrain == Terrain.FOREST and rand < repr(Terrain.FOREST):
        return False
    return True


def solve6():
    """
    Agent 6
    """
    global gridworld, cardinaldirections, trajectorylen

    agent = 6

    # check if cell is blocked

    # check if cell is target
    istarget()

    # update cell prob


def solve7():
    """
    Agent 7
    """
    global gridworld, cardinaldirections, trajectorylen

    agent = 7


def solve8():
    """
    Agent 8
    """
    global gridworld, cardinaldirections, trajectorylen

    agent = 8


def get_weighted_manhattan_distance(x1, y1, x2, y2):
    """Manhattan: d((x1, y1),(x2, y2)) = abs(x1 - x2) + abs(y1 - y2)"""

    return 2*(abs(x1-x2) + abs(y1-y2))


def is_in_bounds(curr):
    """Determines whether next move is within bounds"""
    global gridworld
    return 0 <= curr[0] < len(gridworld) and 0 <= curr[1] < len(gridworld[0])


def get_num_neighbors(x, y):
    if (x == 0 and y == 0) or (x == 0 and y == dim-1) or (x == dim-1 and y == 0) or (x == dim-1 and y == dim-1):
        return 3
    elif (x == 0) or (y == 0) or (y == dim-1) or (x == dim-1):
        return 5
    else:
        return 8
