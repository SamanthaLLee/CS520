import random
from array import *
from queue import PriorityQueue
from cell import Cell
from terrain import Terrain
import time
import numpy as np
from multiprocessing import Pool
import sys

# Global gridworld of Cell objects
gridworld = []

probabilities = []
prob_of_finding = []

terrainprobabilities = [0.2, 0.5, 0.8, 1]

priorityqueue = PriorityQueue()

# Global start and goal cells
goal = None
start = None

# Vectors that represent directions
cardinaldirections = [(1, 0), (-1, 0), (0, 1), (0, -1)]

numcellsprocessed = 0
trajectorylen = 0
actions = 0
dim = 0
totalplanningtime = 0
finaldiscovered = False
fullgridworld = False


def generategridworld(d):
    """Generates a random gridworld based on user inputs"""
    global goal, start, gridworld, probabilities, dim
    dim = d
    p = 0.3
    # Cells are constructed in the following way:
    # Cell(g, h, f, blocked, seen, parent)
    gridworld = [[Cell(x, y) for y in range(dim)] for x in range(dim)]
    probabilities = [[1/(dim*dim) for y in range(dim)] for x in range(dim)]
    id = 0

    terrains = [Terrain.FLAT, Terrain.HILLY, Terrain.FOREST]

    # Let each cell independently be blocked with probability p, and empty with probability 1−p.
    for i in range(dim):
        for j in range(dim):
            curr = gridworld[i][j]

            # Determine block status, randomly decide terrain
            rand = random.random()
            if rand < p:
                curr.blocked = True
                curr.terrain = Terrain.BLOCKED
            else:
                curr.terrain = random.choice(terrains)

            # Assign ID
            curr.id = id
            id += 1

    # Set the start and goal nodes
    while goal is None:
        x = random.randrange(dim)
        y = random.randrange(dim)
        if not gridworld[x][y].blocked:
            goal = gridworld[x][y]

    while start is None:
        x = random.randrange(dim)
        y = random.randrange(dim)
        if not gridworld[x][y].blocked:
            start = gridworld[x][y]

    # Initialize starting cell values
    start.g = 1
    start.h = get_weighted_manhattan_distance(0, 0, goal.x, goal.y)
    start.f = gridworld[0][0].g + gridworld[0][0].h
    start.seen = True


def printGridworld():
    """Prints out the current state of the gridworld.
    Key:
            B: blocked and seen
            b: blocked and unseen
            F: flat and seen
            f: flat and unseen
            H: hilly and seen
            h: blocked and unseen
            O: forest and seen
            o: forest and unseen

            *: current path
            ' ': free space
    """
    global gridworld
    leng = len(gridworld)

    string = ''
    for i in range(leng):
        string += ('-'*(leng*3+1) + '\n')
        for j in range(leng):
            string += '|'
            curr = gridworld[i][j]
            if curr.blocked and curr.seen:
                string += ' '
            elif curr.blocked and not curr.seen:
                string += ' '
            elif curr.terrain == Terrain.FLAT and curr.seen:
                string += 'F'
            elif curr.terrain == Terrain.FLAT and not curr.seen:
                string += 'f'
            elif curr.terrain == Terrain.HILLY and curr.seen:
                string += 'H'
            elif curr.terrain == Terrain.HILLY and not curr.seen:
                string += 'h'
            elif curr.terrain == Terrain.FOREST and curr.seen:
                string += 'O'
            elif curr.terrain == Terrain.FOREST and not curr.seen:
                string += 'o'
            else:
                string += ' '

            if goal.x == i and goal.y == j:
                string += '!'
            elif not curr.blocked and curr.seen:
                string += '*'
            else:
                string += ' '

        string += '|\n'
    string += ('-'*(leng*3+1))
    print(string)


def astar(start, maxcell, agent):
    """Performs the A* algorithm on the gridworld
    Args:
        start (Cell): The cell from which A* will find a path to the goal
    Returns:
        Cell: The head of a Cell linked list containing the shortest path
        int: Length of returned final path
    """
    global gridworld, finaldiscovered, fullgridworld, cardinaldirections, numcellsprocessed, totalplanningtime
    starttime = time.time()
    fringe = PriorityQueue()
    fringeSet = set()
    seenSet = set()
    astarlen = 0
    goalfound = False

    curr = start
    fringe.put((curr.f, curr))
    fringeSet.add(curr.id)

    if start.id == maxcell.id:
        return start, 0

    # Generate all valid children and add to fringe
    # Terminate loop if fringe is empty or if path has reached goal
    while len(fringeSet) != 0:
        f, curr = fringe.get()
        if curr is maxcell:
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
                # Add children to fringe if inbounds AND unblocked and unseen
                if not (nextCell.blocked and nextCell.seen):
                    # Add child if not already in fringe
                    # If in fringe, update child in fringe if old g value > new g value
                    if(((not nextCell.id in fringeSet) or (nextCell.g > curr.g + 1)) and nextCell.id not in seenSet):
                        nextCell.parent = curr
                        nextCell.g = curr.g + 1
                        nextCell.h = get_weighted_manhattan_distance(
                            xx, yy, maxcell.x, maxcell.y)
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
        parentPtr = maxcell
        childPtr = None
        oldParent = start.parent
        start.parent = None
        print("START PATH from", start)
        while(parentPtr is not None):
            print(parentPtr)
            astarlen += 1
            parentPtr.child = childPtr
            childPtr = parentPtr
            parentPtr = parentPtr.parent
        print("END PATH to", maxcell)
        start.parent = oldParent
        endtime = time.time()
        totalplanningtime += endtime - starttime
        return start, astarlen
    else:
        endtime = time.time()
        totalplanningtime += endtime - starttime
        return None, 0


def istarget(curr):
    global actions
    actions += 1
    if curr is goal:
        rand = random.random()
        if curr.terrain == Terrain.FLAT and rand < terrainprobabilities[int(Terrain.FLAT)]:
            return False
        elif curr.terrain == Terrain.HILLY and rand < terrainprobabilities[int(Terrain.HILLY)]:
            return False
        elif curr.terrain == Terrain.FOREST and rand < terrainprobabilities[int(Terrain.FOREST)]:
            return False
        return True
    return False


def solve6():
    """
    Agent 6
    """
    global start, gridworld, cardinaldirections, trajectorylen, actions

    agent = 6

    printGridworld()
    print(probabilities)

    maxcell = getmaxcell(start, agent)

    path, pathlen = astar(start, maxcell, agent)

    curr = path
    while True:

        # A* failed, no solution
        # REVISE/RETHINK THIS!!! WHEN DO WE STOF???
        if curr is None:
            print("fail")
            return None

        # Goal found
        if istarget(curr):
            print("checked w/ success", curr)
            return path

        print("checked", curr)
        # Pre-process cell
        curr.seen = True
        trajectorylen += 1

        # Update probs by sensing terrain
        updateprobabilities(curr, agent)

        # Run into blocked cell
        if curr.blocked:
            print("blocked cell")
            trajectorylen -= 2  # avoid counting block and re-counting parent
            if curr.id == maxcell.id:
                maxcell = getmaxcell(curr, agent)
            path, len = astar(curr.parent, maxcell, agent)
            if len == 0:
                curr = path
            else:
                curr = path.child
        else:
            # Check if maximum probability has changed
            newmaxcell = getmaxcell(curr, agent)
            if maxcell.id is not newmaxcell.id:
                print("max p update", newmaxcell)
                print(probabilities)
                maxcell = newmaxcell
                trajectorylen -= 1  # avoid re-counting curr
                path, len = astar(curr, maxcell, agent)
                if len == 0:
                    curr = path
                else:
                    curr = path.child
            # Continue along A* path
            elif curr.child is not None:
                # Move onto next cell along A* path
                print("cont path")
                curr = curr.child
                actions += 1
            elif maxcell.id == newmaxcell.id:
                path, len = astar(curr, maxcell, agent)
                if len == 0:
                    curr = path
                else:
                    curr = path.child


def solve7():
    """
    Agent 7
    """
    global start, gridworld, cardinaldirections, trajectorylen

    agent = 7
    prob_of_finding = gridworld = [[0 for _ in range(dim)] for _ in range(dim)]

    maxcell = getmaxcell(start)

    path, pathlen = astar(start, maxcell, agent)

    curr = path
    while True:

        # A* failed
        if curr is None:
            return None

        # Goal found
        if istarget(curr):
            return path

        # Pre-process cell
        curr.seen = True
        trajectorylen += 1

        # Update probs by sensing terrain
        updateprobabilities(curr, agent)

        newmaxcell = getmaxcell()

        # Run into blocked cell
        if maxcell is not newmaxcell:
            trajectorylen -= 1
            path, len = astar(curr, newmaxcell, agent)
            curr = path
        elif curr.blocked == True:
            trajectorylen -= 2
            path, len = astar(curr.parent, maxcell, agent)
            curr = path
        # Continue along A* path
        else:
            # Move onto next cell along A* path
            curr = curr.child


def solve8():
    """
    Agent 8
    """
    global start, gridworld, cardinaldirections, trajectorylen

    agent = 8

    maxcell = getmaxcell(start)

    path, pathlen = astar(start, maxcell, agent)

    curr = path
    while True:

        # A* failed
        if curr is None:
            return None

        # Goal found
        if istarget(curr):
            return path

        # Pre-process cell
        curr.seen = True
        trajectorylen += 1

        # Update probs by sensing terrain
        updateprobabilities(curr, agent)

        newmaxcell = getmaxcell()

        # Run into blocked cell
        if maxcell is not newmaxcell:
            trajectorylen -= 1
            path, len = astar(curr, newmaxcell, agent)
            curr = path
        elif curr.blocked == True:
            trajectorylen -= 2
            path, len = astar(curr.parent, maxcell, agent)
            curr = path
        # Continue along A* path
        else:
            # Move onto next cell along A* path
            curr = curr.child


def getmaxcell(curr, agent):
    global prob_of_finding, probabilities

    p = None
    if agent == 6:
        p = np.array(probabilities)
    if agent == 7:
        for i in range(dim):
            for j in range(dim):
                if gridworld[i][j].seen:
                    prob_of_finding[i][j] = probabilities[i][j] * \
                        (1-terrainprobabilities[int(gridworld[i][j].terrain)])
        p = np.array(prob_of_finding)

    maxp = np.amax(p)
    occs = list(zip(*np.where(p == maxp)))
    if len(occs) > 1:
        equidistant = []
        mindist = sys.maxsize
        for occ in occs:
            currdist = get_weighted_manhattan_distance(
                curr.x, curr.y, occ[0], occ[1])
            if currdist < mindist:
                mindist = currdist
                equidistant = [occ]
            elif currdist == mindist:
                equidistant.append(occ)
        i, j = random.choice(equidistant)
        return gridworld[i][j]
    return gridworld[occs[0][0]][occs[0][1]]


def updateprobability(x, y, curr, probabilities):
    factor = 1 - terrainprobabilities[int(curr.terrain)]
    denom = 1 - (factor * probabilities[curr.x][curr.y])
    return probabilities[x][y] / denom


def squash_updateprobability(args):
    return updateprobability(*args)


def updateprobabilities(curr, agent):
    global probabilities

    # Update probabilities of all other cells
    pool = Pool(processes=10)
    results = pool.map(squash_updateprobability, ((i, j, curr, probabilities) for i in range(dim)
                                                  for j in range(dim)))
    probabilities = np.array(results).reshape(dim, dim)
    pool.close()

    # Update probability of current cell
    probabilities[curr.x][curr.y] *= terrainprobabilities[int(curr.terrain)]


def get_weighted_manhattan_distance(x1, y1, x2, y2):
    """Manhattan: d((x1, y1),(x2, y2)) = abs(x1 - x2) + abs(y1 - y2)"""

    return 2*(abs(x1-x2) + abs(y1-y2))


def is_in_bounds(curr):
    """Determines whether next move is within bounds"""
    global gridworld
    return 0 <= curr[0] < len(gridworld) and 0 <= curr[1] < len(gridworld[0])
