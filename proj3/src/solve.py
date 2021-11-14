import random
from array import *
from queue import PriorityQueue
from cell import Cell
from terrain import Terrain
import time
import numpy as np
import multiprocessing
import sys

# Global gridworld of Cell objects
gridworld = []

probabilities = []
prob_of_finding = []
utilities = []

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

maxcelltime = 0
updateptime = 0
updatepfindtime = 0


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
    # Todo: check slim chance that all cells blocked
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

    start = gridworld[2][1]
    goal = gridworld[1][0]

    # Initialize starting cell values
    start.g = 1
    start.h = get_weighted_manhattan_distance(0, 0, goal.x, goal.y)
    start.f = gridworld[0][0].g + gridworld[0][0].h
    start.seen = True

    gridworld[0][0].blocked = False
    gridworld[0][1].blocked = False
    gridworld[0][2].blocked = False

    gridworld[0][0].terrain = Terrain.FLAT
    gridworld[0][1].terrain = Terrain.FLAT
    gridworld[0][2].terrain = Terrain.FOREST

    gridworld[1][0].blocked = False
    gridworld[1][1].blocked = False
    gridworld[1][2].blocked = True

    gridworld[1][0].terrain = Terrain.FOREST
    gridworld[1][1].terrain = Terrain.FLAT
    gridworld[1][2].terrain = Terrain.BLOCKED

    gridworld[2][0].blocked = False
    gridworld[2][1].blocked = False
    gridworld[2][2].blocked = False

    gridworld[2][0].terrain = Terrain.FOREST
    gridworld[2][1].terrain = Terrain.FOREST
    gridworld[2][2].terrain = Terrain.FOREST


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
        # check if we need to set child/parent to null?
        # print("from", start, "to", maxcell, "START = MAXCELL")
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
        return None, -1

    # Starting from goal cell, work backwards and reassign child attributes correctly
    if goalfound:
        parentPtr = maxcell
        childPtr = None
        oldParent = start.parent
        start.parent = None
        while(parentPtr is not None):
            # print(parentPtr)
            astarlen += 1
            parentPtr.child = childPtr
            childPtr = parentPtr
            parentPtr = parentPtr.parent
        start.parent = oldParent
        endtime = time.time()
        totalplanningtime += endtime - starttime
        print("from", start, "to", maxcell)
        maxcell.child = None
        return start, astarlen
    else:
        endtime = time.time()
        totalplanningtime += endtime - starttime
        return None, -1


def istarget(curr):
    global actions
    # actions += 1
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
    global start, gridworld, cardinaldirections, trajectorylen, actions, updateptime

    agent = 6

    maxcell = getmaxcell(start, agent)

    path, pathlen = astar(start, maxcell, agent)
    laststartcell = start

    inf = 0

    # Terminate if maze discovered to be unsolvable
    curr = path
    while not goal.unreachable:
        # inf += 1
        # if inf > 100:
        #     print("blah")
        #     return None

        # If path DNE, then the current maxcell is unreachable and must be updated
        while curr is None:
            if goal.unreachable:
                return None
            # print("mark", maxcell, "unreachable")
            maxcell.unreachable = True
            # Prevent cell from being chosen as maxiumum again
            probabilities[maxcell.x][maxcell.y] *= -1
            maxcell = getmaxcell(laststartcell, agent)
            # print("curr is none, create path from",
            #       laststartcell, "to", maxcell)
            path, len = astar(laststartcell, maxcell, agent)

        # do not update probabilities if you're not examining

        if curr.id == maxcell.id:
            if istarget(curr):
                return path
            updateprobabilities(curr)

        # Goal found
        # if istarget(curr):
        #     if curr.id == maxcell.id:
        #         actions += 1
        #     # print("checked w/ success", curr)
        #     return path

        # print("checked", curr, "maxcell is", maxcell)
        # print(probabilities)
        # printGridworld()
        # Pre-process cell
        curr.seen = True
        trajectorylen += 1

        # starttime = time.time()
        # # Update probs by sensing terrain

        # endtime = time.time()
        # updateptime += endtime - starttime

        # Run into blocked cell
        if curr.blocked:
            updateprobabilities(curr)
            # print("blocked cell")
            trajectorylen -= 2  # avoid counting block and re-counting parent

            # if maxcell is the current blocked cell, must update maxcell
            if curr.id == maxcell.id:
                maxcell = getmaxcell(curr.parent, agent)

            # replan starting from cell right before block to maxcell
            path, len = astar(curr.parent, maxcell, agent)
            laststartcell = curr.parent

            # if len == 0, then curr.parent IS the maxcell, and we should check it again
            if len == 0 or path is None:
                curr = path
            # otherwise, we must look at the second cell in the path because we don't want to examine curr.parent again
            else:
                curr = path.child
        else:
            # Check if maximum probability has changed
            newmaxcell = getmaxcell(curr, agent)

            # If there's a new maxcell, we must replan from the current cell
            # if maxcell.id != newmaxcell.id:
            if probabilities[maxcell.x][maxcell.y] != probabilities[newmaxcell.x][newmaxcell.y]:
                # print("max p update", newmaxcell)
                # print(probabilities)
                maxcell = newmaxcell
                trajectorylen -= 1  # avoid re-counting curr
                path, len = astar(curr, maxcell, agent)
                laststartcell = curr

                # if len == 0, then curr.parent IS the maxcell, and we should check it again
                if len == 0 or path is None:
                    curr = path
                # otherwise, we must look at the second cell in the path because we don't want to examine curr.parent again
                else:
                    curr = path.child

            # If there is a path to follow, continue to follow it
            elif curr.child is not None:
                # print("cont path")
                curr = curr.child
                actions += 1

            # If there is no path to follow, and we must create a new one
            # In the case that curr is the maxcell, no need to update
            elif curr.id != maxcell.id:
                # print("create new path")
                path, len = astar(curr, maxcell, agent)
                laststartcell = curr
                # we must look at the second cell in the path because we don't want to examine curr.parent again
                curr = path.child


def solve7():
    """
    Agent 7
    """
    global start, gridworld, cardinaldirections, trajectorylen, actions, prob_of_finding, probabilities, updateptime, updatepfindtime

    agent = 7
    prob_of_finding = [[1/(dim*dim) for _ in range(dim)] for _ in range(dim)]

    maxcell = getmaxcell(start, agent)

    path, pathlen = astar(start, maxcell, agent)
    laststartcell = start

    # Terminate if maze discovered to be unsolvable
    curr = path
    while not goal.unreachable:

        counter = 0
        # If path DNE, then the current maxcell is unreachable and must be updated
        while curr is None:
            if goal.unreachable:
                return None
            # print("curr is None")  # occasional inf loop? can't find out why
            counter += 1
            if counter > 10:
                return None
            maxcell.unreachable = True
            # Prevent cell from being chosen as maxiumum again
            probabilities[maxcell.x][maxcell.y] *= -1
            maxcell = getmaxcell(laststartcell, agent)
            curr, len = astar(laststartcell, maxcell, agent)

        if curr.id == maxcell.id:
            if istarget(curr):
                return path
            updateprobabilities(curr)

        # # Goal found
        # if curr.id == maxcell.id and istarget(curr):
        #     print("checked w/ success", curr)
        #     return path

        # print("checked", curr)
        # Pre-process cell
        curr.seen = True
        trajectorylen += 1

        # starttime = time.time()
        # # Update probs by sensing terrain
        # updateprobabilities(curr)
        # endtime = time.time()
        # updateptime += endtime - starttime

        starttime = time.time()
        updateprobabilitiesoffinding(curr)
        updateutilities(curr)
        endtime = time.time()
        updatepfindtime += endtime - starttime

        # Run into blocked cell
        if curr.blocked:
            updateprobabilities(curr)
            # print("blocked cell")
            trajectorylen -= 2  # avoid counting block and re-counting parent

            # if maxcell is the current blocked cell, must update maxcell
            if curr.id == maxcell.id:
                maxcell = getmaxcell(curr.parent, agent)

            # replan starting from cell right before block to maxcell
            path, len = astar(curr.parent, maxcell, agent)
            laststartcell = curr.parent

            # if len == 0, then curr.parent IS the maxcell, and we should check it again
            if len == 0 or path is None:
                curr = path
            # otherwise, we must look at the second cell in the path because we don't want to examine curr.parent again
            else:
                curr = path.child
        else:
            # Check if maximum probability has changed
            newmaxcell = getmaxcell(curr, agent)

            # If there's a new maxcell, we must replan from the current cell
            # if maxcell.id != newmaxcell.id:
            # getmaxcell() is partly random, so we check probabilities (we get occasional infinite loops otherwise)
            if prob_of_finding[maxcell.x][maxcell.y] != prob_of_finding[newmaxcell.x][newmaxcell.y]:
                # print("max p update", newmaxcell)
                # print(probabilities)
                maxcell = newmaxcell
                trajectorylen -= 1  # avoid re-counting curr
                path, len = astar(curr, maxcell, agent)
                laststartcell = curr

                # if len == 0, then curr.parent IS the maxcell, and we should check it again
                if len == 0 or path is None:
                    curr = path
                # otherwise, we must look at the second cell in the path because we don't want to examine curr.parent again
                else:
                    curr = path.child

            # If there is a path to follow, continue to follow it
            elif curr.child is not None:
                # print("cont path")
                curr = curr.child
                actions += 1

            # If there is no path to follow, and we must create a new one
            # In the case that curr is the maxcell, no need to update
            elif curr.id != maxcell.id:
                # print("new path")
                path, len = astar(curr, maxcell, agent)
                laststartcell = curr
                # we must look at the second cell in the path because we don't want to examine curr.parent again
                curr = path.child


def solve8():
    """
    Agent 8
    """
    global start, gridworld, cardinaldirections, trajectorylen, actions, prob_of_finding, probabilities, utilities, updateptime, updatepfindtime

    agent = 8
    prob_of_finding = [[1/(dim*dim) for _ in range(dim)] for _ in range(dim)]
    utilities = [[0 for _ in range(dim)] for _ in range(dim)]
    updateutilities(start)

    mincell = getmincell(start, agent)

    path, pathlen = astar(start, mincell, agent)
    laststartcell = start
    inf = 0
    # Terminate if maze discovered to be unsolvable
    curr = path
    while not goal.unreachable:
        inf += 1
        # if inf > 30:
        #     print("blah")
        #     return None
        counter = 0
        # If path DNE, then the current maxcell is unreachable and must be updated
        while curr is None:
            if goal.unreachable:
                print("goal unreachable")
                return None
              # occasional inf loop? can't find out why
            counter += 1
            if counter > 10:
                print("counter > 10")
                return None
            mincell.unreachable = True
            print(mincell, "unreachable")
            # Prevent cell from being chosen as maxiumum again
            probabilities[mincell.x][mincell.y] *= -1
            utilities[mincell.x][mincell.y] = sys.maxsize
            updateutilities(laststartcell)
            mincell = getmincell(laststartcell, agent)
            curr, len = astar(laststartcell, mincell, agent)

        if curr.id == mincell.id:
            if istarget(curr):
                print("return path")
                return path
            updateprobabilities(curr)

        print("checked", curr)

        # Pre-process cell
        curr.seen = True
        trajectorylen += 1

        starttime = time.time()
        updateutilities(curr)
        endtime = time.time()
        updatepfindtime += endtime - starttime

        # Run into blocked cell
        if curr.blocked:
            print("blocked cell")
            updateprobabilities(curr)
            trajectorylen -= 2  # avoid counting block and re-counting parent

            # if maxcell is the current blocked cell, must update maxcell
            if curr.id == mincell.id:
                mincell = getmincell(curr.parent, agent)

            # replan starting from cell right before block to maxcell
            path, len = astar(curr.parent, mincell, agent)
            laststartcell = curr.parent

            # if len == 0, then curr.parent IS the maxcell, and we should check it again
            if len == 0 or path is None:
                curr = path
            # otherwise, we must look at the second cell in the path because we don't want to examine curr.parent again
            else:
                curr = path.child
        else:
            # Check if maximum probability has changed
            newmincell = getmincell(curr, agent)

            # If there's a new maxcell, we must replan from the current cell
            # if maxcell.id != newmaxcell.id:
            # getmaxcell() is partly random, so we check probabilities (we get occasional infinite loops otherwise)
            if utilities[mincell.x][mincell.y] != utilities[newmincell.x][newmincell.y]:
                print("max p update", newmincell)
                print(utilities)
                mincell = newmincell
                trajectorylen -= 1  # avoid re-counting curr
                path, len = astar(curr, mincell, agent)
                laststartcell = curr

                # if len == 0, then curr.parent IS the maxcell, and we should check it again
                if len == 0 or path is None:
                    curr = path
                # otherwise, we must look at the second cell in the path because we don't want to examine curr.parent again
                else:
                    curr = path.child

            # If there is a path to follow, continue to follow it
            elif curr.child is not None:
                print("cont path")
                curr = curr.child
                actions += 1

            # If there is no path to follow, and we must create a new one
            # In the case that curr is the maxcell, no need to update
            elif curr.id != mincell.id:
                print("new path")
                path, len = astar(curr, mincell, agent)
                laststartcell = curr
                # we must look at the second cell in the path because we don't want to examine curr.parent again
                curr = path.child


def getmaxcell(curr, agent):
    global utilities, prob_of_finding, probabilities, maxcelltime

    starttime = time.time()

    p = None
    if agent == 6:
        p = np.array(probabilities)
    elif agent == 7:
        # consider making concurrent
        p = np.array(prob_of_finding)
    elif agent == 8:
        p = np.array(utilities)

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
        endtime = time.time()
        maxcelltime += endtime-starttime
        return gridworld[i][j]
    endtime = time.time()
    maxcelltime += endtime-starttime
    return gridworld[occs[0][0]][occs[0][1]]


def getmincell(curr, agent):
    global utilities, prob_of_finding, probabilities, maxcelltime

    starttime = time.time()

    temp = utilities[curr.x][curr.y]
    utilities[curr.x][curr.y] = sys.maxsize
    p = np.array(utilities)
    utilities[curr.x][curr.y] = temp

    print(utilities)
    minp = np.amin(p)

    print("min p is", minp)

    occs = list(zip(*np.where(p == minp)))
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
        endtime = time.time()
        maxcelltime += endtime-starttime
        print("mincell", gridworld[i][j])
        return gridworld[i][j]
    endtime = time.time()
    maxcelltime += endtime-starttime
    print("mincell", gridworld[occs[0][0]][occs[0][1]])
    return gridworld[occs[0][0]][occs[0][1]]


def updateprobability(x, y, curr, probabilities):
    factor = 1 - terrainprobabilities[int(curr.terrain)]
    denom = 1 - (factor * probabilities[curr.x][curr.y])
    return probabilities[x][y] / denom


def squash_updateprobability(args):
    return updateprobability(*args)


def updateprobabilities(curr):
    global probabilities

    for i in range(dim):
        for j in range(dim):
            factor = 1 - terrainprobabilities[int(curr.terrain)]
            denom = 1 - (factor * probabilities[curr.x][curr.y])
            probabilities[i][j] /= denom

    # # Update probabilities of all other cells
    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # # print(multiprocessing.cpu_count())

    # results = pool.map(squash_updateprobability, ((i, j, curr, probabilities) for i in range(dim)
    #                                               for j in range(dim)))
    # probabilities = np.array(results).reshape(dim, dim)
    # pool.close()

    # Update probability of current cell
    if curr.blocked:
        probabilities[curr.x][curr.y] = 0
    else:
        probabilities[curr.x][curr.y] *= terrainprobabilities[int(
            curr.terrain)]


def updateprobabilityoffinding(x, y, probabilities, gridworld):
    if gridworld[x][y].seen:
        return probabilities[x][y] * \
            (1-terrainprobabilities[int(gridworld[x][y].terrain)])
    else:
        return probabilities[x][y]


def squash_updateprobabilityoffinding(args):
    return updateprobabilityoffinding(*args)


def updateprobabilitiesoffinding(curr):
    global probabilities, prob_of_finding, gridworld

    # # Update probabilities of all other cells
    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    # results = pool.map(squash_updateprobabilityoffinding, ((i, j, probabilities, gridworld) for i in range(dim)
    #                                                        for j in range(dim)))
    # print(results)
    # prob_of_finding = np.array(results).reshape(dim, dim)
    # pool.close()

    for i in range(dim):
        for j in range(dim):
            if gridworld[i][j].seen:
                prob_of_finding[i][j] = probabilities[i][j] * \
                    (1-terrainprobabilities[int(gridworld[i][j].terrain)])

    # Update probability of current cell
    if curr.blocked:
        prob_of_finding[curr.x][curr.y] = 0
    else:
        prob_of_finding[curr.x][curr.y] *= terrainprobabilities[int(
            curr.terrain)]

    # print(prob_of_finding)
    # return


def updateutilities(curr):
    global utilities, prob_of_finding, gridworld

    for i in range(dim):
        for j in range(dim):
            if gridworld[i][j].seen:
                prob_of_finding[i][j] = probabilities[i][j] * \
                    (1-terrainprobabilities[int(gridworld[i][j].terrain)])
            if gridworld[i][j] is curr:
                if curr.blocked:
                    prob_of_finding[i][j] = 0
                else:
                    prob_of_finding[i][j] *= terrainprobabilities[int(
                        curr.terrain)]
            if prob_of_finding[i][j] != 0 and utilities[i][j] != sys.maxsize:
                dist = get_weighted_manhattan_distance(
                    curr.x, curr.y, i, j)
                utilities[i][j] = dist / prob_of_finding[i][j]


def get_weighted_manhattan_distance(x1, y1, x2, y2):
    """Manhattan: d((x1, y1),(x2, y2)) = abs(x1 - x2) + abs(y1 - y2)"""

    return (abs(x1-x2) + abs(y1-y2))


def is_in_bounds(curr):
    """Determines whether next move is within bounds"""
    global gridworld
    return 0 <= curr[0] < len(gridworld) and 0 <= curr[1] < len(gridworld[0])
