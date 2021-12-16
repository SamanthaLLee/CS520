import random
from array import *
from queue import PriorityQueue
from cell import Cell
import itertools
import time
import numpy as np
import copy


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
# finaldiscovered = False
fullgridworld = False

totalplanningtime = 0
numplans = 0

input_states = []
output_states = []

ignore_right = False


def generategridworld(d):
    """Generates a random gridworld based on user inputs"""
    global goal, gridworld, dim
    dim = d
    p = .3
    # Cells are constructed in the following way:
    # Cell(g, h, f, blocked, seen, parent)

    while(True):
        gridworld = [[Cell(x, y) for y in range(dim)] for x in range(dim)]
        id = 0

        # Let each cell independently be blocked with probability p, and empty with probability 1−p.
        for i in range(dim):
            for j in range(dim):
                curr = gridworld[i][j]

                # Determined block status
                rand = random.random()
                if rand < p:
                    curr.blocked = 1

                # Assign ID
                curr.id = id
                id += 1

                # Set N and H
                curr.N = get_num_neighbors(i, j)
                curr.H = curr.N

        # Set the goal node
        goal = gridworld[dim-1][dim-1]

        # Ensure that the start and end positions are unblocked
        gridworld[0][0].blocked = 0
        goal.blocked = 0

        # Initialize starting cell values
        gridworld[0][0].g = 1
        gridworld[0][0].h = get_weighted_manhattan_distance(
            0, 0, goal.x, goal.y)
        gridworld[0][0].f = gridworld[0][0].g + gridworld[0][0].h
        gridworld[0][0].seen = True

        result, len = astar(gridworld[0][0], 0)

        if result is not None:
            break


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
    global goal, gridworld, fullgridworld, cardinaldirections, numcellsprocessed, totalplanningtime, numplans
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
                validchild = ((agent == 1) and not (nextCell.blocked and nextCell.seen)) or (
                    (agent == 2) and not (nextCell.blocked and nextCell.confirmed)) or (
                        (agent == 0) and not nextCell.blocked)

                # Add children to fringe if inbounds AND unblocked and unseen
                # if nextCell.seen:
                if validchild:
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
        numplans += 1
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
        numplans += 1

        return start, astarlen
    else:
        endtime = time.time()
        totalplanningtime += endtime - starttime
        numplans += 1
        return None, 0


def solve1():
    """
    Agent 1 - 4-Neighbor Agent: Can see all 4 cardinal neighbors at once
    """
    global gridworld, cardinaldirections, trajectorylen, dim

    agent = 1

    path, len = astar(gridworld[0][0], agent)

    currstate = np.full((dim, dim, 2), -1)

    currstateslice = np.full((3, 3), -1)

    # Initial A* failed - unsolvable gridworld
    if path is None:
        return None

    # Attempt to move along planned path
    curr = path
    while True:

        # A* failed - unsolvable gridworld
        if curr is None:
            return None

        # Pre-process cell
        curr.seen = True
        trajectorylen += 1

        # Goal found
        if curr.child is None:
            return path

        # Run into blocked cell
        if curr.blocked == True:
            currstate[curr.x][curr.y][0] = 1
            trajectorylen -= 1
            output_states.append(get_action(curr, curr.parent))

            trajectorylen -= 1
            path, len = astar(curr.parent, agent)
            curr = path

        # Continue along A* path
        else:
            # Take note of environment within viewing distance (adjacent cells)
            currstate[curr.x][curr.y][0] = 0
            for dx, dy in cardinaldirections:
                xx, yy = curr.x + dx, curr.y + dy

                # Only mark blocked neighbors as seen
                if is_in_bounds([xx, yy]) and gridworld[xx][yy].blocked:
                    neighbor = gridworld[xx][yy]
                    currstate[xx][yy][0] = 1
                    neighbor.seen = True
            # Move onto next cell along A* path
            output_states.append(get_action(curr, curr.child))
            curr = curr.child

        currstate[curr.x][curr.y][1] = 1
        currstate2 = copy.deepcopy(currstate)

        for dx, dy in cardinaldirections:
            xx, yy = curr.x + dx, curr.y + dy
            currstateslice[1][1] = [curr.x][curr.y][1]
            # Only mark blocked neighbors as seen
            if is_in_bounds([xx, yy]):
                currstateslice[1+dx][1+dy] = currstate[xx][yy][0]

        input_states.append(currstateslice)
        currstate[curr.x][curr.y][1] = -1


def solve2():
    """
    Agent 2 - Example Inference Agent
    """
    global goal, gridworld, alldirections, trajectorylen

    agent = 2

    path, len = astar(gridworld[0][0], agent)

    currstate = np.full((2, dim, dim), -1)
    curr_knowledge = np.full((5, dim, dim), 0)
    currstate = np.concatenate([currstate, curr_knowledge])

    if path is None:
        return None

    # Traverse through planned path
    curr = path
    while True:

        if(curr is None):
            return None

        # Pre-process current cell
        curr.seen = True
        curr.confirmed = True
        trajectorylen += 1

        # Goal found
        if curr.child is None:
            return path

        # Update neighbors and cascade inferences
        updatekb(curr, currstate)

        # Replan if agent has run into blocked cell
        if curr.blocked == True:

            currstate[0][curr.x][curr.y] = 1
            output_states.append(get_action(curr, curr.parent))

            trajectorylen -= 1
            curr, len = astar(curr.parent, agent)
        else:
            currstate[0][curr.x][curr.y] = 0

            # Sense number of blocked and confirmed neighbors for curr
            senseorcount(curr, True, currstate)
            # Make inferences from this sensing
            infer(curr, currstate)

            # Replan if agent finds inferred block in path
            ptr = curr.child
            replanned = False
            while ptr.child is not None:
                if ptr.confirmed and ptr.blocked:
                    curr, len = astar(curr, agent)
                    trajectorylen -= 1
                    replanned = True
                    break

                ptr = ptr.child

            output_states.append(get_action(curr, curr.child))

            # Otherwise, continue along A* path
            if not replanned:
                curr = curr.child

        currstate[1][curr.x][curr.y] = 1
        input_states.append(copy.deepcopy(currstate))
        currstate[1][curr.x][curr.y] = -1


def senseorcount(curr: Cell, sense, currstate):
    """Sets curr's C, E, H, B values based on current KB
    Args:
        curr (cell): current cell
    """

    if curr.blocked:
        return

    if sense:
        curr.C = 0
    curr.E = 0
    curr.B = 0
    curr.H = get_num_neighbors(curr.x, curr.y)

    for x, y in alldirections:
        xx = curr.x + x
        yy = curr.y + y

        if is_in_bounds([xx, yy]):
            neighbor = gridworld[xx][yy]
            # Only sense if agent is in curr
            if neighbor.blocked and sense:
                curr.C += 1
            # Count up all confirmed neighbors
            if neighbor.confirmed:
                if neighbor.blocked:
                    curr.B += 1
                else:
                    curr.E += 1
                curr.H -= 1

    currstate[2][curr.x][curr.y] = curr.N
    currstate[3][curr.x][curr.y] = curr.C
    currstate[4][curr.x][curr.y] = curr.B
    currstate[5][curr.x][curr.y] = curr.E
    currstate[6][curr.x][curr.y] = curr.H


def infer(curr, currstate):
    """Tests for the 3 given inferences
    Args:
        curr (cell): current cell to make inferences on
    """

    inferencemade = False
    if curr.H > 0:
        # More inferences possible on unconfirmed neighboring cells
        if curr.C == curr.B:
            inferencemade = True
            # All remaining hidden neighbors are empty
            for x, y in alldirections:
                xx = curr.x + x
                yy = curr.y + y
                if is_in_bounds([xx, yy]):
                    if not gridworld[xx][yy].confirmed:
                        gridworld[xx][yy].confirmed = True
                        curr.E += 1
                        curr.H -= 1

                        currstate[0][xx][yy] = 0
                        currstate[5][curr.x][curr.y] += 1
                        currstate[6][curr.x][curr.y] -= 1

        elif curr.N - curr.C == curr.E:
            inferencemade = True
            # All remaining hidden neighbors are blocked
            for x, y in alldirections:
                xx = curr.x + x
                yy = curr.y + y
                if is_in_bounds([xx, yy]):
                    if not gridworld[xx][yy].confirmed:
                        gridworld[xx][yy].confirmed = True
                        curr.B += 1
                        curr.H -= 1

                        currstate[0][xx][yy] = 1
                        currstate[4][curr.x][curr.y] += 1
                        currstate[6][curr.x][curr.y] -= 1
        return inferencemade


def updatekb(curr, currstate):
    for x, y in alldirections:
        xx = curr.x + x
        yy = curr.y + y
        if is_in_bounds([xx, yy]):
            # Find all inbounds neighbors of curr
            neighbor = gridworld[xx][yy]
            senseorcount(neighbor, False, currstate)
            if neighbor.seen and neighbor.blocked == 0 and neighbor.H > 0:
                if infer(neighbor, currstate):
                    for x, y in alldirections:
                        xx2 = neighbor.x + x
                        yy2 = neighbor.y + y
                        if is_in_bounds([xx2, yy2]):
                            updatekb(gridworld[xx2][yy2], currstate)


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


def get_weighted_manhattan_distance(x1, y1, x2, y2):
    """Manhattan: d((x1, y1),(x2, y2)) = abs(x1 - x2) + abs(y1 - y2)"""

    return 2*(abs(x1-x2) + abs(y1-y2))


def get_action(curr, next):
    if curr.y < next.y:
        return 3  # right
    elif curr.y > next.y:
        return 2  # left
    elif curr.x < next.x:
        return 1  # down
    else:
        return 0  # up
