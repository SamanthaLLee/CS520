import random
from array import *
from queue import PriorityQueue
from cell import Cell

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


def generategridworld(dim, p):
    """Generates a random gridworld based on user inputs"""
    global goal, gridworld

    # Cells are constructed in the following way:
    # Cell(g, h, f, blocked, seen, parent)
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
            if iscorner(i, j, dim):
                curr.N = 3
                curr.H = 3
            elif isborder(i, j, dim):
                curr.N = 5
                curr.N = 5
            else:
                curr.N = 8
                curr.N = 8

    # Set the goal node
    goal = gridworld[dim-1][dim-1]

    # Ensure that the start and end positions are unblocked
    gridworld[0][0].blocked = 0
    goal.blocked = 0

    # Initialize starting cell values
    gridworld[0][0].g = 1
    gridworld[0][0].h = getManhattanDistance(0, 0, goal.x, goal.y)
    gridworld[0][0].f = gridworld[0][0].g + gridworld[0][0].h
    gridworld[0][0].seen = True


def astar(start):
    """Performs the A* algorithm on the gridworld

    Args:
        start (Cell): The cell from which A* will find a path to the goal

    Returns:
        Cell: The head of a Cell linked list containing the shortest path
        int: Length of returned final path
    """
    global goal, gridworld, cardinaldirections, numcellsprocessed
    fringe = PriorityQueue()
    fringeSet = set()
    seenSet = set()

    curr = start
    fringe.put((curr.f, curr))
    fringeSet.add(curr.id)

    # Generate all valid children and add to fringe
    # Terminate loop if fringe is empty or if path has reached goal
    while len(fringeSet) != 0:
        f, curr = fringe.get()
        if curr is goal:
            break
        if curr.id not in fringeSet:
            continue
        # print("removing", curr)
        fringeSet.remove(curr.id)
        seenSet.add(curr.id)
        numcellsprocessed = numcellsprocessed + 1
        for x, y in cardinaldirections:
            xx = curr.x + x
            yy = curr.y + y

            if isinbounds([xx, yy]):
                nextCell = gridworld[xx][yy]
                # Add children to fringe if inbounds AND unblocked and unseen

                if not (nextCell.blocked and nextCell.seen):
                    # Add child if not already in fringe
                    # If in fringe, update child in fringe if old g value > new g value
                    if(((not nextCell.id in fringeSet) or (nextCell.g > curr.g + 1)) and nextCell.id not in seenSet):
                        nextCell.parent = curr
                        nextCell.g = curr.g + 1
                        nextCell.h = getManhattanDistance(
                            xx, yy, goal.x, goal.y)
                        nextCell.f = nextCell.g + nextCell.h
                        fringe.put((nextCell.f, nextCell))
                        fringeSet.add(nextCell.id)

    # Return None if no solution exists
    if len(fringeSet) == 0:
        return None, None

    # Starting from goal cell, work backwards and reassign child attributes correctly
    parentPtr = goal
    childPtr = None
    oldParent = start.parent
    start.parent = None
    astarlen = 0
    while(parentPtr is not None):
        astarlen = astarlen + 1
        parentPtr.child = childPtr
        childPtr = parentPtr
        parentPtr = parentPtr.parent
    start.parent = oldParent

    return start, astarlen


def solve1():
    """
    Agent 1 - The Blindfolded Agent: bumps into walls
    """
    global gridworld, cardinaldirections, trajectorylen

    path, len = astar(gridworld[0][0])

    # Initial A* failed - unsolvable gridworld
    if path is None:
        return None

    # Attempt to move along planned path
    curr = path
    while True:
        # A* failed - unsolvable gridworld
        if curr is None:
            return None

        trajectorylen += 1
        # Goal found
        if curr.child is None:
            curr.seen = True
            return path

        # Run into blocked cell
        if curr.blocked == True:
            trajectorylen -= 2
            curr.seen = True
            path, len = astar(curr.parent)
            curr = path

        # Continue along A* path
        else:
            # Mark current cell as seen and move onto next cell along A* path
            curr.seen = True
            curr = curr.child


def solve2():
    """
    Agent 2 - 4-Neighbor Agent: Can see all 4 cardinal neighbors at once
    """
    global gridworld, cardinaldirections, trajectorylen

    path, len = astar(gridworld[0][0])

    # Initial A* failed - unsolvable gridworld
    if path is None:
        return None

    # Attempt to move along planned path
    curr = path
    while True:
        # A* failed - unsolvable gridworld
        if curr is None:
            return None

        trajectorylen += 1
        # Goal found
        if curr.child is None:
            curr.seen = True
            return path

        # Run into blocked cell
        if curr.blocked == True:
            trajectorylen -= 2
            curr.seen = True
            path, len = astar(curr.parent)
            curr = path

        # Continue along A* path
        else:
            # Take note of environment within viewing distance (adjacent cells)
            for dx, dy in cardinaldirections:
                xx, yy = curr.x + dx, curr.y + dy

                # Only mark blocked neighbors as seen
                if isinbounds([xx, yy]) and gridworld[xx][yy].blocked:
                    neighbor = gridworld[xx][yy]
                    neighbor.seen = True
            # Mark current cell as seen and move onto next cell along A* path
            curr.seen = True
            curr = curr.child



def solve_a3():
    """
    Agent 3 - Example Inference Agent
    """
    global goal, gridworld, alldirections, trajectorylen

    path, len = astar(gridworld[0][0])

    if path is None:
        return None

    # Traverse through planned path
    curr = path
    while True:

        if(curr is None):
            return None

        curr.seen = True
        curr.confirmed = True
        trajectorylen = trajectorylen + 1

        # Goal found
        if(curr.child is None):
            return path

        # Sense number of blocked neighbors
        for x, y in alldirections:
            xx = curr.x + x
            yy = curr.y + y

            if isinbounds([xx, yy]):
                neighbor = gridworld[xx, yy]
                if neighbor.blocked:
                    curr.C += 1
                if neighbor.confirmed:
                    if neighbor.blocked:
                        curr.B += 1
                    else:
                        curr.E += 1

        # Make inferences from this sensing
        if curr.C == curr.B:
            # All remaining hidden neighbors are empty
            for x, y in alldirections:
                xx = curr.x + x
                yy = curr.y + y
                if isinbounds([xx, yy]):
                    if gridworld[xx, yy].confirmed == False:
                        gridworld[xx, yy].confirmed = True
                        curr.H -= 1
                        curr.E += 1
        elif curr.N - curr.C == curr.E:
            # All remaining hidden neighbors are blocked
            for x, y in alldirections:
                xx = curr.x + x
                yy = curr.y + y
                if isinbounds([xx, yy]):
                    if gridworld[xx, yy].confirmed == False:
                        gridworld[xx, yy].confirmed = True
                        curr.H -= 1
                        curr.B += 1
        elif curr.H == 0:
            # Nothing left to be inferred
            print("aight")

        # Replan if agent has run into blocked cell
        if curr.blocked == True:
            trajectorylen = trajectorylen - 2
            curr, len = astar(curr.parent)
            continue

        # Replan if agent finds inferred block in path
        ptr = curr.child
        replanned = False
        while ptr.child is not None:
            if ptr.confirmed and ptr.blocked:
                curr, len = astar(curr)
                replanned = True
                break
            ptr = ptr.child

        # Otherwise, continue along A* path
        if not replanned:
            curr = curr.child


def solve_a4():
    """
    Agent 4 - Own Inference Agent - design down agent that beats Example Agent
    """
    global goal, gridworld, alldirections, trajectorylen

    path, len = astar(gridworld[0][0])

    if path is None:
        return None

    curr = path
    while(True):

        if(curr is None):
            return None

        # print("curr", curr.x, curr.y)

        trajectorylen = trajectorylen + 1
        # Goal found
        if(curr.child is None):
            curr.seen = True
            return path

        # Run into blocked cell
        if curr.blocked == True:
            trajectorylen = trajectorylen - 2
            curr.seen = True
            path, len = astar(curr.parent)
            curr = path

        # Continue along A* path
        else:
            # Take note of environment within viewing distance (adjacent cells)
            for dx, dy in alldirections:
                xx, yy = curr.x + dx, curr.y + dy

                # Only mark blocked neighbors as seen
                if isinbounds([xx, yy]) and gridworld[xx][yy].blocked:
                    neighbor = gridworld[xx][yy]
                    neighbor.seen = True
            # Mark current cell as seen and move onto next cell along A* path
            curr.seen = True
            curr = curr.child


def getManhattanDistance(x1, y1, x2, y2):
    """Manhattan: d((x1, y1),(x2, y2)) = abs(x1 - x2) + abs(y1 - y2)"""
    return (abs(x1-x2) + abs(y1-y2))


def isinbounds(curr):
    """Determines whether next move is within bounds"""
    global gridworld
    return 0 <= curr[0] < len(gridworld) and 0 <= curr[1] < len(gridworld[0])


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


def iscorner(x, y, dim):
    return (x == 0 and y == 0) or (x == 0 and y == dim-1) or (x == dim-1 and y == 0) or (x == dim-1 and y == dim-1)


def isborder(x, y, dim):
    return (x == 0) or (y == 0) or (y == dim-1) or (x == dim-1)
