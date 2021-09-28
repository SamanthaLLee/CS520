import random
from array import *
from queue import PriorityQueue
from heuristics import *
from cell import Cell

# Global gridworld of Cell objects
gridworld = []

# Global goal cell
goal = None

# Vectors that represent the four cardinal directions
directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

heuristicweight = 1
haslimitedview = True
numcellsprocessed = 0
trajectorylen = 0
checkfullgridworld = False


def generategridworld(dim, p, heuristic):
    """Generates a random gridworld based on user inputs"""
    global goal, gridworld

    # Cells are constructed in the following way:
    # Cell(g, h, f, blocked, seen, parent)
    gridworld = [[Cell(x, y) for y in range(dim)] for x in range(dim)]
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
    gridworld[0][0].g = 1
    gridworld[0][0].h = heuristic(0, 0, goal.x, goal.y, heuristicweight)
    gridworld[0][0].f = gridworld[0][0].g + gridworld[0][0].h
    gridworld[0][0].seen = True


def astar(start, heuristic):
    """Performs the A* algorithm on the gridworld

    Args:
        start (Cell): The cell from which A* will find a path to the goal

    Returns:
        Cell: The head of a Cell linked list containing the shortest path
        int: Length of returned final path
    """
    global goal, gridworld, directions, numcellsprocessed
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
        for x, y in directions:
            xx = curr.x + x
            yy = curr.y + y

            if isinbounds([xx, yy]):
                nextCell = gridworld[xx][yy]
                # Add children to fringe if inbounds AND unblocked and unseen

                if (not (nextCell.blocked and nextCell.seen) and not checkfullgridworld) or (not nextCell.blocked and checkfullgridworld):
                    # Add child if not already in fringe
                    # If in fringe, update child in fringe if old g value > new g value
                    if(((not nextCell.id in fringeSet) or (nextCell.g > curr.g + 1)) and nextCell.id not in seenSet):
                        nextCell.parent = curr
                        nextCell.g = curr.g + 1
                        nextCell.h = heuristic(
                            xx, yy, goal.x, goal.y, heuristicweight)
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


def solve(heuristic):
    """
    Solves the gridworld using Repeated Forward A*.
    """
    global goal, gridworld, directions, trajectorylen

    path, len = astar(gridworld[0][0], heuristic)

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
            path, len = astar(curr.parent, heuristic)
            curr = path

        # Continue along A* path
        else:
            if not haslimitedview:
                # Take note of environment within viewing distance (adjacent cells)
                for dx, dy in directions:
                    xx, yy = curr.x + dx, curr.y + dy

                    # Only mark blocked neighbors as seen
                    if isinbounds([xx, yy]) and gridworld[xx][yy].blocked:
                        neighbor = gridworld[xx][yy]
                        neighbor.seen = True
            # Mark current cell as seen and move onto next cell along A* path
            curr.seen = True
            curr = curr.child


def has_valid_neighbors(cell):
    """Determines if a cell has any valid neighbors. Valid is defined as being in bounds and free and unseen
    Args:
        cell (cell): input cell

    Returns:
        boolean: If valid neighbors exist True, else False
    """
    global gridworld, directions

    if cell is None:
        return False

    for x, y in directions:
        xx, yy = cell.x + x, cell.y + y
        # To be valid, neighbor must be inbounds
        if isinbounds([xx, yy]):
            neighbor = gridworld[xx][yy]
            # Must free AND unseen
            if not neighbor.blocked and not neighbor.seen:
                return True
    return False


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


def astar_backtracking(start, heuristic):
    """Performs the A* algorithm on the gridworld

    Args:
        start (Cell): The cell from which A* will find a path to the goal

    Returns:
        Cell: The head of a Cell linked list containing the shortest path
    """
    global goal, gridworld, directions, numcellsprocessed, trajectorylen

    fringe = PriorityQueue()
    fringeSet = set()
    seenSet = set()

    # Add start to fringe
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
        fringeSet.remove(curr.id)
        seenSet.add(curr.id)
        numcellsprocessed = numcellsprocessed + 1
        for x, y in directions:
            xx = curr.x + x
            yy = curr.y + y

            # Add children to fringe if inbounds AND unseen
            if isinbounds([xx, yy]) and not gridworld[xx][yy].seen:
                nextCell = gridworld[xx][yy]
                # Add unexplored neighbor if not in seenSet AND ((not in fringe) or (in fringe but better))
                # If in fringe, update child in fringe if old g value > new g value
                if(((not nextCell.id in fringeSet) or (nextCell.g > curr.g + 1)) and nextCell.id not in seenSet):
                    nextCell.parent = curr
                    nextCell.g = curr.g + 1
                    nextCell.h = heuristic(
                        xx, yy, goal.x, goal.y, heuristicweight)
                    nextCell.f = nextCell.g + nextCell.h
                    fringe.put((nextCell.f, nextCell))
                    fringeSet.add(nextCell.id)

                    # print("fringe add: ", nextCell)

    # Return None if no solution exists - shouldn't happen since should only return None when referencing [0][0]'s parent
    if len(fringeSet) == 0:
        print("A* back: len fringe = 0, unsolvable")
        print(fringeSet)
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


def solve_back():
    """
    Solves the gridworld using Repeated Forward A*.
    """
    global goal, gridworld, directions, trajectorylen

    path, len = astar_backtracking(gridworld[0][0], getManhattanDistance)

    if path is None:
        print("solve_back: unsolvable gridworld")
        return None
    print('AFTER INIT SEARCCHHHHH')
    # if path is not None:
    #     print("solvable")

    # printer = path
    # while(printer is not None):
    #     print(printer.x, printer.y, printer.h, printer.f)
    #     printer = printer.child

    curr = path
    while(True):

        if(curr is None):
            # print("unsolvable gridworld")
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
            curr = curr.parent
            print("redo astar at: ", curr)

            # Backtrack if curr is stuck until a parent has a valid, unexplored neighbor cell
            while not has_valid_neighbors(curr):
                print("BACKTRACKINGGGGGGGGGGGGGGGGGGGGG")
                # print(f"cell: ({start.x}, {start.y}) doesn't have valid neighbors.")
                curr = curr.parent
                trajectorylen += 1

                # Unsolvable if no valid neighbors are found - backtracks to gridworld's starting cell's parent
                if curr is None:
                    print("A* ret none")
                    return None, None
                print("restarting astar at: ", curr)

            path, len = astar_backtracking(curr, getManhattanDistance)
            curr = path

        # Continue along A* path
        else:
            # Take note of environment within viewing distance (adjacent cells)
            for dx, dy in directions:
                xx, yy = curr.x + dx, curr.y + dy

                # Only mark blocked neighbors as seen
                if isinbounds([xx, yy]) and gridworld[xx][yy].blocked:
                    neighbor = gridworld[xx][yy]
                    neighbor.seen = True
            # Mark current cell as seen and move onto next cell along A* path
            curr.seen = True
            curr = curr.child
