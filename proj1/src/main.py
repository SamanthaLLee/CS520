from array import *
from cell import Cell
from queue import PriorityQueue
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time

# Global gridworld of Cell objects
gridworld = []

# Global goal cell
goal = None

# Vectors that represent the four cardinal directions
directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

heuristicweight = 1
haslimitedview = False
numcellsprocessed = 0
trajectorylen = 0
checkfullgridworld = False
shortestpathlen = 0
shortestpathindiscoveredlen = 0


def generategridworld2():
    global goal, gridworld
    dim = 3
    gridworld = [[Cell(x, y) for y in range(dim)] for x in range(dim)]

    id = 0

    # Let each cell independently be blocked with probability p, and empty with probability 1−p.
    for i in range(dim):
        for j in range(dim):
            gridworld[i][j].id = id
            id = id + 1

     # Set the goal node
    goal = gridworld[dim-1][dim-1]

    # Ensure that the start and end positions are unblocked
    gridworld[0][0].blocked = 0
    goal.blocked = 0

    # Initialize starting cell values
    gridworld[0][0].g = 1
    gridworld[0][0].h = heuristic(0, 0)
    gridworld[0][0].f = gridworld[0][0].g + gridworld[0][0].h
    gridworld[0][0].seen = True

    gridworld[0][2].blocked = 1
    gridworld[1][2].blocked = 1
    gridworld[2][0].blocked = 1
    gridworld[2][1].blocked = 1


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
    gridworld[0][0].h = heuristic(0, 0)
    gridworld[0][0].f = gridworld[0][0].g + gridworld[0][0].h
    gridworld[0][0].seen = True


def astar(start, heuristic):
    """Performs the A* algorithm on the gridworld

    Args:
        start (Cell): The cell from which A* will find a path to the goal

    Returns:
        Cell: The head of a Cell linked list containing the shortest path
    """
    global goal, gridworld, directions, numcellsprocessed
    fringe = PriorityQueue()
    fringeSet = set()
    seenSet = set()

    # infcount = 0
    # # Backtrack if start is stuck until a parent has a valid, unexplored neighbor cell
    # while not hasValidNeighbors(start):
    #     if infcount > 10:
    #         exit()
    #     # print(f"cell: ({start.x}, {start.y}) doesn't have valid neighbors.")
    #     start = start.parent

    #     # Unsolvable if no valid neighbors are found - backtracks to gridworld's starting cell's parent
    #     if start is None:
    #         print("A* ret none")
    #         return None, None
    #     else:
    #         print(start.x, start.y)  # infinite loop
    #     infcount = infcount+1
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
            print("pls dear god don't print this")
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
                        nextCell.h = heuristic(xx, yy)
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
        print("unsolvable gridworld")
        return None
    # if path is not None:
    #     print("shouldnt be here")

    # printer = path
    # while(printer is not None):
    #     print(printer.x, printer.y, printer.h, printer.f)
    #     printer = printer.child

    curr = path
    while(True):

        if(curr is None):
            print("unsolvable gridworld")
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
            print("redo astar")
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


def hasValidNeighbors(cell):
    """Determines if a cell has any valid neighbors. Valid is defined as being in bounds and not (blocked and seen)
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
            # Must be unseen if free
            if not neighbor.blocked and not neighbor.seen:
                return True
                # if the neighbor is unblocked, according to current
                # if not (neighbor.blocked and neighbor.seen):
                #
    return False


def isinbounds(curr):
    """Determines whether next move is within bounds"""
    global gridworld
    return 0 <= curr[0] < len(gridworld) and 0 <= curr[1] < len(gridworld[0])


def getManhattanDistance(x, y):
    """Manhattan: d((x1, y1),(x2, y2)) = abs(x1 - x2) + abs(y1 - y2)"""
    global goal
    return (abs(x-goal.x) + abs(y-goal.y))*heuristicweight


def getEuclideanDistance(x, y):
    """Euclidean: d((x1, y1),(x2, y2)) = sqrt((x1 - x2)2 + (y1 - y2)2)"""
    global goal
    return math.sqrt((x-goal.x)**2 + (y-goal.y)**2)*heuristicweight


def getChebyshevDistance(x, y):
    """Chebyshev: d((x1, y1),(x2, y2)) = max((x1 - x2), (y1 - y2))"""
    global goal
    return max((x - goal.x), (y - goal.y))*heuristicweight


def isfloat(str):
    """Determines whether a given string can be converted to float"""
    try:
        float(str)
        return True
    except ValueError:
        return False


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


def solvability(heuristic):
    """Automates Question 4: plot density vs solvability of various p values to find a value 
        p0 such that p>p0 most mazes are solvable and p<p0 most mazes are unsolvable 

    Args:
        heuristic (function([int][int])): passes heuristic  into generategridworld
    """
    # Initialize results matrix where arg1 is p value, arg2 is number of solvable gridworlds out of 10
    results = [[0 for x in range(100)] for y in range(2)]
    for x in range(100):
        results[0][x] = x

    # Solve gridworlds
    for x in range(100):
        for _ in range(10):
            generategridworld(101, float(x/100), heuristic)
            if solve(heuristic) is None:
                results[1][x] += 1

    # Plot results
    plt.scatter(results[0], results[1])  # plotting the column as histogram
    plt.show()


def densityvtrajectorylength(heuristic):
    """Automates Question 7: plot density vs trajectory 

    Args:
        heuristic (function([int][int])): passes heuristic  into generategridworld
    """
    global trajectorylen

    trialsperp = 20

    # Initialize results matrix where arg1 is p value, arg2 is avg trajectory len
    results = [[0 for x in range(10)] for y in range(2)]
    for x in range(10):
        results[0][x] = x

    # Solve gridworlds
    for x in range(10):  # probability
        tempsum = 0
        for _ in range(trialsperp):
            trajectorylen = 0
            generategridworld(10, float(x/100), heuristic)
            result = solve(heuristic)
            if result is None:
                trialsperp = trialsperp-1
            else:
                tempsum = tempsum + trajectorylen
        results[1][x] = tempsum/trialsperp

    print(results)
    # Plot results
    plt.scatter(results[0], results[1])  # plotting the column as histogram
    plt.show()


def densityvavg1(heuristic):
    """Automates Question 7: plot Density vs Average (Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld)

    Args:
        heuristic (function([int][int])): passes heuristic  into generategridworld
    """
    global trajectorylen

    trialsperp = 20
    # Initialize results matrix where arg1 is p value, arg2 is avg trajectory len
    results = [[0 for x in range(10)] for y in range(2)]
    for x in range(10):
        results[0][x] = x

    # Solve gridworlds
    for x in range(10):  # probability
        tempsum = 0
        for _ in range(trialsperp):
            trajectorylen = 0
            generategridworld(10, float(x/100), heuristic)
            result = solve(heuristic)
            if result is None:
                trialsperp = trialsperp - 1
            else:
                path, pathlen = astar(
                    gridworld[0][0], heuristic)
                currratio = trajectorylen/pathlen
                tempsum = tempsum + currratio
        results[1][x] = tempsum/trialsperp

    print(results)
    # Plot results
    plt.scatter(results[0], results[1])  # plotting the column as histogram
    plt.show()


def densityvavg2(heuristic):
    """Automates Question 7: plot Density vs Average (Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld)

    Args:
        heuristic (function([int][int])): passes heuristic  into generategridworld
    """
    global checkfullgridworld
    trialsperp = 20
    # Initialize results matrix where arg1 is p value, arg2 is avg trajectory len
    results = [[0 for x in range(10)] for y in range(2)]
    for x in range(10):
        results[0][x] = x

    # Solve gridworlds
    for x in range(10):  # probability
        tempsum = 0
        for _ in range(trialsperp):
            generategridworld(30, float(x/100), heuristic)
            result = solve(heuristic)
            if result is None:
                trialsperp = trialsperp - 1
            else:
                discoveredpath, discoveredpathlen = astar(
                    gridworld[0][0], heuristic)
                checkfullgridworld = True
                fullpath, fullpathlen = astar(
                    gridworld[0][0], heuristic)
                currratio = discoveredpathlen/fullpathlen
                tempsum = tempsum + currratio
        results[1][x] = tempsum/trialsperp

    print(results)
    # Plot results
    plt.scatter(results[0], results[1])  # plotting the column as histogram
    plt.show()


def densityvcellsprocessed(heuristic):
    """Automates Question 7: plot density vs trajectory 

    Args:
        heuristic (function([int][int])): passes heuristic  into generategridworld
    """
    global numcellsprocessed

    trialsperp = 20

    # Initialize results matrix where arg1 is p value, arg2 is avg trajectory len
    results = [[0 for x in range(10)] for y in range(2)]
    for x in range(10):
        results[0][x] = x

    # Solve gridworlds
    for x in range(10):  # probability
        tempsum = 0
        for _ in range(trialsperp):
            numcellsprocessed = 0
            generategridworld(10, float(x/100), heuristic)
            result = solve(heuristic)
            if result is None:
                trialsperp = trialsperp-1
            else:
                tempsum = tempsum + numcellsprocessed
        results[1][x] = tempsum/trialsperp

    print(results)
    # Plot results
    plt.scatter(results[0], results[1])  # plotting the column as histogram
    plt.show()


if __name__ == "__main__":
    # dim = input("What is the length of your gridworld? ")
    # while not dim.isdigit() or int(dim) < 2:
    #     dim = input("Enter a valid length. ")

    # p = input("With what probability will a cell be blocked? ")
    # while not isfloat(p) or float(p) > 1 or float(p) < 0:
    #     p = input("Enter a valid probability. ")

    # # Question 7
    # v = input("Set field of view to 1? Y/N ")
    # while v != 'Y' and v != 'y' and v != 'N' and v != 'n':
    #     v = input("Enter a valid input. ")
    # fieldofview = True if v != 'Y' or v != 'y' else False

    # # Question 9
    # w = input("Assign a weight to the heuristic (enter '1' for default). ")
    # while not isfloat(p) or float(p) > 1 or float(p) < 0:
    #     w = input("Enter a valid weight. ")
    # heuristicweight = float(w)

    # heuristic = getManhattanDistance

    # generategridworld(int(dim), float(p), heuristic)
    # # generategridworld2()
    # printGridworld()
    # starttime = time.time()
    # result = solve(heuristic)
    # printGridworld()
    # endtime = time.time()
    # if (result is None):
    #     print("No solution.")

    # trajectorylen = trajectorylen if result is not None else None
    # print("Trajectory length:", trajectorylen)
    # print("Cells processed: ", numcellsprocessed)
    # print("Runtime: ", endtime - starttime, "s")

    # shortestpathindiscovered, shortestpathindiscoveredlen = astar(
    #     gridworld[0][0], heuristic)
    # print("Length of Shortest Path in Final Discovered Gridworld: ",
    #       shortestpathindiscoveredlen)

    # checkfullgridworld = True
    # shortestpath, shortestpathlen = astar(
    #     gridworld[0][0], heuristic)
    # print("Length of Shortest Path in Full Gridworld: ",
    #       shortestpathlen)

    # Question 4
    # solvability(getManhattanDistance)
    # densityvtrajectorylength(getManhattanDistance)
    # densityvavg2(getManhattanDistance)
    densityvcellsprocessed(getManhattanDistance)
