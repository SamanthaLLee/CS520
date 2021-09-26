from array import *
from cell import Cell
from queue import PriorityQueue
import random
import math

# Global gridworld of Cell objects
gridworld = []

# Global goal cell
goal = []

# Vectors that represent the four cardinal directions
directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def generategridworld(dim, p, heuristic):
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
    global goal, gridworld, directions
    fringe = PriorityQueue()
    fringeSet = set()
    seenSet = set()

    # Backtrack if start is stuck until a parent has a valid, unexplored neighbor cell
    while not hasValidNeighbors(start):
        start = start.parent

    # Unsolvable if no valid neighbors are found - backtracks to gridworld's starting cell's parent
    if start is None:
        return None

    # Add start to fringe
    curr = start
    fringe.put((curr.f, curr))
    fringeSet.add(curr.id)

    # Generate all valid children and add to fringe
    # Terminate loop if fringe is empty or if path has reached goal
    while len(fringeSet) != 0:
        f, curr = fringe.get()
        # print("picking", curr.x, curr.y, curr.f)
        if curr is goal:
            break

        fringeSet.remove(curr.id)
        seenSet.add(curr.id)
        for x, y in directions:
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
                        nextCell.h = heuristic(xx, yy)
                        nextCell.f = nextCell.g + nextCell.h
                        # print("adding", xx,
                        #       yy, nextCell.g, nextCell.h, nextCell.f)
                        fringe.put((nextCell.f, nextCell))
                        fringeSet.add(nextCell.id)

    # Return None if no solution exists
    if len(fringeSet) == 0:
        return None

    # Starting from goal cell, work backwards and reassign child attributes correctly
    parentPtr = goal
    childPtr = None
    while(parentPtr is not None):
        print("set return val", parentPtr.x,
              parentPtr.y, parentPtr.h, parentPtr.f)
        parentPtr.child = childPtr
        childPtr = parentPtr
        parentPtr = parentPtr.parent

    print("CHECKING GOAL CHILD JUST IN CASE", goal.child)

    return start


def solve(heuristic):
    """
    Solves the gridworld using Repeated Forward A*.
    """
    global goal, gridworld, directions

    path = astar(gridworld[0][0], heuristic)
    curr = path

    if path is None:
        print("handle case")
        return

    # printer = path
    # while(printer is not None):
    #     print(printer.x, printer.y, printer.h, printer.f)
    #     printer = printer.child

    while(True):
        print("curr", curr.x, curr.y)
        # Goal found
        if(curr.child is None):
            print("curr.child is none for", curr.x, curr.y)
            return path

        # Run into blocked cell
        if curr.blocked == True:
            path = astar(curr.parent, heuristic)
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


def hasValidNeighbors(cell):
    """Determines if a cell has any valid neighbors. Valid is defined as being in bounds and not (blocked and seen)
    Args:
        cell (cell): input cell

    Returns:
        boolean: If valid neighbors exist True, else False
    """
    global gridworld, directions
    for x, y in directions:
        xx, yy = cell.x + x, cell.y + y
        # To be valid, neighbor must be inbounds
        if isinbounds([xx, yy]):
            neighbor = gridworld[xx][yy]
            # Must be unseen if blocked - don't think this is possible since
            # we are only looking at adjacent cells within viewing distance
            if neighbor.blocked and not neighbor.seen:
                return True
            # Must be unseen if free
            if not neighbor.blocked and not neighbor.seen:
                return True
    return False


def isinbounds(curr):
    """Determines whether next move is within bounds"""
    global gridworld
    return 0 <= curr[0] < len(gridworld) and 0 <= curr[1] < len(gridworld[0])


def getManhattanDistance(x, y):
    """Manhattan: d((x1, y1),(x2, y2)) = abs(x1 - x2) + abs(y1 - y2)"""
    global goal
    return abs(x-goal.x) + abs(y-goal.y)


def getEuclideanDistance(x, y):
    """Euclidean: d((x1, y1),(x2, y2)) = sqrt((x1 - x2)2 + (y1 - y2)2)"""
    global goal
    return math.sqrt((x-goal.x)**2 + (y-goal.y)**2)


def getChebyshevDistance(x, y):
    """Chebyshev: d((x1, y1),(x2, y2)) = max((x1 - x2), (y1 - y2))"""
    global goal
    return max((x - goal.x), (y - goal.y))


def isfloat(str):
    """Determines whether a given string can be converted to float"""
    try:
        float(str)
        return True
    except ValueError:
        return False


def printGridworld():
    global gridworld
    leng = len(gridworld)

    string = ''
    for i in range(leng):
        string += ('-'*(leng*2+1) + '\n')
        for j in range(leng):
            string += '|'
            if gridworld[i][j].blocked:
                string += 'B'
            else:
                string += ' '
        string += '|\n'
    string += ('-'*(leng*2+1))
    print(string)


if __name__ == "__main__":
    dim = input("What is the length of your gridworld? ")
    while not dim.isdigit() or int(dim) < 0:
        dim = input("Enter a valid length. ")
    p = input("With what probability will a cell be blocked? ")
    while not isfloat(p) or float(p) > 1 or float(p) < 0:
        p = input("Enter a valid probability. ")
    heuristic = getManhattanDistance
    generategridworld(int(dim), float(p), heuristic)
    printGridworld()
    solve(heuristic)
