from array import *
from cell import Cell
from queue import PriorityQueue
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import timeit

# Global gridworld of Cell objects
gridworld = []

# Global goal cell
goal = None

# Vectors that represent the four cardinal directions
directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def generategridworld2():
    global goal, gridworld
    dim = 10
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

    gridworld[0][9].blocked = 1
    gridworld[0][8].blocked = 1
    gridworld[0][7].blocked = 1
    gridworld[0][4].blocked = 1
    gridworld[0][3].blocked = 1
    gridworld[0][2].blocked = 1

    gridworld[1][6].blocked = 1
    gridworld[1][5].blocked = 1
    gridworld[1][4].blocked = 1
    gridworld[1][3].blocked = 1

    gridworld[2][8].blocked = 1
    gridworld[2][9].blocked = 1

    gridworld[3][1].blocked = 1
    gridworld[3][2].blocked = 1
    gridworld[3][3].blocked = 1
    gridworld[3][5].blocked = 1

    gridworld[4][0].blocked = 1
    gridworld[4][3].blocked = 1
    gridworld[4][9].blocked = 1

    gridworld[5][0].blocked = 1
    gridworld[5][2].blocked = 1
    gridworld[5][8].blocked = 1
    gridworld[5][6].blocked = 1

    gridworld[6][2].blocked = 1
    gridworld[6][3].blocked = 1
    gridworld[6][5].blocked = 1
    gridworld[6][6].blocked = 1
    gridworld[6][7].blocked = 1

    gridworld[7][2].blocked = 1
    gridworld[7][3].blocked = 1
    gridworld[7][4].blocked = 1
    gridworld[7][9].blocked = 1
    gridworld[7][7].blocked = 1

    gridworld[8][0].blocked = 1
    gridworld[8][2].blocked = 1
    gridworld[8][3].blocked = 1
    gridworld[8][6].blocked = 1

    gridworld[9][7].blocked = 1

    # gridworld[0][9].blocked = 1
    # gridworld[0][7].blocked = 1
    # gridworld[1][9].blocked = 1
    # gridworld[1][7].blocked = 1
    # gridworld[1][6].blocked = 1
    # gridworld[1][2].blocked = 1
    # gridworld[2][4].blocked = 1
    # gridworld[2][0].blocked = 1
    # gridworld[2][2].blocked = 1
    # gridworld[3][9].blocked = 1
    # gridworld[3][5].blocked = 1
    # gridworld[3][4].blocked = 1
    # gridworld[4][9].blocked = 1
    # gridworld[4][6].blocked = 1
    # gridworld[4][4].blocked = 1
    # gridworld[4][1].blocked = 1
    # gridworld[5][9].blocked = 1
    # gridworld[5][7].blocked = 1
    # gridworld[5][5].blocked = 1
    # gridworld[5][4].blocked = 1
    # gridworld[5][2].blocked = 1
    # gridworld[5][1].blocked = 1
    # gridworld[6][1].blocked = 1
    # gridworld[6][7].blocked = 1
    # gridworld[6][8].blocked = 1
    # gridworld[7][3].blocked = 1
    # gridworld[7][2].blocked = 1
    # gridworld[7][5].blocked = 1
    # gridworld[7][9].blocked = 1
    # gridworld[8][6].blocked = 1
    # gridworld[9][3].blocked = 1
    # gridworld[9][5].blocked = 1
    # gridworld[9][6].blocked = 1

def generategridworld3():
    global goal, gridworld
    dim = 5
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

    gridworld[0][1].blocked = 1
    gridworld[0][4].blocked = 1
    gridworld[2][1].blocked = 1
    gridworld[2][2].blocked = 1
    gridworld[2][4].blocked = 1
    gridworld[3][1].blocked = 1
    gridworld[3][3].blocked = 1
    gridworld[4][3].blocked = 1

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
    start = gridworld[0][0]
    # Ensure that the start and end positions are unblocked
    start.blocked = 0
    goal.blocked = 0

    # Initialize starting cell values
    start.g = 1
    start.h = heuristic(0, 0)
    start.f = start.g + start.h
    start.seen = True

    # Only mark blocked neighbors as seen
    if isinbounds([1, 0]) and gridworld[1][0].blocked:
        gridworld[1][0].seen = True
    if isinbounds([0, 1]) and gridworld[0][1].blocked:
        gridworld[0][1].seen = True


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

    # Backtrack if start is stuck until a parent has a valid, unexplored, unblocked neighbor cell
    while not hasValidNeighbors(start):
        print(f"cell: ({start.x}, {start.y}) doesn't have valid neighbors. Moving to parent.")
        start = start.parent

        # Unsolvable if no valid neighbors are found - backtracks to gridworld's starting cell's parent
        if start is None:
            print("A* ret none")
            return None
        else:
            print("onto the parent")  # infinite loop
    print(f"CELL: ({start.x}, {start.y}) HAS VALID NEIGHBORS.")
    
    # Add start to fringe
    curr = start
    fringe.put((curr.f, curr))
    fringeSet.add(curr.id)

    # Generate all valid children and add to fringe
    # Terminate loop if fringe is empty or if path has reached goal
    while len(fringeSet) != 0:
        # print("fringe contains", fringeSet)
        # print("fringe contains", fringe.queue)
        f, curr = fringe.get()
        print(f"popping cell: ({curr.x}, {curr.y})")
        if curr is goal:
            break
        # if curr.id not in fringeSet:
        #     continue
        fringeSet.remove(curr.id)
        seenSet.add(curr.id)


        count = 0
        # Iterate through curr's neighbors to find children to add to the fringe
        for x, y in directions:
            xx, yy = curr.x + x, curr.y + y
            count+=1
            print(count)
            if isinbounds([xx, yy]):
                nextCell = gridworld[xx][yy]
                # Child must be unseen if it is blocked
                # Child must be unseen if unblocked
                if not nextCell.seen:
                    
                    # Add child if not already in fringe, or must be better than child already in fringe
                    # If in fringe, update child in fringe if old g value > new g value
                    # Child must not have been part of A*'s path (seenSet is keeping track of seen unblocked cells)
                    if(((not nextCell.id in fringeSet) or (nextCell.g > curr.g + 1)) and nextCell.id not in seenSet):
                        if(nextCell.g > curr.g + 1 and nextCell.id in fringeSet):
                            print(nextCell.id, "requires update")
                        nextCell.parent = curr
                        nextCell.g = curr.g + 1
                        nextCell.h = heuristic(xx, yy)
                        nextCell.f = nextCell.g + nextCell.h
                        # print("adding", xx,
                        #       yy, nextCell.g, nextCell.h, nextCell.f)
                        fringe.put((nextCell.f, nextCell))
                        fringeSet.add(nextCell.id)
                        print(f"    adding cell: ({nextCell.x}, {nextCell.y})")

    # Return None if no solution exists
    if len(fringeSet) == 0:
        print("empty fringe - ran out of things to pop before if statement returns True on \"curr is goal\"")
        return None
    print("nonempty fringe")
    
    # Starting from goal cell, work backwards and reassign child attributes correctly
    parentPtr = goal
    childPtr = None
    
    temp = start.parent
    if start.parent is not None:
        print("storing away start's parent" + str(start.parent.x) + " " + str(start.parent.y))
    start.parent = None
    while(parentPtr is not None):
        # if childPtr is not None:
        # print(parentPtr.x, parentPtr.y,
        #       "is parent of", childPtr.x, childPtr.y)
        parentPtr.child = childPtr
        childPtr = parentPtr
        parentPtr = parentPtr.parent
    start.parent = temp
    if start.parent is not None:
        print("setting start's parent to " + str(temp.x) + " " + str(temp.y))
    
    iterptr = goal
    infcount = 0
    while iterptr is not None:
        print(iterptr.x, iterptr.y)
        iterptr = iterptr.parent
    
        if infcount > 100:
            printGridworld()
            exit()
        infcount = infcount+1
    return start


def solve(heuristic):
    """
    Solves the gridworld using Repeated Forward A*.
    """
    global goal, gridworld, directions

    path = astar(gridworld[0][0], heuristic)

    # printer = path
    # while(printer is not None):
    #     print(printer.x, printer.y, printer.h, printer.f)
    #     printer = printer.child

    curr = path
    while(True):

        if(curr is None):
            print("unsolvable gridworld")
            return None

        print("curr", curr.x, curr.y)
        # Goal found
        if(curr.child is None):
            curr.seen = True
            return path

        # Run into blocked cell
        if curr.blocked == True:
            print("redo astar")
            curr.seen = True
            path = astar(curr.parent, heuristic)
            curr = path

        # Continue along A* path
        else:
            # Take note of environment within viewing distance (adjacent cells)
            for dx, dy in directions:
                xx, yy = curr.x + dx, curr.y + dy

                # Only mark blocked neighbors as seen
                if isinbounds([xx, yy]) and gridworld[xx][yy].blocked:
                    gridworld[xx][yy].seen = True

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
            # Must be unseen if free
            if not gridworld[xx][yy].blocked and not gridworld[xx][yy].seen:
                return True
                
    print(f"WTF IS cell: ({cell.x}, {cell.y}) DOING HERE? IT DOESN'T HAVE ANY VALID NEIGHBORS")
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

def compareHeuristics():
    """Automates Question 5: compares the 3 different heuristics runtimes on graphs of varying densities
    """
    # Initialize results matrix - results[1][3]: Euclidean runtime on graph 4
    results = [[0 for x in range(10)] for y in range(3)]
    
    heuristics = [getManhattanDistance, getEuclideanDistance, getChebyshevDistance]
    # For a range of [0,9] p values, generate gridworlds
    for p in range(10):
        generategridworld(20, float(p/10), heuristic) # not sure what to do about the gridworld being unique to 1 heuristic

        # For each heuristic, solve the gridworld 5 times and average the times
        for index, heuristic in enumerate(heuristics):
            for _ in range(5):
                start = timeit.default_timer()
                solve(heuristic)
                stop = timeit.default_timer()
                results[index][p] += stop - start
            results[index][p] /= 5

    # Plot results
    N = 3
    ind = np.arange(N) 
    width = 0.25
    
    xvals = results[0]
    bar1 = plt.bar(ind, xvals, width, color = 'r')
    
    yvals = results[1]
    bar2 = plt.bar(ind+width, yvals, width, color='g')
    
    zvals = results[2]
    bar3 = plt.bar(ind+width*2, zvals, width, color = 'b')
    
    plt.xlabel('p')
    plt.ylabel('Average Time')
    
    plt.xticks(ind+width,['0', '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9'])
    plt.legend( (bar1, bar2, bar3), ('Manhattan', 'Euclidean', 'Chebyshev') )
    plt.show()

if __name__ == "__main__":
    dim = input("What is the length of your gridworld? ")
    while not dim.isdigit() or int(dim) < 0:
        dim = input("Enter a valid length. ")
    p = input("With what probability will a cell be blocked? ")
    while not isfloat(p) or float(p) > 1 or float(p) < 0:
        p = input("Enter a valid probability. ")
    heuristic = getManhattanDistance
    generategridworld(int(dim), float(p), heuristic)
    # generategridworld2()
    # generategridworld3()
    printGridworld()
    solve(heuristic)
    printGridworld()

    # Question 4
    # solvability(getManhattanDistance)
