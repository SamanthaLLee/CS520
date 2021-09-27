import numpy as np
import matplotlib.pyplot as plt
import timeit
import time
import solve


def solvability(heuristic):
    """Automates Question 4: plot density vs solvability of various p values to find a value 
        p0 such that p>p0 most mazes are solvable and p<p0 most mazes are unsolvable 

    Args:
        heuristic (function([int][int])): passes heuristic  into generategridworld
    """
    # Initialize results matrix where arg0 is p value, arg1 is number of solvable gridworlds out of 10
    results = [[0 for _ in range(100)] for _ in range(2)]
    for x in range(100):
        results[0][x] = x

    # Solve gridworlds
    for p in range(100):
        for _ in range(10):
            solve.generategridworld(101, float(p/100), heuristic)
            if solve(heuristic) is None:
                results[1][p] += 1

    # Plot results
    plt.scatter(results[0], results[1])  # plotting the column as histogram
    plt.show()


def compareHeuristics():
    """Automates Question 5: compares the 3 different heuristics runtimes on graphs of varying densities
    """
    global gridworld, checkfullgridworld

    # As per directions, "you may take each gridworld as known, and thus only search once"
    checkfullgridworld = True

    # Initialize results matrix - eg: results[1][3] --> Euclidean runtime on graph 4
    results = [[0 for _ in range(10)] for _ in range(3)]

    heuristics = [solve.getManhattanDistance,
                  solve.getEuclideanDistance, solve.getChebyshevDistance]
    # For a range of [0,9] p values, generate gridworlds
    for p in range(10):
        # For 5 gridworlds for each p value
        i = 0
        while i < 5:
            # Generate gridworld as Manhattan distance but manually set later
            solve.generategridworld(
                20, float(p/10), solve.getManhattanDistance)

            # For each heuristic, solve the gridworld 5 times and average the times
            for heur_num, heuristic in enumerate(heuristics):

                # Initialize starting cell value for each heuristic
                solve.gridworld[0][0].h = heuristic(
                    0, 0, solve.goal.x, solve.goal.y, 1)
                solve.gridworld[0][0].f = solve.gridworld[0][0].g + \
                    solve.gridworld[0][0].h

                # Time the solve
                start = timeit.default_timer()
                # If the gridworld is unsolvable, decrement i so 5 solvable gridworlds are tested
                if solve(heuristic) is None:
                    i -= 1
                    break
                stop = timeit.default_timer()
                results[heur_num][p] += stop - start
            i += 1

        # Average out times
        for x in range(3):
            results[x][p] /= 5

    # Set back to false
    checkfullgridworld = False

    # Plot results
    N = 3
    ind = np.arange(N)
    width = 0.25

    xvals = results[0]
    bar1 = plt.bar(ind, xvals, width, color='r')

    yvals = results[1]
    bar2 = plt.bar(ind+width, yvals, width, color='g')

    zvals = results[2]
    bar3 = plt.bar(ind+width*2, zvals, width, color='b')

    plt.xlabel('p')
    plt.ylabel('Average Time')

    plt.xticks(ind+width, ['0', '.1', '.2', '.3',
               '.4', '.5', '.6', '.7', '.8', '.9'])
    plt.legend((bar1, bar2, bar3), ('Manhattan', 'Euclidean', 'Chebyshev'))
    plt.show()


def densityvtrajectorylength(heuristic):
    """Automates Question 7: plot density vs trajectory 

    Args:
        heuristic (function([int][int])): passes heuristic  into generategridworld
    """

    trialsperp = 20

    # Initialize results matrix where arg1 is p value, arg2 is avg trajectory len
    results = [[0 for x in range(10)] for y in range(2)]
    for x in range(10):
        results[0][x] = x/10

    # Solve gridworlds
    for x in range(10):  # probability
        tempsum = 0
        for _ in range(trialsperp):
            solve.trajectorylen = 0
            solve.generategridworld(10, float(x/100), heuristic)
            result = solve.solve(heuristic)
            print(solve.trajectorylen)
            if result is None:
                trialsperp = trialsperp-1
            else:
                tempsum = tempsum + solve.trajectorylen
        results[1][x] = tempsum/trialsperp

    print(results)
    # Plot results

    plt.xlabel('Density')
    plt.ylabel('Avg Trajectory Length')

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
        results[0][x] = x/10

    # Solve gridworlds
    for x in range(10):  # probability
        tempsum = 0
        for _ in range(trialsperp):
            trajectorylen = 0
            solve.generategridworld(10, float(x/100), heuristic)
            result = solve(heuristic)
            if result is None:
                trialsperp = trialsperp - 1
            else:
                path, pathlen = solve.astar(
                    solve.gridworld[0][0], heuristic)
                currratio = trajectorylen/pathlen
                tempsum = tempsum + currratio
        results[1][x] = tempsum/trialsperp

    print(results)
    # Plot results
    plt.xlabel('Density')
    plt.ylabel(
        'Avg (Trajectory / Shortest Path in Discovered Gridworld)')
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
        results[0][x] = x/10

    # Solve gridworlds
    for x in range(10):  # probability
        tempsum = 0
        for _ in range(trialsperp):
            solve.generategridworld(50, float(x/100), heuristic)
            result = solve(heuristic)
            if result is None:
                trialsperp = trialsperp - 1
            else:
                discoveredpath, discoveredpathlen = solve.astar(
                    solve.gridworld[0][0], heuristic)
                checkfullgridworld = True
                fullpath, fullpathlen = solve.astar(
                    solve.gridworld[0][0], heuristic)
                currratio = discoveredpathlen/fullpathlen
                tempsum = tempsum + currratio
        results[1][x] = tempsum/trialsperp

    print(results)
    # Plot results
    plt.xlabel('Density')
    plt.ylabel(
        'Avg (Shortest Path in Discovered Gridworld / Shortest Path in Full Gridworld)')

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
        results[0][x] = x/10

    # Solve gridworlds
    for x in range(10):  # probability
        tempsum = 0
        for _ in range(trialsperp):
            numcellsprocessed = 0
            solve.generategridworld(10, float(x/100), heuristic)
            result = solve(heuristic)
            if result is None:
                trialsperp = trialsperp-1
            else:
                tempsum = tempsum + numcellsprocessed
        results[1][x] = tempsum/trialsperp

    print(results)
    plt.xlabel('Density')
    plt.ylabel('Avg Number of Cells Processed by Repeated A*')
    # Plot results
    plt.scatter(results[0], results[1])  # plotting the column as histogram
    plt.show()


def isfloat(str):
    """Determines whether a given string can be converted to float"""
    try:
        float(str)
        return True
    except ValueError:
        return False


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
    densityvtrajectorylength(solve.getManhattanDistance)
