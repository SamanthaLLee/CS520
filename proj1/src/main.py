import numpy as np
import matplotlib.pyplot as plt
import timeit
import time
import solve


def solvability_range(heuristic):
    """Automates Question 4: plot density vs solvability of range of p values to find a value 
        p0 such that p>p0 most mazes are solvable and p<p0 most mazes are unsolvable 

    Args:
        heuristic (function([int][int][int][int][int])): passes heuristic  into generategridworld
    """

    # Initialize constants: range [start, end), difference, # of gridworlds per p
    start = 15  # inclusive
    end = 35    # exclusive
    diff = end - start
    cycles = 100

    solve.checkfullgridworld = True

    # Initialize results matrix where arg0 is p value, arg1 is number of solvable gridworlds out of 10
    results = [[0 for _ in range(diff)] for _ in range(2)]
    for x in range(diff):
        results[0][x] = (x+15)/100
    print(results)

    # Solve gridworlds
    for p in range(start, end):
        p_index = p - start
        print(p, p_index)
        for _ in range(cycles):
            # Generate and solve
            solve.generategridworld(101, float(p/100), heuristic)

            if solve.solve(heuristic) is not None:
                results[1][p_index] += 1
        results[1][p_index] = (results[1][p_index]/cycles)*100

    print(results)
    # Plot results
    plt.title('Density vs. Solvability')
    plt.xlabel('Density')
    plt.ylabel('Percent of Solvable Gridworlds')
    plt.scatter(results[0], results[1])  # plotting the column as histogram
    plt.show()


def compare_heuristics():
    """Automates Question 5: compares the 3 different heuristics runtimes on graphs of varying densities
    """

    # As per directions, "you may take each gridworld as known, and thus only search once"
    solve.checkfullgridworld = True

    # Initialize constants:
    #   range [start, end), difference, # of gridworlds per p
    #   cycles - # of gridworlds per p, max_redos - # of unsolvable gridworlds allowed before breaking
    start = 0  # inclusive
    end = 46    # exclusive
    step = 5
    diff = end - 1 - start
    cycles = 10
    max_redos = 30

    # Initialize results matrix - eg: results[1][3] --> Euclidean runtime on graph 4
    results = [[0 for _ in range((end - 1 - start)//step + 1)]
               for _ in range(3)]

    heuristics = [solve.getManhattanDistance,
                  solve.getEuclideanDistance, solve.getChebyshevDistance]

    # For a range of [0,.45] p values, generate gridworlds
    for p in range(start, end, step):
        p_index = int(p/step)
        print("NEXT P")
        print(p, p_index)

        # "cycles" # of gridworlds for each p value
        i = 0
        redos = 0
        break_var = False

        # Keep making new gridworlds until desired # of solvable gridworlds are made
        while i < cycles:

            print("density:", p/100)
            # Generate gridworld as Manhattan distance but manually set later
            solve.generategridworld(
                10, float(p/100), solve.getManhattanDistance)

            # Solve the gridworld with each heuristic
            for heur_num, heuristic in enumerate(heuristics):

                # Initialize starting cell value for each heuristic
                solve.gridworld[0][0].h = heuristic(
                    0, 0, solve.goal.x, solve.goal.y, 1)
                solve.gridworld[0][0].f = solve.gridworld[0][0].g + \
                    solve.gridworld[0][0].h

                # Time the solve
                start_time = timeit.default_timer()
                # If the gridworld is unsolvable, inc redos, dec i and
                # break -> moves onto next gridworld
                if solve.solve(heuristic) is None:
                    redos += 1
                    if i > 0:
                        i -= 1
                    print(redos, i)

                    # Go to next p if max_redos met/exceeded
                    if redos >= max_redos:
                        print("max_redos met")
                        break_var = True
                    break
                # Continues with the timer
                stop_time = timeit.default_timer()
                results[heur_num][p_index] += stop_time - start_time
            # Ran thru each heuristic so incr i and next generate new gridworld
            i += 1

            # Too many unsolvable gridworlds made - give up on this p value
            if break_var:
                break_var = False
                print("break break: moving on to next p")
                break

        # Average out times for each p
        for x in range(3):
            if i != 0:
                results[x][p_index] /= i
        print(str(i) + "gridworlds made for p = " + str(p))

    print(results)
    # Set back to false
    checkfullgridworld = False

    # Plot results
    N = 3
    ind = np.arange(min(len(results[0]), len(results[1]), len(results[2])))
    width = 0.25

    xvals = results[0]
    bar1 = plt.bar(ind, xvals, width, color='r')

    yvals = results[1]
    bar2 = plt.bar(ind+width, yvals, width, color='g')

    zvals = results[2]
    bar3 = plt.bar(ind+width*2, zvals, width, color='b')

    plt.title('Density vs. Runtime by Heuristic')
    plt.xlabel('Density')
    plt.ylabel('Average Time (s)')

    # Make xticks list
    xtick_list = []
    for i, x in enumerate(range(start, end, step)):
        xtick_list.append(str(x/100))
    plt.xticks(ind+width, xtick_list)
    plt.legend((bar1, bar2, bar3), ('Manhattan', 'Euclidean', 'Chebyshev'))
    plt.show()


def compare_heuristics_no_redos():
    """Automates Question 5: compares the 3 different heuristics runtimes on graphs of varying densities
    """

    # As per directions, "you may take each gridworld as known, and thus only search once"
    solve.checkfullgridworld = True

    # Initialize constants:
    #   range [start, end), difference, # of gridworlds per p
    #   cycles - # of gridworlds per p, max_redos - # of unsolvable gridworlds allowed before breaking
    start = 0  # inclusive
    end = 46    # exclusive
    step = 5
    diff = end - 1 - start
    cycles = 10
    max_fails = 5

    # Initialize results matrix - eg: results[1][3] --> Euclidean runtime on graph 4
    results = [[0 for _ in range((end - 1 - start)//step + 1)]
               for _ in range(3)]

    heuristics = [solve.getManhattanDistance,
                  solve.getEuclideanDistance, solve.getChebyshevDistance]

    # For a range of [0,.45] p values, generate gridworlds
    for p_index, p in enumerate(range(start, end, step)):
        print(str(p_index) + "th P: " + str(p))

        num_fail = 0

        # Keep making new gridworlds until desired # of solvable gridworlds are made
        for _ in range(cycles):

            # Generate gridworld as Manhattan distance but manually set later
            solve.generategridworld(101, float(
                p/100), solve.getManhattanDistance)

            # Solve the gridworld with each heuristic
            for heur_num, heuristic in enumerate(heuristics):

                # Initialize starting cell value for each heuristic
                solve.gridworld[0][0].h = heuristic(
                    0, 0, solve.goal.x, solve.goal.y, 1)
                solve.gridworld[0][0].f = solve.gridworld[0][0].g + \
                    solve.gridworld[0][0].h

                # Time the solve
                start_time = timeit.default_timer()
                # If the gridworld is unsolvable, break -> moves onto next gridworld
                if solve.solve(heuristic) is None:
                    num_fail += 1
                    break
                # Continues with the timer
                stop_time = timeit.default_timer()
                results[heur_num][p_index] += stop_time - start_time

        num_solv = cycles - num_fail

        # Average out times for each p
        for x in range(3):
            if num_solv != 0:
                results[x][p_index] /= num_solv
        print(str(num_solv) + "gridworlds succeeded for p = " + str(p))

    print(results)
    # Set back to false
    checkfullgridworld = False

    # Plot results
    N = 3
    ind = np.arange(min(len(results[0]), len(results[1]), len(results[2])))
    width = 0.25

    xvals = results[0]
    bar1 = plt.bar(ind, xvals, width, color='r')

    yvals = results[1]
    bar2 = plt.bar(ind+width, yvals, width, color='g')

    zvals = results[2]
    bar3 = plt.bar(ind+width*2, zvals, width, color='b')

    plt.title('Density vs. Runtime by Heuristic')
    plt.xlabel('Density')
    plt.ylabel('Average Time (s)')

    # Make xticks list
    xtick_list = []
    for i, x in enumerate(range(start, end, step)):
        xtick_list.append(str(x/100))
    plt.xticks(ind+width, xtick_list)
    plt.legend((bar1, bar2, bar3), ('Manhattan', 'Euclidean', 'Chebyshev'))
    plt.show()


def compareHeuristics_old():
    """Automates Question 5: compares the 3 different heuristics runtimes on graphs of varying densities
    """

    # As per directions, "you may take each gridworld as known, and thus only search once"
    solve.checkfullgridworld = True

    # Initialize results matrix - eg: results[1][3] --> Euclidean runtime on graph 4
    results = [[0 for _ in range(10)] for _ in range(3)]

    heuristics = [solve.getManhattanDistance,
                  solve.getEuclideanDistance, solve.getChebyshevDistance]
    # For a range of [0,9] p values, generate gridworlds
    for p in range(10):
        print(p)
        # For 5 gridworlds for each p value
        fails = 0
        for _ in range(5):
            # Generate gridworld as Manhattan distance but manually set later
            solve.generategridworld(
                101, float(p/10), solve.getManhattanDistance)

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
                if solve.solve(heuristic) is None:
                    break
                stop = timeit.default_timer()
                results[heur_num][p] += stop - start

        # Average out times
        for x in range(3):
            results[x][p] /= 5

    print(results)
    # Set back to false
    solve.checkfullgridworld = False

    # Plot results
    N = 3
    ind = np.arange(min(len(results[0]), len(results[1]), len(results[2])))
    width = 0.25

    xvals = results[0]
    bar1 = plt.bar(ind, xvals, width, color='r')

    yvals = results[1]
    bar2 = plt.bar(ind+width, yvals, width, color='g')

    zvals = results[2]
    bar3 = plt.bar(ind+width*2, zvals, width, color='b')

    plt.title('Density vs. Runtime by Heuristic')
    plt.xlabel('Density')
    plt.ylabel('Average Time (s)')

    plt.xticks(ind+width, ['0', '.1', '.2', '.3',
               '.4', '.5', '.6', '.7', '.8', '.9'])
    plt.legend((bar1, bar2, bar3), ('Manhattan', 'Euclidean', 'Chebyshev'))
    plt.show()


def densityvtrajectorylength(heuristic):
    """Automates Question 7: plot density vs trajectory 

    Args:
        heuristic (function([int][int])): passes heuristic  into generategridworld
    """

    trialsperp = 40

    # Initialize results matrix where arg1 is p value, arg2 is avg trajectory len
    interval = .33/10
    p = 0
    results = [[0 for x in range(10)] for y in range(2)]
    for x in range(10):
        results[0][x] = p
        p += interval

    p = 0
    # # Solve gridworlds
    for x in range(10):  # probability
        tempsum = 0
        for _ in range(trialsperp):
            solve.trajectorylen = 0
            solve.generategridworld(101, p, heuristic)
            result = solve.solve(heuristic)
            # print(solve.trajectorylen)
            if result is None:
                trialsperp = trialsperp-1
            else:
                tempsum = tempsum + solve.trajectorylen
        p += interval
        results[1][x] = tempsum/trialsperp
        trialsperp = 40
        print(x, "probabilities done")

    # print(results)
    # Plot results
    plt.title('Density vs. Trajectory')
    plt.xlabel('Density')
    plt.ylabel('Avg Trajectory Length')

    plt.scatter(results[0], results[1])  # plotting the column as histogram
    plt.show()


def densityvavg1(heuristic):
    """Automates Question 7: plot Density vs Average (Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld)

    Args:
        heuristic (function([int][int])): passes heuristic  into generategridworld
    """

    trialsperp = 40
    # Initialize results matrix where arg1 is p value, arg2 is avg trajectory len
    interval = .33/10
    p = 0
    results = [[0 for x in range(10)] for y in range(2)]
    for x in range(10):
        results[0][x] = p
        p += interval

    p = 0

    # Solve gridworlds
    for x in range(10):  # probability
        tempsum = 0
        for _ in range(trialsperp):
            solve.trajectorylen = 0
            solve.generategridworld(101, p, heuristic)
            result = solve.solve(heuristic)
            if result is None:
                trialsperp = trialsperp - 1
            else:
                path, pathlen = solve.astar(
                    solve.gridworld[0][0], heuristic)
                currratio = solve.trajectorylen/pathlen
                tempsum = tempsum + currratio
        p += interval
        results[1][x] = tempsum/trialsperp
        trialsperp = 40
        print(x, "probabilities done")

    # print(results)
    # Plot results
    plt.title('Density vs. Trajectory/Shortest Path in Discovered Gridworld')
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
    interval = .33/10
    p = 0
    # Initialize results matrix where arg1 is p value, arg2 is avg trajectory len
    results = [[0 for x in range(10)] for y in range(2)]
    for x in range(10):
        results[0][x] = p
        p += interval

    p = 0
    # Solve gridworlds
    for x in range(10):  # probability
        tempsum = 0
        for _ in range(trialsperp):
            solve.generategridworld(70, p, heuristic)
            result = solve.solve(heuristic)
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
        print("done with", x)
        results[1][x] = tempsum/trialsperp
        trialsperp = 20
        p += interval

    print(results)
    # Plot results
    plt.title(
        'Density vs. Shortest Path in Discovered Gridworld/Shortest Path in Full Gridworld')
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

    trialsperp = 40
    interval = .33/10
    p = 0

    # Initialize results matrix where arg1 is p value, arg2 is avg trajectory len
    results = [[0 for x in range(10)] for y in range(2)]
    for x in range(10):
        results[0][x] = p
        p += interval

    p = 0
    # Solve gridworlds
    for x in range(10):  # probability
        tempsum = 0
        for _ in range(trialsperp):
            solve.numcellsprocessed = 0
            solve.generategridworld(101, p, heuristic)
            result = solve.solve(heuristic)
            if result is None:
                trialsperp = trialsperp-1
            else:
                tempsum = tempsum + solve.numcellsprocessed
        results[1][x] = tempsum/trialsperp
        p += interval
        trialsperp = 40
        print("done with", x)

    plt.title('Density vs. Cells Processed')
    plt.xlabel('Density')
    plt.ylabel('Avg Number of Cells Processed by Repeated A*')
    # Plot results
    plt.scatter(results[0], results[1])  # plotting the column as histogram
    plt.show()


def compare_weighted_heuristics():
    """Automates Question 9: compares heuristic weight and density vs avg traj and avg runtime
    """

    # Compare 4 p's per weight vs avg length
    p_list = [0, .1, .2, .3]
    weight_list = [1, 2, 3, 4]

    # As per directions, "you may take each gridworld as known, and thus only search once"
    # solve.checkfulgridworld = True

    # Initialize results matrix - eg: results[1][3] --> weight's (avg trajectory, runtime) at p=.3
    results = [[[0, 0] for _ in p_list] for _ in weight_list]

    # For each weight, set solve.py's heuristicweight
    for w_index, weight in enumerate(weight_list):
        # Set the heuristic weight
        solve.heuristicweight = weight

        # For a range of [0,.3] p values, generate gridworlds
        for p_index, p in enumerate(p_list):
            # print(w_index, p_index)

            # For n different gridworlds
            num_solvable = 0
            for _ in range(50):
                # Generate gridworld with Manhattan heuristic
                solve.generategridworld(101, p, solve.getManhattanDistance)

                # Time the solve
                start_time = timeit.default_timer()
                solve.trajectorylen = 0
                # Solve the gridworld with each weight and
                if solve.solve(solve.getManhattanDistance) is None:
                    continue

                # Continues with the timer
                stop_time = timeit.default_timer()
                # print("traj:", solve.trajectorylen)
                results[w_index][p_index][0] += solve.trajectorylen
                num_solvable += 1
                results[w_index][p_index][1] += stop_time - start_time

            # Average out data for current cell by num_solvable

            for y in range(2):
                if num_solvable != 0:
                    results[w_index][p_index][y] /= num_solvable
            # print(str(num_solvable) + "gridworlds succeeded for p = " + str(p))

    print(results)
    # Set back to false
    checkfullgridworld = False

    # Group avg trajs by weight (arg0) then by density (arg1)
    temp1, temp2, temp3, temp4 = [], [], [], []
    for i in range(4):
        temp1.append(results[0][i][0])
        temp2.append(results[1][i][0])
        temp3.append(results[2][i][0])
        temp4.append(results[3][i][0])
    # print(temp1)
    # print(temp2)
    # print(temp3)
    # print(temp4)

    # Plot avg traj
    N = 4
    ind = np.arange(N)
    width = 0.20
    bar1 = plt.bar(ind, temp1, width, color='r')
    bar2 = plt.bar(ind+width, temp2, width, color='g')
    bar3 = plt.bar(ind+width*2, temp3, width, color='b')
    bar4 = plt.bar(ind+width*3, temp4, width)

    plt.title('Density vs. Average Trajectory by Weight')
    plt.xlabel('Density')
    plt.ylabel('Average Trajectory')
    plt.xticks(ind+width, p_list)
    plt.legend((bar1, bar2, bar3, bar4), ('Weight = 1',
               'Weight = 2', 'Weight = 3', 'Weight = 4'))
    plt.show()

    # Group avg runtimes by weight (arg0) then by density (arg1)
    temp1, temp2, temp3, temp4 = [], [], [], []
    for i in range(4):
        temp1.append(results[0][i][1])
        temp2.append(results[1][i][1])
        temp3.append(results[2][i][1])
        temp4.append(results[3][i][1])
    # print(temp1)
    # print(temp2)
    # print(temp3)
    # print(temp4)

    # Plot avg runtime
    N = 4
    ind = np.arange(N)
    width = 0.20
    bar1 = plt.bar(ind, temp1, width, color='r')
    bar2 = plt.bar(ind+width, temp2, width, color='g')
    bar3 = plt.bar(ind+width*2, temp3, width, color='b')
    bar4 = plt.bar(ind+width*3, temp4, width)

    plt.title('Density vs. Average Runtime by Weight')
    plt.xlabel('Density')
    plt.ylabel('Average Runtime')
    plt.xticks(ind+width, p_list)
    plt.legend((bar1, bar2, bar3, bar4), ('Weight = 1',
               'Weight = 2', 'Weight = 3', 'Weight = 4'))
    plt.show()


def isfloat(str):
    """Determines whether a given string can be converted to float"""
    try:
        float(str)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    dim = input("What is the length of your gridworld? ")
    while not dim.isdigit() or int(dim) < 2:
        dim = input("Enter a valid length. ")

    p = input("With what probability will a cell be blocked? ")
    while not isfloat(p) or float(p) > 1 or float(p) < 0:
        p = input("Enter a valid probability. ")

    # Question 7
    v = input("Set field of view to 1? Y/N ")
    while v != 'Y' and v != 'y' and v != 'N' and v != 'n':
        v = input("Enter a valid input. ")
    fieldofview = True if v != 'Y' or v != 'y' else False

    # Question 9
    w = input("Assign a weight to the heuristic (enter '1' for default). ")
    while not isfloat(p) or float(p) > 1 or float(p) < 0:
        w = input("Enter a valid weight. ")
    heuristicweight = float(w)

    heuristic = solve.getManhattanDistance

    solve.generategridworld(int(dim), float(p), heuristic)
    # generategridworld2()
    solve.printGridworld()
    starttime = time.time()
    result = solve.solve(heuristic)
    solve.printGridworld()
    endtime = time.time()
    if (result is None):
        print("No solution.")

    solve.trajectorylen = solve.trajectorylen if result is not None else None
    print("Trajectory length:", solve.trajectorylen)
    print("Cells processed: ", solve.numcellsprocessed)
    print("Runtime: ", endtime - starttime, "s")

    shortestpathindiscovered, shortestpathindiscoveredlen = solve.astar(
        solve.gridworld[0][0], heuristic)
    print("Length of Shortest Path in Final Discovered Gridworld: ",
          shortestpathindiscoveredlen)

    solve.checkfullgridworld = True
    shortestpath, shortestpathlen = solve.astar(
        solve.gridworld[0][0], heuristic)
    print("Length of Shortest Path in Full Gridworld: ",
          shortestpathlen)

    # Question 4
    # solvability(solve.getManhattanDistance)
    # solvability_range(solve.getManhattanDistance)
    # compare_weighted_heuristics()
    # compare_heuristics()
    # compare_heuristics_no_redos()
    # solve.haslimitedview = True
    # densityvtrajectorylength(solve.getChebyshevDistance)
    # densityvavg1(solve.getChebyshevDistance)
    # densityvavg2(solve.getChebyshevDistance)
    # densityvcellsprocessed(solve.getChebyshevDistance)
