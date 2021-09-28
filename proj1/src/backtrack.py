import solve
import timeit
import numpy as np
import matplotlib.pyplot as plt

def backtracking_runtime():
    """Automates Question 8: part 1
    """

    # Initialize constants:
    #   range [start, end), difference, # of gridworlds per p
    #   cycles - # of gridworlds per p, max_redos - # of unsolvable gridworlds allowed before breaking
    start = 0  # inclusive
    end = 46    # exclusive
    step = 5
    diff = end - 1 - start
    cycles = 10
    max_fails = 5

    # Initialize results matrix - eg: results[1][3] --> backtracking runtime on graph 4
    results = [[0 for _ in range((end - 1 - start)//step + 1)]
               for _ in range(2)]

    solves = [solve.solve, solve.solve_back]

    # For a range of [0,.45] p values, generate gridworlds
    for p_index, p in enumerate(range(start, end, step)):

        num_fail = 0

        # Keep making new gridworlds until desired # of solvable gridworlds are made
        for _ in range(cycles):

            # Generate gridworld as Manhattan distance but manually set later
            solve.generategridworld(101, float(
                p/100), solve.getManhattanDistance)

            # Solve the gridworld with each heuristic
            for solve_num, which_solve in enumerate(solves):

                # Time the solve
                start_time = timeit.default_timer()
                # If the gridworld is unsolvable, break -> moves onto next gridworld
                if which_solve(solve.getManhattanDistance) is None:
                    num_fail += 1
                    break
                # Continues with the timer
                stop_time = timeit.default_timer()
                results[solve_num][p_index] += stop_time - start_time

        num_solv = cycles - num_fail

        # Average out times for each p
        for x in range(2):
            if num_solv != 0:
                results[x][p_index] /= num_solv

    # Set back to false
    checkfullgridworld = False

    # Plot results
    N = 3
    ind = np.arange(min(len(results[0]), len(results[1])))
    width = 0.30

    xvals = results[0]
    bar1 = plt.bar(ind, xvals, width, color='r')

    yvals = results[1]
    bar2 = plt.bar(ind+width, yvals, width, color='g')

    plt.title('Density vs. Runtime by Method')
    plt.xlabel('Density')
    plt.ylabel('Average Time (s)')

    # Make xticks list
    xtick_list = []
    for i, x in enumerate(range(start, end, step)):
        xtick_list.append(str(x/100))
    plt.xticks(ind+width, xtick_list)
    plt.legend((bar1, bar2), ('Regular', 'Backtracking'))
    plt.show()


def backtracking_average_traj():
    """Automates Question 8: part 2
    """

    # Initialize constants:
    #   range [start, end), difference, # of gridworlds per p
    #   cycles - # of gridworlds per p, max_redos - # of unsolvable gridworlds allowed before breaking
    start = 0  # inclusive
    end = 46    # exclusive
    step = 5
    diff = end - 1 - start
    cycles = 10
    max_fails = 5

    # Initialize results matrix - eg: results[1][3] --> backtracking runtime on graph 4
    results = [[0 for _ in range((end - 1 - start)//step + 1)]
               for _ in range(2)]

    solves = [solve.solve, solve.solve_back]

    # For a range of [0,.45] p values, generate gridworlds
    for p_index, p in enumerate(range(start, end, step)):

        num_fail = 0

        # Keep making new gridworlds until desired # of solvable gridworlds are made
        for _ in range(cycles):

            # Generate gridworld as Manhattan distance but manually set later
            solve.generategridworld(101, float(
                p/100), solve.getManhattanDistance)

            # Solve the gridworld with each heuristic
            for solve_num, which_solve in enumerate(solves):

                solve.trajectorylen = 0
                # If the gridworld is unsolvable, break -> moves onto next gridworld
                if which_solve(solve.getManhattanDistance) is None:
                    num_fail += 1
                    break
                results[solve_num][p_index] += solve.trajectorylen

        num_solv = cycles - num_fail

        # Average out times for each p
        for x in range(2):
            if num_solv != 0:
                results[x][p_index] /= num_solv

    # Set back to false
    checkfullgridworld = False

    # Plot results
    N = 3
    ind = np.arange(min(len(results[0]), len(results[1])))
    width = 0.30

    xvals = results[0]
    bar1 = plt.bar(ind, xvals, width, color='r')

    yvals = results[1]
    bar2 = plt.bar(ind+width, yvals, width, color='g')

    plt.title('Density vs. Average Trajectory Length by Method')
    plt.xlabel('Density')
    plt.ylabel('Average Trajectory Length')

    # Make xticks list
    xtick_list = []
    for i, x in enumerate(range(start, end, step)):
        xtick_list.append(str(x/100))
    plt.xticks(ind+width, xtick_list)
    plt.legend((bar1, bar2), ('Regular', 'Backtracking'))
    plt.show()

