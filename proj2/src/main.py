import numpy as np
import matplotlib.pyplot as plt
import solve
import time

# agents = [solve.solve1, solve.solve2, solve.solve3, solve.solve4]
agents = [solve.solve1, solve.solve2, solve.solve3, solve.solve4test]


def density_v_trajectory_length():
    # Have to make bar
    """Density vs Total Trajectory Length for each agent
    """

    global agents

    # Initialize constants:
    curr_p = 0
    interval = 33/10
    trials_per_p = 40

    # Initialize results matrix - range[2][5] = agent 3's runtime at p=.033*5=.165
    results = [[0 for _ in range(11)] for _ in range(4)]

    # For a range of [0,.33] p values, generate gridworlds
    # Total # of gridworlds = (# p values) * trials_per_p * num_agents = 11 * 40 * 4 = 1760
    for p_index in range(11):
        print(str(p_index) + "th P: " + str(curr_p))
        num_fail = 0

        # Create trials_per_p # of gridworlds
        for _ in range(trials_per_p):

            # Solve the gridworld with each agent
            for agent_num, agent in enumerate(agents):

                # Generate and solve new gridworld with current agent
                solve.trajectorylen = 0
                solve.generategridworld(101, curr_p)

                if agent() is None:
                    num_fail += 1
                    break
                else:
                    results[agent_num][p_index] += solve.trajectorylen

        # Average for each p
        num_success = trials_per_p - num_fail
        if num_success != 0:
            for x in range(4):
                results[x][p_index] /= num_success
        print(str(num_success) + " gridworlds succeeded for p = " + str(curr_p))
        curr_p += interval

    print(results)

    # Plot results
    N = 4
    ind = np.arange(11)
    width = 0.20

    bar1 = plt.bar(ind, results[0], width, color='r')
    bar2 = plt.bar(ind+width, results[1], width, color='g')
    bar3 = plt.bar(ind+width*2, results[2], width, color='b')
    bar4 = plt.bar(ind+width*3, results[3], width)

    plt.title('Density vs. Average Trajectory Length by Agent')
    plt.xlabel('Density')
    plt.ylabel('Average Trajectory Length')

    # Make xticks list
    xtick_list = []
    curr_p = 0
    for _ in range(11):
        xtick_list.append(str(curr_p))
        curr_p += interval
    plt.xticks(ind+width, xtick_list)
    plt.legend((bar1, bar2, bar3, bar4),
               ('Agent1', 'Agent2', 'Agent3', 'Agent4'))
    plt.show()


def density_v_avg1():
    """Density vs Average Length of Shortest Path in Final Discovered Gridworld
    """

    global agents

    # Initialize constants:
    curr_p = 0
    interval = 33/10
    trials_per_p = 40

    # Initialize results matrix - range[2][5] = agent 3's runtime at p=.033*5=.165
    results = [[0 for _ in range(11)] for _ in range(4)]

    # For a range of [0,.33] p values, generate gridworlds
    # Total # of gridworlds = (# p values) * trials_per_p * num_agents = 11 * 40 * 4 = 1760
    for p_index in range(11):
        print(str(p_index) + "th P: " + str(curr_p))
        num_fail = 0

        # Create trials_per_p # of gridworlds
        for _ in range(trials_per_p):

            # Solve the gridworld with each agent
            for agent_num, agent in enumerate(agents):

                # Generate and solve new gridworld with current agent
                solve.trajectorylen = 0
                solve.generategridworld(101, curr_p)

                if agent() is None:
                    num_fail += 1
                    break
                else:
                    path, pathlen = solve.astar(
                        solve.gridworld[0][0], agent_num)
                    results[agent_num][p_index] += pathlen

        # Average for each p
        num_success = trials_per_p - num_fail
        if num_success != 0:
            for x in range(4):
                results[x][p_index] /= num_success
        print(str(num_success) + "gridworlds succeeded for p = " + str(curr_p))
        curr_p += interval

    print(results)

    # Plot results
    N = 4
    ind = np.arange(11)
    width = 0.20

    bar1 = plt.bar(ind, results[0], width, color='r')
    bar2 = plt.bar(ind+width, results[1], width, color='g')
    bar3 = plt.bar(ind+width*2, results[2], width, color='b')
    bar4 = plt.bar(ind+width*3, results[3], width)

    plt.title(
        'Density vs. Average Path Length Through Discovered Gridworld by Agent')
    plt.xlabel('Density')
    plt.ylabel('Average Path Length')

    # Make xticks list
    xtick_list = []
    curr_p = 0
    for _ in range(11):
        xtick_list.append(str(curr_p))
        curr_p += interval
    plt.xticks(ind+width, xtick_list)
    plt.legend((bar1, bar2, bar3, bar4),
               ('Agent1', 'Agent2', 'Agent3', 'Agent4'))
    plt.show()


def density_v_runtime():
    """Density vs Total Runtime (total planning time listed in write up but idk how to time that) 
    """
    global agents
    # Initialize constants:
    #   range [start, end), difference, # of gridworlds per p
    #   trials_per_p - # of gridworlds per p, max_redos - # of unsolvable gridworlds allowed before breaking
    start = 0  # inclusive
    end = .33    # exclusive
    step = end/10
    trials_per_p = 40

    # Initialize results matrix - range[2][5] = agent 3's runtime at p=.033*5=.165
    results = [[0 for _ in range(10)] for _ in range(4)]

    # For a range of [0,.33] p values, generate gridworlds
    for p_index, p in enumerate(range(start, end, step)):
        print(str(p_index) + "th P: " + str(p))
        num_fail = 0

        # Keep making new gridworlds until desired # of solvable gridworlds are made
        for _ in range(trials_per_p):

            # Solve the gridworld with each agent
            for agent_num, agent in enumerate(agents):

                # Generate gridworld
                solve.generategridworld(101, p)

                # Time the solve
                start_time = time.time()

                # If the gridworld is unsolvable, break -> moves onto next gridworld
                if agent() is None:
                    num_fail += 1
                    break

                # Continues with the timer
                stop_time = time.time()
                results[agent_num][p_index] += stop_time - start_time

        # Average out times for each p
        num_success = trials_per_p - num_fail
        if num_success != 0:
            for x in range(4):
                results[x][p_index] /= num_success
        print(str(num_success) + "gridworlds succeeded for p = " + str(p))

    print(results)

    # Plot results
    N = 4
    ind = np.arange(11)
    width = 0.20

    bar1 = plt.bar(ind, results[0], width, color='r')
    bar2 = plt.bar(ind+width, results[1], width, color='g')
    bar3 = plt.bar(ind+width*2, results[2], width, color='b')
    bar4 = plt.bar(ind+width*3, results[3], width)

    plt.title('Density vs. Runtime by Agent')
    plt.xlabel('Density')
    plt.ylabel('Average Time (s)')

    # Make xticks list
    xtick_list = []
    for p in range(start, end, step):
        xtick_list.append(str(p))
    plt.xticks(ind+width, xtick_list)
    plt.legend((bar1, bar2, bar3, bar4),
               ('Agent1', 'Agent2', 'Agent3', 'Agent4'))
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

    solve.generategridworld(int(dim), float(p))

    starttime = time.time()
    result = solve.solve3()
    solve.printGridworld()
    endtime = time.time()
    if (result is None):
        print("No solution.")

    solve.trajectorylen = solve.trajectorylen if result is not None else None
    print("Trajectory length:", solve.trajectorylen)
    print("Cells processed: ", solve.numcellsprocessed)
    print("Runtime: ", endtime - starttime, "s")

    # density_v_trajectory_length()
    # density_v_avg1()
    # density_v_runtime()
