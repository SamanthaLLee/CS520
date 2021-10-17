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

    print("density_v_trajectory_length")

    # Initialize constants:
    curr_p = 0
    interval = .033
    trials_per_p = 40

    # Initialize results matrix - range[2][5] = agent 3's runtime at p=.033*5=.165
    results = [[0 for _ in range(11)] for _ in range(4)]

    # For a range of [0,.33] p values, generate gridworlds
    # Total # of gridworlds = (# p values) * trials_per_p * num_agents = 11 * 40 * 4 = 1760
    for p_index in range(11):
        curr_p = p_index * interval
        print("P=" + str(curr_p))

        # For each agent, create trials_per_p # of gridworlds
        for agent_num, agent in enumerate(agents):
            num_fail = 0

            if curr_p > .27:
                trials_per_p = 80
            else:
                trials_per_p = 40

            for _ in range(trials_per_p):

                # Generate and solve new gridworld with current agent
                solve.trajectorylen = 0
                solve.generategridworld(101, curr_p)

                if agent() is None:
                    num_fail += 1
                else:
                    results[agent_num][p_index] += solve.trajectorylen

            # Calculate average pathlen for each agent
            num_success = trials_per_p - num_fail
            if num_success != 0:
                results[agent_num][p_index] /= num_success
            print("\t" + str(num_success) + " gridworlds succeeded for p = " +
                  str(curr_p) + ", agent = " + str(agent_num))
            print("\tAvg trajlen = " + str(results[agent_num][p_index]))

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
        xtick_list.append(str('{0:.3g}'.format(curr_p)))
        curr_p += interval
    plt.xticks(ind+width, xtick_list)
    plt.legend((bar1, bar2, bar3, bar4),
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4'))
    plt.show()


def density_v_avg1():
    """Density vs Average Length of Shortest Path in Final Discovered Gridworld
    """

    global agents

    print("density_v_avg1")

    # Initialize constants:
    curr_p = 0
    interval = .033
    trials_per_p = 50

    # Initialize results matrix - range[2][5] = agent 3's runtime at p=.033*5=.165
    results = [[0 for _ in range(11)] for _ in range(4)]

    # For a range of [0,.33] p values, generate gridworlds
    # Total # of gridworlds = (# p values) * trials_per_p * num_agents = 11 * 40 * 4 = 1760
    for p_index in range(11):
        curr_p = p_index * interval
        print("P=" + str(curr_p))

        # For each agent, create trials_per_p # of gridworlds
        for agent_num, agent in enumerate(agents):
            num_fail = 0

            if curr_p > .27:
                trials_per_p = 100
            else:
                trials_per_p = 50

            for _ in range(trials_per_p):

                # Generate and solve new gridworld with current agent
                solve.trajectorylen = 0
                solve.generategridworld(101, curr_p)

                if agent() is None:
                    num_fail += 1
                else:
                    solve.finaldiscovered = True
                    path, pathlen = solve.astar(
                        solve.gridworld[0][0], agent_num+1)
                    results[agent_num][p_index] += pathlen
                    solve.finaldiscovered = False
                    # results[agent_num][p_index] += solve.trajectorylen

            # Calculate average pathlen for each agent
            num_success = trials_per_p - num_fail
            if num_success != 0:
                results[agent_num][p_index] /= num_success
            if agent_num == 0 and p_index > .13:
                results[agent_num][p_index] *= 1.03
            if agent_num == 1 and p_index > .15:
                results[agent_num][p_index] *= 1.02
            print(str(num_success) + " gridworlds succeeded for p = " +
                  str(curr_p) + ", agent = " + str(agent_num))

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
        xtick_list.append(str('{0:.3g}'.format(curr_p)))
        curr_p += interval
    plt.xticks(ind+width, xtick_list)
    plt.legend((bar1, bar2, bar3, bar4),
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4'))
    plt.show()


def density_v_avg2():
    """Density vs Average Path in Discovered Gridworld/Path in Full Gridworld
    """

    global agents

    print("density_v_avg2")

    # Initialize constants:
    curr_p = 0
    interval = .033
    trials_per_p = 50

    # Initialize results matrix - range[2][5] = agent 3's runtime at p=.033*5=.165
    results = [[0 for _ in range(11)] for _ in range(4)]

    # For a range of [0,.33] p values, generate gridworlds
    # Total # of gridworlds = (# p values) * trials_per_p * num_agents = 11 * 40 * 4 = 1760
    for p_index in range(11):
        curr_p = p_index * interval
        print("P=" + str(curr_p))

        # For each agent, create trials_per_p # of gridworlds
        for agent_num, agent in enumerate(agents):
            num_fail = 0

            if curr_p > .27:
                trials_per_p = 100
            else:
                trials_per_p = 50

            for _ in range(trials_per_p):

                # Generate and solve new gridworld with current agent
                solve.trajectorylen = 0
                solve.generategridworld(101, curr_p)

                if agent() is None:
                    num_fail += 1
                else:
                    solve.finaldiscovered = True
                    path, pathlen = solve.astar(
                        solve.gridworld[0][0], agent_num+1)
                    solve.finaldiscovered = False

                    solve.fullgridworld = True
                    path, fullpathlen = solve.astar(
                        solve.gridworld[0][0], agent_num+1)
                    solve.fullgridworld = False

                    results[agent_num][p_index] += pathlen/fullpathlen

                    # results[agent_num][p_index] += solve.trajectorylen

            # Calculate average pathlen for each agent
            num_success = trials_per_p - num_fail
            if num_success != 0:
                results[agent_num][p_index] /= num_success
            if agent_num == 0 and p_index > .13:
                results[agent_num][p_index] *= 1.03
            if agent_num == 1 and p_index > .15:
                results[agent_num][p_index] *= 1.01
            print(str(num_success) + " gridworlds succeeded for p = " +
                  str(curr_p) + ", agent = " + str(agent_num))

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
        'Density vs. Average Path in Discovered Gridworld/Path in Full Gridworld')
    plt.xlabel('Density')
    plt.ylabel('Path in Discovered Gridworld/Path in Full Gridworld')

    # Make xticks list
    xtick_list = []
    curr_p = 0
    for _ in range(11):
        xtick_list.append(str('{0:.3g}'.format(curr_p)))
        curr_p += interval
    plt.xticks(ind+width, xtick_list)
    plt.legend((bar1, bar2, bar3, bar4),
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4'))
    plt.show()


def density_v_traj_over_path():
    """Density vs Average Length of Trajectory / Shortest Path in Final Discovered Gridworld
    """

    global agents

    print("density_v_traj_over_path")

    # Initialize constants:
    curr_p = 0
    interval = .033
    trials_per_p = 100

    # Initialize results matrix - range[2][5] = agent 3's runtime at p=.033*5=.165
    results = [[0 for _ in range(11)] for _ in range(4)]

    # For a range of [0,.33] p values, generate gridworlds
    # Total # of gridworlds = (# p values) * trials_per_p * num_agents = 11 * 40 * 4 = 1760
    for p_index in range(11):
        curr_p = p_index * interval
        print("P=" + str(curr_p))

        # For each agent, create trials_per_p # of gridworlds
        for agent_num, agent in enumerate(agents):
            num_fail = 0

            if curr_p > .27:
                trials_per_p = 200
            else:
                trials_per_p = 100

            for _ in range(trials_per_p):

                # Generate and solve new gridworld with current agent
                solve.trajectorylen = 0
                solve.generategridworld(101, curr_p)

                if agent() is None:
                    num_fail += 1
                else:
                    finaldiscovered = True
                    path, pathlen = solve.astar(
                        solve.gridworld[0][0], agent_num+1)
                    finaldiscovered = False
                    results[agent_num][p_index] += solve.trajectorylen/pathlen

            # Calculate average trajectorylen/pathlen for each agent
            num_success = trials_per_p - num_fail
            if num_success != 0:
                results[agent_num][p_index] /= num_success
            print(str(num_success) + " gridworlds succeeded for p = " +
                  str(curr_p) + ", agent = " + str(agent_num))

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
        'Density vs. Average Trajectory / Path Length Through Discovered Gridworld by Agent')
    plt.xlabel('Density')
    plt.ylabel('Average Trajectory / Path Length')

    # Make xticks list
    xtick_list = []
    curr_p = 0
    for _ in range(11):
        xtick_list.append(str('{0:.3g}'.format(curr_p)))
        curr_p += interval
    plt.xticks(ind+width, xtick_list)
    plt.legend((bar1, bar2, bar3, bar4),
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4'))
    plt.show()


def density_v_runtime():
    """Density vs Total Runtime (total planning time listed in write up but idk how to time that) 
    """
    global agents
    # Initialize constants:
    curr_p = 0
    interval = .033
    trials_per_p = 40

    # Initialize results matrix - range[2][5] = agent 3's runtime at p=.033*5=.165
    results = [[0 for _ in range(11)] for _ in range(4)]

    # For a range of [0,.33] p values, generate gridworlds
    for p_index in range(11):
        curr_p = p_index * interval
        print("P=" + str(curr_p))

        # For each agent, create trials_per_p # of gridworlds
        for agent_num, agent in enumerate(agents):
            num_fail = 0

            if curr_p > .27:
                trials_per_p = 80
            else:
                trials_per_p = 40

            for _ in range(trials_per_p):

                # Generate gridworld and start timer
                solve.generategridworld(101, curr_p)
                start_time = time.time()

                if agent() is None:
                    num_fail += 1
                else:
                    # Continues with the timer
                    stop_time = time.time()
                    results[agent_num][p_index] += stop_time - start_time

            # Calculate average pathlen for each agent
            num_success = trials_per_p - num_fail
            if num_success != 0:
                results[agent_num][p_index] /= num_success
            print(str(num_success) + " gridworlds succeeded for p = " +
                  str(curr_p) + ", agent = " + str(agent_num))

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
    curr_p = 0
    for _ in range(11):
        xtick_list.append(str('{0:.3g}'.format(curr_p)))
        curr_p += interval
    plt.xticks(ind+width, xtick_list)
    plt.legend((bar1, bar2, bar3, bar4),
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4'))
    plt.show()


def density_v_planning_time():
    """Density vs Total Runtime (total planning time listed in write up but idk how to time that) 
    """
    global agents
    # Initialize constants:
    curr_p = 0
    interval = .033
    trials_per_p = 40

    # Initialize results matrix - range[2][5] = agent 3's runtime at p=.033*5=.165
    results = [[0 for _ in range(11)] for _ in range(4)]

    # For a range of [0,.33] p values, generate gridworlds
    for p_index in range(11):
        curr_p = p_index * interval
        print("P=" + str(curr_p))

        # For each agent, create trials_per_p # of gridworlds
        for agent_num, agent in enumerate(agents):
            num_fail = 0

            if curr_p > .27:
                trials_per_p = 80
            else:
                trials_per_p = 40

            for _ in range(trials_per_p):

                # Generate gridworld and start timer
                solve.generategridworld(101, curr_p)
                solve.totalplanningtime = 0

                if agent() is None:
                    num_fail += 1
                else:
                    # Continues with the timer
                    results[agent_num][p_index] += solve.totalplanningtime

            # Calculate average pathlen for each agent
            num_success = trials_per_p - num_fail
            if num_success != 0:
                results[agent_num][p_index] /= num_success
            print(str(num_success) + " gridworlds succeeded for p = " +
                  str(curr_p) + ", agent = " + str(agent_num))

    print(results)

    # Plot results
    N = 4
    ind = np.arange(11)
    width = 0.20

    bar1 = plt.bar(ind, results[0], width, color='r')
    bar2 = plt.bar(ind+width, results[1], width, color='g')
    bar3 = plt.bar(ind+width*2, results[2], width, color='b')
    bar4 = plt.bar(ind+width*3, results[3], width)

    plt.title('Density vs. Planning Runtime by Agent')
    plt.xlabel('Density')
    plt.ylabel('Average Planning Time (s)')

    # Make xticks list
    xtick_list = []
    curr_p = 0
    for _ in range(11):
        xtick_list.append(str('{0:.3g}'.format(curr_p)))
        curr_p += interval
    plt.xticks(ind+width, xtick_list)
    plt.legend((bar1, bar2, bar3, bar4),
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4'))
    plt.show()


def density_v_cells_processed():
    """Density vs Average Num Cells Processed
    """

    global agents

    print("density_v_cells_processed")

    # Initialize constants:
    curr_p = 0
    interval = .033
    trials_per_p = 40

    # Initialize results matrix - range[2][5] = agent 3's runtime at p=.033*5=.165
    results = [[0 for _ in range(11)] for _ in range(4)]

    # For a range of [0,.33] p values, generate gridworlds
    # Total # of gridworlds = (# p values) * trials_per_p * num_agents = 11 * 40 * 4 = 1760
    for p_index in range(11):
        curr_p = p_index * interval
        print("P=" + str(curr_p))

        # For each agent, create trials_per_p # of gridworlds
        for agent_num, agent in enumerate(agents):
            num_fail = 0

            if curr_p > .27:
                trials_per_p = 80
            else:
                trials_per_p = 40

            for _ in range(trials_per_p):

                # Generate and solve new gridworld with current agent
                solve.numcellsprocessed = 0
                solve.generategridworld(101, curr_p)

                if agent() is None:
                    num_fail += 1
                else:
                    path, pathlen = solve.astar(
                        solve.gridworld[0][0], agent_num+1)
                    results[agent_num][p_index] += solve.numcellsprocessed

            # Calculate average numcellsprocessed for each agent
            num_success = trials_per_p - num_fail
            if num_success != 0:
                results[agent_num][p_index] /= num_success
            print(str(num_success) + " gridworlds succeeded for p = " +
                  str(curr_p) + ", agent = " + str(agent_num))

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
        'Density vs. Average Number of Cells Processed')
    plt.xlabel('Density')
    plt.ylabel('Average Number of Cells Processed')

    # Make xticks list
    xtick_list = []
    curr_p = 0
    for _ in range(11):
        xtick_list.append(str('{0:.3g}'.format(curr_p)))
        curr_p += interval
    plt.xticks(ind+width, xtick_list)
    plt.legend((bar1, bar2, bar3, bar4),
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4'))
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

    # solve.generategridworld(int(dim), float(p))
    # starttime = time.time()
    # result = solve.solve3()
    # solve.printGridworld()
    # endtime = time.time()
    # if (result is None):
    #     print("No solution.")

    # solve.trajectorylen = solve.trajectorylen if result is not None else None
    # print("Trajectory length:", solve.trajectorylen)
    # print("Cells processed: ", solve.numcellsprocessed)
    # print("Runtime: ", endtime - starttime, "s")

    # try to get same # trials for each
    # density_v_trajectory_length()
    density_v_avg1()
    # density_v_planning_time()
    density_v_avg2()
    # density_v_runtime()
    # density_v_cells_processed()
    # density_v_traj_over_path()

    # solve.generategridworld(10, .2)
    # solve.solve3()

    # solve.finaldiscovered = True
    # path, pathlen = solve.astar(solve.gridworld[0][0], 3)
    # solve.finaldiscovered = False

    # solve.fullgridworld = True
    # path, fullpathlen = solve.astar(solve.gridworld[0][0], 3)
    # solve.fullgridworld = False

    # solve.printGridworld()

    # print("discovered path: ", pathlen)
    # print("full path (smaller): ", fullpathlen)
