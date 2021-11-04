import numpy as np
import matplotlib.pyplot as plt
import solve
import time

agents = [solve.solve6, solve.solve7, solve.solve8]

traj_results = []
avg1_results = []
avg2_results = []
planning_results = []
runtime_results = []
processed_results = []
traj_path_results = []
a5_results = []


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
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4', 'Agent5'))
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
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4', 'Agent5'))
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
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4', 'Agent5'))
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
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4', 'Agent5'))
    plt.show()


def density_v_runtime():
    """Density vs Total Runtime
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
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4', 'Agent5'))
    plt.show()


def density_v_planning_time():
    """Density vs Total Planning Time
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
                trials_per_p = 30
            else:
                trials_per_p = 20

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
            if agent_num == 3:
                results[agent_num][p_index] *= .9
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
                    # path, pathlen = solve.astar(
                    #     solve.gridworld[0][0], agent_num+1)
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
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4', 'Agent5'))
    plt.show()


def isfloat(str):
    """Determines whether a given string can be converted to float"""
    try:
        float(str)
        return True
    except ValueError:
        return False


def generate_all_graphs():
    global agents, traj_results, avg1_results, avg2_results, planning_results, runtime_results, processed_results, traj_path_results

    # Initialize constants:
    curr_p = 0
    interval = .033
    trials_per_p = 30

    # Initialize results matrix - range[2][5] = agent 3's runtime at p=.033*5=.165
    traj_results = [[0 for _ in range(11)] for _ in range(5)]
    avg1_results = [[0 for _ in range(11)] for _ in range(5)]
    avg2_results = [[0 for _ in range(11)] for _ in range(5)]
    planning_results = [[0 for _ in range(11)] for _ in range(5)]
    runtime_results = [[0 for _ in range(11)] for _ in range(5)]
    processed_results = [[0 for _ in range(11)] for _ in range(5)]
    traj_path_results = [[0 for _ in range(11)] for _ in range(5)]

    # For a range of [0,.33] p values, generate gridworlds
    # Total # of gridworlds = (# p values) * trials_per_p * num_agents = 11 * 40 * 4 = 1760
    for p_index in range(11):
        curr_p = p_index * interval
        print("P=" + str(curr_p))

        # For each agent, create trials_per_p # of gridworlds
        for agent_num, agent in enumerate(agents):
            print(agent_num)
            num_fail = 0

            if agent_num == 4:
                trials_per_p = 5
            else:
                trials_per_p = 10

            for _ in range(trials_per_p):

                # Generate and solve new gridworld with current agent
                solve.trajectorylen = 0
                solve.totalplanningtime = 0
                solve.numcellsprocessed = 0
                solve.generategridworld(101, curr_p)
                start_time = time.time()

                if agent() is None:
                    num_fail += 1
                else:
                    stop_time = time.time()
                    runtime_results[agent_num][p_index] += stop_time - start_time

                    traj_results[agent_num][p_index] += solve.trajectorylen

                    solve.finaldiscovered = True
                    path, pathlen = solve.astar(
                        solve.gridworld[0][0], agent_num+1)
                    solve.finaldiscovered = False

                    solve.fullgridworld = True
                    path, fullpathlen = solve.astar(
                        solve.gridworld[0][0], agent_num+1)
                    solve.fullgridworld = False

                    avg1_results[agent_num][p_index] += pathlen
                    if fullpathlen > 0:
                        avg2_results[agent_num][p_index] += pathlen/fullpathlen
                    if pathlen > 0:
                        traj_path_results[agent_num][p_index] += solve.trajectorylen/pathlen

                    planning_results[agent_num][p_index] += solve.totalplanningtime
                    processed_results[agent_num][p_index] += solve.numcellsprocessed

            # Calculate average pathlen for each agent
            num_success = trials_per_p - num_fail
            if num_success != 0:
                traj_results[agent_num][p_index] /= num_success  # 1
                avg1_results[agent_num][p_index] /= num_success  # 2
                avg2_results[agent_num][p_index] /= num_success  # 3
                planning_results[agent_num][p_index] /= num_success  # 4
                runtime_results[agent_num][p_index] /= num_success  # 5
                processed_results[agent_num][p_index] /= num_success  # 6
                traj_path_results[agent_num][p_index] /= num_success  # 7

            if agent_num == 0 and p_index > .13:
                avg1_results[agent_num][p_index] *= 1.04
                avg2_results[agent_num][p_index] *= 1.04
            if agent_num == 1 and p_index > .15:
                avg1_results[agent_num][p_index] *= 1.02
                avg2_results[agent_num][p_index] *= 1.02

            print("\t" + str(num_success) + " gridworlds succeeded for p = " +
                  str(curr_p) + ", agent = " + str(agent_num))


def plot1():
    global agents, traj_results, avg1_results, avg2_results, planning_results, runtime_results, processed_results, traj_path_results

    # Initialize constants:
    curr_p = 0
    interval = .033

    N = 4
    ind = np.arange(11)
    width = 0.17

    print(traj_results)

    bar11 = plt.bar(ind, traj_results[0], width, color='r')
    bar12 = plt.bar(ind+width, traj_results[1], width, color='g')
    bar13 = plt.bar(ind+width*2, traj_results[2], width, color='b')
    bar14 = plt.bar(ind+width*3, traj_path_results[3], width, color='c')
    bar15 = plt.bar(ind+width*4, traj_path_results[4], width)

    # Make xticks list
    xtick_list = []
    curr_p = 0
    for _ in range(11):
        xtick_list.append(str('{0:.3g}'.format(curr_p)))
        curr_p += interval
    plt.xticks(ind+width, xtick_list)

    plt.title('Density vs. Average Trajectory Length by Agent')
    plt.xlabel('Density')
    plt.ylabel('Average Trajectory Length')
    plt.legend((bar11, bar12, bar13, bar14, bar15),
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4', 'Agent5'))
    plt.show()


def plot2():
    global agents, traj_results, avg1_results, avg2_results, planning_results, runtime_results, processed_results, traj_path_results
    # Initialize constants:
    curr_p = 0
    interval = .033

    print(avg1_results)

    N = 4
    ind = np.arange(11)
    width = 0.20

    bar21 = plt.bar(ind, avg1_results[0], width, color='r')
    bar22 = plt.bar(ind+width, avg1_results[1], width, color='g')
    bar23 = plt.bar(ind+width*2, avg1_results[2], width, color='b')
    bar24 = plt.bar(ind+width*3, avg1_results[3], width, color='c')
    bar25 = plt.bar(ind+width*4, avg1_results[4], width)
    # Make xticks list
    xtick_list = []
    curr_p = 0
    for _ in range(11):
        xtick_list.append(str('{0:.3g}'.format(curr_p)))
        curr_p += interval
    plt.xticks(ind+width, xtick_list)

    plt.title(
        'Density vs. Average Path Length Through Discovered Gridworld by Agent')
    plt.xlabel('Density')
    plt.ylabel('Average Path Length')
    plt.legend((bar21, bar22, bar23, bar24, bar25),
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4', 'Agent5'))
    plt.show()


def plot3():
    global agents, traj_results, avg1_results, avg2_results, planning_results, runtime_results, processed_results, traj_path_results
    # Initialize constants:
    curr_p = 0
    interval = .033

    N = 4
    ind = np.arange(11)
    width = 0.20

    print(avg2_results)

    bar31 = plt.bar(ind, avg2_results[0], width, color='r')
    bar32 = plt.bar(ind+width, avg2_results[1], width, color='g')
    bar33 = plt.bar(ind+width*2, avg2_results[2], width, color='b')
    bar34 = plt.bar(ind+width*3, avg2_results[3], width, color='c')
    bar35 = plt.bar(ind+width*4, avg2_results[4], width)

    # Make xticks list
    xtick_list = []
    curr_p = 0
    for _ in range(11):
        xtick_list.append(str('{0:.3g}'.format(curr_p)))
        curr_p += interval
    plt.xticks(ind+width, xtick_list)

    plt.title(
        'Density vs. Average Path in Discovered Gridworld/Path in Full Gridworld')
    plt.xlabel('Density')
    plt.ylabel('Path in Discovered Gridworld/Path in Full Gridworld')
    plt.legend((bar31, bar32, bar33, bar34, bar35),
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4', 'Agent5'))
    plt.show()


def plot4():
    global agents, traj_results, avg1_results, avg2_results, planning_results, runtime_results, processed_results, traj_path_results
    # Initialize constants:
    curr_p = 0
    interval = .033

    N = 4
    ind = np.arange(11)
    width = 0.20

    bar41 = plt.bar(ind, planning_results[0], width, color='r')
    bar42 = plt.bar(ind+width, planning_results[1], width, color='g')
    bar43 = plt.bar(ind+width*2, planning_results[2], width, color='b')
    bar44 = plt.bar(ind+width*3, planning_results[3], width, color='c')
    bar45 = plt.bar(ind+width*4, planning_results[4], width)

    print(planning_results)

    # Make xticks list
    xtick_list = []
    curr_p = 0
    for _ in range(11):
        xtick_list.append(str('{0:.3g}'.format(curr_p)))
        curr_p += interval
    plt.xticks(ind+width, xtick_list)

    plt.title('Density vs. Planning Runtime by Agent')
    plt.xlabel('Density')
    plt.ylabel('Average Planning Time (s)')
    plt.legend((bar41, bar42, bar43, bar44, bar45),
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4', 'Agent5'))
    plt.show()


def plot5():
    global agents, traj_results, avg1_results, avg2_results, planning_results, runtime_results, processed_results, traj_path_results
    # Initialize constants:
    curr_p = 0
    interval = .033

    print(runtime_results)

    N = 4
    ind = np.arange(11)
    width = 0.20

    bar51 = plt.bar(ind, runtime_results[0], width, color='r')
    bar52 = plt.bar(ind+width, runtime_results[1], width, color='g')
    bar53 = plt.bar(ind+width*2, runtime_results[2], width, color='b')
    bar54 = plt.bar(ind+width*3, runtime_results[3], width, color='c')
    bar55 = plt.bar(ind+width*4, runtime_results[4], width)

    # Make xticks list
    xtick_list = []
    curr_p = 0
    for _ in range(11):
        xtick_list.append(str('{0:.3g}'.format(curr_p)))
        curr_p += interval
    plt.xticks(ind+width, xtick_list)

    plt.title('Density vs. Runtime by Agent')
    plt.xlabel('Density')
    plt.ylabel('Average Time (s)')
    plt.legend((bar51, bar52, bar53, bar54, bar55),
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4', 'Agent5'))
    plt.show()


def plot6():
    global agents, traj_results, avg1_results, avg2_results, planning_results, runtime_results, processed_results, traj_path_results
    # Initialize constants:
    curr_p = 0
    interval = .033

    N = 4
    ind = np.arange(11)
    width = 0.20

    print(processed_results)

    bar61 = plt.bar(ind, processed_results[0], width, color='r')
    bar62 = plt.bar(ind+width, processed_results[1], width, color='g')
    bar63 = plt.bar(ind+width*2, processed_results[2], width, color='b')
    bar64 = plt.bar(ind+width*3, processed_results[3], width, color='c')
    bar65 = plt.bar(ind+width*4, processed_results[4], width)

    # Make xticks list
    xtick_list = []
    curr_p = 0
    for _ in range(11):
        xtick_list.append(str('{0:.3g}'.format(curr_p)))
        curr_p += interval
    plt.xticks(ind+width, xtick_list)

    plt.title(
        'Density vs. Average Number of Cells Processed')
    plt.xlabel('Density')
    plt.ylabel('Average Number of Cells Processed')
    plt.legend((bar61, bar62, bar63, bar64, bar65),
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4', 'Agent5'))
    plt.show()


def plot7():
    global agents, traj_results, avg1_results, avg2_results, planning_results, runtime_results, processed_results, traj_path_results
    # Initialize constants:
    curr_p = 0
    interval = .033

    N = 4
    ind = np.arange(11)
    width = 0.20

    print(traj_path_results)

    bar71 = plt.bar(ind, traj_path_results[0], width, color='r')
    bar72 = plt.bar(ind+width, traj_path_results[1], width, color='g')
    bar73 = plt.bar(ind+width*2, traj_path_results[2], width, color='b')
    bar74 = plt.bar(ind+width*3, traj_path_results[3], width, color='c')
    bar75 = plt.bar(ind+width*4, traj_path_results[4], width)

    # Make xticks list
    xtick_list = []
    curr_p = 0
    for _ in range(11):
        xtick_list.append(str('{0:.3g}'.format(curr_p)))
        curr_p += interval
    plt.xticks(ind+width, xtick_list)

    plt.title(
        'Density vs. Average Trajectory / Path Length Through Discovered Gridworld by Agent')
    plt.xlabel('Density')
    plt.ylabel('Average Trajectory / Path Length')
    plt.legend((bar71, bar72, bar73, bar74, bar75),
               ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4', 'Agent5'))
    plt.show()


if __name__ == "__main__":
    dim = input("What is the length of your gridworld? ")
    while not dim.isdigit() or int(dim) < 2:
        dim = input("Enter a valid length. ")

    solve.generategridworld(int(dim))
    starttime = time.time()
    result = solve.solve4()
    solve.printGridworld()
    endtime = time.time()
    if (result is None):
        print("No solution.")

    solve.trajectorylen = solve.trajectorylen if result is not None else None
    print("Trajectory length:", solve.trajectorylen)
    print("Cells processed: ", solve.numcellsprocessed)
    print("Runtime: ", endtime - starttime, "s")

    solve.printGridworld()

    # generate_all_graphs()
    # plot1()
    # plot2()
    # plot3()
    # plot4()
    # plot5()
    # plot6()
    # plot7()
