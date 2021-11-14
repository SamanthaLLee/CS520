import numpy as np
import matplotlib.pyplot as plt
import solve
import time
from terrain import Terrain

# agents = [solve.solve6, solve.solve7, solve.solve8]
agents = [solve.solve8]

actions_results = []
runtime_results = []
processed_results = []


def isfloat(str):
    """Determines whether a given string can be converted to float"""
    try:
        float(str)
        return True
    except ValueError:
        return False


def generate_all_graphs():
    global agents, actions_results, runtime_results, processed_results

    # Initialize constants:
    trials_per_agent = 100

    # Initialize results matrix - range[2][5] = agent 3's runtime at p=.033*5=.165
    actions_results = [0 for _ in range(3)]
    runtime_results = [0 for _ in range(3)]
    processed_results = [0 for _ in range(3)]

    # For each agent, create trials_per_p # of gridworlds
    for agent_num, agent in enumerate(agents):
        print(agent_num)
        num_fail = 0

        for _ in range(trials_per_agent):
            # Generate and solve new gridworld with current agent
            solve.goal = None
            solve.start = None
            solve.actions = 0
            solve.numcellsprocessed = 0
            solve.generategridworld(50)

            start_time = time.time()
            test = agent()
            stop_time = time.time()
            solve.printGridworld()

            if test is None:
                num_fail += 1
            else:
                runtime_results[agent_num] += stop_time - start_time
                actions_results[agent_num] += solve.actions
                processed_results[agent_num] += solve.numcellsprocessed

        # Calculate average pathlen for each agent
        num_success = trials_per_agent - num_fail
        if num_success != 0:
            actions_results[agent_num] /= num_success
            runtime_results[agent_num] /= num_success
            processed_results[agent_num] /= num_success

        print("\t" + str(num_success) +
              " gridworlds succeeded for agent = " + str(agent_num))


def plot_actions():
    global agents, actions_results
    # Initialize constants:
    # curr_p = 0

    # fig = plt.figure()
    # ax = fig.add_axes([0, 0, 1, 1])
    # langs = ['Agent 6', 'Agent 7', 'Agent 8']
    # ax.bar(langs, actions_results)
    # plt.show()

    print("actions", actions_results)

    # N = 4
    # ind = np.arange(1)
    # width = 0.20

    # bar1 = plt.bar(ind, actions_results[0], width, color='r')
    # bar2 = plt.bar(ind+width, actions_results[1], width, color='g')
    # bar3 = plt.bar(ind+width*2, actions_results[2], width, color='b')

    # plt.title('Density vs. Runtime by Agent')
    # plt.xlabel('Density')
    # plt.ylabel('Average Time (s)')
    # plt.legend((bar1, bar2, bar3),
    #            ('Agent 6', 'Agent 7', 'Agent 8'))
    # plt.show()


def plot_runtime():
    global agents, traj_results, avg1_results, avg2_results, planning_results, runtime_results, processed_results, traj_path_results
    # Initialize constants:
    curr_p = 0
    interval = .033

    N = 4
    ind = np.arange(11)
    width = 0.20

    print("runtime", runtime_results)

    # bar61 = plt.bar(ind, processed_results[0], width, color='r')
    # bar62 = plt.bar(ind+width, processed_results[1], width, color='g')
    # bar63 = plt.bar(ind+width*2, processed_results[2], width, color='b')
    # bar64 = plt.bar(ind+width*3, processed_results[3], width, color='c')
    # bar65 = plt.bar(ind+width*4, processed_results[4], width)

    # # Make xticks list
    # xtick_list = []
    # curr_p = 0
    # for _ in range(11):
    #     xtick_list.append(str('{0:.3g}'.format(curr_p)))
    #     curr_p += interval
    # plt.xticks(ind+width, xtick_list)

    # plt.title(
    #     'Density vs. Average Number of Cells Processed')
    # plt.xlabel('Density')
    # plt.ylabel('Average Number of Cells Processed')
    # plt.legend((bar61, bar62, bar63, bar64, bar65),
    #            ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4', 'Agent5'))
    # plt.show()


def plot_processed():
    global agents, traj_results, avg1_results, avg2_results, planning_results, runtime_results, processed_results, traj_path_results
    # Initialize constants:
    curr_p = 0
    interval = .033

    N = 4
    ind = np.arange(11)
    width = 0.20

    print("processed", processed_results)

    # bar71 = plt.bar(ind, traj_path_results[0], width, color='r')
    # bar72 = plt.bar(ind+width, traj_path_results[1], width, color='g')
    # bar73 = plt.bar(ind+width*2, traj_path_results[2], width, color='b')
    # bar74 = plt.bar(ind+width*3, traj_path_results[3], width, color='c')
    # bar75 = plt.bar(ind+width*4, traj_path_results[4], width)

    # # Make xticks list
    # xtick_list = []
    # curr_p = 0
    # for _ in range(11):
    #     xtick_list.append(str('{0:.3g}'.format(curr_p)))
    #     curr_p += interval
    # plt.xticks(ind+width, xtick_list)

    # plt.title(
    #     'Density vs. Average Trajectory / Path Length Through Discovered Gridworld by Agent')
    # plt.xlabel('Density')
    # plt.ylabel('Average Trajectory / Path Length')
    # plt.legend((bar71, bar72, bar73, bar74, bar75),
    #            ('Agent1 - Blindfolded', 'Agent2 - 4-Neighbor', 'Agent3', 'Agent4', 'Agent5'))
    # plt.show()


if __name__ == "__main__":

    dim = input("What is the length of your gridworld? ")
    while not dim.isdigit() or int(dim) < 2:
        dim = input("Enter a valid length. ")

    solve.generategridworld(int(dim))
    solve.printGridworld()
    starttime = time.time()
    result = solve.solve8()
    endtime = time.time()
    solve.printGridworld()
    if (result is None):
        print("No solution.")

    # # solve.trajectorylen = solve.trajectorylen if result is not None else None
    # # print("Trajectory length:", solve.trajectorylen)
    # print("Num actions: ", solve.actions)
    # print("Cells processed: ", solve.numcellsprocessed)
    # print("Runtime: ", endtime - starttime, "s")
    # print("Total planning time: ", solve.totalplanningtime)
    # print("Total maxcell time: ", solve.maxcelltime)
    # print("Total update time: ", solve.updateptime)
    # print("Total update find time: ", solve.updatepfindtime)

    # solve.printGridworld()

    # generate_all_graphs()
    # plot_actions()
    # plot_processed()
    # plot_runtime()
    # plot2()
    # plot3()
    # plot4()
    # plot5()
    # plot6()
    # plot7()
